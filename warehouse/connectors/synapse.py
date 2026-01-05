"""
Azure Synapse Analytics Connector

Specialized connector for Azure Synapse Analytics with:
- SQL pool support
- Spark pool support
- Azure AD authentication
- Managed identity support
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from .base import (
    WarehouseConnector,
    WarehouseConnectionError,
    WarehouseQueryError,
    WarehouseInfo,
    TableInfo,
    QueryResult
)

logger = logging.getLogger(__name__)


class SynapseConnector(WarehouseConnector):
    """
    Connector for Azure Synapse Analytics.

    Supports:
    - Dedicated SQL pools
    - Serverless SQL pools
    - SQL authentication
    - Azure AD authentication
    - Managed identity authentication
    """

    def __init__(
        self,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        schema: Optional[str] = None,
        authentication: Optional[str] = None,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        driver: Optional[str] = None,
        connection_timeout: int = 30,
        query_timeout: int = 300,
        pool_size: int = 5,
        max_overflow: int = 10
    ):
        super().__init__(connection_timeout, query_timeout, pool_size, max_overflow)

        # Load from environment if not provided
        self.server = server or os.getenv("SYNAPSE_SERVER")
        self.database = database or os.getenv("SYNAPSE_DATABASE")
        self.username = username or os.getenv("SYNAPSE_USERNAME") or os.getenv("SYNAPSE_USER")
        self.password = password or os.getenv("SYNAPSE_PASSWORD")
        self.schema = schema or os.getenv("SYNAPSE_SCHEMA", "dbo")
        self.authentication = authentication or os.getenv("SYNAPSE_AUTHENTICATION", "sql")  # sql, ad, msi
        self.tenant_id = tenant_id or os.getenv("SYNAPSE_TENANT_ID") or os.getenv("AZURE_TENANT_ID")
        self.client_id = client_id or os.getenv("SYNAPSE_CLIENT_ID") or os.getenv("AZURE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SYNAPSE_CLIENT_SECRET") or os.getenv("AZURE_CLIENT_SECRET")
        self.driver = driver or os.getenv("SYNAPSE_DRIVER", "ODBC Driver 17 for SQL Server")

        self._cursor = None

    @property
    def warehouse_type(self) -> str:
        return "synapse"

    def _get_azure_ad_token(self) -> str:
        """Get Azure AD access token for authentication."""
        try:
            from azure.identity import ClientSecretCredential
        except ImportError:
            raise ImportError(
                "azure-identity required for AD authentication: pip install azure-identity"
            )

        if not all([self.tenant_id, self.client_id, self.client_secret]):
            raise WarehouseConnectionError(
                "Azure AD authentication requires tenant_id, client_id, and client_secret"
            )

        credential = ClientSecretCredential(
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret
        )

        # Get token for Azure SQL Database resource
        token = credential.get_token("https://database.windows.net/.default")
        return token.token

    def _get_connection_string(self) -> str:
        """Build the ODBC connection string."""
        parts = [
            f"DRIVER={{{self.driver}}}",
            f"SERVER={self.server}",
            f"DATABASE={self.database}",
            "Encrypt=yes",
            "TrustServerCertificate=no",
            f"Connection Timeout={self.connection_timeout}"
        ]

        if self.authentication == "sql":
            parts.extend([
                f"UID={self.username}",
                f"PWD={self.password}"
            ])
        elif self.authentication == "ad":
            parts.append("Authentication=ActiveDirectoryPassword")
            parts.extend([
                f"UID={self.username}",
                f"PWD={self.password}"
            ])
        elif self.authentication == "msi":
            parts.append("Authentication=ActiveDirectoryMsi")
        elif self.authentication == "service_principal":
            # Use access token
            pass

        return ";".join(parts)

    def connect(self) -> None:
        """Establish connection to Synapse."""
        if self._connected:
            return

        try:
            import pyodbc
        except ImportError:
            raise ImportError("pyodbc required: pip install pyodbc")

        if not self.server:
            raise WarehouseConnectionError("Synapse server is required")

        try:
            if self.authentication == "service_principal":
                # Use access token authentication
                token = self._get_azure_ad_token()
                connection_string = self._get_connection_string()

                # Create connection with access token
                import struct
                token_bytes = token.encode("utf-16-le")
                token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)

                self._connection = pyodbc.connect(
                    connection_string,
                    attrs_before={1256: token_struct}  # SQL_COPT_SS_ACCESS_TOKEN
                )
            else:
                connection_string = self._get_connection_string()
                self._connection = pyodbc.connect(connection_string)

            self._connection.autocommit = True
            self._cursor = self._connection.cursor()
            self._connected = True
            logger.info(f"Connected to Synapse: {self.server}/{self.database}")

        except Exception as e:
            raise WarehouseConnectionError(f"Failed to connect to Synapse: {str(e)}")

    def disconnect(self) -> None:
        """Close the Synapse connection."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._connection:
            self._connection.close()
            self._connection = None
        self._connected = False
        logger.info("Disconnected from Synapse")

    def test_connection(self) -> bool:
        """Test if the connection is valid."""
        try:
            if not self._connected:
                self.connect()
            self._cursor.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def get_info(self) -> WarehouseInfo:
        """Get information about the Synapse connection."""
        if not self._connected:
            self.connect()

        self._cursor.execute("SELECT @@VERSION")
        version = self._cursor.fetchone()[0]

        self._cursor.execute("SELECT DB_NAME(), SCHEMA_NAME()")
        row = self._cursor.fetchone()

        return WarehouseInfo(
            warehouse_type=self.warehouse_type,
            name=self.server,
            host=self.server,
            database=row[0],
            schema=row[1],
            connected=True,
            version=version,
            metadata={
                "authentication": self.authentication
            }
        )

    def list_databases(self) -> List[str]:
        """List all accessible databases."""
        if not self._connected:
            self.connect()

        self._cursor.execute("""
            SELECT name FROM sys.databases
            WHERE state = 0
            ORDER BY name
        """)
        return [row[0] for row in self._cursor.fetchall()]

    def list_schemas(self, database: Optional[str] = None) -> List[str]:
        """List all schemas in a database."""
        if not self._connected:
            self.connect()

        self._cursor.execute("""
            SELECT schema_name FROM information_schema.schemata
            WHERE schema_name NOT IN ('sys', 'INFORMATION_SCHEMA', 'guest')
            ORDER BY schema_name
        """)
        return [row[0] for row in self._cursor.fetchall()]

    def list_tables(
        self,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        include_views: bool = True
    ) -> List[TableInfo]:
        """List all tables in a schema."""
        if not self._connected:
            self.connect()

        sch = schema or self.schema

        table_types = "'BASE TABLE'"
        if include_views:
            table_types += ", 'VIEW'"

        self._cursor.execute(f"""
            SELECT table_name, table_schema, table_type
            FROM information_schema.tables
            WHERE table_schema = ?
            AND table_type IN ({table_types})
            ORDER BY table_name
        """, (sch,))

        tables = []
        for row in self._cursor.fetchall():
            tables.append(TableInfo(
                name=row[0],
                schema=row[1],
                database=self.database,
                table_type="VIEW" if row[2] == "VIEW" else "TABLE"
            ))

        return tables

    def get_table_info(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> TableInfo:
        """Get detailed information about a table."""
        if not self._connected:
            self.connect()

        sch = schema or self.schema
        full_name = f"[{sch}].[{self._validate_identifier(table_name)}]"

        # Get row count (approximate for large tables in Synapse)
        try:
            self._cursor.execute(f"""
                SELECT SUM(row_count) as row_count
                FROM sys.dm_pdw_nodes_db_partition_stats
                WHERE object_id = OBJECT_ID('{sch}.{table_name}')
                AND index_id < 2
            """)
            result = self._cursor.fetchone()
            row_count = result[0] if result and result[0] else 0
        except Exception:
            # Fallback for serverless pool
            self._cursor.execute(f"SELECT COUNT(*) FROM {full_name}")
            row_count = self._cursor.fetchone()[0]

        columns = self.get_table_schema(table_name, schema)

        return TableInfo(
            name=table_name,
            schema=sch,
            database=self.database,
            row_count=row_count,
            columns=columns,
            table_type="TABLE"
        )

    def get_table_schema(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get table column schema."""
        if not self._connected:
            self.connect()

        sch = schema or self.schema

        self._cursor.execute("""
            SELECT
                c.COLUMN_NAME,
                c.DATA_TYPE,
                c.IS_NULLABLE,
                c.COLUMN_DEFAULT,
                CASE WHEN pk.COLUMN_NAME IS NOT NULL THEN 1 ELSE 0 END as IS_PRIMARY_KEY
            FROM INFORMATION_SCHEMA.COLUMNS c
            LEFT JOIN (
                SELECT ku.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku
                    ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
                WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                AND tc.TABLE_SCHEMA = ?
                AND tc.TABLE_NAME = ?
            ) pk ON c.COLUMN_NAME = pk.COLUMN_NAME
            WHERE c.TABLE_SCHEMA = ? AND c.TABLE_NAME = ?
            ORDER BY c.ORDINAL_POSITION
        """, (sch, table_name, sch, table_name))

        columns = []
        for row in self._cursor.fetchall():
            columns.append({
                "name": row[0],
                "type": row[1],
                "nullable": row[2] == "YES",
                "default": row[3],
                "primary_key": bool(row[4]),
                "comment": None
            })

        return columns

    def read_sample(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        n_rows: int = 1000
    ) -> QueryResult:
        """Read a sample of rows from a table."""
        sch = schema or self.schema
        full_name = f"[{self._validate_identifier(sch)}].[{self._validate_identifier(table_name)}]"

        query = f"SELECT TOP {n_rows} * FROM {full_name}"
        return self.execute_query(query, max_rows=n_rows)

    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        max_rows: Optional[int] = None
    ) -> QueryResult:
        """Execute a SQL query and return results."""
        if not self._connected:
            self.connect()

        start_time = time.time()

        try:
            if parameters:
                # Convert dict parameters to tuple for pyodbc
                param_values = tuple(parameters.values())
                self._cursor.execute(query, param_values)
            else:
                self._cursor.execute(query)

            columns = [desc[0] for desc in self._cursor.description] if self._cursor.description else []

            if max_rows:
                rows = self._cursor.fetchmany(max_rows)
                truncated = len(rows) == max_rows
            else:
                rows = self._cursor.fetchall()
                truncated = False

            execution_time = (time.time() - start_time) * 1000

            # Convert to list of dicts
            row_dicts = [dict(zip(columns, row)) for row in rows]

            return QueryResult(
                columns=columns,
                rows=row_dicts,
                row_count=len(row_dicts),
                execution_time_ms=execution_time,
                truncated=truncated
            )

        except Exception as e:
            raise WarehouseQueryError(f"Query execution failed: {str(e)}")

    def get_row_count(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> int:
        """Get the row count for a table."""
        if not self._connected:
            self.connect()

        sch = schema or self.schema

        # Try to get from system views first (faster for large tables)
        try:
            self._cursor.execute(f"""
                SELECT SUM(row_count) as row_count
                FROM sys.dm_pdw_nodes_db_partition_stats
                WHERE object_id = OBJECT_ID('{sch}.{table_name}')
                AND index_id < 2
            """)
            result = self._cursor.fetchone()
            if result and result[0]:
                return int(result[0])
        except Exception:
            pass

        # Fallback to COUNT(*)
        full_name = f"[{self._validate_identifier(sch)}].[{self._validate_identifier(table_name)}]"
        self._cursor.execute(f"SELECT COUNT(*) FROM {full_name}")
        return self._cursor.fetchone()[0]

    def get_table_statistics(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get table statistics."""
        if not self._connected:
            self.connect()

        sch = schema or self.schema
        full_name = f"[{self._validate_identifier(sch)}].[{self._validate_identifier(table_name)}]"

        # Get table size and distribution info
        try:
            self._cursor.execute(f"""
                SELECT
                    SUM(row_count) as row_count,
                    SUM(used_page_count) * 8 / 1024 as size_mb
                FROM sys.dm_pdw_nodes_db_partition_stats
                WHERE object_id = OBJECT_ID('{sch}.{table_name}')
                AND index_id < 2
            """)
            table_stats = self._cursor.fetchone()
            row_count = int(table_stats[0]) if table_stats and table_stats[0] else 0
            size_mb = table_stats[1] if table_stats else None
        except Exception:
            row_count = self.get_row_count(table_name, schema)
            size_mb = None

        stats = {
            "table_name": full_name,
            "row_count": row_count,
            "size_mb": size_mb,
            "column_stats": {}
        }

        # Get column statistics
        columns = self.get_table_schema(table_name, schema)

        for col in columns:
            col_name = col["name"]
            col_type = col["type"].upper()

            try:
                # Get null count and distinct count
                self._cursor.execute(f"""
                    SELECT
                        COUNT(*) - COUNT([{col_name}]) as null_count,
                        COUNT(DISTINCT [{col_name}]) as distinct_count
                    FROM {full_name}
                """)
                row = self._cursor.fetchone()

                col_stats = {
                    "null_count": row[0],
                    "distinct_count": row[1],
                    "data_type": col_type
                }

                # Get min/max for numeric and date types
                if any(t in col_type for t in ["INT", "FLOAT", "NUMERIC", "DECIMAL", "DATE", "TIME", "MONEY", "REAL"]):
                    self._cursor.execute(f'SELECT MIN([{col_name}]), MAX([{col_name}]) FROM {full_name}')
                    minmax = self._cursor.fetchone()
                    col_stats["min"] = str(minmax[0]) if minmax[0] is not None else None
                    col_stats["max"] = str(minmax[1]) if minmax[1] is not None else None

                stats["column_stats"][col_name] = col_stats

            except Exception as e:
                logger.warning(f"Failed to get stats for column {col_name}: {str(e)}")
                stats["column_stats"][col_name] = {"error": str(e)}

        return stats

    def get_distribution_info(self, table_name: str, schema: Optional[str] = None) -> Dict[str, Any]:
        """Get table distribution information (Synapse-specific)."""
        if not self._connected:
            self.connect()

        sch = schema or self.schema

        try:
            self._cursor.execute(f"""
                SELECT
                    distribution_policy_desc,
                    distribution_ordinal
                FROM sys.pdw_table_distribution_properties
                WHERE object_id = OBJECT_ID('{sch}.{table_name}')
            """)
            dist = self._cursor.fetchone()

            if dist:
                return {
                    "distribution_type": dist[0],
                    "distribution_ordinal": dist[1]
                }
        except Exception:
            pass

        return {"distribution_type": "UNKNOWN"}

    def get_query_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent query history."""
        if not self._connected:
            self.connect()

        try:
            self._cursor.execute(f"""
                SELECT TOP {limit}
                    request_id,
                    status,
                    submit_time,
                    start_time,
                    end_time,
                    total_elapsed_time,
                    command
                FROM sys.dm_pdw_exec_requests
                WHERE status NOT IN ('Running', 'Queued')
                ORDER BY submit_time DESC
            """)

            return [
                {
                    "request_id": row[0],
                    "status": row[1],
                    "submit_time": row[2].isoformat() if row[2] else None,
                    "start_time": row[3].isoformat() if row[3] else None,
                    "end_time": row[4].isoformat() if row[4] else None,
                    "elapsed_ms": row[5],
                    "command": row[6][:500] if row[6] else None
                }
                for row in self._cursor.fetchall()
            ]
        except Exception:
            # Fallback for serverless pool
            return []
