"""
Snowflake Connector

Specialized connector for Snowflake Data Cloud with:
- Native Snowflake driver support
- SQLAlchemy integration
- Warehouse/role management
- Query profiling
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


class SnowflakeConnector(WarehouseConnector):
    """
    Connector for Snowflake Data Cloud.

    Supports:
    - Account-based authentication
    - Key-pair authentication
    - OAuth authentication
    - Role and warehouse switching
    """

    def __init__(
        self,
        account: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        warehouse: Optional[str] = None,
        role: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key_passphrase: Optional[str] = None,
        authenticator: Optional[str] = None,
        connection_timeout: int = 30,
        query_timeout: int = 300,
        pool_size: int = 5,
        max_overflow: int = 10
    ):
        super().__init__(connection_timeout, query_timeout, pool_size, max_overflow)

        # Load from environment if not provided
        self.account = account or os.getenv("SNOWFLAKE_ACCOUNT")
        self.username = username or os.getenv("SNOWFLAKE_USERNAME") or os.getenv("SNOWFLAKE_USER")
        self.password = password or os.getenv("SNOWFLAKE_PASSWORD")
        self.database = database or os.getenv("SNOWFLAKE_DATABASE")
        self.schema = schema or os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC")
        self.warehouse = warehouse or os.getenv("SNOWFLAKE_WAREHOUSE")
        self.role = role or os.getenv("SNOWFLAKE_ROLE")
        self.private_key_path = private_key_path or os.getenv("SNOWFLAKE_PRIVATE_KEY_PATH")
        self.private_key_passphrase = private_key_passphrase or os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
        self.authenticator = authenticator or os.getenv("SNOWFLAKE_AUTHENTICATOR")

        self._cursor = None

    @property
    def warehouse_type(self) -> str:
        return "snowflake"

    def connect(self) -> None:
        """Establish connection to Snowflake."""
        if self._connected:
            return

        try:
            import snowflake.connector
        except ImportError:
            raise ImportError(
                "snowflake-connector-python required: pip install snowflake-connector-python"
            )

        if not self.account:
            raise WarehouseConnectionError("Snowflake account is required")

        connect_params = {
            "account": self.account,
            "user": self.username,
            "database": self.database,
            "schema": self.schema,
            "warehouse": self.warehouse,
            "login_timeout": self.connection_timeout,
            "network_timeout": self.query_timeout,
        }

        # Authentication method
        if self.private_key_path:
            # Key-pair authentication
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization

            with open(self.private_key_path, "rb") as key_file:
                p_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=self.private_key_passphrase.encode() if self.private_key_passphrase else None,
                    backend=default_backend()
                )
            connect_params["private_key"] = p_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        elif self.authenticator:
            connect_params["authenticator"] = self.authenticator
            if self.authenticator != "externalbrowser":
                connect_params["password"] = self.password
        else:
            connect_params["password"] = self.password

        if self.role:
            connect_params["role"] = self.role

        try:
            self._connection = snowflake.connector.connect(**connect_params)
            self._cursor = self._connection.cursor()
            self._connected = True
            logger.info(f"Connected to Snowflake account: {self.account}")
        except Exception as e:
            raise WarehouseConnectionError(f"Failed to connect to Snowflake: {str(e)}")

    def disconnect(self) -> None:
        """Close the Snowflake connection."""
        if self._cursor:
            self._cursor.close()
            self._cursor = None
        if self._connection:
            self._connection.close()
            self._connection = None
        self._connected = False
        logger.info("Disconnected from Snowflake")

    def test_connection(self) -> bool:
        """Test if the connection is valid."""
        try:
            if not self._connected:
                self.connect()
            self._cursor.execute("SELECT CURRENT_VERSION()")
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False

    def get_info(self) -> WarehouseInfo:
        """Get information about the Snowflake connection."""
        if not self._connected:
            self.connect()

        self._cursor.execute("SELECT CURRENT_VERSION(), CURRENT_ACCOUNT(), CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_WAREHOUSE(), CURRENT_ROLE()")
        row = self._cursor.fetchone()

        return WarehouseInfo(
            warehouse_type=self.warehouse_type,
            name=self.account,
            host=f"{self.account}.snowflakecomputing.com",
            database=row[2],
            schema=row[3],
            connected=True,
            version=row[0],
            metadata={
                "account": row[1],
                "warehouse": row[4],
                "role": row[5]
            }
        )

    def list_databases(self) -> List[str]:
        """List all accessible databases."""
        if not self._connected:
            self.connect()

        self._cursor.execute("SHOW DATABASES")
        return [row[1] for row in self._cursor.fetchall()]

    def list_schemas(self, database: Optional[str] = None) -> List[str]:
        """List all schemas in a database."""
        if not self._connected:
            self.connect()

        db = database or self.database
        if db:
            self._cursor.execute(f"SHOW SCHEMAS IN DATABASE {self._validate_identifier(db)}")
        else:
            self._cursor.execute("SHOW SCHEMAS")

        return [row[1] for row in self._cursor.fetchall()]

    def list_tables(
        self,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        include_views: bool = True
    ) -> List[TableInfo]:
        """List all tables in a schema."""
        if not self._connected:
            self.connect()

        db = database or self.database
        sch = schema or self.schema

        query = "SHOW TABLES"
        if sch:
            query += f" IN SCHEMA {self._validate_identifier(db)}.{self._validate_identifier(sch)}" if db else f" IN SCHEMA {self._validate_identifier(sch)}"

        self._cursor.execute(query)
        tables = []

        for row in self._cursor.fetchall():
            tables.append(TableInfo(
                name=row[1],
                schema=row[3],
                database=row[2],
                table_type="TABLE",
                metadata={"kind": row[4], "comment": row[6] if len(row) > 6 else None}
            ))

        if include_views:
            query = "SHOW VIEWS"
            if sch:
                query += f" IN SCHEMA {self._validate_identifier(db)}.{self._validate_identifier(sch)}" if db else f" IN SCHEMA {self._validate_identifier(sch)}"

            self._cursor.execute(query)
            for row in self._cursor.fetchall():
                tables.append(TableInfo(
                    name=row[1],
                    schema=row[3],
                    database=row[2],
                    table_type="VIEW",
                    metadata={"comment": row[6] if len(row) > 6 else None}
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

        full_name = self._format_full_table_name(
            self._validate_identifier(table_name),
            self._validate_identifier(schema) if schema else self.schema,
            self._validate_identifier(database) if database else self.database
        )

        # Get row count
        self._cursor.execute(f"SELECT COUNT(*) FROM {full_name}")
        row_count = self._cursor.fetchone()[0]

        # Get table metadata
        self._cursor.execute(f"SHOW TABLES LIKE '{table_name}' IN SCHEMA {schema or self.schema}")
        table_row = self._cursor.fetchone()

        columns = self.get_table_schema(table_name, schema, database)

        return TableInfo(
            name=table_name,
            schema=schema or self.schema,
            database=database or self.database,
            row_count=row_count,
            columns=columns,
            table_type="TABLE" if table_row else "VIEW",
            metadata={
                "comment": table_row[6] if table_row and len(table_row) > 6 else None
            }
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

        full_name = self._format_full_table_name(
            self._validate_identifier(table_name),
            self._validate_identifier(schema) if schema else self.schema,
            self._validate_identifier(database) if database else self.database
        )

        self._cursor.execute(f"DESCRIBE TABLE {full_name}")
        columns = []

        for row in self._cursor.fetchall():
            columns.append({
                "name": row[0],
                "type": row[1],
                "nullable": row[3] == "Y",
                "default": row[4],
                "primary_key": row[5] == "Y" if len(row) > 5 else False,
                "comment": row[8] if len(row) > 8 else None
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
        full_name = self._format_full_table_name(
            self._validate_identifier(table_name),
            self._validate_identifier(schema) if schema else self.schema,
            self._validate_identifier(database) if database else self.database
        )

        # Use SAMPLE for better distribution
        query = f"SELECT * FROM {full_name} SAMPLE ({n_rows} ROWS)"
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
                self._cursor.execute(query, parameters)
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
                query_id=self._cursor.sfqid,
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

        full_name = self._format_full_table_name(
            self._validate_identifier(table_name),
            self._validate_identifier(schema) if schema else self.schema,
            self._validate_identifier(database) if database else self.database
        )

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

        full_name = self._format_full_table_name(
            self._validate_identifier(table_name),
            self._validate_identifier(schema) if schema else self.schema,
            self._validate_identifier(database) if database else self.database
        )

        # Get row count and basic stats
        stats = {
            "table_name": full_name,
            "row_count": self.get_row_count(table_name, schema, database),
            "column_stats": {}
        }

        # Get column statistics
        columns = self.get_table_schema(table_name, schema, database)

        for col in columns:
            col_name = col["name"]
            col_type = col["type"].upper()

            try:
                # Get null count
                self._cursor.execute(f"SELECT COUNT(*) - COUNT({col_name}) as null_count FROM {full_name}")
                null_count = self._cursor.fetchone()[0]

                # Get distinct count
                self._cursor.execute(f"SELECT COUNT(DISTINCT {col_name}) FROM {full_name}")
                distinct_count = self._cursor.fetchone()[0]

                col_stats = {
                    "null_count": null_count,
                    "distinct_count": distinct_count,
                    "data_type": col_type
                }

                # Get min/max for numeric and date types
                if any(t in col_type for t in ["INT", "FLOAT", "NUMBER", "DECIMAL", "DATE", "TIME"]):
                    self._cursor.execute(f"SELECT MIN({col_name}), MAX({col_name}) FROM {full_name}")
                    row = self._cursor.fetchone()
                    col_stats["min"] = str(row[0]) if row[0] is not None else None
                    col_stats["max"] = str(row[1]) if row[1] is not None else None

                stats["column_stats"][col_name] = col_stats

            except Exception as e:
                logger.warning(f"Failed to get stats for column {col_name}: {str(e)}")
                stats["column_stats"][col_name] = {"error": str(e)}

        return stats

    def use_warehouse(self, warehouse_name: str) -> None:
        """Switch to a different warehouse."""
        if not self._connected:
            self.connect()

        self._cursor.execute(f"USE WAREHOUSE {self._validate_identifier(warehouse_name)}")
        self.warehouse = warehouse_name
        logger.info(f"Switched to warehouse: {warehouse_name}")

    def use_role(self, role_name: str) -> None:
        """Switch to a different role."""
        if not self._connected:
            self.connect()

        self._cursor.execute(f"USE ROLE {self._validate_identifier(role_name)}")
        self.role = role_name
        logger.info(f"Switched to role: {role_name}")

    def use_database(self, database_name: str) -> None:
        """Switch to a different database."""
        if not self._connected:
            self.connect()

        self._cursor.execute(f"USE DATABASE {self._validate_identifier(database_name)}")
        self.database = database_name
        logger.info(f"Switched to database: {database_name}")

    def get_ddl(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        include_dependencies: bool = False
    ) -> str:
        """Get DDL for a table using Snowflake's GET_DDL function."""
        if not self._connected:
            self.connect()

        db = self._validate_identifier(database) if database else self.database
        sch = self._validate_identifier(schema) if schema else self.schema
        tbl = self._validate_identifier(table_name)

        # Build fully qualified name for GET_DDL
        if db and sch:
            full_name = f"'{db}.{sch}.{tbl}'"
        elif sch:
            full_name = f"'{sch}.{tbl}'"
        else:
            full_name = f"'{tbl}'"

        try:
            self._cursor.execute(f"SELECT GET_DDL('TABLE', {full_name})")
            result = self._cursor.fetchone()
            return result[0] if result else ""
        except Exception as e:
            # Try as VIEW if TABLE fails
            try:
                self._cursor.execute(f"SELECT GET_DDL('VIEW', {full_name})")
                result = self._cursor.fetchone()
                return result[0] if result else ""
            except Exception:
                logger.warning(f"Could not get DDL for {table_name}: {str(e)}")
                return ""

    def get_query_history(
        self,
        days: int = 7,
        limit: int = 1000,
        database_filter: Optional[str] = None,
        schema_filter: Optional[str] = None,
        table_filter: Optional[str] = None,
        user_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get query history from Snowflake's QUERY_HISTORY view."""
        if not self._connected:
            self.connect()

        # Build query with filters
        query = """
        SELECT
            QUERY_ID,
            QUERY_TEXT,
            USER_NAME,
            ROLE_NAME,
            DATABASE_NAME,
            SCHEMA_NAME,
            START_TIME,
            END_TIME,
            TOTAL_ELAPSED_TIME as EXECUTION_TIME_MS,
            ROWS_PRODUCED,
            BYTES_SCANNED,
            EXECUTION_STATUS,
            ERROR_MESSAGE,
            WAREHOUSE_NAME,
            QUERY_TYPE
        FROM SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY
        WHERE START_TIME >= DATEADD(day, -%s, CURRENT_TIMESTAMP())
        """

        params = [days]

        if database_filter:
            query += " AND DATABASE_NAME = %s"
            params.append(database_filter)

        if schema_filter:
            query += " AND SCHEMA_NAME = %s"
            params.append(schema_filter)

        if table_filter:
            query += " AND QUERY_TEXT ILIKE %s"
            params.append(f"%{table_filter}%")

        if user_filter:
            query += " AND USER_NAME = %s"
            params.append(user_filter)

        query += f" ORDER BY START_TIME DESC LIMIT {limit}"

        try:
            self._cursor.execute(query, params)
            columns = [desc[0] for desc in self._cursor.description]
            rows = self._cursor.fetchall()

            results = []
            for row in rows:
                record = dict(zip(columns, row))
                results.append({
                    "query_id": record.get("QUERY_ID"),
                    "query_text": record.get("QUERY_TEXT"),
                    "user_name": record.get("USER_NAME"),
                    "role_name": record.get("ROLE_NAME"),
                    "database_name": record.get("DATABASE_NAME"),
                    "schema_name": record.get("SCHEMA_NAME"),
                    "start_time": record.get("START_TIME"),
                    "end_time": record.get("END_TIME"),
                    "execution_time_ms": record.get("EXECUTION_TIME_MS"),
                    "rows_produced": record.get("ROWS_PRODUCED"),
                    "bytes_scanned": record.get("BYTES_SCANNED"),
                    "status": "success" if record.get("EXECUTION_STATUS") == "SUCCESS" else "failed",
                    "error_message": record.get("ERROR_MESSAGE"),
                    "warehouse_name": record.get("WAREHOUSE_NAME"),
                    "query_type": record.get("QUERY_TYPE"),
                    "tables_accessed": []  # Would need additional parsing
                })

            return results

        except Exception as e:
            logger.warning(f"Could not get query history: {str(e)}")
            # Try alternative view (requires less privileges)
            try:
                alt_query = """
                SELECT *
                FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY(
                    END_TIME_RANGE_START => DATEADD(day, -%s, CURRENT_TIMESTAMP()),
                    RESULT_LIMIT => %s
                ))
                ORDER BY START_TIME DESC
                """
                self._cursor.execute(alt_query, [days, limit])
                columns = [desc[0] for desc in self._cursor.description]
                rows = self._cursor.fetchall()

                results = []
                for row in rows:
                    record = dict(zip(columns, row))
                    results.append({
                        "query_id": record.get("QUERY_ID"),
                        "query_text": record.get("QUERY_TEXT"),
                        "user_name": record.get("USER_NAME"),
                        "start_time": record.get("START_TIME"),
                        "end_time": record.get("END_TIME"),
                        "execution_time_ms": record.get("TOTAL_ELAPSED_TIME"),
                        "rows_produced": record.get("ROWS_PRODUCED"),
                        "bytes_scanned": record.get("BYTES_SCANNED"),
                        "status": "success" if record.get("EXECUTION_STATUS") == "SUCCESS" else "failed",
                        "tables_accessed": []
                    })

                return results
            except Exception as e2:
                logger.error(f"Could not get query history from alternative view: {str(e2)}")
                return []
