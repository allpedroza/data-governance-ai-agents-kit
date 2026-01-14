# /// script
# dependencies = [
#   "azure-identity>=1.12.0",
#   "azure-storage-blob>=12.14.0",
#   "black>=22.0.0",
#   "boto3>=1.26.0",
#   "chromadb>=0.4.0",
#   "cryptography>=41.0.0",
#   "databricks-sdk>=0.5.0",
#   "faiss-cpu>=1.7.0",
#   "flake8>=5.0.0",
#   "google-cloud-bigquery-storage>=2.0.0",
#   "google-cloud-bigquery>=3.0.0",
#   "google-cloud-storage>=2.7.0",
#   "isort>=5.0.0",
#   "kaleido>=0.2.0",
#   "matplotlib>=3.6.0",
#   "mypy>=1.0.0",
#   "networkx>=3.0",
#   "numpy>=1.24.0",
#   "openai>=1.0.0",
#   "openpyxl>=3.0.0",
#   "pandas>=2.0.0",
#   "plotly>=5.0.0",
#   "psycopg2-binary>=2.9.0",
#   "pyarrow>=14.0.0",
#   "pyodbc>=4.0.0",
#   "pyspark>=3.3.0",
#   "pytest-cov>=4.0.0",
#   "pytest>=7.0.0",
#   "python-dotenv>=1.0.0",
#   "python-igraph>=0.10.0",
#   "pyyaml>=6.0",
#   "redshift-connector>=2.0.0",
#   "requests>=2.31.0",
#   "scikit-learn>=1.0.0",
#   "seaborn>=0.12.0",
#   "sentence-transformers>=2.2.0",
#   "snowflake-connector-python>=3.0.0",
#   "snowflake-sqlalchemy>=1.5.0",
#   "spacy>=3.5.0; extra == "spacy"",
#   "sphinx-rtd-theme>=1.0.0",
#   "sphinx>=5.0.0",
#   "sqlalchemy-bigquery>=1.6.0",
#   "sqlalchemy-redshift>=0.8.0",
#   "sqlalchemy>=2.0.0",
#   "sqlparse>=0.4.0",
#   "streamlit>=1.32.0",
#   "tqdm>=4.65.0",
# ]
# ///
"""
Base Warehouse Connector

Abstract base class for data warehouse connectors with:
- Common interface for all warehouses
- Permission checking integration
- Connection pooling
- Query execution with timeout
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Iterator
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class WarehouseConnectionError(Exception):
    """Raised when warehouse connection fails."""
    pass


class WarehouseQueryError(Exception):
    """Raised when query execution fails."""
    pass


class WarehousePermissionError(Exception):
    """Raised when operation is not permitted."""
    pass


@dataclass
class WarehouseInfo:
    """Information about a warehouse connection."""
    warehouse_type: str
    name: str
    host: Optional[str] = None
    database: Optional[str] = None
    schema: Optional[str] = None
    connected: bool = False
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TableInfo:
    """Information about a database table."""
    name: str
    schema: Optional[str] = None
    database: Optional[str] = None
    full_name: str = ""
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    table_type: str = "TABLE"  # TABLE, VIEW, EXTERNAL, MATERIALIZED_VIEW
    columns: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.full_name:
            parts = []
            if self.database:
                parts.append(self.database)
            if self.schema:
                parts.append(self.schema)
            parts.append(self.name)
            self.full_name = ".".join(parts)


@dataclass
class QueryResult:
    """Result of a query execution."""
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: float
    query_id: Optional[str] = None
    truncated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class WarehouseConnector(ABC):
    """
    Abstract base class for data warehouse connectors.

    All warehouse-specific connectors must implement this interface.
    """

    def __init__(
        self,
        connection_timeout: int = 30,
        query_timeout: int = 300,
        pool_size: int = 5,
        max_overflow: int = 10
    ):
        self.connection_timeout = connection_timeout
        self.query_timeout = query_timeout
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self._connected = False
        self._connection = None
        self._engine = None

    @property
    @abstractmethod
    def warehouse_type(self) -> str:
        """Return the warehouse type identifier."""
        pass

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the warehouse."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close the warehouse connection."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the connection is valid."""
        pass

    @abstractmethod
    def get_info(self) -> WarehouseInfo:
        """Get information about the warehouse connection."""
        pass

    @abstractmethod
    def list_databases(self) -> List[str]:
        """List all accessible databases."""
        pass

    @abstractmethod
    def list_schemas(self, database: Optional[str] = None) -> List[str]:
        """List all schemas in a database."""
        pass

    @abstractmethod
    def list_tables(
        self,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        include_views: bool = True
    ) -> List[TableInfo]:
        """List all tables in a schema."""
        pass

    @abstractmethod
    def get_table_info(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> TableInfo:
        """Get detailed information about a table."""
        pass

    @abstractmethod
    def get_table_schema(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get table column schema.

        Returns list of column definitions with:
        - name: Column name
        - type: Data type
        - nullable: Whether null values are allowed
        - default: Default value if any
        - primary_key: Whether part of primary key
        - comment: Column comment/description
        """
        pass

    @abstractmethod
    def read_sample(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        n_rows: int = 1000
    ) -> QueryResult:
        """Read a sample of rows from a table."""
        pass

    @abstractmethod
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        max_rows: Optional[int] = None
    ) -> QueryResult:
        """Execute a SQL query and return results."""
        pass

    @abstractmethod
    def get_row_count(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> int:
        """Get the row count for a table."""
        pass

    @abstractmethod
    def get_table_statistics(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get table statistics.

        Returns statistics including:
        - row_count: Number of rows
        - size_bytes: Table size in bytes
        - column_stats: Per-column statistics (min, max, nulls, distinct)
        """
        pass

    @abstractmethod
    def get_ddl(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        include_dependencies: bool = False
    ) -> str:
        """
        Get the DDL (CREATE statement) for a table.

        Args:
            table_name: Name of the table
            schema: Schema name (optional)
            database: Database name (optional)
            include_dependencies: Include dependent objects DDL

        Returns:
            DDL statement as string
        """
        pass

    @abstractmethod
    def get_query_history(
        self,
        days: int = 7,
        limit: int = 1000,
        database_filter: Optional[str] = None,
        schema_filter: Optional[str] = None,
        table_filter: Optional[str] = None,
        user_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get query execution history from warehouse audit tables.

        Args:
            days: Number of days to look back
            limit: Maximum number of queries to return
            database_filter: Filter by database name
            schema_filter: Filter by schema name
            table_filter: Filter by table name
            user_filter: Filter by user name

        Returns:
            List of query history records with:
            - query_id: Unique query identifier
            - query_text: SQL text
            - user_name: User who ran the query
            - start_time: Query start time
            - end_time: Query end time
            - execution_time_ms: Duration in milliseconds
            - rows_produced: Number of rows returned
            - bytes_scanned: Data scanned in bytes
            - status: Query status (success/failed)
            - tables_accessed: List of tables accessed
        """
        pass

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected

    @contextmanager
    def connection(self):
        """Context manager for connection handling."""
        try:
            if not self._connected:
                self.connect()
            yield self
        finally:
            pass  # Keep connection open for reuse

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def _format_full_table_name(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> str:
        """Format a fully qualified table name."""
        parts = []
        if database:
            parts.append(database)
        if schema:
            parts.append(schema)
        parts.append(table_name)
        return ".".join(parts)

    def _validate_identifier(self, identifier: str) -> str:
        """Validate and sanitize an identifier (table/schema/database name)."""
        # Basic SQL injection prevention
        if not identifier:
            raise ValueError("Identifier cannot be empty")

        # Allow only alphanumeric, underscore, and dot
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_\.]*$', identifier):
            raise ValueError(f"Invalid identifier: {identifier}")

        return identifier
