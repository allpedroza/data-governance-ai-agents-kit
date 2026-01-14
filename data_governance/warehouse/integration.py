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
Warehouse Integration Module

Provides integration layer between warehouse connectors and data governance modules:
- Lineage
- Discovery
- Enrichment
- Classification
- Quality
- Asset Value

Handles permission validation and connection management.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Type
from datetime import datetime
from functools import wraps

from .config import WarehouseConfig, WarehouseType, WarehouseConfigManager
from .permissions import (
    WarehousePermissionManager,
    WarehousePermission,
    ModuleAccess,
    DataModule,
    WarehouseOperation,
    WarehouseAccessLevel
)
from .connectors import (
    WarehouseConnector,
    create_warehouse_connector,
    WarehouseConnectionError,
    WarehouseQueryError
)
from .connectors.base import WarehousePermissionError, QueryResult, TableInfo

logger = logging.getLogger(__name__)


@dataclass
class WarehouseSession:
    """Represents an active warehouse session with permission context."""
    session_id: str
    warehouse_name: str
    connector: WarehouseConnector
    permission: WarehousePermission
    module: DataModule
    module_access: ModuleAccess
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: Optional[datetime] = None
    query_count: int = 0
    rows_read: int = 0

    def record_query(self, rows: int = 0) -> None:
        """Record a query execution."""
        self.query_count += 1
        self.rows_read += rows
        self.last_activity = datetime.utcnow()


class WarehouseIntegration:
    """
    Central integration layer for warehouse access from data governance modules.

    This class:
    - Manages warehouse connections
    - Validates permissions for each operation
    - Provides a unified interface for all modules
    - Handles connection pooling and session management
    """

    def __init__(
        self,
        config_dir: str = ".warehouse_configs",
        permissions_dir: str = ".warehouse_permissions"
    ):
        self._config_manager = WarehouseConfigManager(config_dir)
        self._permission_manager = WarehousePermissionManager(permissions_dir)
        self._sessions: Dict[str, WarehouseSession] = {}
        self._connectors: Dict[str, WarehouseConnector] = {}

    @property
    def config_manager(self) -> WarehouseConfigManager:
        """Get the configuration manager."""
        return self._config_manager

    @property
    def permission_manager(self) -> WarehousePermissionManager:
        """Get the permission manager."""
        return self._permission_manager

    def add_warehouse(self, config: WarehouseConfig) -> None:
        """Add a warehouse configuration."""
        self._config_manager.add_config(config)
        logger.info(f"Added warehouse configuration: {config.name}")

    def remove_warehouse(self, name: str) -> bool:
        """Remove a warehouse configuration."""
        # Close any open connections
        if name in self._connectors:
            self._connectors[name].disconnect()
            del self._connectors[name]

        return self._config_manager.remove_config(name)

    def list_warehouses(self) -> List[WarehouseConfig]:
        """List all configured warehouses."""
        return self._config_manager.list_configs()

    def get_connector(
        self,
        warehouse_name: str,
        module: DataModule,
        permission_id: str
    ) -> WarehouseSession:
        """
        Get a warehouse connector with permission validation.

        Args:
            warehouse_name: Name of the warehouse configuration
            module: Data module requesting access
            permission_id: Permission ID to validate

        Returns:
            WarehouseSession with active connection

        Raises:
            WarehousePermissionError: If access is not allowed
            WarehouseConnectionError: If connection fails
        """
        # Validate permission
        permission = self._permission_manager.get_permission(permission_id)
        if not permission or not permission.is_valid():
            raise WarehousePermissionError(f"Invalid or expired permission: {permission_id}")

        module_access = permission.get_module_access(module, warehouse_name)
        if not module_access:
            raise WarehousePermissionError(
                f"Module {module.value} does not have access to warehouse {warehouse_name}"
            )

        # Get warehouse configuration
        config = self._config_manager.get_config(warehouse_name)
        if not config:
            raise WarehouseConnectionError(f"Warehouse not found: {warehouse_name}")

        if not config.is_active:
            raise WarehouseConnectionError(f"Warehouse is not active: {warehouse_name}")

        # Create or reuse connector
        if warehouse_name not in self._connectors:
            connector = create_warehouse_connector(
                warehouse_type=config.credentials.warehouse_type.value,
                **self._get_connector_params(config)
            )
            connector.connect()
            self._connectors[warehouse_name] = connector
        else:
            connector = self._connectors[warehouse_name]
            if not connector.is_connected:
                connector.connect()

        # Create session
        import secrets
        session_id = f"ws_{secrets.token_hex(8)}"

        session = WarehouseSession(
            session_id=session_id,
            warehouse_name=warehouse_name,
            connector=connector,
            permission=permission,
            module=module,
            module_access=module_access
        )

        self._sessions[session_id] = session
        logger.info(f"Created warehouse session: {session_id} for {module.value} on {warehouse_name}")

        return session

    def _get_connector_params(self, config: WarehouseConfig) -> Dict[str, Any]:
        """Extract connector parameters from configuration."""
        creds = config.credentials
        params = {
            "connection_timeout": creds.connection_timeout,
            "query_timeout": creds.query_timeout,
            "pool_size": creds.pool_size,
            "max_overflow": creds.max_overflow,
        }

        if creds.warehouse_type == WarehouseType.SNOWFLAKE:
            params.update({
                "account": creds.account,
                "username": creds.username,
                "password": creds.password,
                "database": creds.database,
                "schema": creds.schema,
                "warehouse": creds.warehouse,
                "role": creds.role,
            })
        elif creds.warehouse_type == WarehouseType.REDSHIFT:
            params.update({
                "host": creds.host,
                "port": creds.port,
                "database": creds.database,
                "username": creds.username,
                "password": creds.password,
                "schema": creds.schema,
                "cluster_identifier": creds.cluster_identifier,
                "region": creds.region,
            })
        elif creds.warehouse_type == WarehouseType.BIGQUERY:
            params.update({
                "project_id": creds.project_id,
                "dataset": creds.dataset,
                "credentials_path": creds.credentials_path,
                "credentials_json": creds.credentials_json,
            })
        elif creds.warehouse_type == WarehouseType.SYNAPSE:
            params.update({
                "server": creds.server,
                "database": creds.database,
                "username": creds.username,
                "password": creds.password,
                "schema": creds.schema,
                "authentication": creds.authentication,
                "tenant_id": creds.tenant_id,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
            })

        return params

    def close_session(self, session_id: str) -> None:
        """Close a warehouse session."""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            logger.info(
                f"Closing session {session_id}: {session.query_count} queries, "
                f"{session.rows_read} rows read"
            )
            del self._sessions[session_id]

    def close_all_sessions(self) -> None:
        """Close all active sessions."""
        for session_id in list(self._sessions.keys()):
            self.close_session(session_id)

    def disconnect_all(self) -> None:
        """Disconnect all warehouse connections."""
        self.close_all_sessions()
        for name, connector in self._connectors.items():
            try:
                connector.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting {name}: {e}")
        self._connectors.clear()

    # Module-specific access methods

    def list_tables(
        self,
        session: WarehouseSession,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> List[TableInfo]:
        """
        List tables accessible to the module.

        Validates:
        - READ_SCHEMA operation is allowed
        - Schema access is allowed (if restricted)
        """
        self._validate_operation(session, WarehouseOperation.READ_SCHEMA)

        if schema and not session.module_access.can_access_schema(schema):
            raise WarehousePermissionError(f"Access to schema {schema} is not allowed")

        tables = session.connector.list_tables(schema, database)

        # Filter tables based on allowed_tables if set
        if session.module_access.allowed_tables:
            tables = [t for t in tables if session.module_access.can_access_table(t.name, t.schema)]

        session.record_query()
        return tables

    def get_table_schema(
        self,
        session: WarehouseSession,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get table schema with permission validation.

        Validates:
        - READ_SCHEMA operation is allowed
        - Table access is allowed
        """
        self._validate_operation(session, WarehouseOperation.READ_SCHEMA)
        self._validate_table_access(session, table_name, schema)

        result = session.connector.get_table_schema(table_name, schema, database)
        session.record_query()
        return result

    def read_sample(
        self,
        session: WarehouseSession,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        n_rows: Optional[int] = None
    ) -> QueryResult:
        """
        Read sample data with permission validation.

        Validates:
        - READ_SAMPLE operation is allowed
        - Table access is allowed
        - Row limit is respected
        """
        self._validate_operation(session, WarehouseOperation.READ_SAMPLE)
        self._validate_table_access(session, table_name, schema)

        # Apply row limit from permissions
        max_rows = min(
            n_rows or session.module_access.max_sample_rows,
            session.module_access.max_sample_rows
        )

        result = session.connector.read_sample(table_name, schema, database, max_rows)
        session.record_query(result.row_count)
        return result

    def read_full(
        self,
        session: WarehouseSession,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
        max_rows: Optional[int] = None
    ) -> QueryResult:
        """
        Read full table data with permission validation.

        Validates:
        - READ_FULL operation is allowed
        - Table access is allowed
        - Query row limit is respected
        """
        self._validate_operation(session, WarehouseOperation.READ_FULL)
        self._validate_table_access(session, table_name, schema)

        # Apply row limit from permissions
        limit = min(
            max_rows or session.module_access.max_query_rows,
            session.module_access.max_query_rows
        )

        full_name = session.connector._format_full_table_name(table_name, schema, database)
        query = f"SELECT * FROM {full_name}"

        result = session.connector.execute_query(query, max_rows=limit)
        session.record_query(result.row_count)
        return result

    def execute_query(
        self,
        session: WarehouseSession,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """
        Execute a custom query with permission validation.

        Validates:
        - EXECUTE_QUERY operation is allowed
        - Query row limit is respected
        """
        self._validate_operation(session, WarehouseOperation.EXECUTE_QUERY)

        result = session.connector.execute_query(
            query,
            parameters,
            max_rows=session.module_access.max_query_rows
        )
        session.record_query(result.row_count)
        return result

    def get_table_statistics(
        self,
        session: WarehouseSession,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get table statistics with permission validation.

        Validates:
        - READ_STATS operation is allowed
        - Table access is allowed
        """
        self._validate_operation(session, WarehouseOperation.READ_STATS)
        self._validate_table_access(session, table_name, schema)

        result = session.connector.get_table_statistics(table_name, schema, database)
        session.record_query()
        return result

    def get_row_count(
        self,
        session: WarehouseSession,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None
    ) -> int:
        """
        Get table row count with permission validation.

        Validates:
        - READ_STATS operation is allowed
        - Table access is allowed
        """
        self._validate_operation(session, WarehouseOperation.READ_STATS)
        self._validate_table_access(session, table_name, schema)

        result = session.connector.get_row_count(table_name, schema, database)
        session.record_query()
        return result

    def _validate_operation(self, session: WarehouseSession, operation: WarehouseOperation) -> None:
        """Validate that an operation is allowed for the session."""
        if not session.module_access.can_perform(operation):
            raise WarehousePermissionError(
                f"Operation {operation.name} is not allowed for module {session.module.value}"
            )

    def _validate_table_access(
        self,
        session: WarehouseSession,
        table_name: str,
        schema: Optional[str]
    ) -> None:
        """Validate that table access is allowed."""
        if not session.module_access.can_access_table(table_name, schema):
            raise WarehousePermissionError(
                f"Access to table {schema}.{table_name} is not allowed"
            )

    # Convenience methods for quick setup

    def setup_default_permissions(self) -> Dict[str, WarehousePermission]:
        """
        Set up default permissions for all modules.

        Creates appropriate permission sets for:
        - Lineage: Read schema/metadata only
        - Discovery: Read schema and sample data
        - Enrichment: Read schema and sample data
        - Classification: Read sample data for classification
        - Quality: Full read access for quality metrics
        - Asset Value: Read metadata for value assessment
        """
        return self._permission_manager.create_default_permissions()

    def quick_connect(
        self,
        warehouse_type: str,
        module: DataModule,
        **kwargs
    ) -> WarehouseSession:
        """
        Quick connection to a warehouse without explicit configuration.

        Creates temporary configuration and permission for immediate use.
        Useful for development and testing.

        Args:
            warehouse_type: Type of warehouse (snowflake, redshift, bigquery, synapse)
            module: Data module requesting access
            **kwargs: Warehouse-specific connection parameters

        Returns:
            WarehouseSession ready for use
        """
        import secrets
        from .config import WarehouseCredentials

        # Create temporary configuration
        temp_name = f"temp_{secrets.token_hex(4)}"

        wh_type = WarehouseType(warehouse_type.lower())
        credentials = WarehouseCredentials(
            warehouse_type=wh_type,
            **kwargs
        )

        config = WarehouseConfig(
            name=temp_name,
            credentials=credentials,
            description="Temporary connection"
        )

        self._config_manager.add_config(config)

        # Create permission
        permission = self._permission_manager.create_permission(
            name=f"temp_perm_{temp_name}",
            description="Temporary permission"
        )

        # Grant module access based on module type
        module_configs = {
            DataModule.LINEAGE: (WarehouseAccessLevel.READ_METADATA, WarehouseOperation.READ_SCHEMA | WarehouseOperation.READ_STATS),
            DataModule.DISCOVERY: (WarehouseAccessLevel.READ_SAMPLE, WarehouseOperation.READ_ONLY),
            DataModule.ENRICHMENT: (WarehouseAccessLevel.READ_SAMPLE, WarehouseOperation.READ_ONLY),
            DataModule.CLASSIFICATION: (WarehouseAccessLevel.READ_SAMPLE, WarehouseOperation.READ_SCHEMA | WarehouseOperation.READ_SAMPLE),
            DataModule.QUALITY: (WarehouseAccessLevel.READ_FULL, WarehouseOperation.READ_ALL | WarehouseOperation.EXECUTE_QUERY),
            DataModule.ASSET_VALUE: (WarehouseAccessLevel.READ_METADATA, WarehouseOperation.READ_SCHEMA | WarehouseOperation.READ_STATS),
        }

        access_level, operations = module_configs.get(
            module,
            (WarehouseAccessLevel.READ_SAMPLE, WarehouseOperation.READ_ONLY)
        )

        self._permission_manager.grant_module_access(
            permission_id=permission.permission_id,
            module=module,
            warehouse_name=temp_name,
            access_level=access_level,
            operations=operations
        )

        # Get session
        return self.get_connector(temp_name, module, permission.permission_id)

    def get_session_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all active sessions."""
        return [
            {
                "session_id": s.session_id,
                "warehouse": s.warehouse_name,
                "module": s.module.value,
                "created_at": s.created_at.isoformat(),
                "last_activity": s.last_activity.isoformat() if s.last_activity else None,
                "query_count": s.query_count,
                "rows_read": s.rows_read
            }
            for s in self._sessions.values()
        ]


# Convenience function for module integration
def create_warehouse_integration() -> WarehouseIntegration:
    """Create a WarehouseIntegration instance with environment-based configuration."""
    integration = WarehouseIntegration()

    # Load configurations from environment
    env_manager = WarehouseConfigManager.load_from_env()
    for config in env_manager.list_configs():
        integration.add_warehouse(config)

    return integration
