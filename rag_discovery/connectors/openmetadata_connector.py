"""
OpenMetadata Connector
Fetches table metadata from OpenMetadata catalog for use in Data Discovery Agent

OpenMetadata is an open-source metadata platform for data discovery,
data observability, and data governance.

References:
- https://docs.open-metadata.org/latest/sdk/python
- https://github.com/open-metadata/openmetadata-sdk
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class OpenMetadataConfig:
    """Configuration for OpenMetadata connection"""

    # Server configuration
    server_host: str = "http://localhost:8585"
    api_version: str = "v1"

    # Authentication
    auth_type: str = "jwt"  # jwt, basic, google, okta, azure, auth0, custom-oidc
    jwt_token: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    # Filters
    database_filter: Optional[List[str]] = None  # Filter by database names
    schema_filter: Optional[List[str]] = None    # Filter by schema names
    table_filter: Optional[List[str]] = None     # Filter by table name patterns
    service_filter: Optional[List[str]] = None   # Filter by service names

    # Options
    include_columns: bool = True
    include_tags: bool = True
    include_owners: bool = True
    include_lineage: bool = False
    include_sample_data: bool = False

    # Pagination
    limit: int = 100  # Results per page

    @classmethod
    def from_env(cls) -> "OpenMetadataConfig":
        """Create config from environment variables"""
        return cls(
            server_host=os.getenv("OPENMETADATA_HOST", "http://localhost:8585"),
            jwt_token=os.getenv("OPENMETADATA_JWT_TOKEN"),
            username=os.getenv("OPENMETADATA_USERNAME"),
            password=os.getenv("OPENMETADATA_PASSWORD"),
            auth_type=os.getenv("OPENMETADATA_AUTH_TYPE", "jwt")
        )


@dataclass
class OMTableMetadata:
    """Table metadata from OpenMetadata"""
    id: str
    name: str
    fully_qualified_name: str
    display_name: str = ""
    description: str = ""
    database: str = ""
    database_schema: str = ""
    service: str = ""
    service_type: str = ""
    columns: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    owners: List[str] = field(default_factory=list)
    table_type: str = ""
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    location: str = ""
    partition_columns: List[str] = field(default_factory=list)

    def to_discovery_format(self) -> Dict[str, Any]:
        """Convert to Data Discovery Agent TableMetadata format"""
        # Parse FQN to get database and schema
        fqn_parts = self.fully_qualified_name.split(".")

        if len(fqn_parts) >= 4:
            # service.database.schema.table
            database = fqn_parts[1]
            schema = fqn_parts[2]
        elif len(fqn_parts) == 3:
            # database.schema.table
            database = fqn_parts[0]
            schema = fqn_parts[1]
        else:
            database = self.database
            schema = self.database_schema

        return {
            "name": self.name,
            "database": database,
            "schema": schema,
            "description": self.description or self.display_name or "",
            "columns": self.columns,
            "row_count": self.row_count,
            "size_bytes": self.size_bytes,
            "owner": ", ".join(self.owners) if self.owners else "",
            "tags": self.tags,
            "location": self.location,
            "format": self.service_type,
            "partition_keys": self.partition_columns,
            # Extra metadata from OpenMetadata
            "openmetadata_id": self.id,
            "openmetadata_fqn": self.fully_qualified_name,
            "service": self.service,
            "table_type": self.table_type
        }


class OpenMetadataConnector:
    """
    Connector to fetch metadata from OpenMetadata

    Supports two modes:
    1. Using official OpenMetadata Python SDK (recommended)
    2. Using direct REST API calls (fallback)

    Usage:
        config = OpenMetadataConfig(
            server_host="http://openmetadata:8585",
            jwt_token="your-jwt-token"
        )

        connector = OpenMetadataConnector(config)
        tables = connector.get_tables()

        # Use with Data Discovery Agent
        agent.index_metadata(tables)
    """

    def __init__(self, config: Optional[OpenMetadataConfig] = None):
        """
        Initialize OpenMetadata connector

        Args:
            config: OpenMetadata connection configuration
        """
        self.config = config or OpenMetadataConfig.from_env()
        self._client = None
        self._use_sdk = True

        # Try to initialize SDK client
        try:
            self._init_sdk_client()
        except ImportError:
            logger.warning(
                "OpenMetadata SDK not installed. Falling back to REST API. "
                "Install with: pip install openmetadata-ingestion"
            )
            self._use_sdk = False
        except Exception as e:
            logger.warning(f"Failed to initialize SDK client: {e}. Falling back to REST API.")
            self._use_sdk = False

    def _init_sdk_client(self):
        """Initialize OpenMetadata SDK client"""
        from metadata.ingestion.ometa.ometa_api import OpenMetadata
        from metadata.generated.schema.entity.services.connections.metadata.openMetadataConnection import (
            OpenMetadataConnection,
            AuthProvider
        )
        from metadata.generated.schema.security.client.openMetadataJWTClientConfig import (
            OpenMetadataJWTClientConfig
        )

        # Build security config based on auth type
        if self.config.auth_type == "jwt" and self.config.jwt_token:
            security_config = OpenMetadataJWTClientConfig(
                jwtToken=self.config.jwt_token
            )
            auth_provider = AuthProvider.openmetadata
        else:
            security_config = None
            auth_provider = AuthProvider.no_auth

        # Create connection config
        server_config = OpenMetadataConnection(
            hostPort=self.config.server_host,
            authProvider=auth_provider,
            securityConfig=security_config
        )

        # Create client
        self._client = OpenMetadata(server_config)

        # Test connection
        if not self._client.health_check():
            raise ConnectionError(
                f"Cannot connect to OpenMetadata at {self.config.server_host}"
            )

        logger.info(f"Connected to OpenMetadata at {self.config.server_host}")

    def _init_rest_client(self):
        """Initialize REST API client (fallback)"""
        try:
            import requests
        except ImportError:
            raise ImportError("requests not installed. Install with: pip install requests")

        self._session = requests.Session()

        # Set authentication headers
        if self.config.jwt_token:
            self._session.headers["Authorization"] = f"Bearer {self.config.jwt_token}"
        elif self.config.username and self.config.password:
            self._session.auth = (self.config.username, self.config.password)

        self._session.headers["Content-Type"] = "application/json"

        # Test connection
        response = self._session.get(
            f"{self.config.server_host}/api/{self.config.api_version}/system/version"
        )
        if response.status_code != 200:
            raise ConnectionError(
                f"Cannot connect to OpenMetadata: {response.status_code}"
            )

        logger.info(f"Connected to OpenMetadata via REST API")

    def get_tables(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        service: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[OMTableMetadata]:
        """
        Fetch all tables from OpenMetadata

        Args:
            database: Filter by database name
            schema: Filter by schema name
            service: Filter by service name
            limit: Maximum number of tables to fetch

        Returns:
            List of OMTableMetadata objects
        """
        if self._use_sdk:
            return self._get_tables_sdk(database, schema, service, limit)
        else:
            return self._get_tables_rest(database, schema, service, limit)

    def _get_tables_sdk(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        service: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[OMTableMetadata]:
        """Fetch tables using SDK"""
        from metadata.generated.schema.entity.data.table import Table
        from metadata.ingestion.ometa.utils import model_str

        tables_result = []
        after = None
        page_limit = self.config.limit
        total_limit = limit
        fetched = 0

        while True:
            # Fetch page of tables
            response = self._client.list_entities(
                entity=Table,
                after=after,
                limit=page_limit,
                fields=[
                    "columns",
                    "tags",
                    "owner",
                    "database",
                    "databaseSchema",
                    "service",
                    "tablePartition",
                    "tableProfile"
                ]
            )

            for table in response.entities:
                # Apply filters
                if database and table.database and database not in model_str(table.database.name):
                    continue
                if schema and table.databaseSchema and schema not in model_str(table.databaseSchema.name):
                    continue
                if service and table.service and service not in model_str(table.service.name):
                    continue

                # Convert to our format
                om_table = self._convert_sdk_table(table)
                tables_result.append(om_table)
                fetched += 1

                if total_limit and fetched >= total_limit:
                    break

            if total_limit and fetched >= total_limit:
                break

            # Check for more pages
            after = response.after
            if not after:
                break

        logger.info(f"Fetched {len(tables_result)} tables from OpenMetadata")
        return tables_result

    def _convert_sdk_table(self, table) -> OMTableMetadata:
        """Convert SDK Table entity to OMTableMetadata"""
        from metadata.ingestion.ometa.utils import model_str

        # Extract columns
        columns = []
        if table.columns:
            for col in table.columns:
                col_data = {
                    "name": model_str(col.name),
                    "type": model_str(col.dataType) if col.dataType else "unknown",
                    "description": col.description or ""
                }
                if col.tags:
                    col_data["tags"] = [model_str(t.tagFQN) for t in col.tags]
                columns.append(col_data)

        # Extract tags
        tags = []
        if table.tags:
            tags = [model_str(t.tagFQN) for t in table.tags]

        # Extract owners
        owners = []
        if table.owner:
            if hasattr(table.owner, 'name'):
                owners.append(model_str(table.owner.name))
            elif hasattr(table.owner, 'displayName'):
                owners.append(model_str(table.owner.displayName))

        # Extract partition columns
        partition_cols = []
        if table.tablePartition and table.tablePartition.columns:
            partition_cols = [model_str(c) for c in table.tablePartition.columns]

        # Extract profile data
        row_count = None
        size_bytes = None
        if table.profile:
            row_count = table.profile.rowCount
            size_bytes = table.profile.sizeInBytes

        return OMTableMetadata(
            id=model_str(table.id),
            name=model_str(table.name),
            fully_qualified_name=model_str(table.fullyQualifiedName),
            display_name=table.displayName or "",
            description=table.description or "",
            database=model_str(table.database.name) if table.database else "",
            database_schema=model_str(table.databaseSchema.name) if table.databaseSchema else "",
            service=model_str(table.service.name) if table.service else "",
            service_type=model_str(table.serviceType) if table.serviceType else "",
            columns=columns,
            tags=tags,
            owners=owners,
            table_type=model_str(table.tableType) if table.tableType else "",
            row_count=row_count,
            size_bytes=size_bytes,
            partition_columns=partition_cols
        )

    def _get_tables_rest(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        service: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[OMTableMetadata]:
        """Fetch tables using REST API (fallback)"""
        import requests

        if not hasattr(self, '_session'):
            self._init_rest_client()

        tables_result = []
        after = None
        page_limit = self.config.limit
        total_limit = limit
        fetched = 0

        base_url = f"{self.config.server_host}/api/{self.config.api_version}/tables"

        while True:
            # Build query params
            params = {
                "limit": page_limit,
                "fields": "columns,tags,owner,database,databaseSchema,service,tablePartition"
            }
            if after:
                params["after"] = after
            if database:
                params["database"] = database

            # Fetch page
            response = self._session.get(base_url, params=params)

            if response.status_code != 200:
                logger.error(f"Failed to fetch tables: {response.status_code} - {response.text}")
                break

            data = response.json()

            for table_data in data.get("data", []):
                # Apply filters
                fqn = table_data.get("fullyQualifiedName", "")
                if schema and schema not in fqn:
                    continue
                if service and service not in fqn:
                    continue

                # Convert to our format
                om_table = self._convert_rest_table(table_data)
                tables_result.append(om_table)
                fetched += 1

                if total_limit and fetched >= total_limit:
                    break

            if total_limit and fetched >= total_limit:
                break

            # Check for more pages
            paging = data.get("paging", {})
            after = paging.get("after")
            if not after:
                break

        logger.info(f"Fetched {len(tables_result)} tables from OpenMetadata (REST)")
        return tables_result

    def _convert_rest_table(self, data: Dict[str, Any]) -> OMTableMetadata:
        """Convert REST API response to OMTableMetadata"""
        # Extract columns
        columns = []
        for col in data.get("columns", []):
            col_data = {
                "name": col.get("name", ""),
                "type": col.get("dataType", "unknown"),
                "description": col.get("description", "")
            }
            if col.get("tags"):
                col_data["tags"] = [t.get("tagFQN", "") for t in col["tags"]]
            columns.append(col_data)

        # Extract tags
        tags = [t.get("tagFQN", "") for t in data.get("tags", [])]

        # Extract owners
        owners = []
        owner = data.get("owner")
        if owner:
            owners.append(owner.get("name") or owner.get("displayName", ""))

        # Extract partition columns
        partition_cols = []
        partition = data.get("tablePartition")
        if partition and partition.get("columns"):
            partition_cols = partition["columns"]

        return OMTableMetadata(
            id=data.get("id", ""),
            name=data.get("name", ""),
            fully_qualified_name=data.get("fullyQualifiedName", ""),
            display_name=data.get("displayName", ""),
            description=data.get("description", ""),
            database=data.get("database", {}).get("name", ""),
            database_schema=data.get("databaseSchema", {}).get("name", ""),
            service=data.get("service", {}).get("name", ""),
            service_type=data.get("serviceType", ""),
            columns=columns,
            tags=tags,
            owners=owners,
            table_type=data.get("tableType", ""),
            row_count=data.get("profile", {}).get("rowCount") if data.get("profile") else None,
            size_bytes=data.get("profile", {}).get("sizeInBytes") if data.get("profile") else None,
            partition_columns=partition_cols
        )

    def get_tables_as_discovery_format(
        self,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        service: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch tables and convert to Data Discovery Agent format

        Args:
            database: Filter by database name
            schema: Filter by schema name
            service: Filter by service name
            limit: Maximum tables to fetch

        Returns:
            List of table metadata dictionaries ready for indexing
        """
        om_tables = self.get_tables(database, schema, service, limit)
        return [t.to_discovery_format() for t in om_tables]

    def export_to_json(
        self,
        output_path: str,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        service: Optional[str] = None,
        limit: Optional[int] = None
    ) -> int:
        """
        Export metadata to JSON file

        Args:
            output_path: Path to output JSON file
            database: Filter by database name
            schema: Filter by schema name
            service: Filter by service name
            limit: Maximum tables to export

        Returns:
            Number of tables exported
        """
        tables = self.get_tables_as_discovery_format(database, schema, service, limit)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(tables, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(tables)} tables to {output_path}")
        return len(tables)

    def export_to_catalog_txt(
        self,
        output_path: str,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        service: Optional[str] = None,
        limit: Optional[int] = None
    ) -> int:
        """
        Export table names to TXT file for catalog validation

        Args:
            output_path: Path to output TXT file
            database: Filter by database name
            schema: Filter by schema name
            service: Filter by service name
            limit: Maximum tables to export

        Returns:
            Number of tables exported
        """
        om_tables = self.get_tables(database, schema, service, limit)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for table in om_tables:
                f.write(f"{table.fully_qualified_name}\n")
                if table.description:
                    f.write(f"{table.description}\n")
                f.write("\n")

        logger.info(f"Exported {len(om_tables)} table names to {output_path}")
        return len(om_tables)

    def get_table_by_fqn(self, fqn: str) -> Optional[OMTableMetadata]:
        """
        Get a specific table by fully qualified name

        Args:
            fqn: Fully qualified name (e.g., "service.database.schema.table")

        Returns:
            OMTableMetadata or None if not found
        """
        if self._use_sdk:
            return self._get_table_by_fqn_sdk(fqn)
        else:
            return self._get_table_by_fqn_rest(fqn)

    def _get_table_by_fqn_sdk(self, fqn: str) -> Optional[OMTableMetadata]:
        """Get table by FQN using SDK"""
        from metadata.generated.schema.entity.data.table import Table

        try:
            table = self._client.get_by_name(
                entity=Table,
                fqn=fqn,
                fields=[
                    "columns",
                    "tags",
                    "owner",
                    "database",
                    "databaseSchema",
                    "service",
                    "tablePartition",
                    "tableProfile"
                ]
            )
            if table:
                return self._convert_sdk_table(table)
        except Exception as e:
            logger.warning(f"Failed to get table {fqn}: {e}")

        return None

    def _get_table_by_fqn_rest(self, fqn: str) -> Optional[OMTableMetadata]:
        """Get table by FQN using REST API"""
        import urllib.parse

        if not hasattr(self, '_session'):
            self._init_rest_client()

        encoded_fqn = urllib.parse.quote(fqn, safe="")
        url = f"{self.config.server_host}/api/{self.config.api_version}/tables/name/{encoded_fqn}"

        params = {
            "fields": "columns,tags,owner,database,databaseSchema,service,tablePartition"
        }

        try:
            response = self._session.get(url, params=params)
            if response.status_code == 200:
                return self._convert_rest_table(response.json())
        except Exception as e:
            logger.warning(f"Failed to get table {fqn}: {e}")

        return None

    def get_services(self) -> List[Dict[str, Any]]:
        """Get list of database services"""
        if self._use_sdk:
            return self._get_services_sdk()
        else:
            return self._get_services_rest()

    def _get_services_sdk(self) -> List[Dict[str, Any]]:
        """Get services using SDK"""
        from metadata.generated.schema.entity.services.databaseService import DatabaseService
        from metadata.ingestion.ometa.utils import model_str

        services = []
        response = self._client.list_entities(
            entity=DatabaseService,
            limit=100
        )

        for svc in response.entities:
            services.append({
                "id": model_str(svc.id),
                "name": model_str(svc.name),
                "service_type": model_str(svc.serviceType) if svc.serviceType else "",
                "description": svc.description or ""
            })

        return services

    def _get_services_rest(self) -> List[Dict[str, Any]]:
        """Get services using REST API"""
        if not hasattr(self, '_session'):
            self._init_rest_client()

        url = f"{self.config.server_host}/api/{self.config.api_version}/services/databaseServices"

        try:
            response = self._session.get(url, params={"limit": 100})
            if response.status_code == 200:
                data = response.json()
                return [
                    {
                        "id": svc.get("id", ""),
                        "name": svc.get("name", ""),
                        "service_type": svc.get("serviceType", ""),
                        "description": svc.get("description", "")
                    }
                    for svc in data.get("data", [])
                ]
        except Exception as e:
            logger.warning(f"Failed to get services: {e}")

        return []

    def health_check(self) -> bool:
        """Check if OpenMetadata is reachable"""
        if self._use_sdk and self._client:
            return self._client.health_check()
        else:
            try:
                if not hasattr(self, '_session'):
                    self._init_rest_client()
                response = self._session.get(
                    f"{self.config.server_host}/api/{self.config.api_version}/system/version"
                )
                return response.status_code == 200
            except Exception:
                return False


# Convenience function
def fetch_openmetadata_tables(
    host: str = "http://localhost:8585",
    jwt_token: Optional[str] = None,
    database: Optional[str] = None,
    schema: Optional[str] = None,
    service: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch tables from OpenMetadata

    Args:
        host: OpenMetadata server URL
        jwt_token: JWT token for authentication
        database: Filter by database name
        schema: Filter by schema name
        service: Filter by service name
        limit: Maximum tables to fetch

    Returns:
        List of table metadata in Data Discovery format
    """
    config = OpenMetadataConfig(
        server_host=host,
        jwt_token=jwt_token or os.getenv("OPENMETADATA_JWT_TOKEN")
    )

    connector = OpenMetadataConnector(config)
    return connector.get_tables_as_discovery_format(
        database=database,
        schema=schema,
        service=service,
        limit=limit
    )
