"""
Warehouse Configuration Module

Centralized configuration for data warehouse connections including:
- Snowflake
- Amazon Redshift
- Google BigQuery
- Azure Synapse Analytics
"""

import os
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime


class WarehouseType(Enum):
    """Supported data warehouse types."""
    SNOWFLAKE = "snowflake"
    REDSHIFT = "redshift"
    BIGQUERY = "bigquery"
    SYNAPSE = "synapse"


@dataclass
class WarehouseCredentials:
    """Secure credentials container for warehouse connections."""
    warehouse_type: WarehouseType

    # Common fields
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    schema: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None

    # Snowflake specific
    account: Optional[str] = None
    warehouse: Optional[str] = None
    role: Optional[str] = None

    # BigQuery specific
    project_id: Optional[str] = None
    dataset: Optional[str] = None
    credentials_path: Optional[str] = None
    credentials_json: Optional[Dict[str, Any]] = None

    # Redshift specific
    cluster_identifier: Optional[str] = None
    region: Optional[str] = None
    iam_role: Optional[str] = None

    # Synapse specific
    server: Optional[str] = None
    authentication: Optional[str] = None  # sql, ad, msi
    tenant_id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

    # Connection options
    ssl_enabled: bool = True
    connection_timeout: int = 30
    query_timeout: int = 300
    pool_size: int = 5
    max_overflow: int = 10

    # Extra parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_connection_string(self) -> str:
        """Generate SQLAlchemy connection string based on warehouse type."""
        if self.warehouse_type == WarehouseType.SNOWFLAKE:
            return self._snowflake_connection_string()
        elif self.warehouse_type == WarehouseType.REDSHIFT:
            return self._redshift_connection_string()
        elif self.warehouse_type == WarehouseType.BIGQUERY:
            return self._bigquery_connection_string()
        elif self.warehouse_type == WarehouseType.SYNAPSE:
            return self._synapse_connection_string()
        else:
            raise ValueError(f"Unsupported warehouse type: {self.warehouse_type}")

    def _snowflake_connection_string(self) -> str:
        """Generate Snowflake connection string."""
        # Format: snowflake://user:password@account/database/schema?warehouse=WH&role=ROLE
        base = f"snowflake://{self.username}:{self.password}@{self.account}"

        if self.database:
            base += f"/{self.database}"
            if self.schema:
                base += f"/{self.schema}"

        params = []
        if self.warehouse:
            params.append(f"warehouse={self.warehouse}")
        if self.role:
            params.append(f"role={self.role}")

        for key, value in self.extra_params.items():
            params.append(f"{key}={value}")

        if params:
            base += "?" + "&".join(params)

        return base

    def _redshift_connection_string(self) -> str:
        """Generate Amazon Redshift connection string."""
        # Format: postgresql+psycopg2://user:password@host:port/database
        port = self.port or 5439
        return f"postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{port}/{self.database}"

    def _bigquery_connection_string(self) -> str:
        """Generate Google BigQuery connection string."""
        # Format: bigquery://project/dataset
        base = f"bigquery://{self.project_id}"

        if self.dataset:
            base += f"/{self.dataset}"

        params = []
        if self.credentials_path:
            params.append(f"credentials_path={self.credentials_path}")

        for key, value in self.extra_params.items():
            params.append(f"{key}={value}")

        if params:
            base += "?" + "&".join(params)

        return base

    def _synapse_connection_string(self) -> str:
        """Generate Azure Synapse connection string."""
        # Format: mssql+pyodbc://user:password@server/database?driver=ODBC+Driver+17+for+SQL+Server
        driver = self.extra_params.get("driver", "ODBC Driver 17 for SQL Server")

        if self.authentication == "sql":
            base = f"mssql+pyodbc://{self.username}:{self.password}@{self.server}/{self.database}"
        else:
            # Azure AD authentication
            base = f"mssql+pyodbc://@{self.server}/{self.database}"

        params = [f"driver={driver.replace(' ', '+')}", "encrypt=yes", "TrustServerCertificate=no"]

        if self.authentication == "ad":
            params.append("Authentication=ActiveDirectoryPassword")
        elif self.authentication == "msi":
            params.append("Authentication=ActiveDirectoryMsi")

        for key, value in self.extra_params.items():
            if key != "driver":
                params.append(f"{key}={value}")

        return base + "?" + "&".join(params)

    def mask_sensitive(self) -> Dict[str, Any]:
        """Return configuration with sensitive data masked."""
        data = {
            "warehouse_type": self.warehouse_type.value,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "schema": self.schema,
            "username": self.username,
            "password": "***MASKED***" if self.password else None,
            "ssl_enabled": self.ssl_enabled,
        }

        if self.warehouse_type == WarehouseType.SNOWFLAKE:
            data.update({
                "account": self.account,
                "warehouse": self.warehouse,
                "role": self.role,
            })
        elif self.warehouse_type == WarehouseType.BIGQUERY:
            data.update({
                "project_id": self.project_id,
                "dataset": self.dataset,
                "credentials_path": self.credentials_path,
            })
        elif self.warehouse_type == WarehouseType.REDSHIFT:
            data.update({
                "cluster_identifier": self.cluster_identifier,
                "region": self.region,
            })
        elif self.warehouse_type == WarehouseType.SYNAPSE:
            data.update({
                "server": self.server,
                "authentication": self.authentication,
                "client_secret": "***MASKED***" if self.client_secret else None,
            })

        return data


@dataclass
class WarehouseConfig:
    """
    Centralized warehouse configuration management.

    Supports loading configurations from:
    - Environment variables
    - Configuration files (JSON)
    - Direct initialization
    """

    name: str
    credentials: WarehouseCredentials
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (sensitive data masked)."""
        return {
            "name": self.name,
            "warehouse_type": self.credentials.warehouse_type.value,
            "credentials": self.credentials.mask_sensitive(),
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
            "metadata": self.metadata,
        }

    @classmethod
    def from_env(cls, prefix: str = "", warehouse_type: WarehouseType = WarehouseType.SNOWFLAKE) -> "WarehouseConfig":
        """
        Load configuration from environment variables.

        Environment variable naming convention:
        - {PREFIX}_HOST, {PREFIX}_PORT, {PREFIX}_DATABASE, etc.

        Examples:
            SNOWFLAKE_ACCOUNT, SNOWFLAKE_USERNAME, SNOWFLAKE_PASSWORD
            REDSHIFT_HOST, REDSHIFT_PORT, REDSHIFT_DATABASE
            BIGQUERY_PROJECT_ID, BIGQUERY_DATASET
            SYNAPSE_SERVER, SYNAPSE_DATABASE
        """
        prefix = prefix.upper() + "_" if prefix else ""

        credentials = WarehouseCredentials(
            warehouse_type=warehouse_type,
            # Common
            host=os.getenv(f"{prefix}HOST"),
            port=int(os.getenv(f"{prefix}PORT", "0")) or None,
            database=os.getenv(f"{prefix}DATABASE"),
            schema=os.getenv(f"{prefix}SCHEMA"),
            username=os.getenv(f"{prefix}USERNAME") or os.getenv(f"{prefix}USER"),
            password=os.getenv(f"{prefix}PASSWORD"),
            # Snowflake
            account=os.getenv(f"{prefix}ACCOUNT"),
            warehouse=os.getenv(f"{prefix}WAREHOUSE"),
            role=os.getenv(f"{prefix}ROLE"),
            # BigQuery
            project_id=os.getenv(f"{prefix}PROJECT_ID") or os.getenv(f"{prefix}PROJECT"),
            dataset=os.getenv(f"{prefix}DATASET"),
            credentials_path=os.getenv(f"{prefix}CREDENTIALS_PATH") or os.getenv(f"{prefix}GOOGLE_APPLICATION_CREDENTIALS"),
            # Redshift
            cluster_identifier=os.getenv(f"{prefix}CLUSTER_IDENTIFIER") or os.getenv(f"{prefix}CLUSTER"),
            region=os.getenv(f"{prefix}REGION") or os.getenv(f"{prefix}AWS_REGION"),
            iam_role=os.getenv(f"{prefix}IAM_ROLE"),
            # Synapse
            server=os.getenv(f"{prefix}SERVER"),
            authentication=os.getenv(f"{prefix}AUTHENTICATION", "sql"),
            tenant_id=os.getenv(f"{prefix}TENANT_ID"),
            client_id=os.getenv(f"{prefix}CLIENT_ID"),
            client_secret=os.getenv(f"{prefix}CLIENT_SECRET"),
            # Options
            ssl_enabled=os.getenv(f"{prefix}SSL_ENABLED", "true").lower() == "true",
            connection_timeout=int(os.getenv(f"{prefix}CONNECTION_TIMEOUT", "30")),
            query_timeout=int(os.getenv(f"{prefix}QUERY_TIMEOUT", "300")),
        )

        return cls(
            name=os.getenv(f"{prefix}NAME", warehouse_type.value),
            credentials=credentials,
            description=os.getenv(f"{prefix}DESCRIPTION"),
        )

    @classmethod
    def from_file(cls, file_path: str) -> "WarehouseConfig":
        """Load configuration from JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(path, "r") as f:
            data = json.load(f)

        warehouse_type = WarehouseType(data.get("warehouse_type", "snowflake"))

        credentials = WarehouseCredentials(
            warehouse_type=warehouse_type,
            **{k: v for k, v in data.get("credentials", {}).items() if k != "warehouse_type"}
        )

        return cls(
            name=data.get("name", warehouse_type.value),
            credentials=credentials,
            description=data.get("description"),
            tags=data.get("tags", []),
            is_active=data.get("is_active", True),
            metadata=data.get("metadata", {}),
        )


class WarehouseConfigManager:
    """
    Manages multiple warehouse configurations.

    Features:
    - Load/save configurations
    - Manage multiple warehouses
    - Environment-based configuration
    """

    def __init__(self, config_dir: str = ".warehouse_configs"):
        self._config_dir = Path(config_dir)
        self._configs: Dict[str, WarehouseConfig] = {}
        self._ensure_storage()
        self._load_configs()

    def _ensure_storage(self) -> None:
        """Create storage directory with secure permissions."""
        if not self._config_dir.exists():
            self._config_dir.mkdir(parents=True, mode=0o700)

    def add_config(self, config: WarehouseConfig) -> None:
        """Add or update a warehouse configuration."""
        config.updated_at = datetime.utcnow()
        self._configs[config.name] = config
        self._save_configs()

    def get_config(self, name: str) -> Optional[WarehouseConfig]:
        """Get a warehouse configuration by name."""
        return self._configs.get(name)

    def remove_config(self, name: str) -> bool:
        """Remove a warehouse configuration."""
        if name in self._configs:
            del self._configs[name]
            self._save_configs()
            return True
        return False

    def list_configs(self) -> List[WarehouseConfig]:
        """List all warehouse configurations."""
        return list(self._configs.values())

    def get_by_type(self, warehouse_type: WarehouseType) -> List[WarehouseConfig]:
        """Get all configurations of a specific warehouse type."""
        return [
            cfg for cfg in self._configs.values()
            if cfg.credentials.warehouse_type == warehouse_type
        ]

    def _save_configs(self) -> None:
        """Save configurations to storage."""
        configs_file = self._config_dir / "configs.json"

        # Don't save actual passwords to file - use environment variables
        data = {}
        for name, config in self._configs.items():
            data[name] = {
                "name": config.name,
                "warehouse_type": config.credentials.warehouse_type.value,
                "description": config.description,
                "tags": config.tags,
                "created_at": config.created_at.isoformat(),
                "updated_at": config.updated_at.isoformat() if config.updated_at else None,
                "is_active": config.is_active,
                "metadata": config.metadata,
                # Store non-sensitive credential info
                "credentials": {
                    "host": config.credentials.host,
                    "port": config.credentials.port,
                    "database": config.credentials.database,
                    "schema": config.credentials.schema,
                    "account": config.credentials.account,
                    "warehouse": config.credentials.warehouse,
                    "project_id": config.credentials.project_id,
                    "dataset": config.credentials.dataset,
                    "server": config.credentials.server,
                    "cluster_identifier": config.credentials.cluster_identifier,
                    "region": config.credentials.region,
                    "ssl_enabled": config.credentials.ssl_enabled,
                }
            }

        with open(configs_file, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(configs_file, 0o600)

    def _load_configs(self) -> None:
        """Load configurations from storage."""
        configs_file = self._config_dir / "configs.json"
        if not configs_file.exists():
            return

        with open(configs_file, "r") as f:
            data = json.load(f)

        for name, cfg_data in data.items():
            warehouse_type = WarehouseType(cfg_data.get("warehouse_type", "snowflake"))
            creds_data = cfg_data.get("credentials", {})

            credentials = WarehouseCredentials(
                warehouse_type=warehouse_type,
                host=creds_data.get("host"),
                port=creds_data.get("port"),
                database=creds_data.get("database"),
                schema=creds_data.get("schema"),
                account=creds_data.get("account"),
                warehouse=creds_data.get("warehouse"),
                project_id=creds_data.get("project_id"),
                dataset=creds_data.get("dataset"),
                server=creds_data.get("server"),
                cluster_identifier=creds_data.get("cluster_identifier"),
                region=creds_data.get("region"),
                ssl_enabled=creds_data.get("ssl_enabled", True),
            )

            self._configs[name] = WarehouseConfig(
                name=cfg_data.get("name", name),
                credentials=credentials,
                description=cfg_data.get("description"),
                tags=cfg_data.get("tags", []),
                created_at=datetime.fromisoformat(cfg_data["created_at"]) if cfg_data.get("created_at") else datetime.utcnow(),
                updated_at=datetime.fromisoformat(cfg_data["updated_at"]) if cfg_data.get("updated_at") else None,
                is_active=cfg_data.get("is_active", True),
                metadata=cfg_data.get("metadata", {}),
            )

    @classmethod
    def load_from_env(cls) -> "WarehouseConfigManager":
        """
        Load all warehouse configurations from environment variables.

        Looks for:
        - SNOWFLAKE_* variables
        - REDSHIFT_* variables
        - BIGQUERY_* variables
        - SYNAPSE_* variables
        """
        manager = cls()

        # Try loading each warehouse type from environment
        warehouse_envs = [
            ("SNOWFLAKE", WarehouseType.SNOWFLAKE),
            ("REDSHIFT", WarehouseType.REDSHIFT),
            ("BIGQUERY", WarehouseType.BIGQUERY),
            ("SYNAPSE", WarehouseType.SYNAPSE),
        ]

        for prefix, wh_type in warehouse_envs:
            try:
                # Check if any env vars exist for this prefix
                has_config = any(
                    os.getenv(f"{prefix}_{suffix}")
                    for suffix in ["HOST", "ACCOUNT", "PROJECT_ID", "SERVER", "DATABASE"]
                )

                if has_config:
                    config = WarehouseConfig.from_env(prefix, wh_type)
                    manager.add_config(config)
            except Exception:
                pass  # Skip if configuration is incomplete

        return manager
