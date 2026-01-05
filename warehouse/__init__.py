"""
Warehouse Module - Centralized Data Warehouse Connections and Permissions

This module provides:
- Unified warehouse connection management
- Role-based access control for warehouse operations
- Support for Snowflake, Amazon Redshift, Google BigQuery, and Azure Synapse
- Integration with Lineage, Discovery, Enrichment, Classification, Quality, and Asset Value modules
"""

from .config import WarehouseConfig, WarehouseType
from .permissions import (
    WarehousePermission,
    WarehouseAccessLevel,
    WarehousePermissionManager,
    ModuleAccess
)
from .connectors import (
    WarehouseConnector,
    SnowflakeConnector,
    RedshiftConnector,
    BigQueryConnector,
    SynapseConnector,
    create_warehouse_connector
)
from .integration import WarehouseIntegration

__all__ = [
    # Config
    "WarehouseConfig",
    "WarehouseType",
    # Permissions
    "WarehousePermission",
    "WarehouseAccessLevel",
    "WarehousePermissionManager",
    "ModuleAccess",
    # Connectors
    "WarehouseConnector",
    "SnowflakeConnector",
    "RedshiftConnector",
    "BigQueryConnector",
    "SynapseConnector",
    "create_warehouse_connector",
    # Integration
    "WarehouseIntegration",
]
