"""
Warehouse Connectors Module

Provides specialized connectors for:
- Snowflake
- Amazon Redshift
- Google BigQuery
- Azure Synapse Analytics
"""

from .base import WarehouseConnector, WarehouseConnectionError
from .snowflake import SnowflakeConnector
from .redshift import RedshiftConnector
from .bigquery import BigQueryConnector
from .synapse import SynapseConnector


def create_warehouse_connector(
    warehouse_type: str,
    **kwargs
) -> WarehouseConnector:
    """
    Factory function to create appropriate warehouse connector.

    Args:
        warehouse_type: One of 'snowflake', 'redshift', 'bigquery', 'synapse'
        **kwargs: Connector-specific arguments

    Returns:
        WarehouseConnector instance
    """
    connectors = {
        "snowflake": SnowflakeConnector,
        "redshift": RedshiftConnector,
        "bigquery": BigQueryConnector,
        "synapse": SynapseConnector,
    }

    warehouse_type = warehouse_type.lower()
    if warehouse_type not in connectors:
        raise ValueError(f"Unknown warehouse type: {warehouse_type}. Supported: {list(connectors.keys())}")

    return connectors[warehouse_type](**kwargs)


__all__ = [
    "WarehouseConnector",
    "WarehouseConnectionError",
    "SnowflakeConnector",
    "RedshiftConnector",
    "BigQueryConnector",
    "SynapseConnector",
    "create_warehouse_connector",
]
