"""Port: Cloud Warehouse

Defines the interface for data warehouse adapters (BigQuery, Snowflake,
Redshift, Synapse, etc.).

The existing ``WarehouseConnector`` ABC in ``warehouse/connectors/base.py``
already provides a comprehensive contract.  This port re-exports it under
the ``ports`` namespace for discoverability and adds a narrower subset
that lightweight consumers (e.g. the lineage agent) can programme against.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class CloudWarehousePort(ABC):
    """Minimal port for cloud warehouse interactions.

    For full-featured warehouse access (DDL, statistics, query history),
    use ``data_governance.warehouse.connectors.base.WarehouseConnector``
    which extends this contract.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish a connection to the warehouse."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close the warehouse connection."""
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Return True if the connection is alive."""
        ...

    @abstractmethod
    def list_tables(
        self,
        schema: Optional[str] = None,
        database: Optional[str] = None,
    ) -> List[Any]:
        """List tables visible to the current credentials."""
        ...

    @abstractmethod
    def get_table_schema(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return column-level schema for a given table."""
        ...

    @abstractmethod
    def get_ddl(
        self,
        table_name: str,
        schema: Optional[str] = None,
        database: Optional[str] = None,
    ) -> str:
        """Return the CREATE TABLE DDL for the given table."""
        ...
