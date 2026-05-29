"""Port: Metadata Catalog

Defines the interface that any metadata catalog adapter must implement
(OpenMetadata, Apache Atlas, Unity Catalog, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TableMetadataDTO:
    """Lightweight, adapter-agnostic representation of a catalog table.

    Adapters convert their native responses into this DTO so that agents
    never depend on a specific catalog SDK.
    """

    name: str
    database: str = ""
    schema: str = ""
    description: str = ""
    columns: List[Dict[str, str]] = field(default_factory=list)
    owner: str = ""
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


class MetadataCatalogPort(ABC):
    """Abstract port for metadata catalog integrations.

    Implementations must convert their native table objects into
    :class:`TableMetadataDTO` instances.
    """

    @abstractmethod
    def fetch_tables(
        self,
        max_tables: int = 200,
        service_filter: Optional[str] = None,
    ) -> List[TableMetadataDTO]:
        """Retrieve tables from the catalog.

        Args:
            max_tables: Upper bound on the number of tables returned.
            service_filter: Optional filter applied to the catalog service.

        Returns:
            List of table metadata DTOs.
        """
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Verify that the catalog is reachable.

        Returns:
            True if the connection succeeds.

        Raises:
            ConnectionError or subclass on failure.
        """
        ...
