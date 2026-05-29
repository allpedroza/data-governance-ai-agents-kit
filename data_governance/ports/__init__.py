"""Abstract port interfaces for external connectors.

These ports define the contracts that adapters (concrete connectors)
must implement.  Agents depend only on these abstractions, enabling:

- Easy swapping of catalog / warehouse backends
- Simplified unit testing via mock implementations
- Clear separation of concerns (Hexagonal Architecture)

Example::

    from data_governance.ports import MetadataCatalogPort

    class MyCustomCatalog(MetadataCatalogPort):
        ...

    agent = DataDiscoveryRAGAgent(catalog=MyCustomCatalog())
"""

from .catalog import MetadataCatalogPort
from .warehouse import CloudWarehousePort

__all__ = [
    "MetadataCatalogPort",
    "CloudWarehousePort",
]
