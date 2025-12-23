"""
Data Discovery Connectors
Integrations with external metadata catalogs
"""

from .openmetadata_connector import (
    OpenMetadataConnector,
    OpenMetadataConfig
)

__all__ = [
    "OpenMetadataConnector",
    "OpenMetadataConfig"
]
