"""
Data Connectors for Quality Agent

Provides unified interface to read data from various sources
for quality evaluation.
"""

from .data_connector import (
    DataConnector,
    DataSource,
    ParquetConnector,
    CSVConnector,
    SQLConnector,
    DeltaConnector,
    create_connector
)

__all__ = [
    "DataConnector",
    "DataSource",
    "ParquetConnector",
    "CSVConnector",
    "SQLConnector",
    "DeltaConnector",
    "create_connector"
]
