"""
Metadata Enrichment Agent

Generates table and column descriptions, tags, and classifications
using RAG over architecture standards and data sampling.
"""

from .agent import MetadataEnrichmentAgent, EnrichmentResult, ColumnEnrichment

__all__ = [
    "MetadataEnrichmentAgent",
    "EnrichmentResult",
    "ColumnEnrichment"
]
