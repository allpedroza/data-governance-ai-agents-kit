"""
Data Discovery RAG Agent
Sistema de IA para descoberta de dados usando RAG com banco vetorizado
"""

from .data_discovery_rag_agent import (
    DataDiscoveryRAGAgent,
    TableMetadata,
    SearchResult,
    create_sample_metadata
)

__version__ = "1.0.0"

__all__ = [
    'DataDiscoveryRAGAgent',
    'TableMetadata',
    'SearchResult',
    'create_sample_metadata'
]
