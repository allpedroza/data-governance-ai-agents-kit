"""
Providers - Abstrações para embeddings, LLM e vector stores
Permite trocar de provider sem alterar código principal (sem vendor lock-in)
"""

from .base import (
    EmbeddingProvider,
    EmbeddingResult,
    LLMProvider,
    LLMResponse,
    VectorStoreProvider
)

__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "LLMProvider",
    "LLMResponse",
    "VectorStoreProvider"
]
