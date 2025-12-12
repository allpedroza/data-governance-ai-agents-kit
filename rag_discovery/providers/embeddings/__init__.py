"""
Embedding Providers - Implementações de geradores de embeddings
"""

from .sentence_transformer import SentenceTransformerEmbeddings
from .openai_embeddings import OpenAIEmbeddings

__all__ = [
    "SentenceTransformerEmbeddings",
    "OpenAIEmbeddings"
]
