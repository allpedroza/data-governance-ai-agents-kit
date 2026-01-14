"""
Data Discovery RAG Agent v2
Sistema de IA para descoberta de dados usando RAG híbrido com Dartboard Ranking

Features:
- Dartboard Ranking (semantic + lexical + importance)
- Table validation against catalog
- Pluggable providers (no vendor lock-in)
- Diversity filtering
- Structured logging

Usage:
    from rag_discovery import DataDiscoveryAgent
    from rag_discovery.providers.embeddings import SentenceTransformerEmbeddings
    from rag_discovery.providers.llm import OpenAILLM
    from rag_discovery.providers.vectorstore import ChromaStore

    agent = DataDiscoveryAgent(
        embedding_provider=SentenceTransformerEmbeddings(),
        llm_provider=OpenAILLM(),
        vector_store=ChromaStore(),
        catalog_source="./catalog.txt"
    )
"""

__version__ = "2.0.0"

# New v2 exports
from .agent import (
    DataDiscoveryAgent,
    TableMetadata,
    DiscoveryResult
)

# Providers
from .providers import (
    EmbeddingProvider,
    EmbeddingResult,
    LLMProvider,
    LLMResponse,
    VectorStoreProvider
)

# Retrieval
from .retrieval import (
    HybridRetriever,
    RetrievalResult,
    LexicalIndex,
    DiversityFilter
)

# Validation
from .validation import (
    TableValidator,
    ValidationResult
)

# Utils
from .utils import (
    StructuredLogger,
    QueryLog,
    PORTUGUESE_STOPWORDS
)

# Legacy v1 compatibility (deprecated)
try:
    from .data_discovery_rag_agent import (
        DataDiscoveryRAGAgent,
        SearchResult,
        create_sample_metadata
    )
    _LEGACY_AVAILABLE = True
except ImportError:
    _LEGACY_AVAILABLE = False
    DataDiscoveryRAGAgent = None
    SearchResult = None
    create_sample_metadata = None

__all__ = [
    # Version
    "__version__",

    # Main agent
    "DataDiscoveryAgent",
    "TableMetadata",
    "DiscoveryResult",

    # Providers
    "EmbeddingProvider",
    "EmbeddingResult",
    "LLMProvider",
    "LLMResponse",
    "VectorStoreProvider",

    # Retrieval
    "HybridRetriever",
    "RetrievalResult",
    "LexicalIndex",
    "DiversityFilter",

    # Validation
    "TableValidator",
    "ValidationResult",

    # Utils
    "StructuredLogger",
    "QueryLog",
    "PORTUGUESE_STOPWORDS",

    # Legacy (deprecated)
    "DataDiscoveryRAGAgent",
    "SearchResult",
    "create_sample_metadata",
]
