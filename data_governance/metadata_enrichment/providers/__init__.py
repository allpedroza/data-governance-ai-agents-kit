"""
Providers for Metadata Enrichment Agent
Reuses providers from rag_discovery for consistency
"""

# Import from rag_discovery to avoid duplication
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from rag_discovery.providers.base import (
        EmbeddingProvider,
        LLMProvider,
        VectorStoreProvider,
        EmbeddingResult,
        LLMResponse,
        VectorSearchResult
    )
except ImportError:
    from data_governance.rag_discovery.providers.base import (
        EmbeddingProvider,
        LLMProvider,
        VectorStoreProvider,
        EmbeddingResult,
        LLMResponse,
        VectorSearchResult
    )

try:
    from rag_discovery.providers.embeddings import (
        SentenceTransformerEmbeddings,
        OpenAIEmbeddings
    )
except ImportError:
    from data_governance.rag_discovery.providers.embeddings import (
        SentenceTransformerEmbeddings,
        OpenAIEmbeddings
    )

try:
    from rag_discovery.providers.llm import OpenAILLM
except ImportError:
    from data_governance.rag_discovery.providers.llm import OpenAILLM

try:
    from rag_discovery.providers.llm import VertexAILLM
except ImportError:
    try:
        from data_governance.rag_discovery.providers.llm import VertexAILLM
    except ImportError:
        VertexAILLM = None

try:
    from rag_discovery.providers.vectorstore import ChromaStore
except ImportError:
    from data_governance.rag_discovery.providers.vectorstore import ChromaStore

try:
    from rag_discovery.providers.vectorstore import FAISSStore
except ImportError:
    try:
        from data_governance.rag_discovery.providers.vectorstore import FAISSStore
    except ImportError:
        FAISSStore = None

__all__ = [
    # Base classes
    "EmbeddingProvider",
    "LLMProvider",
    "VectorStoreProvider",
    "EmbeddingResult",
    "LLMResponse",
    "VectorSearchResult",
    # Implementations
    "SentenceTransformerEmbeddings",
    "OpenAIEmbeddings",
    "OpenAILLM",
    "VertexAILLM",
    "ChromaStore",
    "FAISSStore"
]
