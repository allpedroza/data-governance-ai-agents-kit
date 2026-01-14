# /// script
# dependencies = [
#   "azure-identity>=1.12.0",
#   "azure-storage-blob>=12.14.0",
#   "black>=22.0.0",
#   "boto3>=1.26.0",
#   "chromadb>=0.4.0",
#   "cryptography>=41.0.0",
#   "databricks-sdk>=0.5.0",
#   "faiss-cpu>=1.7.0",
#   "flake8>=5.0.0",
#   "google-cloud-bigquery-storage>=2.0.0",
#   "google-cloud-bigquery>=3.0.0",
#   "google-cloud-storage>=2.7.0",
#   "isort>=5.0.0",
#   "kaleido>=0.2.0",
#   "matplotlib>=3.6.0",
#   "mypy>=1.0.0",
#   "networkx>=3.0",
#   "numpy>=1.24.0",
#   "openai>=1.0.0",
#   "openpyxl>=3.0.0",
#   "pandas>=2.0.0",
#   "plotly>=5.0.0",
#   "psycopg2-binary>=2.9.0",
#   "pyarrow>=14.0.0",
#   "pyodbc>=4.0.0",
#   "pyspark>=3.3.0",
#   "pytest-cov>=4.0.0",
#   "pytest>=7.0.0",
#   "python-dotenv>=1.0.0",
#   "python-igraph>=0.10.0",
#   "pyyaml>=6.0",
#   "redshift-connector>=2.0.0",
#   "requests>=2.31.0",
#   "scikit-learn>=1.0.0",
#   "seaborn>=0.12.0",
#   "sentence-transformers>=2.2.0",
#   "snowflake-connector-python>=3.0.0",
#   "snowflake-sqlalchemy>=1.5.0",
#   "spacy>=3.5.0; extra == "spacy"",
#   "sphinx-rtd-theme>=1.0.0",
#   "sphinx>=5.0.0",
#   "sqlalchemy-bigquery>=1.6.0",
#   "sqlalchemy-redshift>=0.8.0",
#   "sqlalchemy>=2.0.0",
#   "sqlparse>=0.4.0",
#   "streamlit>=1.32.0",
#   "tqdm>=4.65.0",
# ]
# ///
"""
Base classes for providers - Abstract interfaces for embeddings, LLM, and vector stores
Enables swapping providers without changing main code (no vendor lock-in)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class EmbeddingResult:
    """Standardized embedding result"""
    vector: List[float]
    model: str
    tokens_used: int = 0
    dimension: int = field(init=False)

    def __post_init__(self):
        self.dimension = len(self.vector)


@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = field(init=False)

    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens


class EmbeddingProvider(ABC):
    """
    Abstract interface for embedding providers

    Implementations:
    - SentenceTransformerEmbeddings (local, no API cost)
    - OpenAIEmbeddings (API-based)
    - AzureOpenAIEmbeddings (Azure)
    - BedrockEmbeddings (AWS)
    """

    @abstractmethod
    def embed(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text

        Args:
            text: Text to embed

        Returns:
            EmbeddingResult with vector and metadata
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts (more efficient)

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResult
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name/identifier"""
        pass


class LLMProvider(ABC):
    """
    Abstract interface for LLM providers

    Implementations:
    - OpenAILLM (GPT-4, GPT-4o, etc.)
    - VertexAILLM (Gemini)
    - AzureOpenAILLM
    - BedrockLLM (Claude, etc.)
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text response from LLM

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and metadata
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name/identifier"""
        pass


@dataclass
class VectorSearchResult:
    """Result from vector store search"""
    ids: List[List[str]]
    documents: List[List[str]]
    metadatas: List[List[Dict[str, Any]]]
    distances: List[List[float]]


class VectorStoreProvider(ABC):
    """
    Abstract interface for vector stores

    Implementations:
    - ChromaStore (ChromaDB)
    - FAISSStore (FAISS)
    - PineconeStore (Pinecone)
    """

    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Add documents to the vector store

        Args:
            ids: Unique identifiers for documents
            embeddings: Pre-computed embedding vectors
            documents: Original text documents
            metadatas: Metadata dictionaries for each document
        """
        pass

    @abstractmethod
    def search(
        self,
        embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> VectorSearchResult:
        """
        Search for similar documents

        Args:
            embedding: Query embedding vector
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            VectorSearchResult with ids, documents, metadatas, distances
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID"""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of documents in store"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all documents from store"""
        pass

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Collection/index name"""
        pass
