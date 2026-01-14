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
OpenAI Embeddings Provider
API-based embeddings with high quality
"""

import os
from typing import List, Optional

from ..base import EmbeddingProvider, EmbeddingResult


class OpenAIEmbeddings(EmbeddingProvider):
    """
    OpenAI API embeddings

    Models available:
    - text-embedding-3-small (1536 dim, $0.02/1M tokens)
    - text-embedding-3-large (3072 dim, $0.13/1M tokens)
    - text-embedding-ada-002 (1536 dim, legacy)

    Advantages:
    - High quality embeddings
    - Scalable
    - Supports Azure OpenAI with base_url
    """

    # Model dimensions mapping
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None
    ):
        """
        Initialize OpenAI embeddings

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom API base URL (for Azure or proxies)
            organization: OpenAI organization ID
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai not installed. Install with: pip install openai"
            )

        self._model_name = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_API_URL")

        if not self._api_key:
            raise ValueError(
                "OpenAI API key not provided. "
                "Set OPENAI_API_KEY environment variable or pass api_key parameter."
            )

        # Initialize client
        client_kwargs = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        if organization:
            client_kwargs["organization"] = organization

        self._client = OpenAI(**client_kwargs)

        # Set dimension based on model
        self._dimension = self.MODEL_DIMENSIONS.get(model, 1536)

    def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text"""
        response = self._client.embeddings.create(
            input=text,
            model=self._model_name
        )

        return EmbeddingResult(
            vector=response.data[0].embedding,
            model=self._model_name,
            tokens_used=response.usage.total_tokens
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts"""
        if not texts:
            return []

        # OpenAI supports batch embedding natively
        response = self._client.embeddings.create(
            input=texts,
            model=self._model_name
        )

        # Distribute tokens across results
        tokens_per_text = response.usage.total_tokens // len(texts)

        results = []
        for item in response.data:
            results.append(EmbeddingResult(
                vector=item.embedding,
                model=self._model_name,
                tokens_used=tokens_per_text
            ))

        return results

    @property
    def dimension(self) -> int:
        """Embedding vector dimension"""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Model name"""
        return self._model_name
