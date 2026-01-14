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
SentenceTransformer Embeddings Provider
Local embeddings - no API cost, low latency (~5-10ms)
"""

from typing import List
import numpy as np

from ..base import EmbeddingProvider, EmbeddingResult


class SentenceTransformerEmbeddings(EmbeddingProvider):
    """
    Local embeddings using SentenceTransformers

    Advantages:
    - No API cost
    - Low latency (~5-10ms per embedding)
    - Works offline
    - No rate limits

    Models available:
    - all-MiniLM-L6-v2 (384 dim, fast, good quality)
    - all-mpnet-base-v2 (768 dim, slower, better quality)
    - paraphrase-multilingual-MiniLM-L12-v2 (384 dim, multilingual)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize SentenceTransformer embeddings

        Args:
            model_name: HuggingFace model name
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text"""
        vector = self._model.encode(text, show_progress_bar=False)

        # Convert numpy array to list
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        return EmbeddingResult(
            vector=vector,
            model=self._model_name,
            tokens_used=0  # Local model, no token tracking
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts (more efficient)"""
        if not texts:
            return []

        vectors = self._model.encode(texts, show_progress_bar=False)

        results = []
        for vector in vectors:
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()

            results.append(EmbeddingResult(
                vector=vector,
                model=self._model_name,
                tokens_used=0
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

    def normalize(self, vectors: List[List[float]]) -> List[List[float]]:
        """Normalize vectors to unit length (for cosine similarity)"""
        arr = np.array(vectors)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = arr / norms
        return normalized.tolist()
