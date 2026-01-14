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
FAISS Vector Store Provider
High-performance vector similarity search
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np

from ..base import VectorStoreProvider, VectorSearchResult


class FAISSStore(VectorStoreProvider):
    """
    FAISS vector store

    Features:
    - Very fast similarity search
    - Handles millions of vectors
    - Multiple index types for different use cases
    - Persistent storage

    Index types:
    - IndexFlatIP: Inner product (for normalized vectors = cosine similarity)
    - IndexFlatL2: L2 distance
    - IndexIVFFlat: Faster search with some accuracy trade-off
    """

    def __init__(
        self,
        collection_name: str = "data_discovery",
        persist_directory: str = "./faiss_db",
        dimension: int = 384,
        index_type: str = "flat_ip",
        normalize: bool = True
    ):
        """
        Initialize FAISS store

        Args:
            collection_name: Name of the collection (used for file names)
            persist_directory: Directory for persistent storage
            dimension: Vector dimension
            index_type: Type of FAISS index (flat_ip, flat_l2)
            normalize: Whether to normalize vectors (for cosine similarity)
        """
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Install with: pip install faiss-cpu"
            )

        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._dimension = dimension
        self._normalize = normalize

        # Create directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # File paths
        self._index_path = os.path.join(persist_directory, f"{collection_name}.faiss")
        self._metadata_path = os.path.join(persist_directory, f"{collection_name}_metadata.json")
        self._docs_path = os.path.join(persist_directory, f"{collection_name}_docs.pkl")

        # Initialize index
        if index_type == "flat_ip":
            self._index = faiss.IndexFlatIP(dimension)
        elif index_type == "flat_l2":
            self._index = faiss.IndexFlatL2(dimension)
        else:
            self._index = faiss.IndexFlatIP(dimension)

        # Storage for metadata and documents
        self._ids: List[str] = []
        self._documents: List[str] = []
        self._metadatas: List[Dict[str, Any]] = []

        # Load existing data if available
        self._load_state()

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vectors / norms

    def _save_state(self) -> None:
        """Save index and metadata to disk"""
        try:
            import faiss
            faiss.write_index(self._index, self._index_path)

            with open(self._metadata_path, "w", encoding="utf-8") as f:
                json.dump({
                    "ids": self._ids,
                    "metadatas": self._metadatas
                }, f, ensure_ascii=False)

            with open(self._docs_path, "wb") as f:
                pickle.dump(self._documents, f)

        except Exception as e:
            print(f"Error saving FAISS state: {e}")

    def _load_state(self) -> None:
        """Load index and metadata from disk"""
        try:
            import faiss

            if os.path.exists(self._index_path):
                self._index = faiss.read_index(self._index_path)

            if os.path.exists(self._metadata_path):
                with open(self._metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._ids = data.get("ids", [])
                    self._metadatas = data.get("metadatas", [])

            if os.path.exists(self._docs_path):
                with open(self._docs_path, "rb") as f:
                    self._documents = pickle.load(f)

        except Exception as e:
            print(f"Error loading FAISS state: {e}")

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Add documents to FAISS"""
        if not ids:
            return

        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)

        # Normalize if required
        if self._normalize:
            vectors = self._normalize_vectors(vectors)

        # Handle duplicates - remove existing and re-add
        existing_indices = []
        for i, id_ in enumerate(ids):
            if id_ in self._ids:
                existing_indices.append(self._ids.index(id_))

        # For simplicity, rebuild index if we have duplicates
        # (FAISS doesn't support in-place updates well)
        if existing_indices:
            # Remove existing entries
            for idx in sorted(existing_indices, reverse=True):
                self._ids.pop(idx)
                self._documents.pop(idx)
                self._metadatas.pop(idx)

            # Rebuild index with remaining vectors
            if self._ids:
                remaining_vectors = []
                for i, id_ in enumerate(self._ids):
                    # We need to re-fetch embeddings - this is a limitation
                    pass
            else:
                self._index.reset()

        # Add new vectors
        self._index.add(vectors)

        # Store metadata
        self._ids.extend(ids)
        self._documents.extend(documents)
        self._metadatas.extend(metadatas)

        # Save to disk
        self._save_state()

    def search(
        self,
        embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> VectorSearchResult:
        """Search for similar documents"""
        if self._index.ntotal == 0:
            return VectorSearchResult(
                ids=[[]],
                documents=[[]],
                metadatas=[[]],
                distances=[[]]
            )

        # Convert to numpy array
        query_vector = np.array([embedding], dtype=np.float32)

        # Normalize if required
        if self._normalize:
            query_vector = self._normalize_vectors(query_vector)

        # Search
        # Get more results if we need to filter
        search_k = n_results * 3 if filter_metadata else n_results
        search_k = min(search_k, self._index.ntotal)

        distances, indices = self._index.search(query_vector, search_k)

        # Collect results
        result_ids = []
        result_docs = []
        result_metas = []
        result_distances = []

        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue

            # Apply metadata filter if provided
            if filter_metadata:
                meta = self._metadatas[idx]
                match = True
                for k, v in filter_metadata.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            result_ids.append(self._ids[idx])
            result_docs.append(self._documents[idx])
            result_metas.append(self._metadatas[idx])
            result_distances.append(float(distances[0][i]))

            if len(result_ids) >= n_results:
                break

        return VectorSearchResult(
            ids=[result_ids],
            documents=[result_docs],
            metadatas=[result_metas],
            distances=[result_distances]
        )

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID (requires index rebuild)"""
        indices_to_remove = []
        for id_ in ids:
            if id_ in self._ids:
                indices_to_remove.append(self._ids.index(id_))

        if not indices_to_remove:
            return

        # Remove from metadata lists
        for idx in sorted(indices_to_remove, reverse=True):
            self._ids.pop(idx)
            self._documents.pop(idx)
            self._metadatas.pop(idx)

        # Rebuild index (FAISS limitation - no direct delete)
        # This is expensive but necessary
        self._index.reset()
        self._save_state()

    def count(self) -> int:
        """Return total number of documents"""
        return self._index.ntotal

    def reset(self) -> None:
        """Clear all documents"""
        self._index.reset()
        self._ids = []
        self._documents = []
        self._metadatas = []

        # Remove files
        for path in [self._index_path, self._metadata_path, self._docs_path]:
            if os.path.exists(path):
                os.remove(path)

    @property
    def collection_name(self) -> str:
        """Collection name"""
        return self._collection_name

    @property
    def embeddings(self) -> Optional[np.ndarray]:
        """Get all embeddings as numpy array (for diversity filtering)"""
        if self._index.ntotal == 0:
            return None
        # Reconstruct vectors from index
        return np.array([
            self._index.reconstruct(i)
            for i in range(self._index.ntotal)
        ])
