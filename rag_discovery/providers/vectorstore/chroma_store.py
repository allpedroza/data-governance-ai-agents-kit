"""
ChromaDB Vector Store Provider
Persistent vector database with rich metadata filtering
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from ..base import VectorStoreProvider, VectorSearchResult


class ChromaStore(VectorStoreProvider):
    """
    ChromaDB vector store

    Features:
    - Persistent storage
    - Rich metadata filtering
    - Easy to use
    - Good for small to medium datasets

    Note: For very large datasets (>1M vectors), consider FAISS or Pinecone.
    """

    def __init__(
        self,
        collection_name: str = "data_discovery",
        persist_directory: str = "./chroma_db",
        distance_metric: str = "cosine"
    ):
        """
        Initialize ChromaDB store

        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
            distance_metric: Distance metric (cosine, l2, ip)
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb not installed. Install with: pip install chromadb"
            )

        self._collection_name = collection_name
        self._persist_directory = persist_directory

        # Create directory if not exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Map distance metric to ChromaDB format
        space_mapping = {
            "cosine": "cosine",
            "l2": "l2",
            "ip": "ip"  # inner product
        }
        space = space_mapping.get(distance_metric, "cosine")

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": space}
        )

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Add documents to ChromaDB"""
        if not ids:
            return

        # ChromaDB has restrictions on metadata values
        # Convert any None or complex types to strings
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for k, v in meta.items():
                if v is None:
                    clean_meta[k] = ""
                elif isinstance(v, (list, dict)):
                    import json
                    clean_meta[k] = json.dumps(v, ensure_ascii=False)
                elif isinstance(v, bool):
                    clean_meta[k] = str(v).lower()
                else:
                    clean_meta[k] = v
            clean_metadatas.append(clean_meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=clean_metadatas
        )

    def search(
        self,
        embedding: List[float],
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> VectorSearchResult:
        """Search for similar documents"""
        query_kwargs = {
            "query_embeddings": [embedding],
            "n_results": n_results
        }

        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = self._collection.query(**query_kwargs)

        return VectorSearchResult(
            ids=results.get("ids", [[]]),
            documents=results.get("documents", [[]]),
            metadatas=results.get("metadatas", [[]]),
            distances=results.get("distances", [[]])
        )

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID"""
        if ids:
            self._collection.delete(ids=ids)

    def count(self) -> int:
        """Return total number of documents"""
        return self._collection.count()

    def reset(self) -> None:
        """Clear all documents"""
        self._client.delete_collection(name=self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name
        )

    def get(
        self,
        ids: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get documents by ID or all documents"""
        kwargs = {}
        if ids:
            kwargs["ids"] = ids
        if limit:
            kwargs["limit"] = limit

        return self._collection.get(**kwargs)

    @property
    def collection_name(self) -> str:
        """Collection name"""
        return self._collection_name
