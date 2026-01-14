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
Hybrid Retriever with Dartboard Ranking
Combines semantic search, lexical matching, and importance scoring
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np

from .lexical_index import LexicalIndex
from .diversity_filter import DiversityFilter, RetrievalResult, MMRFilter
from ..utils.stopwords import MULTILINGUAL_STOPWORDS

if TYPE_CHECKING:
    from ..providers.base import EmbeddingProvider, VectorStoreProvider


@dataclass
class HybridRetrieverConfig:
    """Configuration for hybrid retriever"""
    # Dartboard weights
    alpha: float = 0.7   # Semantic weight
    beta: float = 0.2    # Lexical (TF-IDF) weight
    gamma: float = 0.1   # Importance weight

    # Retrieval settings
    initial_k: int = 30  # Initial candidates from vector search
    diversity_threshold: float = 0.95  # Similarity threshold for diversity filter
    use_mmr: bool = False  # Use MMR instead of threshold-based diversity
    mmr_lambda: float = 0.7  # MMR lambda parameter

    # Lexical index settings
    ngram_range: tuple = (1, 2)
    min_df: int = 1
    max_df: float = 0.95


class HybridRetriever:
    """
    Hybrid Retriever implementing Dartboard Ranking algorithm

    Combines three signals:
    1. Semantic similarity (vector search)
    2. Lexical matching (TF-IDF)
    3. Document importance (based on content richness)

    Score = α × Semantic + β × Lexical + γ × Importance

    Features:
    - Configurable weights for each signal
    - Diversity filtering to avoid redundant results
    - Support for metadata filtering
    - Adjustable parameters at runtime
    """

    def __init__(
        self,
        embedding_provider: "EmbeddingProvider",
        vector_store: "VectorStoreProvider",
        config: Optional[HybridRetrieverConfig] = None
    ):
        """
        Initialize hybrid retriever

        Args:
            embedding_provider: Provider for generating embeddings
            vector_store: Vector database provider
            config: Configuration options
        """
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.config = config or HybridRetrieverConfig()

        # Initialize lexical index
        self.lexical_index = LexicalIndex(
            stopwords=MULTILINGUAL_STOPWORDS,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df
        )

        # Initialize diversity filter
        if self.config.use_mmr:
            self.diversity_filter = MMRFilter(
                lambda_param=self.config.mmr_lambda
            )
        else:
            self.diversity_filter = DiversityFilter(
                threshold=self.config.diversity_threshold
            )

        # Document storage for TF-IDF and importance
        self._documents: List[str] = []
        self._document_ids: List[str] = []
        self._document_embeddings: Optional[np.ndarray] = None
        self._importance_scores: Dict[str, float] = {}

    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> int:
        """
        Index documents in the hybrid retriever

        Args:
            documents: List of documents with 'id', 'text', 'metadata' keys
            show_progress: Whether to show progress

        Returns:
            Number of documents indexed
        """
        if not documents:
            return 0

        texts = [doc["text"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]

        # 1. Generate embeddings
        if show_progress:
            print(f"Generating embeddings for {len(documents)} documents...")

        embeddings_results = self.embedding_provider.embed_batch(texts)
        vectors = [e.vector for e in embeddings_results]

        # 2. Add to vector store
        if show_progress:
            print("Adding to vector store...")

        self.vector_store.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas
        )

        # 3. Build lexical index
        if show_progress:
            print("Building lexical index (TF-IDF)...")

        self._documents = texts
        self._document_ids = ids
        self.lexical_index.fit(texts)

        # 4. Store embeddings for diversity filtering
        self._document_embeddings = np.array(vectors)

        # 5. Calculate importance scores
        if show_progress:
            print("Calculating importance scores...")

        self._calculate_importance_scores(texts, ids)

        if show_progress:
            print(f"✅ Indexed {len(documents)} documents")

        return len(documents)

    def _calculate_importance_scores(
        self,
        texts: List[str],
        ids: List[str]
    ) -> None:
        """
        Calculate importance scores based on document richness

        Factors considered:
        - Word count
        - Unique terms
        - Information density
        """
        scores = []

        for text in texts:
            words = text.split()
            word_count = len(words)
            unique_words = len(set(words))

            # Richness = unique_words / word_count (information density)
            richness = unique_words / word_count if word_count > 0 else 0

            # Combined score: word count (size) * richness (quality)
            score = word_count * (1 + richness)
            scores.append(score)

        # Normalize to 0-1
        scores = np.array(scores)
        if scores.sum() > 0:
            scores = scores / scores.max()

        self._importance_scores = dict(zip(ids, scores))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        apply_diversity: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using Dartboard Ranking

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            apply_diversity: Whether to apply diversity filtering

        Returns:
            List of RetrievalResult sorted by combined score
        """
        # 1. Semantic search (get initial candidates)
        query_embedding = self.embedding_provider.embed(query)

        initial_k = max(self.config.initial_k, top_k * 3)
        semantic_results = self.vector_store.search(
            embedding=query_embedding.vector,
            n_results=initial_k,
            filter_metadata=filter_metadata
        )

        if not semantic_results.ids[0]:
            return []

        # 2. Calculate lexical scores for candidates
        candidate_indices = []
        for doc_id in semantic_results.ids[0]:
            if doc_id in self._document_ids:
                candidate_indices.append(self._document_ids.index(doc_id))
            else:
                candidate_indices.append(-1)

        lexical_scores = self.lexical_index.score_subset(query, candidate_indices)

        # 3. Combine scores using Dartboard formula
        results = []

        for i, doc_id in enumerate(semantic_results.ids[0]):
            # Semantic score: convert distance to similarity
            distance = semantic_results.distances[0][i]
            # For cosine distance, similarity = 1 - distance
            # For inner product, similarity = distance (already a similarity)
            semantic_score = 1.0 - min(distance / 2.0, 1.0)

            # Lexical score
            lexical_score = lexical_scores[i] if i < len(lexical_scores) else 0.0

            # Importance score
            importance_score = self._importance_scores.get(doc_id, 0.0)

            # Dartboard combined score
            combined_score = (
                self.config.alpha * semantic_score +
                self.config.beta * lexical_score +
                self.config.gamma * importance_score
            )

            results.append(RetrievalResult(
                chunk_id=doc_id,
                text=semantic_results.documents[0][i],
                metadata=semantic_results.metadatas[0][i],
                combined_score=combined_score,
                semantic_score=semantic_score,
                lexical_score=lexical_score,
                importance_score=importance_score
            ))

        # 4. Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        # 5. Apply diversity filtering
        if apply_diversity and self._document_embeddings is not None:
            if isinstance(self.diversity_filter, MMRFilter):
                results = self.diversity_filter.filter(
                    results,
                    self._document_embeddings,
                    self._document_ids,
                    top_k=top_k
                )
            else:
                results = self.diversity_filter.filter(
                    results,
                    self._document_embeddings,
                    self._document_ids
                )

        return results[:top_k]

    def adjust_weights(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None
    ) -> None:
        """
        Adjust Dartboard weights at runtime

        Args:
            alpha: Semantic weight (0-1)
            beta: Lexical weight (0-1)
            gamma: Importance weight (0-1)
        """
        if alpha is not None:
            self.config.alpha = alpha
        if beta is not None:
            self.config.beta = beta
        if gamma is not None:
            self.config.gamma = gamma

        print(f"Weights adjusted: α={self.config.alpha}, β={self.config.beta}, γ={self.config.gamma}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "total_documents": len(self._documents),
            "vector_store_count": self.vector_store.count(),
            "lexical_vocabulary_size": self.lexical_index.vocabulary_size,
            "weights": {
                "alpha": self.config.alpha,
                "beta": self.config.beta,
                "gamma": self.config.gamma
            },
            "diversity_threshold": self.config.diversity_threshold,
            "embedding_dimension": self.embedding_provider.dimension
        }


def create_hybrid_retriever(
    embedding_provider: "EmbeddingProvider",
    vector_store: "VectorStoreProvider",
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1,
    diversity_threshold: float = 0.95,
    use_mmr: bool = False
) -> HybridRetriever:
    """
    Factory function to create a hybrid retriever with custom settings

    Args:
        embedding_provider: Embedding provider
        vector_store: Vector store provider
        alpha: Semantic weight
        beta: Lexical weight
        gamma: Importance weight
        diversity_threshold: Diversity filter threshold
        use_mmr: Use MMR instead of threshold-based diversity

    Returns:
        Configured HybridRetriever instance
    """
    config = HybridRetrieverConfig(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        diversity_threshold=diversity_threshold,
        use_mmr=use_mmr
    )

    return HybridRetriever(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        config=config
    )
