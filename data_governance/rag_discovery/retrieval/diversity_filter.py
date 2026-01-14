"""
Diversity Filter for retrieval results
Removes redundant/similar results to increase information diversity
"""

from typing import List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from ..providers.base import EmbeddingProvider


@dataclass
class RetrievalResult:
    """Standardized retrieval result"""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    combined_score: float
    semantic_score: float
    lexical_score: float
    importance_score: float


class DiversityFilter:
    """
    Filters retrieval results to ensure diversity

    Uses cosine similarity between document embeddings to identify
    and remove redundant results that are too similar to already
    selected documents.

    Algorithm:
    1. Start with highest scored document
    2. For each candidate, check similarity with all selected documents
    3. If similarity > threshold, skip (redundant)
    4. Otherwise, add to selected set
    """

    def __init__(
        self,
        threshold: float = 0.95,
        min_results: int = 1
    ):
        """
        Initialize diversity filter

        Args:
            threshold: Maximum similarity allowed between results (0-1)
                      Higher = more similar results allowed
                      Lower = more diverse results required
            min_results: Minimum results to return even if similar
        """
        self.threshold = threshold
        self.min_results = min_results

    def filter(
        self,
        results: List[RetrievalResult],
        embeddings: np.ndarray,
        document_ids: List[str]
    ) -> List[RetrievalResult]:
        """
        Filter results for diversity

        Args:
            results: List of retrieval results (already sorted by score)
            embeddings: Document embeddings matrix (n_docs x dimension)
            document_ids: List of document IDs matching embeddings

        Returns:
            Filtered list of diverse results
        """
        if not results:
            return []

        if len(results) <= self.min_results:
            return results

        if embeddings is None or len(embeddings) == 0:
            return results

        # Create ID to index mapping
        id_to_idx = {id_: i for i, id_ in enumerate(document_ids)}

        selected: List[RetrievalResult] = []
        selected_embeddings: List[np.ndarray] = []

        for result in results:
            chunk_id = result.chunk_id

            # Get embedding for this result
            if chunk_id not in id_to_idx:
                # If we can't find embedding, include result anyway
                selected.append(result)
                continue

            idx = id_to_idx[chunk_id]
            candidate_embedding = embeddings[idx]

            # Check similarity with already selected results
            is_redundant = False

            for selected_emb in selected_embeddings:
                similarity = self._cosine_similarity(candidate_embedding, selected_emb)

                if similarity > self.threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                selected.append(result)
                selected_embeddings.append(candidate_embedding)

        # Ensure minimum results
        if len(selected) < self.min_results:
            # Add back some results even if similar
            for result in results:
                if result not in selected:
                    selected.append(result)
                    if len(selected) >= self.min_results:
                        break

        return selected

    def filter_with_provider(
        self,
        results: List[RetrievalResult],
        embedding_provider: "EmbeddingProvider"
    ) -> List[RetrievalResult]:
        """
        Filter results using embedding provider to compute similarities

        Args:
            results: List of retrieval results
            embedding_provider: Provider to generate embeddings

        Returns:
            Filtered list of diverse results
        """
        if not results or len(results) <= self.min_results:
            return results

        # Generate embeddings for all result texts
        texts = [r.text for r in results]
        embeddings_results = embedding_provider.embed_batch(texts)
        embeddings = np.array([e.vector for e in embeddings_results])

        # Create pseudo document IDs
        document_ids = [r.chunk_id for r in results]

        return self.filter(results, embeddings, document_ids)

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class MMRFilter:
    """
    Maximal Marginal Relevance (MMR) based diversity filter

    MMR = λ * Relevance - (1-λ) * max(Similarity to selected)

    This provides a more sophisticated balance between relevance
    and diversity compared to simple threshold-based filtering.
    """

    def __init__(
        self,
        lambda_param: float = 0.7,
        min_results: int = 1
    ):
        """
        Initialize MMR filter

        Args:
            lambda_param: Balance between relevance (1) and diversity (0)
            min_results: Minimum results to return
        """
        self.lambda_param = lambda_param
        self.min_results = min_results

    def filter(
        self,
        results: List[RetrievalResult],
        embeddings: np.ndarray,
        document_ids: List[str],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Filter results using MMR

        Args:
            results: List of retrieval results
            embeddings: Document embeddings matrix
            document_ids: List of document IDs
            top_k: Number of results to return

        Returns:
            Filtered list using MMR
        """
        if not results:
            return []

        if len(results) <= top_k:
            return results

        if embeddings is None or len(embeddings) == 0:
            return results[:top_k]

        # Create ID to embedding mapping
        id_to_idx = {id_: i for i, id_ in enumerate(document_ids)}

        # Normalize scores to 0-1 range
        max_score = max(r.combined_score for r in results)
        min_score = min(r.combined_score for r in results)
        score_range = max_score - min_score if max_score != min_score else 1

        selected: List[RetrievalResult] = []
        selected_embeddings: List[np.ndarray] = []
        remaining = list(results)

        while len(selected) < top_k and remaining:
            best_mmr = float('-inf')
            best_idx = 0

            for i, result in enumerate(remaining):
                # Get embedding
                if result.chunk_id not in id_to_idx:
                    continue

                emb_idx = id_to_idx[result.chunk_id]
                candidate_emb = embeddings[emb_idx]

                # Normalized relevance score
                relevance = (result.combined_score - min_score) / score_range

                # Max similarity to selected
                max_sim = 0.0
                for sel_emb in selected_embeddings:
                    sim = DiversityFilter._cosine_similarity(candidate_emb, sel_emb)
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            # Add best result
            best_result = remaining.pop(best_idx)
            selected.append(best_result)

            if best_result.chunk_id in id_to_idx:
                emb_idx = id_to_idx[best_result.chunk_id]
                selected_embeddings.append(embeddings[emb_idx])

        return selected
