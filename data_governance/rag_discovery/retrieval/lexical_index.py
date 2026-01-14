"""
Lexical Index using TF-IDF
Used for hybrid ranking in Dartboard algorithm
"""

from typing import List, Optional
import numpy as np

from ..utils.stopwords import PORTUGUESE_STOPWORDS, MULTILINGUAL_STOPWORDS


class LexicalIndex:
    """
    TF-IDF based lexical index for keyword matching

    Used in hybrid ranking to complement semantic search with
    exact keyword matches (important for technical terms, table names, etc.)
    """

    def __init__(
        self,
        stopwords: Optional[List[str]] = None,
        min_df: int = 1,
        max_df: float = 0.95,
        ngram_range: tuple = (1, 2),
        lowercase: bool = True
    ):
        """
        Initialize lexical index

        Args:
            stopwords: List of stopwords to ignore
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency (proportion)
            ngram_range: N-gram range for tokenization
            lowercase: Whether to lowercase text
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError(
                "scikit-learn not installed. Install with: pip install scikit-learn"
            )

        self._stopwords = stopwords or MULTILINGUAL_STOPWORDS
        self._vectorizer = TfidfVectorizer(
            lowercase=lowercase,
            stop_words=self._stopwords,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )
        self._tfidf_matrix = None
        self._fitted = False

    def fit(self, documents: List[str]) -> None:
        """
        Fit the TF-IDF vectorizer on documents

        Args:
            documents: List of document texts
        """
        if not documents:
            return

        self._tfidf_matrix = self._vectorizer.fit_transform(documents)
        self._fitted = True

    def score(self, query: str, documents: Optional[List[str]] = None) -> np.ndarray:
        """
        Calculate TF-IDF similarity scores for a query

        Args:
            query: Query text
            documents: Optional list of documents (if not fitted)

        Returns:
            Array of similarity scores for each document
        """
        if not self._fitted:
            if documents:
                self.fit(documents)
            else:
                return np.array([])

        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise ImportError("scikit-learn not installed")

        # Transform query
        query_tfidf = self._vectorizer.transform([query])

        # Calculate cosine similarity with all documents
        similarities = cosine_similarity(query_tfidf, self._tfidf_matrix).flatten()

        return similarities

    def score_subset(
        self,
        query: str,
        indices: List[int]
    ) -> np.ndarray:
        """
        Calculate TF-IDF scores for a subset of documents

        Args:
            query: Query text
            indices: Document indices to score

        Returns:
            Array of similarity scores for specified documents
        """
        if not self._fitted or self._tfidf_matrix is None:
            return np.zeros(len(indices))

        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return np.zeros(len(indices))

        query_tfidf = self._vectorizer.transform([query])

        scores = np.zeros(len(indices))
        for i, idx in enumerate(indices):
            if 0 <= idx < self._tfidf_matrix.shape[0]:
                score = cosine_similarity(
                    query_tfidf,
                    self._tfidf_matrix[idx].reshape(1, -1)
                ).item()
                scores[i] = score

        return scores

    def get_top_terms(self, n: int = 10) -> List[str]:
        """Get top terms by IDF score"""
        if not self._fitted:
            return []

        feature_names = self._vectorizer.get_feature_names_out()
        idf_scores = self._vectorizer.idf_

        # Sort by IDF (higher = more distinctive)
        top_indices = np.argsort(idf_scores)[-n:][::-1]

        return [feature_names[i] for i in top_indices]

    @property
    def vocabulary_size(self) -> int:
        """Number of terms in vocabulary"""
        if not self._fitted:
            return 0
        return len(self._vectorizer.vocabulary_)

    @property
    def is_fitted(self) -> bool:
        """Whether the index has been fitted"""
        return self._fitted
