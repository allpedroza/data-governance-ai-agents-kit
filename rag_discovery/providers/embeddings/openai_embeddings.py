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
