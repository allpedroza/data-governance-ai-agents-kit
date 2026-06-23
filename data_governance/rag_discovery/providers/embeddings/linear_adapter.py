"""Linear-adapter wrapper for embedding providers.

Adds a small learned linear transformation on top of an existing
``EmbeddingProvider`` so that query embeddings are nudged closer to the
documents users actually want — without retraining the underlying model.
Based on SantanderAI's ``linear-adapter-trainer`` (Apache 2.0).

Two operating modes:

* **External adapter** — when ``linear_adapter_trainer`` is installed and
  a trained ``adapter.pt`` exists, ``LinearAdapter.load`` provides the
  transformation. Recommended for production.
* **Internal numpy adapter** — pure-numpy fallback that loads a plain
  ``.npz`` (matrix ``W`` + optional bias ``b``) produced by the same
  trainer or by any custom training loop. Lets the framework benefit
  from a trained adapter without forcing the heavy torch dependency on
  every install.

The wrapper is opaque to the rest of the framework: it implements the
same :class:`EmbeddingProvider` contract, so :class:`ChromaStore`,
:class:`DataDiscoveryRAGAgent` and the taxonomy discovery pipeline see
no difference.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, TYPE_CHECKING

from ..base import EmbeddingProvider, EmbeddingResult

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class _NumpyLinearAdapter:
    """``y = x @ W.T + b`` — kept tiny so it ships without torch."""

    def __init__(self, W, b=None) -> None:
        import numpy as np
        self.W = np.asarray(W, dtype="float32")
        self.b = np.asarray(b, dtype="float32") if b is not None else None
        if self.W.ndim != 2:
            raise ValueError(f"W must be 2-D; got shape {self.W.shape}")

    @property
    def in_dim(self) -> int:
        return int(self.W.shape[1])

    @property
    def out_dim(self) -> int:
        return int(self.W.shape[0])

    @classmethod
    def load(cls, path: str) -> "_NumpyLinearAdapter":
        import numpy as np
        data = np.load(path)
        b = data["b"] if "b" in data.files else None
        return cls(W=data["W"], b=b)

    def save(self, path: str) -> None:
        import numpy as np
        if self.b is not None:
            np.savez(path, W=self.W, b=self.b)
        else:
            np.savez(path, W=self.W)

    def transform(self, vectors):
        import numpy as np
        v = np.asarray(vectors, dtype="float32")
        if v.ndim == 1:
            v = v[None, :]
        if v.shape[1] != self.in_dim:
            raise ValueError(
                f"Embedding dim {v.shape[1]} != adapter input dim {self.in_dim}"
            )
        out = v @ self.W.T
        if self.b is not None:
            out = out + self.b
        return out


def _load_external_adapter(path: str):
    """Try to load via the upstream library; return ``None`` if unavailable."""
    try:
        from linear_adapter_trainer import LinearAdapter  # type: ignore
    except ImportError:
        return None
    try:
        return LinearAdapter.load(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Upstream LinearAdapter.load failed (%s); falling back.", exc)
        return None


class LinearAdapterEmbeddings(EmbeddingProvider):
    """Apply a trained linear adapter to every vector produced by ``base``.

    Parameters
    ----------
    base:
        Any concrete :class:`EmbeddingProvider` (Sentence-Transformer,
        OpenAI, Bedrock, …). All actual embedding work is delegated to it.
    adapter_path:
        Path to either:

        * an ``.npz`` produced by :meth:`_NumpyLinearAdapter.save`, or
        * an upstream ``adapter.pt`` from ``linear_adapter_trainer``.
    normalize:
        L2-normalise the transformed vector before returning. Recommended
        when the downstream store uses cosine similarity (Chroma default).
    apply_to_documents:
        If ``False`` (default), only query embeddings are transformed. The
        upstream library actually trains the adapter to align *queries*
        with *documents*, so applying to both halves cancels out the
        signal. Keep ``False`` unless you trained the adapter symmetrically.
    """

    def __init__(
        self,
        base: EmbeddingProvider,
        adapter_path: str,
        normalize: bool = True,
        apply_to_documents: bool = False,
    ) -> None:
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter file not found: {adapter_path}")
        self.base = base
        self.normalize = normalize
        self.apply_to_documents = apply_to_documents
        self._adapter = _load_external_adapter(adapter_path)
        self._is_external = self._adapter is not None
        if self._adapter is None:
            self._adapter = _NumpyLinearAdapter.load(adapter_path)

    # ------------------------------------------------------------------
    # EmbeddingProvider contract
    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        base_name = getattr(self.base, "model_name", "base") or "base"
        return f"{base_name}+adapter"

    @property
    def dimension(self) -> int:
        base_dim = getattr(self.base, "dimension", None)
        if base_dim is not None:
            return base_dim
        return self._adapter.out_dim

    def embed(self, text: str) -> EmbeddingResult:
        result = self.base.embed(text)
        return self._apply(result, is_document=False)

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        results = self.base.embed_batch(texts)
        return [self._apply(r, is_document=False) for r in results]

    # ------------------------------------------------------------------
    # Helpers for explicit document indexing (called by ChromaStore wrapper
    # when ``apply_to_documents=True``).
    # ------------------------------------------------------------------
    def embed_document(self, text: str) -> EmbeddingResult:
        result = self.base.embed(text)
        return self._apply(result, is_document=True)

    def embed_documents_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        results = self.base.embed_batch(texts)
        return [self._apply(r, is_document=True) for r in results]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _apply(self, result: EmbeddingResult, is_document: bool) -> EmbeddingResult:
        if is_document and not self.apply_to_documents:
            return result
        import numpy as np
        v = np.asarray(result.vector, dtype="float32")
        transformed = self._adapter.transform(v)
        out = transformed[0] if hasattr(transformed, "ndim") and transformed.ndim == 2 else transformed
        out = np.asarray(out, dtype="float32")
        if self.normalize:
            n = float(np.linalg.norm(out))
            if n > 1e-12:
                out = out / n
        return EmbeddingResult(
            vector=out.tolist(),
            model=self.model_name,
            tokens_used=result.tokens_used,
        )
