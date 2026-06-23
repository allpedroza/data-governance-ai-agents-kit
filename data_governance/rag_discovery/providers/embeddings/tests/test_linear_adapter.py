"""Tests for the LinearAdapterEmbeddings wrapper (no torch required)."""
from __future__ import annotations

import os
import sys
import tempfile
import types
from typing import List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
if "data_governance" not in sys.modules:
    pkg = types.ModuleType("data_governance"); pkg.__path__ = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    ]
    sys.modules["data_governance"] = pkg

import numpy as np

from data_governance.rag_discovery.providers.base import EmbeddingProvider, EmbeddingResult
from data_governance.rag_discovery.providers.embeddings.linear_adapter import (
    LinearAdapterEmbeddings, _NumpyLinearAdapter,
)


class _FakeEmbedder(EmbeddingProvider):
    """Deterministic 4-dim embedder for testing."""
    model_name = "fake"
    dimension = 4

    def embed(self, text: str) -> EmbeddingResult:
        # Map a few known texts to known vectors
        m = {
            "alpha": [1.0, 0.0, 0.0, 0.0],
            "beta":  [0.0, 1.0, 0.0, 0.0],
            "gamma": [0.5, 0.5, 0.0, 0.0],
        }
        v = m.get(text, [0.25, 0.25, 0.25, 0.25])
        return EmbeddingResult(vector=v, model="fake", tokens_used=0)

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        return [self.embed(t) for t in texts]


def _make_adapter(tmpdir: str) -> str:
    """Write a simple identity-like adapter to disk and return its path."""
    W = np.eye(4, dtype="float32") * 2.0
    b = np.array([0.0, 0.0, 0.0, 0.0], dtype="float32")
    path = os.path.join(tmpdir, "adapter.npz")
    np.savez(path, W=W, b=b)
    return path


def test_adapter_applied_to_query_embeddings():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_adapter(tmpdir)
        wrapped = LinearAdapterEmbeddings(
            base=_FakeEmbedder(), adapter_path=path, normalize=False,
        )
        result = wrapped.embed("alpha")
        # W * 2 → vector should be [2,0,0,0]
        assert result.vector == [2.0, 0.0, 0.0, 0.0]
        assert result.model == "fake+adapter"


def test_adapter_normalizes_when_requested():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_adapter(tmpdir)
        wrapped = LinearAdapterEmbeddings(
            base=_FakeEmbedder(), adapter_path=path, normalize=True,
        )
        result = wrapped.embed("gamma")
        v = np.array(result.vector)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6


def test_documents_not_transformed_by_default():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_adapter(tmpdir)
        wrapped = LinearAdapterEmbeddings(
            base=_FakeEmbedder(), adapter_path=path, normalize=False,
        )
        # embed_document is the explicit document path — default keeps original
        doc = wrapped.embed_document("alpha")
        assert doc.vector == [1.0, 0.0, 0.0, 0.0]
        # query path still transforms
        q = wrapped.embed("alpha")
        assert q.vector == [2.0, 0.0, 0.0, 0.0]


def test_documents_transformed_when_apply_to_documents_true():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_adapter(tmpdir)
        wrapped = LinearAdapterEmbeddings(
            base=_FakeEmbedder(), adapter_path=path,
            normalize=False, apply_to_documents=True,
        )
        doc = wrapped.embed_document("alpha")
        assert doc.vector == [2.0, 0.0, 0.0, 0.0]


def test_dimension_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 3x5 adapter expects 5-dim input but base produces 4-dim
        W = np.zeros((3, 5), dtype="float32")
        path = os.path.join(tmpdir, "bad.npz")
        np.savez(path, W=W)
        wrapped = LinearAdapterEmbeddings(
            base=_FakeEmbedder(), adapter_path=path, normalize=False,
        )
        try:
            wrapped.embed("alpha")
        except ValueError as exc:
            assert "dim" in str(exc).lower()
            return
        raise AssertionError("expected ValueError for dim mismatch")


def test_missing_adapter_file_raises():
    try:
        LinearAdapterEmbeddings(_FakeEmbedder(), "/nonexistent.npz")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_numpy_adapter_save_load_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        W = np.random.rand(8, 4).astype("float32")
        b = np.random.rand(8).astype("float32")
        ad = _NumpyLinearAdapter(W=W, b=b)
        path = os.path.join(tmpdir, "roundtrip.npz")
        ad.save(path)
        loaded = _NumpyLinearAdapter.load(path)
        assert loaded.in_dim == 4 and loaded.out_dim == 8
        out = loaded.transform(np.ones(4, dtype="float32"))
        assert out.shape == (1, 8)
