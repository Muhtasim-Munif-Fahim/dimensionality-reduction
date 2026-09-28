"""Tests for Locally Linear Embedding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lle import LLE, lle_reduce


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _two_blobs(n_per: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=-3.0, scale=0.3, size=(n_per, 5))
    b = rng.normal(loc=3.0, scale=0.3, size=(n_per, 5))
    X = np.vstack([a, b])
    y = np.array([0] * n_per + [1] * n_per)
    return X, y


def _s_curve_ish(n: int = 80, seed: int = 0):
    """1-D manifold bent into an S in 3-D."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-1.5 * np.pi, 1.5 * np.pi, n)
    X = np.column_stack([np.sin(t), 2.0 * t / np.pi, np.sign(t) * (np.cos(t) - 1.0)])
    X = X + 0.02 * rng.normal(size=X.shape)
    return X, t


class TestLLEBasic:
    def test_fit_transform_shape(self):
        X, _ = _two_blobs()
        emb = LLE(n_components=2, n_neighbors=8).fit_transform(X)
        assert emb.shape == (80, 2)
        assert np.all(np.isfinite(emb))

    def test_fit_stores_embedding(self):
        X, _ = _two_blobs()
        model = LLE(n_components=2, n_neighbors=8).fit(X)
        assert model.embedding_ is not None
        assert model.embedding_.shape == (80, 2)
        assert model.reconstruction_weights_ is not None
        assert model.reconstruction_weights_.shape == (80, 80)
        assert model.reconstruction_error_ is not None
        assert np.isfinite(model.reconstruction_error_)

    def test_weights_sum_to_one(self):
        X, _ = _two_blobs(n_per=25)
        model = LLE(n_components=2, n_neighbors=6).fit(X)
        row_sums = model.reconstruction_weights_.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_deterministic(self):
        X, _ = _two_blobs()
        a = LLE(n_components=2, n_neighbors=8).fit_transform(X)
        b = LLE(n_components=2, n_neighbors=8).fit_transform(X)
        assert np.allclose(a, b)

    def test_separates_two_blobs(self):
        X, y = _two_blobs(n_per=35, seed=2)
        emb = LLE(n_components=2, n_neighbors=10).fit_transform(X)
        c0 = emb[y == 0].mean(axis=0)
        c1 = emb[y == 1].mean(axis=0)
        within = 0.5 * (
            np.linalg.norm(emb[y == 0] - c0, axis=1).mean()
            + np.linalg.norm(emb[y == 1] - c1, axis=1).mean()
        )
        between = np.linalg.norm(c0 - c1)
        assert between > within

    def test_dataframe_helper(self):
        X, _ = _two_blobs(n_per=20)
        frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        out = lle_reduce(frame, n_components=2, n_neighbors=6)
        assert list(out.columns) == ["LLE1", "LLE2"]
        assert len(out) == 40

    def test_s_curve_finite(self):
        X, _ = _s_curve_ish()
        emb = LLE(n_components=2, n_neighbors=10).fit_transform(X)
        assert emb.shape == (80, 2)
        assert np.all(np.isfinite(emb))

    def test_n_neighbors_too_large_raises(self):
        X = np.random.default_rng(0).normal(size=(10, 3))
        _raises(ValueError, lambda: LLE(n_neighbors=10).fit_transform(X))

    def test_invalid_params_raise(self):
        _raises(ValueError, lambda: LLE(n_components=0))
        _raises(ValueError, lambda: LLE(n_neighbors=0))
        _raises(ValueError, lambda: LLE(reg=-0.1))

    def test_single_sample_raises(self):
        _raises(ValueError, lambda: LLE(n_neighbors=1).fit_transform(np.zeros((1, 3))))
