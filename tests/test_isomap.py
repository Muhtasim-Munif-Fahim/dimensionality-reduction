"""Tests for Isomap manifold embedding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from isomap import Isomap, isomap_reduce


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _swiss_roll_ish(n: int = 80, seed: int = 0):
    """Two arcs that are close in Euclidean space but far along the manifold."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.5 * np.pi, 1.5 * np.pi, n // 2)
    a = np.column_stack([np.cos(t), np.sin(t), np.zeros(n // 2)])
    b = np.column_stack([np.cos(t) + 3.0, np.sin(t), np.zeros(n // 2)])
    X = np.vstack([a, b]) + 0.02 * rng.normal(size=(n, 3))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def _two_blobs(n_per: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=-3.0, scale=0.3, size=(n_per, 5))
    b = rng.normal(loc=3.0, scale=0.3, size=(n_per, 5))
    X = np.vstack([a, b])
    y = np.array([0] * n_per + [1] * n_per)
    return X, y


class TestIsomapBasic:
    def test_fit_transform_shape(self):
        X, _ = _two_blobs()
        emb = Isomap(n_components=2, n_neighbors=8).fit_transform(X)
        assert emb.shape == (80, 2)
        assert np.all(np.isfinite(emb))

    def test_fit_stores_embedding(self):
        X, _ = _two_blobs()
        model = Isomap(n_components=2, n_neighbors=8).fit(X)
        assert model.embedding_ is not None
        assert model.embedding_.shape == (80, 2)
        assert model.dist_matrix_ is not None
        assert model.reconstruction_error_ is not None
        assert np.isfinite(model.reconstruction_error_)

    def test_deterministic(self):
        X, _ = _two_blobs()
        a = Isomap(n_components=2, n_neighbors=8).fit_transform(X)
        b = Isomap(n_components=2, n_neighbors=8).fit_transform(X)
        assert np.allclose(a, b)

    def test_separates_two_blobs(self):
        X, y = _two_blobs(n_per=35, seed=2)
        emb = Isomap(n_components=2, n_neighbors=10).fit_transform(X)
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
        out = isomap_reduce(frame, n_components=2, n_neighbors=6)
        assert list(out.columns) == ["ISO1", "ISO2"]
        assert len(out) == 40

    def test_n_neighbors_too_large_raises(self):
        X = np.random.default_rng(0).normal(size=(10, 3))
        _raises(ValueError, lambda: Isomap(n_neighbors=10).fit_transform(X))

    def test_invalid_params_raise(self):
        _raises(ValueError, lambda: Isomap(n_components=0))
        _raises(ValueError, lambda: Isomap(n_neighbors=0))
        _raises(ValueError, lambda: Isomap(metric="manhattan"))

    def test_single_sample_raises(self):
        _raises(ValueError, lambda: Isomap(n_neighbors=1).fit_transform(np.zeros((1, 3))))

    def test_bridges_far_clusters(self):
        # Three far-apart pairs: raw 1-NN leaves three components, but the
        # MST bridge should reconnect them so Floyd–Warshall succeeds.
        X = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [100.0, 0.0],
                [100.1, 0.0],
                [200.0, 0.0],
                [200.1, 0.0],
            ]
        )
        emb = Isomap(n_components=2, n_neighbors=1).fit_transform(X)
        assert emb.shape == (6, 2)
        assert np.all(np.isfinite(emb))
