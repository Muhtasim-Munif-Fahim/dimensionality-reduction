"""Tests for classical multidimensional scaling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mds import ClassicalMDS, mds_reduce


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _procrustes_rmse(Y: np.ndarray, X: np.ndarray) -> float:
    """RMSE after optimal orthogonal Procrustes alignment of Y onto X."""
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    U, _, Vt = np.linalg.svd(Yc.T @ Xc, full_matrices=False)
    R = U @ Vt
    aligned = Yc @ R
    return float(np.sqrt(np.mean((aligned - Xc) ** 2)))


def _grid_2d(n: int = 5):
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    pts = np.array([(x, y) for x in xs for y in ys], dtype=float)
    return pts


def _circle(n: int = 40):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


class TestClassicalMDSBasic:
    def test_fit_transform_shape(self):
        X = _grid_2d()
        emb = ClassicalMDS(n_components=2).fit_transform(X)
        assert emb.shape == (25, 2)
        assert np.all(np.isfinite(emb))

    def test_fit_stores_embedding(self):
        X = _grid_2d()
        model = ClassicalMDS(n_components=2).fit(X)
        assert model.embedding_ is not None
        assert model.embedding_.shape == (25, 2)
        assert model.dist_matrix_ is not None
        assert model.eigenvalues_ is not None
        assert model.eigenvalues_.shape == (2,)
        assert model.reconstruction_error_ is not None
        assert np.isfinite(model.reconstruction_error_)

    def test_deterministic(self):
        X = _grid_2d()
        a = ClassicalMDS(n_components=2).fit_transform(X)
        b = ClassicalMDS(n_components=2).fit_transform(X)
        assert np.allclose(a, b)

    def test_recovers_2d_grid(self):
        X = _grid_2d(n=6)
        emb = ClassicalMDS(n_components=2).fit_transform(X)
        rmse = _procrustes_rmse(emb, X)
        assert rmse < 1e-6

    def test_recovers_circle(self):
        X = _circle(n=48)
        emb = ClassicalMDS(n_components=2).fit_transform(X)
        rmse = _procrustes_rmse(emb, X)
        assert rmse < 1e-6
        # Radii should stay roughly constant after centering.
        centered = emb - emb.mean(axis=0)
        radii = np.linalg.norm(centered, axis=1)
        assert radii.std() / radii.mean() < 1e-5

    def test_precomputed_distances(self):
        X = _grid_2d(n=4)
        # Pairwise Euclidean.
        d = np.sqrt(
            np.maximum(
                (X ** 2).sum(1, keepdims=True)
                + (X ** 2).sum(1)
                - 2.0 * (X @ X.T),
                0.0,
            )
        )
        np.fill_diagonal(d, 0.0)
        emb = ClassicalMDS(n_components=2, metric="precomputed").fit_transform(d)
        assert emb.shape == (16, 2)
        assert _procrustes_rmse(emb, X) < 1e-6

    def test_dataframe_helper(self):
        X = _circle(n=20)
        frame = pd.DataFrame(X, columns=["x", "y"])
        out = mds_reduce(frame, n_components=2)
        assert list(out.columns) == ["MDS1", "MDS2"]
        assert len(out) == 20

    def test_n_components_too_large_raises(self):
        X = np.random.default_rng(0).normal(size=(5, 3))
        _raises(ValueError, lambda: ClassicalMDS(n_components=5).fit_transform(X))

    def test_invalid_params_raise(self):
        _raises(ValueError, lambda: ClassicalMDS(n_components=0))
        _raises(ValueError, lambda: ClassicalMDS(metric="cosine"))
        _raises(ValueError, lambda: ClassicalMDS(random_state="bad"))

    def test_negative_precomputed_raises(self):
        d = np.array([[0.0, -1.0], [-1.0, 0.0]])
        _raises(
            ValueError,
            lambda: ClassicalMDS(metric="precomputed").fit_transform(d),
        )
