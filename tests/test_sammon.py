"""Tests for Sammon nonlinear mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sammon import Sammon, sammon_reduce, _sammon_stress


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _procrustes_rmse(Y: np.ndarray, X: np.ndarray) -> float:
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    U, _, Vt = np.linalg.svd(Yc.T @ Xc, full_matrices=False)
    R = U @ Vt
    aligned = Yc @ R
    return float(np.sqrt(np.mean((aligned - Xc) ** 2)))


def _grid_2d(n: int = 5):
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    return np.array([(x, y) for x in xs for y in ys], dtype=float)


def _circle(n: int = 40):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(t), np.sin(t)])


class TestSammonBasic:
    def test_fit_transform_shape(self):
        X = _grid_2d()
        emb = Sammon(n_components=2, max_iter=50).fit_transform(X)
        assert emb.shape == (25, 2)
        assert np.all(np.isfinite(emb))

    def test_fit_stores_attributes(self):
        X = _grid_2d()
        model = Sammon(n_components=2, max_iter=50).fit(X)
        assert model.embedding_ is not None
        assert model.embedding_.shape == (25, 2)
        assert model.dist_matrix_ is not None
        assert model.stress_ is not None
        assert np.isfinite(model.stress_)
        assert model.n_iter_ is not None and model.n_iter_ >= 1
        assert model.reconstruction_error_ == model.stress_

    def test_deterministic_classical_init(self):
        X = _grid_2d()
        a = Sammon(n_components=2, max_iter=40, init="classical").fit_transform(X)
        b = Sammon(n_components=2, max_iter=40, init="classical").fit_transform(X)
        assert np.allclose(a, b)

    def test_recovers_2d_grid(self):
        X = _grid_2d(n=5)
        emb = Sammon(n_components=2, max_iter=80, tol=1e-9).fit_transform(X)
        assert _procrustes_rmse(emb, X) < 0.05

    def test_recovers_circle(self):
        X = _circle(n=36)
        emb = Sammon(n_components=2, max_iter=100, tol=1e-9).fit_transform(X)
        assert _procrustes_rmse(emb, X) < 0.08
        centered = emb - emb.mean(axis=0)
        radii = np.linalg.norm(centered, axis=1)
        assert radii.std() / radii.mean() < 0.05

    def test_stress_decreases(self):
        X = _grid_2d(n=4)
        model = Sammon(n_components=2, max_iter=1, init="classical")
        # Capture initial classical MDS stress by fitting with 0 effective iters
        # via comparing stress after few iters vs classical init.
        from sammon import _classical_mds, _pairwise_euclidean

        d = _pairwise_euclidean(X)
        y0 = _classical_mds(d, 2)
        s0 = _sammon_stress(d, y0)
        model = Sammon(n_components=2, max_iter=60, tol=0.0, init="classical")
        model.fit(X)
        assert model.stress_ <= s0 + 1e-9

    def test_precomputed_distances(self):
        X = _grid_2d(n=4)
        d = _pairwise_euclidean = None
        d = np.sqrt(
            np.maximum(
                (X ** 2).sum(1, keepdims=True)
                + (X ** 2).sum(1)
                - 2.0 * (X @ X.T),
                0.0,
            )
        )
        np.fill_diagonal(d, 0.0)
        emb = Sammon(
            n_components=2, metric="precomputed", max_iter=60
        ).fit_transform(d)
        assert emb.shape == (16, 2)
        assert _procrustes_rmse(emb, X) < 0.08

    def test_dataframe_helper(self):
        X = _circle(n=20)
        frame = pd.DataFrame(X, columns=["x", "y"])
        out = sammon_reduce(frame, n_components=2, max_iter=40)
        assert list(out.columns) == ["SAM1", "SAM2"]
        assert len(out) == 20

    def test_n_components_too_large_raises(self):
        X = np.random.default_rng(0).normal(size=(5, 3))
        _raises(ValueError, lambda: Sammon(n_components=5).fit_transform(X))

    def test_invalid_params_raise(self):
        _raises(ValueError, lambda: Sammon(n_components=0))
        _raises(ValueError, lambda: Sammon(metric="cosine"))
        _raises(ValueError, lambda: Sammon(max_iter=0))
        _raises(ValueError, lambda: Sammon(magic=0.0))
        _raises(ValueError, lambda: Sammon(init="pca"))
        _raises(ValueError, lambda: Sammon(random_state="bad"))
        _raises(ValueError, lambda: Sammon(tol=-1.0))

    def test_negative_precomputed_raises(self):
        d = np.array([[0.0, -1.0], [-1.0, 0.0]])
        _raises(
            ValueError,
            lambda: Sammon(metric="precomputed").fit_transform(d),
        )

    def test_random_init_uses_seed(self):
        X = _grid_2d(n=4)
        a = Sammon(
            n_components=2, init="random", random_state=7, max_iter=30
        ).fit_transform(X)
        b = Sammon(
            n_components=2, init="random", random_state=7, max_iter=30
        ).fit_transform(X)
        assert np.allclose(a, b)
