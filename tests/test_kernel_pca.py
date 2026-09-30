"""Tests for Kernel PCA."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kernel_pca import KernelPCA, kernel_pca_reduce


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _grid_2d(n: int = 5):
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    return np.array([(x, y) for x in xs for y in ys], dtype=float)


class TestKernelPCABasic:
    def test_fit_transform_shape(self):
        X = _grid_2d()
        emb = KernelPCA(n_components=2, kernel="rbf").fit_transform(X)
        assert emb.shape == (25, 2)
        assert np.all(np.isfinite(emb))

    def test_fit_stores_embedding(self):
        X = _grid_2d()
        model = KernelPCA(n_components=2, kernel="linear").fit(X)
        assert model.embedding_ is not None
        assert model.embedding_.shape == (25, 2)
        assert model.eigenvalues_ is not None
        assert model.eigenvalues_.shape == (2,)
        assert model.X_fit_ is not None

    def test_deterministic(self):
        X = _grid_2d()
        a = KernelPCA(n_components=2, kernel="rbf", gamma=0.5).fit_transform(X)
        b = KernelPCA(n_components=2, kernel="rbf", gamma=0.5).fit_transform(X)
        assert np.allclose(a, b)

    def test_linear_kernel_recovers_pca_directions(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 4))
        # Linear KPCA on centered data should span the same subspace as PCA.
        Xc = X - X.mean(axis=0)
        emb = KernelPCA(n_components=2, kernel="linear").fit_transform(Xc)
        # Covariance PCA
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        pca = Xc @ Vt[:2].T
        # Compare subspaces via principal angles (canonical correlations).
        Qa, _ = np.linalg.qr(emb)
        Qb, _ = np.linalg.qr(pca)
        s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
        assert np.min(s) > 0.99

    def test_transform_matches_fit_transform_on_train(self):
        X = _grid_2d(n=6)
        model = KernelPCA(n_components=2, kernel="rbf", gamma=1.0).fit(X)
        out = model.transform(X)
        assert np.allclose(out, model.embedding_, atol=1e-6)

    def test_poly_kernel_finite(self):
        X = np.random.default_rng(1).normal(size=(40, 3))
        emb = KernelPCA(n_components=2, kernel="poly", degree=2, coef0=1.0).fit_transform(X)
        assert emb.shape == (40, 2)
        assert np.all(np.isfinite(emb))

    def test_dataframe_helper(self):
        X = _grid_2d(n=4)
        frame = pd.DataFrame(X, columns=["x", "y"])
        out = kernel_pca_reduce(frame, n_components=2, kernel="linear")
        assert list(out.columns) == ["KPCA1", "KPCA2"]
        assert len(out) == 16

    def test_n_components_too_large_raises(self):
        X = np.random.default_rng(0).normal(size=(5, 3))
        _raises(ValueError, lambda: KernelPCA(n_components=5).fit_transform(X))

    def test_invalid_params_raise(self):
        _raises(ValueError, lambda: KernelPCA(n_components=0))
        _raises(ValueError, lambda: KernelPCA(kernel="sigmoid"))
        _raises(ValueError, lambda: KernelPCA(gamma=0.0))
        _raises(ValueError, lambda: KernelPCA(degree=0))

    def test_transform_before_fit_raises(self):
        _raises(ValueError, lambda: KernelPCA().transform(np.zeros((3, 2))))

    def test_feature_mismatch_raises(self):
        model = KernelPCA(n_components=2).fit(np.zeros((10, 3)))
        _raises(ValueError, lambda: model.transform(np.zeros((4, 2))))
