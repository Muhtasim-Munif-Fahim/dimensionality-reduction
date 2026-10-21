"""Tests for Locality Preserving Projections and Neighborhood Preserving Embedding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lle import _knn_indices, _reconstruction_weights  # noqa: E402
from lpp import (  # noqa: E402
    LocalityPreservingProjections,
    NeighborhoodPreservingEmbedding,
    knn_heat_affinity,
    lpp_projection,
    lpp_reduce,
    npe_reduce,
)


def _blobs(n=240, d=6, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(3, d))
    labels = rng.integers(0, 3, n)
    return centers[labels] + rng.normal(size=(n, d)) * rng.uniform(0.5, 2.0, d), labels


def _laplacian_cost(Y, W):
    """sum_ij W_ij ||y_i - y_j||^2 / (2 * sum_i D_ii ||y_i||^2), per column."""
    deg = W.sum(axis=1)
    num = np.einsum("ij,ijk->k", W, (Y[:, None, :] - Y[None, :, :]) ** 2) / 2.0
    den = np.einsum("i,ik->k", deg, Y**2)
    return num / den


class TestAffinity:
    def test_symmetric_knn_graph(self):
        X, _ = _blobs(60)
        W = knn_heat_affinity(X, n_neighbors=4, weight="binary")
        assert np.allclose(W, W.T) and np.all(np.diag(W) == 0)
        assert np.all(W.sum(axis=1) >= 4)
        d2 = ((X[:, None] - X[None]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.argsort(d2, axis=1)[:, :4]
        for i in range(60):
            assert np.all(W[i, nn[i]] == 1)
        H = knn_heat_affinity(X, n_neighbors=4, t=2.0)
        i, j = np.argwhere(H > 0)[0]
        assert H[i, j] == pytest.approx(np.exp(-((X[i] - X[j]) ** 2).sum() / 2.0))

    def test_validation(self):
        X, _ = _blobs(10)
        with pytest.raises(ValueError):
            knn_heat_affinity(X, n_neighbors=10)
        with pytest.raises(ValueError):
            knn_heat_affinity(X, weight="cosine")


class TestLPP:
    def test_generalised_eigenproblem(self):
        X, _ = _blobs()
        W = knn_heat_affinity(X, 8)
        evals, A = lpp_projection(X, W, n_components=3, reg=0.0)
        D = np.diag(W.sum(axis=1))
        Lap = D - W
        lhs = X.T @ Lap @ X @ A
        rhs = X.T @ D @ X @ A * evals
        assert np.allclose(lhs, rhs, atol=1e-6 * np.abs(lhs).max())
        assert np.allclose(A.T @ X.T @ D @ X @ A, np.eye(3), atol=1e-8)
        assert np.all(np.diff(evals) >= -1e-12)

    def test_minimises_locality_cost(self):
        X, _ = _blobs(seed=1)
        lpp = LocalityPreservingProjections(n_components=1, n_neighbors=8, reg=0.0).fit(X)
        W = lpp.affinity_
        Xc = X - X.mean(axis=0)
        best = _laplacian_cost(Xc @ lpp.components_.T, W)[0]
        assert best == pytest.approx(lpp.eigenvalues_[0], rel=1e-6)
        rng = np.random.default_rng(0)
        for _ in range(200):
            assert _laplacian_cost(Xc @ rng.normal(size=(X.shape[1], 1)), W)[0] >= best - 1e-10
        pca_dir = PCA(1).fit(X).components_.T
        assert _laplacian_cost(Xc @ pca_dir, W)[0] >= best

    def test_out_of_sample_and_reduce(self):
        X, _ = _blobs(seed=2)
        lpp = LocalityPreservingProjections(n_components=2, n_neighbors=6)
        Y = lpp.fit_transform(X)
        assert Y.shape == (len(X), 2)
        assert np.allclose(lpp.transform(X), Y)
        assert np.allclose(lpp_reduce(X, 2, 6), Y)
        new = X[:5] + 0.01
        assert np.allclose(lpp.transform(new), (new - lpp.mean_) @ lpp.components_.T)

    def test_rotation_equivariance(self):
        X, _ = _blobs(seed=3)
        Q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(6, 6)))
        a = LocalityPreservingProjections(2, 7, reg=0.0).fit_transform(X)
        b = LocalityPreservingProjections(2, 7, reg=0.0).fit_transform(X @ Q)
        for j in range(2):
            assert abs(np.corrcoef(a[:, j], b[:, j])[0, 1]) == pytest.approx(1.0, abs=1e-6)

    def test_high_dimensional_with_pca(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(40, 200))  # d > n: X'DX singular without reg / PCA
        lpp = LocalityPreservingProjections(2, 5, pca_components=20).fit(X)
        assert lpp.components_.shape == (2, 200)
        assert np.all(np.isfinite(lpp.embedding_))
        assert np.all(np.isfinite(LocalityPreservingProjections(2, 5, reg=1e-3).fit_transform(X)))

    def test_validation(self):
        X, _ = _blobs(20)
        with pytest.raises(ValueError):
            LocalityPreservingProjections(n_components=7).fit(X)
        with pytest.raises(ValueError):
            LocalityPreservingProjections(n_neighbors=20).fit(X)
        with pytest.raises(RuntimeError):
            LocalityPreservingProjections().transform(X)
        with pytest.raises(ValueError):
            LocalityPreservingProjections().fit(np.array([[1.0, np.nan], [0.0, 1.0]]))


class TestNPE:
    def test_weights_match_lle_module(self):
        X, _ = _blobs(80, seed=5)
        npe = NeighborhoodPreservingEmbedding(2, n_neighbors=7, lle_reg=1e-3).fit(X)
        ref = _reconstruction_weights(X, _knn_indices(X, 7), 1e-3)
        assert np.allclose(npe.reconstruction_weights_, ref)
        assert np.allclose(npe.reconstruction_weights_.sum(axis=1), 1.0)

    def test_generalised_eigenproblem_and_cost(self):
        X, _ = _blobs(seed=6)
        npe = NeighborhoodPreservingEmbedding(2, n_neighbors=8, reg=0.0).fit(X)
        Xc = X - X.mean(axis=0)
        IW = np.eye(len(X)) - npe.reconstruction_weights_
        M = IW.T @ IW
        A = npe.components_.T
        assert np.allclose(Xc.T @ M @ Xc @ A, Xc.T @ Xc @ A * npe.eigenvalues_, atol=1e-6)
        Y = Xc @ A
        cost = ((IW @ Y) ** 2).sum(axis=0) / (Y**2).sum(axis=0)
        assert np.allclose(cost, npe.eigenvalues_)
        pca = Xc @ PCA(1).fit(X).components_.T
        assert ((IW @ pca) ** 2).sum() / (pca**2).sum() >= cost[0]

    def test_out_of_sample(self):
        X, _ = _blobs(seed=7)
        npe = NeighborhoodPreservingEmbedding(3, 10)
        Y = npe.fit_transform(X)
        assert np.allclose(npe.transform(X), Y)
        assert np.allclose(npe_reduce(X, 3, 10), Y)


def test_pipeline_includes_lpp_and_npe(tmp_path):
    from pipeline import run_pipeline

    summary = run_pipeline(output_dir=tmp_path, n_samples=70, n_features=10)
    results = summary["results"] if "results" in summary else summary
    for key in ("lpp", "npe"):
        assert key in results and np.isfinite(results[key]["silhouette"])
