"""Tests for PHATE."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from phate import PHATE, alpha_decay_kernel, find_knee, phate_reduce, von_neumann_entropy


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _blobs(n_per: int = 25, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0] * 5, [12.0] + [0.0] * 4, [0.0, 12.0, 0.0, 0.0, 0.0]])
    X = np.vstack([c + rng.normal(scale=0.5, size=(n_per, 5)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    return X, y


def _curve(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 3.0, n)
    X = np.column_stack([np.cos(s), np.sin(s), s]) + 0.01 * rng.normal(size=(n, 3))
    return X, s


def test_kernel_matches_formula_and_is_symmetric():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(15, 3))
    K, eps = alpha_decay_kernel(X, knn=3, decay=10.0, thresh=0.0)
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    ref_eps = np.sort(d, axis=1)[:, 3]
    np.testing.assert_allclose(eps, ref_eps)
    ref = 0.5 * np.exp(-(d / ref_eps[:, None]) ** 10) + 0.5 * np.exp(-(d / ref_eps[None, :]) ** 10)
    np.fill_diagonal(ref, 1.0)
    np.testing.assert_allclose(K, ref, atol=1e-12)
    np.testing.assert_allclose(K, K.T)
    # At the bandwidth distance the one-sided affinity is exp(-1).
    j = np.argsort(d[0])[3]
    assert 0.5 * np.exp(-1.0) <= K[0, j] + 1e-12


def test_kernel_threshold_and_precomputed():
    X, _ = _blobs()
    K, _ = alpha_decay_kernel(X, knn=5, decay=40.0, thresh=1e-4)
    assert np.all((K == 0.0) | (K >= 1e-4))
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    K2, _ = alpha_decay_kernel(d, knn=5, decay=40.0, thresh=1e-4, precomputed=True)
    np.testing.assert_allclose(K, K2, atol=1e-10)


def test_find_knee_on_piecewise_linear():
    y = np.concatenate([np.linspace(10.0, 2.0, 9), np.linspace(1.9, 1.0, 30)])
    assert abs(find_knee(y) - 8) <= 1
    _raises(ValueError, lambda: find_knee([1.0, 2.0]))


def test_von_neumann_entropy_decreasing():
    X, _ = _curve()
    K, _ = alpha_decay_kernel(X, knn=5)
    P = K / K.sum(axis=1, keepdims=True)
    t, H = von_neumann_entropy(P, t_max=30)
    assert t[0] == 1 and len(H) == 30
    assert np.all(np.diff(H) <= 1e-10)
    assert H[0] <= np.log(len(X)) + 1e-12
    _raises(ValueError, lambda: von_neumann_entropy(K, t_max=5))


def test_fit_attributes_and_row_stochastic():
    X, _ = _blobs()
    model = PHATE(n_components=2, knn=5, random_state=0).fit(X)
    assert model.embedding_.shape == (75, 2)
    np.testing.assert_allclose(model.diff_op_.sum(axis=1), 1.0)
    assert 1 <= model.optimal_t_ <= 100
    assert model.entropy_.shape == (100,)
    assert model.n_features_in_ == 5
    assert np.all(np.isfinite(model.embedding_))
    np.testing.assert_allclose(model.potential_distances_, model.potential_distances_.T, atol=1e-10)


def test_clusters_are_separated():
    X, y = _blobs()
    for mds in ("classic", "metric"):
        emb = PHATE(knn=5, t=10, mds=mds).fit_transform(X)
        cents = np.array([emb[y == c].mean(axis=0) for c in range(3)])
        within = max(np.linalg.norm(emb[y == c] - cents[c], axis=1).max() for c in range(3))
        between = min(
            np.linalg.norm(cents[a] - cents[b]) for a in range(3) for b in range(a + 1, 3)
        )
        assert between > within, mds


def test_trajectory_order_preserved():
    X, s = _curve()
    emb = PHATE(n_components=1, knn=8, t=20, mds="classic").fit_transform(X)
    rho = np.corrcoef(np.argsort(np.argsort(emb[:, 0])), np.arange(len(s)))[0, 1]
    assert abs(rho) > 0.95


def test_gamma_variants_and_precomputed():
    X, _ = _blobs(n_per=10)
    for gamma in (1.0, 0.0, -1.0, 0.5):
        emb = PHATE(knn=4, t=5, gamma=gamma, mds="classic").fit_transform(X)
        assert np.all(np.isfinite(emb))
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    a = PHATE(knn=4, t=5, mds="classic").fit_transform(X)
    b = PHATE(knn=4, t=5, mds="classic", metric="precomputed").fit_transform(d)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_deterministic_and_reduce_frame():
    X, _ = _blobs(n_per=10)
    a = PHATE(knn=4, random_state=0).fit_transform(X)
    b = PHATE(knn=4, random_state=0).fit_transform(X)
    np.testing.assert_allclose(a, b)
    frame = phate_reduce(pd.DataFrame(X, index=[f"r{i}" for i in range(len(X))]), knn=4, t=3)
    assert list(frame.columns) == ["PHATE1", "PHATE2"]
    assert frame.index[0] == "r0"


def test_decay_none_and_validation():
    X, _ = _blobs(n_per=8)
    emb = PHATE(knn=3, decay=None, t=4, mds="classic").fit_transform(X)
    assert emb.shape == (24, 2)
    _raises(ValueError, lambda: PHATE(knn=0))
    _raises(ValueError, lambda: PHATE(t=0))
    _raises(ValueError, lambda: PHATE(t="fast"))
    _raises(ValueError, lambda: PHATE(gamma=2.0))
    _raises(ValueError, lambda: PHATE(mds="nonmetric"))
    _raises(ValueError, lambda: PHATE(decay=-1.0))
    _raises(ValueError, lambda: PHATE(thresh=0.7))
    _raises(ValueError, lambda: PHATE(knn=30).fit(X))
    _raises(ValueError, lambda: PHATE(n_components=30).fit(X))
    _raises(ValueError, lambda: PHATE().fit(np.array([[1.0, np.nan], [0.0, 1.0]])))
