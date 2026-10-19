"""Tests for Robust PCA (Principal Component Pursuit, inexact ALM)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robust_pca import RobustPCA, principal_component_pursuit, rpca_reduce  # noqa: E402


def _corrupted(n=200, d=120, r=4, frac=0.05, magnitude=(5.0, 15.0), seed=0):
    rng = np.random.default_rng(seed)
    L0 = rng.normal(size=(n, r)) @ rng.normal(size=(r, d))
    mask = rng.random((n, d)) < frac
    S0 = np.where(mask, rng.choice([-1.0, 1.0], (n, d)) * rng.uniform(*magnitude, (n, d)), 0.0)
    return L0, S0, mask


def _principal_angle_sin(A, B):
    """Largest sine of the principal angles between row spaces of A and B."""
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(B.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.sqrt(max(0.0, 1.0 - s.min() ** 2)))


def test_pcp_exactly_recovers_low_rank_and_sparse_parts():
    L0, S0, mask = _corrupted()
    L, S, info = principal_component_pursuit(L0 + S0)
    assert info["converged"]
    assert info["rank"] == 4
    assert np.linalg.norm(L - L0) / np.linalg.norm(L0) < 1e-5
    assert np.linalg.norm(S - S0) / np.linalg.norm(S0) < 1e-5
    support = np.abs(S) > 1e-3
    assert np.array_equal(support, mask)


def test_pcp_with_missing_entries_completes_the_matrix():
    L0, S0, _ = _corrupted(seed=1)
    M = L0 + S0
    rng = np.random.default_rng(1)
    missing = rng.random(M.shape) < 0.15
    M[missing] = np.nan
    L, S, info = principal_component_pursuit(M)
    assert info["converged"]
    assert np.all(S[missing] == 0.0)
    assert np.linalg.norm(L - L0) / np.linalg.norm(L0) < 1e-4
    # Imputed (never observed) entries are recovered too.
    assert np.max(np.abs(L[missing] - L0[missing])) < 1e-3


def test_classical_pca_is_rotated_by_corruption_but_robust_pca_is_not():
    L0, S0, _ = _corrupted(n=300, d=60, r=3, frac=0.08, magnitude=(20.0, 40.0), seed=2)
    X = L0 + S0
    truth = np.linalg.svd(L0 - L0.mean(axis=0), full_matrices=False)[2][:3]
    pca_axes = PCA(n_components=3).fit(X).components_
    model = RobustPCA(n_components=3).fit(X)
    assert _principal_angle_sin(model.components_, truth) < 1e-3
    assert _principal_angle_sin(pca_axes, truth) > 0.2
    np.testing.assert_allclose(model.components_ @ model.components_.T, np.eye(3), atol=1e-10)


def test_fit_attributes_and_reconstruction_identity():
    L0, S0, _ = _corrupted(seed=3)
    X = L0 + S0 + 7.0
    model = RobustPCA().fit(X)
    assert model.converged_
    assert model.rank_ == model.n_components_ == 4
    np.testing.assert_allclose(model.low_rank_ + model.sparse_, X, atol=1e-5)
    np.testing.assert_allclose(model.clean(), L0 + 7.0, atol=1e-4 * np.abs(L0).max())
    np.testing.assert_allclose(model.center_, (L0 + 7.0).mean(axis=0), atol=1e-4)
    assert model.corruption_fraction_ == pytest.approx(0.05, abs=0.01)
    assert model.explained_variance_ratio_.sum() == pytest.approx(1.0)
    assert np.all(np.diff(model.explained_variance_) <= 1e-9)


def test_outlier_scores_flag_corrupted_rows():
    rng = np.random.default_rng(4)
    L0 = rng.normal(size=(150, 3)) @ rng.normal(size=(3, 40))
    X = L0.copy()
    bad = rng.choice(150, 8, replace=False)
    X[bad, :5] += 25.0
    model = RobustPCA(n_components=3).fit(X)
    top = np.argsort(-model.outlier_scores_)[:8]
    assert set(top) == set(bad)
    assert np.all(model.outlier_scores_[np.setdiff1d(np.arange(150), bad)] < 1e-3)


def test_transform_and_inverse_transform_roundtrip_on_clean_rows():
    L0, S0, _ = _corrupted(seed=5)
    model = RobustPCA().fit(L0 + S0)
    clean_rows = L0[:10]
    Z = model.transform(clean_rows)
    np.testing.assert_allclose(model.inverse_transform(Z), clean_rows, atol=1e-4 * np.abs(L0).max())
    scores = model.fit_transform(L0 + S0)
    np.testing.assert_allclose(scores[:10], Z, atol=1e-4 * np.abs(L0).max())


def test_fixed_n_components_truncates_axes():
    L0, S0, _ = _corrupted(seed=6)
    model = RobustPCA(n_components=2).fit(L0 + S0)
    assert model.components_.shape == (2, L0.shape[1])
    assert model.rank_ == 4
    assert model.explained_variance_ratio_.sum() < 1.0


def test_center_options():
    L0, S0, _ = _corrupted(seed=7)
    X = L0 + S0 + 3.0
    none = RobustPCA(center=None).fit(X)
    assert np.all(none.center_ == 0)
    assert none.rank_ == 5  # the constant offset adds one direction
    mean = RobustPCA(center="mean").fit(X)
    np.testing.assert_allclose(mean.center_, mean.low_rank_.mean(axis=0))
    assert mean.rank_ == 4
    with pytest.raises(ValueError, match="center"):
        RobustPCA(center="median").fit(X)


def test_rpca_reduce_returns_dataframe():
    L0, S0, _ = _corrupted(n=80, d=30, seed=8)
    df = pd.DataFrame(L0 + S0, index=[f"r{i}" for i in range(80)])
    out = rpca_reduce(df, n_components=2)
    assert list(out.columns) == ["RPCA1", "RPCA2"]
    assert list(out.index) == list(df.index)


def test_zero_matrix_and_validation():
    L, S, info = principal_component_pursuit(np.zeros((4, 3)))
    assert np.all(L == 0) and np.all(S == 0) and info["rank"] == 0
    with pytest.raises(ValueError, match="2 x 2"):
        principal_component_pursuit(np.ones((1, 5)))
    with pytest.raises(ValueError, match="no observed"):
        principal_component_pursuit(np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="lam"):
        principal_component_pursuit(np.ones((3, 3)), lam=0.0)
    with pytest.raises(ValueError, match="rho"):
        principal_component_pursuit(np.ones((3, 3)), rho=1.0)
    with pytest.raises(ValueError, match="n_components"):
        RobustPCA(n_components=0).fit(np.ones((5, 4)))
    with pytest.raises(RuntimeError, match="not fitted"):
        RobustPCA().transform(np.ones((2, 2)))
    model = RobustPCA().fit(np.random.default_rng(0).normal(size=(20, 5)))
    with pytest.raises(ValueError, match="complete rows"):
        model.transform(np.full((1, 5), np.nan))
    with pytest.raises(ValueError, match="different number"):
        model.transform(np.ones((1, 4)))
