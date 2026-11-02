"""Tests for Hessian LLE."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hessian_lle import HessianLLE, hessian_lle_reduce


def _affine_r2(emb: np.ndarray, truth: np.ndarray) -> float:
    """Worst R^2 of regressing each true coordinate on the embedding (affine)."""
    A = np.hstack([np.ones((emb.shape[0], 1)), emb])
    r2 = []
    for j in range(truth.shape[1]):
        coef, *_ = np.linalg.lstsq(A, truth[:, j], rcond=None)
        resid = truth[:, j] - A @ coef
        r2.append(1 - resid.var() / truth[:, j].var())
    return float(min(r2))


def _swiss_roll_with_hole(n=600, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        u, v = rng.uniform(0, 1, size=2)
        if 0.35 < u < 0.65 and 0.35 < v < 0.65:
            continue  # the hole makes the parameter domain non-convex
        pts.append((u, v))
    uv = np.array(pts)
    t = 1.5 * np.pi * (1 + 2 * uv[:, 0])
    X = np.column_stack([t * np.cos(t), 21 * uv[:, 1], t * np.sin(t)])
    return X, np.column_stack([t, uv[:, 1]])


def test_flat_plane_recovered_exactly():
    rng = np.random.default_rng(1)
    uv = rng.uniform(-1, 1, size=(200, 2))
    basis = np.linalg.qr(rng.normal(size=(5, 2)))[0]
    X = uv @ basis.T + 3.0
    model = HessianLLE(n_components=2, n_neighbors=12)
    emb = model.fit_transform(X)
    assert emb.shape == (200, 2)
    assert _affine_r2(emb, uv) > 1 - 1e-8
    np.testing.assert_allclose(emb.T @ emb / 200, np.eye(2), atol=1e-8)
    assert model.reconstruction_error_ < 1e-8


def test_swiss_roll_with_hole_unrolled():
    X, truth = _swiss_roll_with_hole(1000)
    emb = HessianLLE(n_components=2, n_neighbors=12).fit_transform(X)
    assert _affine_r2(emb, truth) > 0.97


def test_matches_sklearn_hessian_lle():
    from sklearn.manifold import LocallyLinearEmbedding

    X, _ = _swiss_roll_with_hole(1000)
    ours = HessianLLE(n_components=2, n_neighbors=12).fit_transform(X)
    ref = LocallyLinearEmbedding(n_components=2, n_neighbors=12, method="hessian",
                                 eigen_solver="dense").fit_transform(X)
    # Same subspace: each embedding is an affine function of the other.
    assert _affine_r2(ours, ref) > 0.999
    assert _affine_r2(ref, ours) > 0.999


def test_one_dimensional_curve_and_dataframe():
    t = np.linspace(0, 3 * np.pi, 150)
    X = np.column_stack([np.cos(t), np.sin(t), 0.3 * t])  # helix, isometric to a line
    frame = pd.DataFrame(X, index=[f"r{i}" for i in range(150)])
    out = hessian_lle_reduce(frame, n_components=1, n_neighbors=8)
    assert list(out.columns) == ["HLLE1"] and list(out.index) == list(frame.index)
    assert abs(np.corrcoef(out["HLLE1"], t)[0, 1]) > 0.999


def test_deterministic_and_attributes():
    X, _ = _swiss_roll_with_hole(150, seed=3)
    a = HessianLLE(n_components=2, n_neighbors=10).fit(X)
    b = HessianLLE(n_components=2, n_neighbors=10).fit(X)
    np.testing.assert_allclose(a.embedding_, b.embedding_)
    assert a.hessian_matrix_.shape == (150, 150)
    np.testing.assert_allclose(a.hessian_matrix_, a.hessian_matrix_.T)
    assert a.eigenvalues_.shape == (3,) and a.n_features_in_ == 3
    assert HessianLLE(n_components=3).n_neighbors == 12


def test_validation():
    with pytest.raises(ValueError):
        HessianLLE(n_components=0)
    with pytest.raises(ValueError, match="n_neighbors must be >"):
        HessianLLE(n_components=2, n_neighbors=5)
    with pytest.raises(ValueError):
        HessianLLE(reg=-1.0)
    with pytest.raises(ValueError):
        HessianLLE(n_neighbors=12).fit(np.random.default_rng(0).normal(size=(10, 3)))
    with pytest.raises(ValueError):
        HessianLLE(n_components=3, n_neighbors=12).fit(np.random.default_rng(0).normal(size=(50, 3)))
    with pytest.raises(ValueError):
        HessianLLE().fit(np.array([[np.nan, 1.0, 2.0]] * 30))
