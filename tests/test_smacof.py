"""Tests for SMACOF iterative metric MDS."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smacof import SMACOF, smacof_reduce


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


def test_fit_transform_shape_and_attrs():
    X = _grid_2d()
    model = SMACOF(n_components=2, max_iter=50)
    emb = model.fit_transform(X)
    assert emb.shape == (25, 2)
    assert model.embedding_ is not None
    assert model.stress_ is not None and model.stress_ >= 0.0
    assert model.n_iter_ is not None and model.n_iter_ >= 1
    assert model.reconstruction_error_ is not None
    assert np.all(np.isfinite(emb))


def test_recovers_euclidean_grid():
    X = _grid_2d(n=6)
    emb = SMACOF(n_components=2, max_iter=100, tol=1e-9).fit_transform(X)
    assert _procrustes_rmse(emb, X) < 1e-5


def test_stress_nonincreasing_on_noisy_distances():
    rng = np.random.default_rng(0)
    X = _grid_2d(n=4)
    # Classical seed already near-optimal; perturb target distances.
    from mds import _pairwise_euclidean

    d = _pairwise_euclidean(X)
    noise = rng.uniform(0.0, 0.05, size=d.shape)
    noise = 0.5 * (noise + noise.T)
    np.fill_diagonal(noise, 0.0)
    d_noisy = d + noise
    model = SMACOF(n_components=2, metric="precomputed", init="random", random_state=0, max_iter=30)
    # Manual loop to track stress
    from smacof import _initial_config, _raw_stress, _guttman_transform

    y = _initial_config(d_noisy, 2, "random", 0)
    prev = _raw_stress(d_noisy, y)
    for _ in range(20):
        y = _guttman_transform(d_noisy, y)
        cur = _raw_stress(d_noisy, y)
        assert cur <= prev + 1e-9
        prev = cur
    model.fit(d_noisy)
    assert model.stress_ is not None


def test_deterministic_classical_init():
    X = _grid_2d()
    a = SMACOF(n_components=2, max_iter=40).fit_transform(X)
    b = SMACOF(n_components=2, max_iter=40).fit_transform(X)
    assert np.allclose(a, b)


def test_precomputed_and_dataframe():
    X = _grid_2d(n=4)
    from mds import _pairwise_euclidean

    d = _pairwise_euclidean(X)
    emb = SMACOF(n_components=2, metric="precomputed", max_iter=40).fit_transform(d)
    assert emb.shape == (16, 2)
    df = pd.DataFrame(X, columns=["a", "b"])
    out = smacof_reduce(df, n_components=2, max_iter=20)
    assert list(out.columns) == ["SMACOF1", "SMACOF2"]
    assert len(out) == 16


def test_validation():
    X = _grid_2d(n=3)
    _raises(ValueError, lambda: SMACOF(n_components=0))
    _raises(ValueError, lambda: SMACOF(metric="cosine"))
    _raises(ValueError, lambda: SMACOF(init="pca"))
    _raises(ValueError, lambda: SMACOF(max_iter=0))
    _raises(ValueError, lambda: SMACOF(n_components=9).fit_transform(X))
