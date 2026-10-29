"""Tests for Neighbourhood Components Analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nca import NeighbourhoodComponentsAnalysis, nca_objective  # noqa: E402


def _noisy_classes(n=300, p_noise=6, seed=0):
    """Two informative dims (3 well-separated classes) plus large-variance noise dims."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    centres = np.array([[0, 0], [3, 0], [0, 3]], float)
    inf = centres[y] + 0.5 * rng.normal(size=(n, 2))
    noise = 5.0 * rng.normal(size=(n, p_noise))
    return np.c_[inf, noise], y


def _knn_loo(Z, y):
    D = np.sum((Z[:, None] - Z[None]) ** 2, -1)
    np.fill_diagonal(D, np.inf)
    return np.mean(y[np.argmin(D, 1)] == y)


def test_gradient_matches_finite_differences():
    X, y = _noisy_classes(40, 2, seed=1)
    rng = np.random.default_rng(0)
    A = 0.3 * rng.normal(size=(2, X.shape[1]))
    f, g = nca_objective(A, X, y)
    eps = 1e-6
    num = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            E = np.zeros_like(A)
            E[i, j] = eps
            num[i, j] = (nca_objective(A + E, X, y)[0] - nca_objective(A - E, X, y)[0]) / (2 * eps)
    assert np.allclose(g, num, rtol=1e-5, atol=1e-6)
    assert 0 <= f <= len(y)


def test_finds_informative_subspace_and_beats_pca():
    X, y = _noisy_classes()
    nca = NeighbourhoodComponentsAnalysis(n_components=2, random_state=0).fit(X, y)
    Z = nca.transform(X)
    assert _knn_loo(Z, y) > 0.95
    Xc = X - X.mean(0)
    pca = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:2].T
    assert _knn_loo(pca, y) < 0.7
    # learned metric concentrates on the two informative features
    M = nca.mahalanobis_matrix_
    assert np.trace(M[:2, :2]) > 20 * np.trace(M[2:, 2:])
    # objective improved over the initialisation and is the LOO soft accuracy
    assert nca.objective_history_[-1] >= nca.objective_history_[0]
    assert nca.leave_one_out_accuracy() == pytest.approx(nca.objective_ / len(y))
    assert nca.leave_one_out_accuracy() > 0.9


def test_generalises_with_knn_predict():
    X, y = _noisy_classes(400, seed=2)
    Xt, yt = _noisy_classes(300, seed=3)
    nca = NeighbourhoodComponentsAnalysis(n_components=2, init="lda").fit(X, y)
    assert nca.score(Xt, yt, n_neighbors=5) > 0.93
    pred = nca.predict(pd.DataFrame(Xt))
    assert set(np.unique(pred)) <= set(np.unique(y))


def test_auto_init_avoids_poor_pca_start():
    X, y = _noisy_classes()
    auto = NeighbourhoodComponentsAnalysis(n_components=2).fit(X, y)
    pca = NeighbourhoodComponentsAnalysis(n_components=2, init="pca").fit(X, y)
    assert auto.objective_ > pca.objective_


def test_matches_scikit_learn_objective():
    sk = pytest.importorskip("sklearn.neighbors")
    X, y = _noisy_classes(150, 3, seed=4)
    ours = NeighbourhoodComponentsAnalysis(n_components=2, init="identity", max_iter=50, tol=1e-6).fit(X, y)
    ref = sk.NeighborhoodComponentsAnalysis(n_components=2, init="identity", max_iter=50, tol=1e-6).fit(X, y)
    f_ref = nca_objective(ref.components_, X, y)[0]
    # same objective; both runs end at a comparable local optimum
    assert ours.objective_ == pytest.approx(f_ref, rel=0.03)


def test_standardize_and_inits_and_string_labels():
    X, y = _noisy_classes(200, seed=5)
    X = X * np.r_[100.0, 100.0, np.ones(X.shape[1] - 2)]
    labels = np.array(["a", "b", "c"])[y]
    for init in ("auto", "pca", "lda", "identity", "random"):
        m = NeighbourhoodComponentsAnalysis(n_components=2, init=init, standardize=True, random_state=0).fit(X, labels)
        assert m.transform(X).shape == (200, 2)
        if init != "random":
            assert m.leave_one_out_accuracy() > 0.85
        else:
            assert m.objective_history_[-1] >= m.objective_history_[0]
    # components_ act on raw features: transform == (X - mean) @ components_.T
    assert np.allclose(m.transform(X), (X - m.mean_) @ m.components_.T)
    full = NeighbourhoodComponentsAnalysis().fit(X[:, :4], labels)
    assert full.components_.shape == (4, 4)


def test_validation():
    X, y = _noisy_classes(30, 1)
    with pytest.raises(ValueError):
        NeighbourhoodComponentsAnalysis(n_components=10).fit(X, y)
    with pytest.raises(ValueError):
        NeighbourhoodComponentsAnalysis().fit(X, np.zeros(30))
    with pytest.raises(ValueError):
        NeighbourhoodComponentsAnalysis().fit(X, y[:-1])
    with pytest.raises(ValueError):
        NeighbourhoodComponentsAnalysis(init="bogus").fit(X, y)
    with pytest.raises(ValueError):
        NeighbourhoodComponentsAnalysis(n_components=2, init=np.eye(3)).fit(X, y)
    with pytest.raises(RuntimeError):
        NeighbourhoodComponentsAnalysis().transform(X)
    m = NeighbourhoodComponentsAnalysis(n_components=1).fit(X, y)
    with pytest.raises(ValueError):
        m.transform(X[:, :2])
