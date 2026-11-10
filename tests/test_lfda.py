"""Tests for Local Fisher Discriminant Analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lfda import LocalFisherDiscriminantAnalysis, lfda_reduce  # noqa: E402


def _bimodal(n_per=60, seed=0):
    """Class 0 = two far-apart modes along x; class 1 sits between, shifted in y."""
    rng = np.random.default_rng(seed)
    a = rng.normal([-6.0, 0.0, 0.0], [0.6, 0.6, 3.0], size=(n_per, 3))
    b = rng.normal([6.0, 0.0, 0.0], [0.6, 0.6, 3.0], size=(n_per, 3))
    c = rng.normal([0.0, 2.5, 0.0], [0.6, 0.6, 3.0], size=(2 * n_per, 3))
    X = np.vstack([a, b, c])
    y = np.array([0] * (2 * n_per) + [1] * (2 * n_per))
    mode = np.array([0] * n_per + [1] * n_per + [2] * (2 * n_per))
    return X, y, mode


def _abs_corr(u, v):
    return abs(np.corrcoef(u, v)[0, 1])


def test_affinity_none_reproduces_lda_direction():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(m, 1.0, size=(50, 4)) for m in ([0, 0, 0, 0], [3, 1, 0, 0], [0, 3, 1, 0])])
    y = np.repeat([0, 1, 2], 50)
    lf = LocalFisherDiscriminantAnalysis(n_components=2, affinity="none").fit_transform(X, y)
    ld = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
    # same 2-D subspace: each LDA axis is a linear combination of LFDA axes
    coef, *_ = np.linalg.lstsq(np.c_[lf, np.ones(len(lf))], ld, rcond=None)
    resid = ld - np.c_[lf, np.ones(len(lf))] @ coef
    assert np.linalg.norm(resid) / np.linalg.norm(ld) < 1e-6


def test_generalized_eigen_property_plain():
    X, y, _ = _bimodal()
    m = LocalFisherDiscriminantAnalysis(n_components=3, embedding="plain").fit(X, y)
    assert np.all(np.diff(m.eigenvalues_) <= 1e-12)
    assert m.components_.shape == (3, 3)


def test_preserves_within_class_modes_and_separates_classes():
    X, y, mode = _bimodal()
    Z = LocalFisherDiscriminantAnalysis(n_components=2).fit_transform(X, y)
    # classes separated along some axis; the two class-0 modes stay apart.
    za, zb = Z[mode == 0].mean(0), Z[mode == 1].mean(0)
    spread = Z[y == 0].std(0).mean()
    assert np.linalg.norm(za - zb) > 3 * spread / 2
    from sklearn.neighbors import KNeighborsClassifier

    acc = KNeighborsClassifier(5).fit(Z, y).score(Z, y)
    assert acc > 0.97


def test_lda_collapses_modes_but_lfda_does_not():
    X, y, mode = _bimodal()
    lda = LinearDiscriminantAnalysis(n_components=1).fit_transform(X, y)[:, 0]
    Z = LocalFisherDiscriminantAnalysis(n_components=2).fit_transform(X, y)
    lda_gap = abs(lda[mode == 0].mean() - lda[mode == 1].mean()) / lda.std()
    lfda_gap = np.linalg.norm(Z[mode == 0].mean(0) - Z[mode == 1].mean(0)) / Z.std(0).mean()
    assert lfda_gap > 2 * lda_gap


def test_transform_matches_embedding_and_orthonormal():
    X, y, _ = _bimodal()
    m = LocalFisherDiscriminantAnalysis(n_components=2, embedding="orthonormalized").fit(X, y)
    np.testing.assert_allclose(m.transform(X), m.embedding_, atol=1e-10)
    np.testing.assert_allclose(m.components_ @ m.components_.T, np.eye(2), atol=1e-10)


def test_reduce_dataframe():
    X, y, _ = _bimodal()
    df = pd.DataFrame(X, index=[f"s{i}" for i in range(len(X))])
    out = lfda_reduce(df, y, n_components=2)
    assert list(out.columns) == ["LFDA1", "LFDA2"]
    assert list(out.index) == list(df.index)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_components": 0},
        {"n_components": 4},
        {"n_neighbors": 0},
        {"affinity": "gauss"},
        {"embedding": "nope"},
        {"reg": -1.0},
    ],
)
def test_invalid_params(kwargs):
    X, y, _ = _bimodal()
    with pytest.raises(ValueError):
        LocalFisherDiscriminantAnalysis(**kwargs).fit(X, y)


def test_single_class_and_bad_y():
    X, y, _ = _bimodal()
    with pytest.raises(ValueError):
        LocalFisherDiscriminantAnalysis().fit(X, np.zeros(len(X)))
    with pytest.raises(ValueError):
        LocalFisherDiscriminantAnalysis().fit(X, y[:-1])
    with pytest.raises(RuntimeError):
        LocalFisherDiscriminantAnalysis().transform(X)
