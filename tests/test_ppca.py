"""Tests for Probabilistic PCA (closed form, Minka/BIC selection, EM with missing data)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ppca import (  # noqa: E402
    ProbabilisticPCA,
    minka_log_evidence,
    minka_mle_dimension,
    ppca_reduce,
)


def _low_rank(n=300, d=10, q=3, noise=0.7, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, q))
    W = rng.normal(size=(q, d)) * 2.0
    return Z @ W + rng.normal(size=(n, d)) * noise + 5.0


def test_closed_form_matches_sklearn_probabilistic_pca():
    X = _low_rank()
    sk = PCA(n_components=3).fit(X)
    model = ProbabilisticPCA(n_components=3, ddof=1).fit(X)
    assert model.score(X) == pytest.approx(sk.score(X), rel=1e-10)
    assert np.allclose(model.get_covariance(), sk.get_covariance())
    assert model.noise_variance_ == pytest.approx(sk.noise_variance_)
    assert np.allclose(np.abs(model.components_), np.abs(sk.components_), atol=1e-8)
    assert np.allclose(model.explained_variance_, sk.explained_variance_)


def test_density_precision_and_reconstruction():
    X = _low_rank(n=200)
    model = ProbabilisticPCA(n_components=3).fit(X)
    C = model.get_covariance()
    assert np.allclose(model.get_precision(), np.linalg.inv(C))
    ref = multivariate_normal(model.mean_, C).logpdf(X[:10])
    assert np.allclose(model.score_samples(X[:10]), ref)
    # W W.T = U (L - sigma2) U.T and loadings rows are orthogonal
    W = model.loadings_.T
    gram = W.T @ W
    assert np.allclose(gram, np.diag(np.diag(gram)), atol=1e-8)
    Z = model.transform(X)
    assert Z.shape == (200, 3)
    recon = model.inverse_transform(Z)
    assert np.mean((recon - X) ** 2) < np.var(X - X.mean(0)) * 0.2
    with pytest.raises(ValueError, match="features"):
        model.transform(X[:, :5])


def test_minka_selection_matches_sklearn_mle():
    for seed, q in [(0, 3), (1, 2), (2, 5)]:
        X = _low_rank(d=12, q=q, seed=seed)
        expected = PCA(n_components="mle").fit(X).n_components_
        assert minka_mle_dimension(X) == expected == q
        model = ProbabilisticPCA(n_components="mle", ddof=1).fit(X)
        assert model.n_components_ == q
        assert model.evidence_.shape == (11,)
        assert int(np.argmax(model.evidence_)) + 1 == q


def test_bic_selection_recovers_rank_and_validation():
    X = _low_rank(n=400, d=8, q=2, seed=3)
    assert ProbabilisticPCA(n_components="bic").fit(X).n_components_ == 2
    with pytest.raises(ValueError):
        ProbabilisticPCA(n_components="aic").fit(X)
    with pytest.raises(ValueError):
        ProbabilisticPCA(n_components=8).fit(X)
    with pytest.raises(ValueError):
        ProbabilisticPCA(n_components=0).fit(X)
    with pytest.raises(ValueError):
        ProbabilisticPCA(n_components=2, ddof=3).fit(X)
    with pytest.raises(ValueError):
        minka_log_evidence(np.ones(4), 4, 10)
    with pytest.raises(RuntimeError):
        ProbabilisticPCA().transform(X)


def test_em_on_complete_data_reaches_closed_form_likelihood():
    X = _low_rank(n=120, d=6, q=2, seed=4)
    closed = ProbabilisticPCA(n_components=2).fit(X)
    em = ProbabilisticPCA(n_components=2, tol=1e-10, max_iter=3000, random_state=0)
    em._fit_em(X.copy(), np.ones_like(X, dtype=bool), 2)
    em.n_features_in_ = X.shape[1]
    assert em.loglike_history_[-1] == pytest.approx(closed.score(X), abs=1e-4)
    assert em.noise_variance_ == pytest.approx(closed.noise_variance_, rel=1e-2)


def test_em_with_missing_values_is_monotone_and_imputes_well():
    X = _low_rank(n=150, d=8, q=2, seed=5)
    rng = np.random.default_rng(5)
    Xm = X.copy()
    Xm[rng.random(X.shape) < 0.2] = np.nan
    model = ProbabilisticPCA(n_components=2, random_state=0).fit(Xm)
    history = model.loglike_history_
    assert model.n_iter_ == history.size >= 2
    assert np.all(np.diff(history) > -1e-9)
    filled = model.impute(Xm)
    mask = np.isnan(Xm)
    assert not np.isnan(filled).any()
    assert np.array_equal(filled[~mask], Xm[~mask])
    mean_fill = np.where(mask, np.nanmean(Xm, axis=0), Xm)
    rmse_ppca = np.sqrt(np.mean((filled - X)[mask] ** 2))
    rmse_mean = np.sqrt(np.mean((mean_fill - X)[mask] ** 2))
    assert rmse_ppca < 0.5 * rmse_mean
    emb = model.fit_transform(Xm)
    assert emb.shape == (150, 2) and np.isfinite(emb).all()
    with pytest.raises(ValueError, match="NaN"):
        model.score(Xm)
    with pytest.raises(ValueError, match="complete data"):
        ProbabilisticPCA(n_components="mle").fit(Xm)


def test_em_rejects_fully_missing_feature():
    X = _low_rank(n=40, d=5, q=2)
    X[:, 2] = np.nan
    with pytest.raises(ValueError, match="feature"):
        ProbabilisticPCA(n_components=2).fit(X)


def test_ppca_reduce_returns_dataframe():
    X = pd.DataFrame(_low_rank(n=60, d=6, q=2), index=[f"r{i}" for i in range(60)])
    frame = ppca_reduce(X, n_components="mle")
    assert list(frame.columns) == ["PPCA1", "PPCA2"]
    assert list(frame.index) == list(X.index)
