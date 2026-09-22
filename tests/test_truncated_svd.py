"""Tests for truncated SVD on a Gaussian blob and sparse TF-IDF matrices."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from truncated_svd import TruncatedSVD


def _gaussian_blob(
    n_samples: int = 160,
    n_features: int = 12,
    intrinsic: int = 4,
    noise: float = 0.08,
    seed: int = 42,
) -> np.ndarray:
    """Low-rank Gaussian blob embedded in a higher-dimensional space."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_samples, intrinsic))
    weights = rng.normal(size=(intrinsic, n_features))
    observed = latent @ weights
    if noise:
        observed = observed + rng.normal(scale=noise, size=observed.shape)
    return observed


def _tfidf_matrix(
    n_docs: int = 70,
    n_terms: int = 18,
    n_topics: int = 4,
    seed: int = 7,
) -> sparse.csr_matrix:
    """Sparse non-negative document-term matrix with TF-IDF weighting."""
    rng = np.random.default_rng(seed)
    topics = rng.random((n_topics, n_terms))
    topics[topics < 0.65] = 0.0
    topics /= np.maximum(topics.sum(axis=1, keepdims=True), 1e-12)
    mixture = rng.dirichlet(np.ones(n_topics) * 0.4, size=n_docs)
    counts = rng.poisson(mixture @ topics * 6.0).astype(np.float64)
    term_frequency = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    document_frequency = np.count_nonzero(counts, axis=0)
    inverse_document_frequency = np.log((1.0 + n_docs) / (1.0 + document_frequency)) + 1.0
    weighted = term_frequency * inverse_document_frequency
    weighted[weighted < 1e-12] = 0.0
    return sparse.csr_matrix(weighted)


def _raises(exc_type, fn) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


class TestGaussianBlob:
    def test_fit_transform_and_inverse_shape(self):
        blob = _gaussian_blob()
        svd = TruncatedSVD(n_components=3)
        embedding = svd.fit_transform(blob)
        restored = svd.inverse_transform(embedding)
        assert embedding.shape == (blob.shape[0], 3)
        assert restored.shape == blob.shape
        assert svd.explained_variance_ratio_.shape == (3,)
        assert svd.n_components_ == 3

    def test_matches_numpy_svd(self):
        blob = _gaussian_blob(seed=1)
        rank = 3
        svd = TruncatedSVD(n_components=rank).fit(blob)
        _, singular, components = np.linalg.svd(blob, full_matrices=False)
        assert np.allclose(svd.singular_values_, singular[:rank], atol=1e-6)
        assert np.allclose(np.abs(svd.components_), np.abs(components[:rank]), atol=1e-6)
        restored = svd.inverse_transform(svd.transform(blob))
        reference = (blob @ components[:rank].T) @ components[:rank]
        assert np.allclose(restored, reference, atol=1e-6)

    def test_components_are_orthonormal(self):
        blob = _gaussian_blob()
        svd = TruncatedSVD(n_components=4).fit(blob)
        gram = svd.components_ @ svd.components_.T
        assert np.allclose(gram, np.eye(4), atol=1e-5)

    def test_explained_variance_matches_score_variance(self):
        blob = _gaussian_blob()
        svd = TruncatedSVD(n_components=4).fit(blob)
        scores = svd.transform(blob)
        total = np.var(blob, axis=0).sum()
        expected = np.var(scores, axis=0) / total
        assert np.allclose(svd.explained_variance_ratio_, expected, atol=1e-8)
        assert np.all(svd.explained_variance_ratio_ >= -1e-12)
        assert svd.explained_variance_ratio_.sum() <= 1.0 + 1e-6

    def test_full_rank_retains_all_variance(self):
        blob = _gaussian_blob(n_samples=90, n_features=6, intrinsic=6, noise=0.0)
        svd = TruncatedSVD(n_components=6).fit(blob)
        restored = svd.inverse_transform(svd.transform(blob))
        assert abs(float(svd.explained_variance_ratio_.sum()) - 1.0) < 1e-6
        assert np.allclose(restored, blob, atol=1e-6)

    def test_more_components_lower_reconstruction_error(self):
        blob = _gaussian_blob()
        errors = []
        for rank in (1, 2, 4):
            svd = TruncatedSVD(n_components=rank).fit(blob)
            restored = svd.inverse_transform(svd.transform(blob))
            errors.append(float(np.mean((blob - restored) ** 2)))
        assert errors[0] > errors[1] > errors[2]

    def test_variance_threshold_keeps_about_95_percent(self):
        blob = _gaussian_blob(
            n_samples=200,
            n_features=15,
            intrinsic=3,
            noise=0.02,
            seed=5,
        )
        svd = TruncatedSVD(n_components=0.95).fit(blob)
        assert svd.explained_variance_ratio_.sum() >= 0.95
        assert 1 <= svd.n_components_ < blob.shape[1]
        if svd.n_components_ > 1:
            smaller = TruncatedSVD(n_components=svd.n_components_ - 1).fit(blob)
            assert smaller.explained_variance_ratio_.sum() < 0.95

    def test_unit_threshold_reconstructs_blob(self):
        blob = _gaussian_blob(n_samples=40, n_features=8, intrinsic=8, noise=0.0, seed=3)
        svd = TruncatedSVD(n_components=1.0).fit(blob)
        restored = svd.inverse_transform(svd.transform(blob))
        assert np.allclose(restored, blob, atol=1e-6)

    def test_does_not_remove_a_constant_offset(self):
        offset = np.full((30, 5), 3.0)
        svd = TruncatedSVD(n_components=1).fit(offset)
        restored = svd.inverse_transform(svd.transform(offset))
        assert np.allclose(restored, offset, atol=1e-8)
        assert abs(float(restored.mean()) - 3.0) < 1e-8

    def test_dataframe_input(self):
        blob = _gaussian_blob(n_samples=40, n_features=5, intrinsic=3)
        frame = pd.DataFrame(blob, columns=[f"f{i}" for i in range(blob.shape[1])])
        embedding = TruncatedSVD(n_components=2).fit_transform(frame)
        assert embedding.shape == (len(frame), 2)


class TestSparseTfIdf:
    def test_sparse_matrix_is_actually_sparse(self):
        tfidf = _tfidf_matrix()
        assert sparse.issparse(tfidf)
        assert tfidf.nnz < tfidf.shape[0] * tfidf.shape[1] * 0.5

    def test_sparse_matches_dense_fit(self):
        tfidf = _tfidf_matrix()
        dense = tfidf.toarray()
        sparse_model = TruncatedSVD(n_components=4).fit(tfidf)
        dense_model = TruncatedSVD(n_components=4).fit(dense)
        assert np.allclose(
            sparse_model.explained_variance_ratio_,
            dense_model.explained_variance_ratio_,
            atol=1e-8,
        )
        sparse_restored = sparse_model.inverse_transform(sparse_model.transform(tfidf))
        dense_restored = dense_model.inverse_transform(dense_model.transform(dense))
        assert np.allclose(sparse_restored, dense_restored, atol=1e-6)

    def test_fit_does_not_mutate_sparse_values(self):
        tfidf = _tfidf_matrix()
        before = tfidf.data.copy()
        TruncatedSVD(n_components=3).fit(tfidf)
        assert np.array_equal(tfidf.data, before)

    def test_variance_threshold_on_tfidf(self):
        tfidf = _tfidf_matrix(n_docs=90, n_terms=12, n_topics=3, seed=11)
        svd = TruncatedSVD(n_components=0.95).fit(tfidf)
        assert svd.explained_variance_ratio_.sum() >= 0.95
        assert svd.n_components_ <= tfidf.shape[1]
        restored = svd.inverse_transform(svd.transform(tfidf))
        assert restored.shape == tfidf.shape

    def test_wide_sparse_matrix(self):
        tfidf = _tfidf_matrix(n_docs=20, n_terms=35, n_topics=3, seed=9)
        assert tfidf.shape[0] < tfidf.shape[1]
        svd = TruncatedSVD(n_components=5).fit(tfidf)
        embedding = svd.transform(tfidf)
        assert embedding.shape == (tfidf.shape[0], 5)
        gram = svd.components_ @ svd.components_.T
        assert np.allclose(gram, np.eye(5), atol=1e-5)


class TestTruncatedSVDValidation:
    def test_rejects_bad_n_components(self):
        blob = _gaussian_blob(n_samples=20, n_features=5, intrinsic=3)
        _raises(ValueError, lambda: TruncatedSVD(n_components=0))
        _raises(ValueError, lambda: TruncatedSVD(n_components=1.5))
        _raises(ValueError, lambda: TruncatedSVD(n_components=-0.2))
        _raises(ValueError, lambda: TruncatedSVD(n_components=True))
        _raises(ValueError, lambda: TruncatedSVD(n_components=6).fit(blob))

    def test_rejects_non_finite_and_wrong_width(self):
        blob = _gaussian_blob(n_samples=20, n_features=5, intrinsic=3)
        svd = TruncatedSVD(n_components=2).fit(blob)
        broken = blob.copy()
        broken[0, 0] = np.nan
        _raises(ValueError, lambda: TruncatedSVD(n_components=2).fit(broken))
        _raises(ValueError, lambda: svd.transform(blob[:, :3]))
        _raises(ValueError, lambda: svd.inverse_transform(np.ones((4, 1))))
        _raises(RuntimeError, lambda: TruncatedSVD(n_components=2).transform(blob))
