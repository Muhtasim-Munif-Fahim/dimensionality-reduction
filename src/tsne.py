"""Classic exact t-SNE in NumPy (van der Maaten & Hinton, 2008).

PCA / TruncatedSVD / FastICA / NMF are linear or parts-based factorizations.
t-SNE instead embeds points so that similarities in the high-dimensional
input match Student-t similarities in a low-dimensional map. This module
is sklearn-free: affinities, early exaggeration, and gradient descent are
all implemented with NumPy.

The exact O(n^2) algorithm is used (suitable for a few thousand points).
``fit_transform`` returns the embedding; there is no out-of-sample
``transform`` because classic t-SNE is a transductive map.
"""

from __future__ import annotations

import numpy as np

__all__ = ["TSNE", "tsne_reduce"]


class TSNE:
    """Exact t-distributed Stochastic Neighbor Embedding.

    Parameters
    ----------
    n_components :
        Embedding dimension (usually 2 or 3).
    perplexity :
        Effective neighbourhood size for the Gaussian affinities. Must be
        strictly less than ``n_samples``.
    learning_rate :
        Gradient step size (the paper's default is 200).
    n_iter :
        Maximum gradient iterations after early exaggeration ends.
    early_exaggeration :
        Multiplier on the input affinities for the first
        ``early_exaggeration_iter`` steps.
    early_exaggeration_iter :
        Number of exaggerated steps before the affinities revert.
    min_grad_norm :
        Stop early when the gradient Frobenius norm falls below this.
    random_state :
        Seed for the initial embedding. ``None`` leaves it unreproducible.

    Attributes
    ----------
    embedding_ :
        Low-dimensional map, shape ``(n_samples, n_components_)``.
    kl_divergence_ :
        Final Kullback–Leibler divergence between the joint affinities.
    n_iter_ :
        Gradient steps actually run.
    n_components_ :
        Embedding dimension used.
    n_features_in_ :
        Number of columns seen during ``fit``.
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        early_exaggeration_iter: int = 250,
        min_grad_norm: float = 1e-7,
        random_state: int | None = None,
    ) -> None:
        self.n_components = _validate_n_components(n_components)
        self.perplexity = _validate_perplexity(perplexity)
        self.learning_rate = _validate_positive(learning_rate, "learning_rate")
        self.n_iter = _validate_n_iter(n_iter)
        self.early_exaggeration = _validate_positive(
            early_exaggeration, "early_exaggeration"
        )
        self.early_exaggeration_iter = _validate_n_iter(early_exaggeration_iter)
        self.min_grad_norm = _validate_positive(min_grad_norm, "min_grad_norm")
        self.random_state = _validate_random_state(random_state)
        self.embedding_: np.ndarray | None = None
        self.kl_divergence_: float | None = None
        self.n_iter_: int | None = None
        self.n_components_: int | None = None
        self.n_features_in_: int | None = None

    def fit(self, X) -> TSNE:
        """Compute a t-SNE embedding of ``X`` and store it on ``embedding_``."""
        self.fit_transform(X)
        return self

    def fit_transform(self, X) -> np.ndarray:
        """Return the low-dimensional embedding of ``X``."""
        matrix = _as_dense(X)
        n_samples, n_features = matrix.shape
        if n_samples < 2:
            raise ValueError("X must contain at least two samples.")
        if self.perplexity >= n_samples:
            raise ValueError(
                f"perplexity={self.perplexity} must be less than n_samples={n_samples}."
            )
        k = int(self.n_components)
        if k >= n_samples:
            raise ValueError(
                f"n_components={k} must be less than n_samples={n_samples}."
            )

        distances = _pairwise_squared_distances(matrix)
        affinities = _binary_search_perplexity(distances, self.perplexity)
        # Symmetrise and normalise to a joint probability matrix.
        p = affinities + affinities.T
        p = np.maximum(p / p.sum(), 1e-12)

        rng = np.random.default_rng(self.random_state)
        y = 1e-4 * rng.standard_normal(size=(n_samples, k))
        gains = np.ones_like(y)
        update = np.zeros_like(y)
        momentum = 0.5
        exaggeration = float(self.early_exaggeration)
        kl = float("nan")
        n_iter_done = 0

        for iteration in range(1, self.n_iter + 1):
            if iteration == self.early_exaggeration_iter + 1:
                exaggeration = 1.0
                momentum = 0.8
            q, num = _student_t_affinities(y)
            pq = exaggeration * p - q
            # Gradient of KL(P || Q) w.r.t. the embedding (van der Maaten).
            grad = np.zeros_like(y)
            for i in range(n_samples):
                diff = y[i] - y
                grad[i] = 4.0 * np.sum((pq[i] * num[i])[:, None] * diff, axis=0)

            grad_norm = float(np.linalg.norm(grad))
            if grad_norm < self.min_grad_norm and iteration > self.early_exaggeration_iter:
                n_iter_done = iteration
                kl = float(np.sum(p * np.log(p / q)))
                break

            gains = np.where(np.sign(grad) != np.sign(update), gains + 0.2, gains * 0.8)
            gains = np.maximum(gains, 0.01)
            update = momentum * update - self.learning_rate * gains * grad
            y = y + update
            y -= y.mean(axis=0)
            n_iter_done = iteration
            if iteration % 50 == 0 or iteration == self.n_iter:
                kl = float(np.sum(p * np.log(p / q)))

        self.embedding_ = np.ascontiguousarray(y)
        self.kl_divergence_ = kl
        self.n_iter_ = n_iter_done
        self.n_components_ = k
        self.n_features_in_ = n_features
        return self.embedding_


def tsne_reduce(
    X,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    random_state: int | None = 0,
):
    """Fit classic t-SNE and return a DataFrame (or ndarray) embedding."""
    import pandas as pd

    embedding = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        random_state=random_state,
    ).fit_transform(X)
    cols = [f"TSNE{i + 1}" for i in range(n_components)]
    if isinstance(X, pd.DataFrame):
        return pd.DataFrame(embedding, index=X.index, columns=cols)
    return pd.DataFrame(embedding, columns=cols)


def _is_integral(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_n_components(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("n_components must be a positive integer.")
    return int(value)


def _validate_perplexity(value: object) -> float:
    try:
        perplexity = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("perplexity must be a positive finite number.") from exc
    if not np.isfinite(perplexity) or perplexity <= 0.0:
        raise ValueError("perplexity must be a positive finite number.")
    return perplexity


def _validate_positive(value: object, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite number.")
    return number


def _validate_n_iter(value: object) -> int:
    if not _is_integral(value) or int(value) < 1:
        raise ValueError("iteration counts must be positive integers.")
    return int(value)


def _validate_random_state(value: object) -> int | None:
    if value is None:
        return None
    if not _is_integral(value):
        raise ValueError("random_state must be an integer or None.")
    return int(value)


def _as_dense(X) -> np.ndarray:
    if hasattr(X, "toarray"):
        matrix = np.asarray(X.toarray(), dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("X must be a 2-D array of shape (n_samples, n_features).")
    if not np.isfinite(matrix).all():
        raise ValueError("X must contain only finite values.")
    return matrix


def _pairwise_squared_distances(X: np.ndarray) -> np.ndarray:
    sq = np.einsum("ij,ij->i", X, X)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return d2


def _binary_search_perplexity(distances: np.ndarray, perplexity: float) -> np.ndarray:
    """Row-wise Gaussian affinities with binary search on the bandwidth."""
    n = distances.shape[0]
    target_entropy = np.log(perplexity)
    affinities = np.zeros((n, n), dtype=float)
    # Reasonable beta (= 1/(2 sigma^2)) search bounds.
    beta_min_base = -np.inf
    beta_max_base = np.inf
    for i in range(n):
        beta = 1.0
        beta_min = beta_min_base
        beta_max = beta_max_base
        row = distances[i].copy()
        row[i] = np.inf
        for _ in range(64):
            probs = np.exp(-row * beta)
            sum_p = float(probs.sum())
            if sum_p <= 0.0 or not np.isfinite(sum_p):
                sum_p = 1e-12
                probs = np.full(n, 1e-12)
            probs /= sum_p
            entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12))))
            diff = entropy - target_entropy
            if abs(diff) < 1e-5:
                break
            if diff > 0.0:
                beta_min = beta
                beta = beta * 2.0 if not np.isfinite(beta_max) else 0.5 * (beta + beta_max)
            else:
                beta_max = beta
                beta = beta / 2.0 if not np.isfinite(beta_min) else 0.5 * (beta + beta_min)
        affinities[i] = probs
        affinities[i, i] = 0.0
    return affinities


def _student_t_affinities(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Q, numerator)`` for the Student-t kernel with one degree."""
    d2 = _pairwise_squared_distances(Y)
    num = 1.0 / (1.0 + d2)
    np.fill_diagonal(num, 0.0)
    total = float(num.sum())
    if total <= 0.0:
        n = Y.shape[0]
        q = np.full((n, n), 1.0 / max(n * (n - 1), 1))
        np.fill_diagonal(q, 0.0)
        return q, num
    q = np.maximum(num / total, 1e-12)
    return q, num
