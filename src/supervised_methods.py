"""LDA and t-SNE dimensionality reduction wrappers."""

from __future__ import annotations

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE


def lda_reduce(X: pd.DataFrame, y: pd.Series, n_components: int | None = None) -> pd.DataFrame:
    model = LinearDiscriminantAnalysis(n_components=n_components)
    projected = model.fit_transform(X, y)
    cols = [f"LD{i+1}" for i in range(projected.shape[1])]
    return pd.DataFrame(projected, columns=cols)


def tsne_reduce(X: pd.DataFrame, n_components: int = 2, perplexity: float = 30.0) -> pd.DataFrame:
    model = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, n_jobs=-1)
    projected = model.fit_transform(X)
    cols = [f"TSNE{i+1}" for i in range(n_components)]
    return pd.DataFrame(projected, columns=cols)
