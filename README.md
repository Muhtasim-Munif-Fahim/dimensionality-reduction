# Dimensionality Reduction

Practice project comparing dimensionality reduction methods for high-dimensional data.

## Project Structure

```
src/                - Source code
data/               - Sample data
tests/              - Unit tests
```

## Methods

- PCA (manual implementation + scikit-learn)
- SVD-based reduction
- Truncated SVD for sparse TF-IDF matrices (no centering)
- FastICA for linear mixtures of independent sources
- LDA (supervised)
- t-SNE (visualization)
- Autoencoder (numpy from scratch)

## Pivot: TruncatedSVD instead of another PCA

PCA is already implemented in `src/pca.py` (`pca_manual`, `pca_svd`, and `variance_retained`). Those routines subtract the column mean before factoring. Centering a bag-of-words or TF-IDF matrix fills in structural zeros and destroys sparsity, so this change does not add a second PCA.

`TruncatedSVD` is the sparse-matrix counterpart. It factors the matrix as stored, accepts SciPy CSR/CSC input, and exposes `fit`, `transform`, `inverse_transform`, and `explained_variance_ratio_`. `n_components` is either an integer rank or a variance fraction such as `0.95`.

Run the example from the repository root with `src` on `PYTHONPATH` (the same layout `run.py` uses).

```python
import numpy as np
from scipy import sparse
from truncated_svd import TruncatedSVD

rng = np.random.default_rng(0)
counts = sparse.csr_matrix(rng.poisson(0.4, size=(80, 30)).astype(float))
document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
idf = np.log((1 + counts.shape[0]) / (1 + document_frequency)) + 1.0
tfidf = counts.multiply(idf)

svd = TruncatedSVD(n_components=0.95)
embedding = svd.fit_transform(tfidf)
restored = svd.inverse_transform(embedding)

print(embedding.shape)  # (80, k)
print(svd.n_components_, round(float(svd.explained_variance_ratio_.sum()), 3))
print(restored.shape)  # (80, 30)
```

An integer rank works the same way: `TruncatedSVD(n_components=10)`. The embedding is dense. `inverse_transform` does not add a mean, because the decomposition never subtracted one.

## FastICA

`FastICA` in `src/ica.py` is a symmetric fixed-point ICA (Hyvärinen's algorithm with the logcosh contrast). PCA and truncated SVD factor variance. ICA whitens that subspace and then rotates it so the coordinates are as independent as the contrast can make them. Use it when the observed columns are an unknown linear mix of a few non-Gaussian sources.

`n_components` is the number of sources. `None` keeps every feature. `fit`, `transform`, and `inverse_transform` follow the same estimator shape as `TruncatedSVD`. Recovered sources and mixing columns are only determined up to sign and order.

Run the example from the repository root with `src` on `PYTHONPATH`.

```python
import numpy as np
from ica import FastICA

rng = np.random.default_rng(0)
t = np.linspace(0, 8, 1500, endpoint=False)
sources = np.column_stack(
    [
        np.sin(2 * np.pi * 0.7 * t),
        np.sign(np.sin(2 * np.pi * 1.5 * t)),
        rng.laplace(size=t.shape[0]),
    ]
)
sources -= sources.mean(axis=0)
sources /= sources.std(axis=0)
observed = sources @ rng.normal(size=(6, 3)).T

ica = FastICA(n_components=3, random_state=0)
recovered = ica.fit_transform(observed)
restored = ica.inverse_transform(recovered)

print(recovered.shape)  # (1500, 3)
print(ica.n_iter_, round(float(np.mean((observed - restored) ** 2)), 6))
```

`components_` is the unmixing matrix, shape `(n_components, n_features)`. `mixing_` maps sources back to the centered features. `inverse_transform` adds the training mean.

## Evaluation

- Variance retention
- Reconstruction error
- Silhouette score on reduced embeddings
- Visual separation quality
