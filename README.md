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

## Evaluation

- Variance retention
- Reconstruction error
- Silhouette score on reduced embeddings
- Visual separation quality
