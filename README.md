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
- NMF (non-negative parts-based factors, multiplicative updates)
- LDA (supervised)
- t-SNE (visualization; sklearn wrapper)
- Classic t-SNE (exact NumPy, perplexity / learning-rate)
- Isomap (geodesic kNN graph + classical MDS)
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

## NMF

`NMF` in `src/nmf.py` factors a non-negative matrix as `X ≈ W @ H` with Lee–Seung multiplicative updates. `W` is the sample embedding and `H` (`components_`) is the basis. Both stay non-negative, so the parts can be read as topics, spectra, or additive features. PCA and FastICA are free to use negative loadings; this one is not.

The implementation is NumPy only. `nmf_reduce` follows `pca_manual`: it returns the embedding and an info dict. A DataFrame comes back as a DataFrame with columns `NMF1`, `NMF2`, ...; an array comes back as an array. `components`, `reconstruction_error` (mean squared residual), and `n_iter` live in the info dict. The class also exposes `fit`, `transform`, `fit_transform`, and `inverse_transform`.

`n_components` is an integer rank, at most `min(n_samples, n_features)`. The default start is non-negative double SVD (`init="nndsvda"`). `init="random"` uses `random_state`. Negative, NaN, or infinite entries are rejected. The comparison pipeline shifts each synthetic column so its minimum is zero before calling `nmf_reduce`, because those features are signed.

Run the example from the repository root with `src` on `PYTHONPATH`.

```python
import numpy as np
from nmf import NMF, nmf_reduce

rng = np.random.default_rng(0)
topics = rng.random((4, 12))
weights = rng.dirichlet(np.ones(4), size=50)
documents = weights @ topics

embedding, info = nmf_reduce(documents, n_components=4, random_state=0)
model = NMF(n_components=4, random_state=0)
codes = model.fit_transform(documents)
restored = model.inverse_transform(codes)

print(embedding.shape)  # (50, 4)
print(info["components"].shape)  # (4, 12)
print(float(embedding.min()), round(info["reconstruction_error"], 6))
print(restored.shape)  # (50, 12)
```

`codes @ model.components_` is the reconstruction. There is no mean to add.

## Evaluation

- Variance retention
- Reconstruction error
- Silhouette score on reduced embeddings
- Visual separation quality


## Classic t-SNE (NumPy)

The sklearn wrapper in `supervised_methods.tsne_reduce` is convenient for
quick plots. `src/tsne.py` adds the classic exact algorithm matching the
style of `FastICA` / `NMF` / `TruncatedSVD`: a `TSNE` class with
`fit` / `fit_transform`, binary-search perplexity affinities, early
exaggeration, and adaptive gains. No sklearn dependency.

```python
import numpy as np
from tsne import TSNE, tsne_reduce

rng = np.random.default_rng(0)
X = rng.normal(size=(120, 8))
embedding = TSNE(n_components=2, perplexity=20.0, learning_rate=200.0, n_iter=500, random_state=0).fit_transform(X)
print(embedding.shape)  # (120, 2)
```

## Isomap

`Isomap` in `src/isomap.py` builds a k-nearest-neighbour graph, approximates
geodesic distances with Floyd–Warshall, and embeds them with classical MDS
(Tenenbaum, de Silva & Langford, 2000). Use it when the data lie on a
nonlinear manifold that is locally Euclidean.

```python
import numpy as np
from isomap import Isomap, isomap_reduce

rng = np.random.default_rng(0)
X = rng.normal(size=(120, 8))
emb = Isomap(n_components=2, n_neighbors=10).fit_transform(X)
print(emb.shape)  # (120, 2)
```

`n_neighbors` controls the neighbourhood graph. If the raw kNN graph is
disconnected, shortest inter-component edges are bridged automatically.
`isomap_reduce` returns a pandas DataFrame with columns `ISO1`, `ISO2`, …

