"""Dimensionality reduction utilities (PCA / Sparse PCA) for feature extraction."""

import logging

import numpy as np
from sklearn.decomposition import PCA, SparsePCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """Implements PCA and Sparse PCA dimensionality reduction with feature scaling."""

    def __init__(self, n_components=20):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.sparse_pca = SparsePCA(n_components=n_components, alpha=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.components_ = None

    def fit_transform(self, X, method="sparse_pca"):
        X_scaled = self.scaler.fit_transform(X)

        if method == "pca":
            self.components_ = self.pca.fit_transform(X_scaled)
        elif method == "sparse_pca":
            self.components_ = self.sparse_pca.fit_transform(X_scaled)
        else:
            raise ValueError(f"Unknown method: {method}")

        explained_var = np.var(self.components_, axis=0).sum() / np.var(X_scaled, axis=0).sum()
        logger.info(f"Explained variance ratio ({method}): {explained_var:.3f}")

        return self.components_
