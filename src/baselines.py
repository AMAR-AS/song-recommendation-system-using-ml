"""Simple baseline recommenders used to benchmark the proposed model."""

import numpy as np
from scipy.sparse.linalg import svds


class BaselineModels:
    """Popularity, random, and plain-SVD baselines."""

    @staticmethod
    def popularity_based(interactions, k=10):
        """Recommend the k most popular songs overall."""
        song_popularity = interactions.sum(axis=0)
        return np.argsort(song_popularity)[-k:][::-1]

    @staticmethod
    def random_recommendation(n_songs, k=10):
        """Recommend k random songs."""
        return np.random.choice(n_songs, k, replace=False)

    @staticmethod
    def svd_based(interactions, rank=10, k=10):
        """Standard (unregularized) truncated SVD matrix factorization."""
        U, s, Vt = svds(interactions, k=rank)
        return U @ np.diag(s) @ Vt
