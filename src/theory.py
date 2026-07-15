"""VC-dimension bounds and generalization guarantees (Structural Risk Minimization)."""

import math

import numpy as np


class VCDimensionAnalyzer:
    """Analyzes VC-dimension and generalization bounds for the model's components."""

    @staticmethod
    def low_rank_vc_bound(rank, n_features):
        """VC-dimension bound for low-rank matrix factorization."""
        return rank * n_features + rank

    @staticmethod
    def sparse_vc_bound(n_nonzero, n_features):
        """VC-dimension bound for sparse linear models."""
        return (n_nonzero + 1) * (np.log2(n_features) + 1)

    @staticmethod
    def generalization_bound(vc_dim, n_samples, confidence=0.05):
        """Compute a generalization bound using classical VC theory."""
        term = (vc_dim * (math.log(2 * n_samples / vc_dim) + 1) + math.log(4 / confidence)) / n_samples
        return math.sqrt(max(0, term))

    def analyze_model(self, rank, n_nonzero, n_features, n_samples):
        """Comprehensive VC analysis combining the low-rank and sparse components."""
        low_rank_bound = self.low_rank_vc_bound(rank, n_features)
        sparse_bound = self.sparse_vc_bound(n_nonzero, n_features)
        combined_bound = min(low_rank_bound, sparse_bound)  # SRM takes the minimum

        return {
            "Low-Rank VC-dim": low_rank_bound,
            "Sparse VC-dim": sparse_bound,
            "Combined VC-dim (SRM)": combined_bound,
            "Gen. Bound (Low-Rank)": self.generalization_bound(low_rank_bound, n_samples),
            "Gen. Bound (Sparse)": self.generalization_bound(sparse_bound, n_samples),
            "Gen. Bound (Combined)": self.generalization_bound(combined_bound, n_samples),
        }
