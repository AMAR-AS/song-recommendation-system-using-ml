"""
Core model: a convex relaxation of the hybrid recommendation problem using
Nuclear Norm (low-rank, collaborative) + L1 (sparse, content-based)
regularization, trained with accelerated proximal SGD (Nesterov's method).
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ConvexRecommendationModel:
    """
    Implements the convex relaxation of the recommendation problem using
    Nuclear Norm (low-rank) + L1 (sparse) regularization.
    """

    def __init__(self, n_users, n_songs, n_features_song, rank_estimate=10, lambda_nuc=0.1, lambda_l1=0.01):
        self.n_users = n_users
        self.n_songs = n_songs
        self.rank_estimate = rank_estimate

        # Regularization parameters (Structural Risk Minimization)
        self.lambda_nuc = lambda_nuc  # Nuclear norm penalty
        self.lambda_l1 = lambda_l1  # L1 penalty for sparsity

        # Model parameters
        self.X = np.zeros((n_users, n_songs))  # Low-rank interaction matrix
        self.w = np.zeros(n_features_song)  # Sparse content-feature weights
        self.losses = []

    def bpr_loss(self, triples, X, w, features):
        """Bayesian Personalized Ranking loss with cross-entropy."""
        loss = 0
        for u, i, j in triples:
            score_diff = X[u, i] - X[u, j]
            if features is not None:
                score_diff += np.dot(w, features[i] - features[j])
            loss += np.log(1 + np.exp(-np.clip(score_diff, -50, 50)))

        return loss / len(triples)

    def bpr_gradient(self, triples, X, w, features, batch_size=100):
        """Stochastic gradient of the BPR loss over a mini-batch."""
        batch_idx = np.random.choice(len(triples), batch_size, replace=False)
        batch = triples[batch_idx]

        grad_X = np.zeros_like(X)
        grad_w = np.zeros_like(w)

        for u, i, j in batch:
            score_diff = X[u, i] - X[u, j]
            if features is not None:
                score_diff += np.dot(w, features[i] - features[j])

            sigmoid_val = 1 / (1 + np.exp(np.clip(score_diff, -50, 50)))
            grad_factor = (1 - sigmoid_val) / batch_size

            grad_X[u, i] -= grad_factor
            grad_X[u, j] += grad_factor

            if features is not None:
                grad_w += grad_factor * (features[i] - features[j])

        return grad_X, grad_w

    def singular_value_thresholding(self, X, tau):
        """Proximal operator for the nuclear norm (SVT)."""
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        s_thresholded = np.maximum(s - tau, 0)
        return U @ np.diag(s_thresholded) @ Vt

    def soft_thresholding(self, w, tau):
        """Proximal operator for the L1 norm."""
        return np.sign(w) * np.maximum(np.abs(w) - tau, 0)

    def fit_accelerated_proximal_sgd(
        self, triples, features, n_epochs=50, batch_size=100, learning_rate=0.01, momentum=0.9
    ):
        """
        Accelerated Proximal Stochastic Gradient Descent.
        Uses Nesterov's acceleration for O(1/k^2) convergence.
        """
        X_k = np.zeros((self.n_users, self.n_songs))
        w_k = np.zeros(features.shape[1] if features is not None else 1)

        X_prev = X_k.copy()
        w_prev = w_k.copy()
        t_k = 1

        losses = []

        for epoch in range(n_epochs):
            grad_X, grad_w = self.bpr_gradient(triples, X_k, w_k, features, batch_size)

            # Gradient step
            X_intermediate = X_k - learning_rate * grad_X
            w_intermediate = w_k - learning_rate * grad_w

            # Proximal step (composite optimization)
            X_new = self.singular_value_thresholding(X_intermediate, learning_rate * self.lambda_nuc)
            w_new = self.soft_thresholding(w_intermediate, learning_rate * self.lambda_l1)

            # Nesterov acceleration
            t_new = (1 + np.sqrt(1 + 4 * t_k**2)) / 2
            X_k = X_new + ((t_k - 1) / t_new) * (X_new - X_prev)
            w_k = w_new + ((t_k - 1) / t_new) * (w_new - w_prev)

            X_prev = X_new
            w_prev = w_new
            t_k = t_new

            loss = self.bpr_loss(triples, X_k, w_k, features)
            losses.append(loss)

            if epoch % 10 == 0:
                nuc_norm = np.linalg.norm(X_k, "nuc")
                sparsity = np.sum(np.abs(w_k) > 1e-5) if features is not None else 0
                logger.info(
                    f"Epoch {epoch}: Loss={loss:.4f}, Nuclear Norm={nuc_norm:.2f}, "
                    f"Sparsity={sparsity}/{len(w_k)}"
                )

        self.X = X_k
        self.w = w_k
        self.losses = losses

        return self

    def predict(self, user_ids, item_ids=None):
        """Generate raw scores for the given users (optionally restricted to items)."""
        if item_ids is None:
            return self.X[user_ids, :]
        return self.X[user_ids, :][:, item_ids]

    def evaluate(self, test_triples, features, k=10):
        """Evaluate using Hit Rate@k and NDCG@k."""
        hits = 0
        ndcg = 0

        user_items = {}
        for u, i, j in test_triples:
            user_items.setdefault(u, set()).add(i)

        n_users = len(user_items)

        for user, true_items in user_items.items():
            scores = self.X[user] + (features @ self.w if features is not None else 0)
            top_k = np.argsort(scores)[-k:][::-1]

            hits += len(set(top_k) & true_items) > 0

            dcg = sum(1 / np.log2(rank + 2) for rank, item in enumerate(top_k) if item in true_items)
            idcg = sum(1 / np.log2(rank + 2) for rank in range(min(k, len(true_items))))
            ndcg += dcg / idcg if idcg > 0 else 0

        return {
            "hit_rate": hits / n_users,
            "ndcg": ndcg / n_users,
        }
