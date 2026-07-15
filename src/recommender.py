"""User-facing recommendation interface, with score explanations."""

import numpy as np
import pandas as pd


class SongRecommender:
    """Final recommendation system interface: top-N recommendations + explanations."""

    def __init__(self, model, song_features, song_names=None, song_urls=None):
        self.model = model
        self.song_features = song_features

        self.song_names = (
            song_names if song_names is not None else [f"Song_{i}" for i in range(len(song_features))]
        )
        self.song_urls = song_urls if song_urls is not None else ["" for _ in range(len(song_features))]

    def recommend_for_user(self, user_id, n_recommendations=10, exclude_listened=True):
        """Generate personalized top-N recommendations for a user."""
        scores = self.model.X[user_id] + (self.song_features @ self.model.w)

        if exclude_listened and hasattr(self, "interactions"):
            listened = np.where(self.interactions[user_id] == 1)[0]
            scores[listened] = -np.inf

        top_indices = np.argsort(scores)[-n_recommendations:][::-1]
        top_scores = scores[top_indices]

        recommendations = [
            {
                "song": self.song_names[idx],
                "score": score,
                "song_id": idx,
                "url": self.song_urls[idx],
            }
            for idx, score in zip(top_indices, top_scores)
        ]

        return pd.DataFrame(recommendations)

    def explain_recommendation(self, user_id, song_id):
        """Decompose a recommendation into its collaborative and content components."""
        collab_score = self.model.X[user_id, song_id]
        content_score = np.dot(self.model.w, self.song_features[song_id])

        feature_contributions = self.model.w * self.song_features[song_id]
        top_features = np.argsort(np.abs(feature_contributions))[-5:][::-1]

        return {
            "collaborative_score": collab_score,
            "content_score": content_score,
            "total_score": collab_score + content_score,
            "top_features": [(i, feature_contributions[i]) for i in top_features],
        }
