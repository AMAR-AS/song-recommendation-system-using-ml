import numpy as np
from ytmusicapi import YTMusic
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SongRecommendationDataGenerator:
    """Generates synthetic data mimicking real song recommendation scenarios."""

    def __init__(self, n_users=1000, n_songs=5000, n_features_song=50, n_features_user=20):
        self.n_users = n_users
        self.n_songs = n_songs
        self.n_features_song = n_features_song
        self.n_features_user = n_features_user
        
        self.U_latent = None
        self.V_latent = None
        self.song_features = None
        self.user_features = None
        self.interactions = None
        self.train_triples = None
        self.test_triples = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def generate_data(self):
        """Generates the latent factors, features, and the highly sparse interaction matrix."""
        # Generate latent factors
        self.U_latent = np.random.randn(self.n_users, 10)
        self.V_latent = np.random.randn(self.n_songs, 10)

        # Song features (audio characteristics with genre clusters)
        self.song_features = np.random.randn(self.n_songs, self.n_features_song)
        genres = np.random.choice([0, 1, 2, 3, 4], self.n_songs)
        for i in range(5):
            mask = genres == i
            self.song_features[mask] += np.random.randn(sum(mask), self.n_features_song) * 0.5 + i

        # User features (demographics with age group clusters)
        self.user_features = np.random.randn(self.n_users, self.n_features_user)
        age_groups = np.random.choice([0, 1, 2], self.n_users)
        for i in range(3):
            mask = age_groups == i
            self.user_features[mask] += np.random.randn(sum(mask), self.n_features_user) * 0.5 + i

        # Generate interaction matrix with latent structure
        interaction_prob = self._sigmoid(self.U_latent @ self.V_latent.T)
        self.interactions = (np.random.random((self.n_users, self.n_songs)) < interaction_prob).astype(float)

        # Apply 98-99% sparsity to mimic a realistic real-world matrix
        sparsity_mask = np.random.random((self.n_users, self.n_songs)) < 0.98
        self.interactions[sparsity_mask] = 0

        logger.info(f"Dataset Generated: {self.n_users} Users, {self.n_songs} Songs.")
        logger.info(f"Matrix Sparsity: {(1 - np.sum(self.interactions)/(self.n_users*self.n_songs))*100:.2f}%")
        
        self._generate_triplets()
        return self

    def _generate_triplets(self):
        """Creates training and testing triples (user, positive_item, negative_item)."""
        train_triples, test_triples = [], []

        for u in range(self.n_users):
            pos_items = np.where(self.interactions[u] == 1)[0]
            neg_items = np.where(self.interactions[u] == 0)[0]

            if len(pos_items) == 0:
                continue

            np.random.shuffle(pos_items)
            n_test = max(1, int(len(pos_items) * 0.2)) 
            train_pos = pos_items[:-n_test]
            test_pos = pos_items[-n_test:]

            for i in train_pos:
                j = np.random.choice(neg_items, min(3, len(neg_items)), replace=False)
                for neg_item in j:
                    train_triples.append((u, i, neg_item))

            for i in test_pos:
                j = np.random.choice(neg_items, min(1, len(neg_items)), replace=False)
                for neg_item in j:
                    test_triples.append((u, i, neg_item))

        self.train_triples = np.array(train_triples)
        self.test_triples = np.array(test_triples)
        logger.info(f"Training triples: {len(self.train_triples)} | Test triples: {len(self.test_triples)}")


def fetch_live_trending_data(country_code='IN'):
    """Attempts to fetch live trending data, fails gracefully if API key/structure changes."""
    try:
        ytmusic = YTMusic()
        trending_data = ytmusic.get_charts(country=country_code)
        # Using .get() prevents KeyError if 'trending' is unexpectedly missing
        trending_items = trending_data.get('trending', {}).get('items', [])
        
        if not trending_items:
            logger.warning("No trending items found. Falling back to synthetic data.")
            return None
            
        logger.info(f"Successfully fetched {len(trending_items)} trending songs.")
        return trending_items
    except Exception as e:
        logger.error(f"Error connecting to YTMusic API: {e}. Falling back to synthetic data.")
        return None
