import numpy as np

from src.data_loader import SongRecommendationDataGenerator


def test_data_generator_shapes():
    """Test that the generated interaction matrix has the correct dimensions."""
    gen = SongRecommendationDataGenerator(n_users=100, n_songs=500)
    gen.generate_data()

    assert gen.interactions.shape == (100, 500), "Interaction matrix shape mismatch."
    assert gen.song_features.shape == (500, 50), "Song features shape mismatch."
    assert gen.user_features.shape == (100, 20), "User features shape mismatch."


def test_data_sparsity():
    """Test that the interaction matrix mimics real-world sparsity (> 95%)."""
    gen = SongRecommendationDataGenerator(n_users=200, n_songs=1000)
    gen.generate_data()

    total_elements = 200 * 1000
    non_zero_elements = np.sum(gen.interactions)
    sparsity = 1.0 - (non_zero_elements / total_elements)

    assert sparsity > 0.95, f"Matrix is too dense. Sparsity is only {sparsity * 100:.2f}%"


def test_triples_generation():
    """Ensure training and testing triples are correctly generated."""
    gen = SongRecommendationDataGenerator(n_users=50, n_songs=200)
    gen.generate_data()

    assert len(gen.train_triples) > 0, "No training triples generated."
    assert len(gen.test_triples) > 0, "No test triples generated."
    assert gen.train_triples.shape[1] == 3, "Triples should have 3 columns."


def test_song_urls_generated():
    """Ensure demo YouTube URLs are attached to every song slot."""
    gen = SongRecommendationDataGenerator(n_users=50, n_songs=200)
    gen.generate_data()

    assert len(gen.song_urls) == 200
    assert all(url.startswith("https://www.youtube.com/") for url in gen.song_urls)
