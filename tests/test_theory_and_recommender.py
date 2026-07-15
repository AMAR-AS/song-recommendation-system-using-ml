import numpy as np

from src.kernels import MultipleKernelLearning
from src.model import ConvexRecommendationModel
from src.recommender import SongRecommender
from src.theory import VCDimensionAnalyzer


def test_vc_bounds_are_positive():
    analyzer = VCDimensionAnalyzer()
    results = analyzer.analyze_model(rank=10, n_nonzero=20, n_features=50, n_samples=5000)

    for key, value in results.items():
        assert value >= 0, f"{key} should be non-negative, got {value}"


def test_combined_bound_is_srm_minimum():
    analyzer = VCDimensionAnalyzer()
    low_rank = analyzer.low_rank_vc_bound(rank=10, n_features=50)
    sparse = analyzer.sparse_vc_bound(n_nonzero=20, n_features=50)
    results = analyzer.analyze_model(rank=10, n_nonzero=20, n_features=50, n_samples=5000)

    assert results["Combined VC-dim (SRM)"] == min(low_rank, sparse)


def test_mkl_combined_kernel_shape_and_weights_sum_to_one():
    X = np.random.randn(20, 5)
    mkl = MultipleKernelLearning(kernel_types=["rbf", "linear"])
    combined = mkl.compute_combined_kernel(X)

    assert combined.shape == (20, 20)
    assert np.isclose(mkl.kernel_weights.sum(), 1.0)


def test_recommender_returns_top_n_with_no_duplicates():
    n_songs = 50
    model = ConvexRecommendationModel(n_users=5, n_songs=n_songs, n_features_song=8)
    model.X = np.random.randn(5, n_songs)
    model.w = np.random.randn(8)

    song_features = np.random.randn(n_songs, 8)
    recommender = SongRecommender(model, song_features)

    recs = recommender.recommend_for_user(user_id=0, n_recommendations=10, exclude_listened=False)

    assert len(recs) == 10
    assert recs["song_id"].nunique() == 10


def test_explain_recommendation_scores_sum_correctly():
    n_songs = 20
    model = ConvexRecommendationModel(n_users=3, n_songs=n_songs, n_features_song=6)
    model.X = np.random.randn(3, n_songs)
    model.w = np.random.randn(6)

    song_features = np.random.randn(n_songs, 6)
    recommender = SongRecommender(model, song_features)

    explanation = recommender.explain_recommendation(user_id=0, song_id=3)
    assert np.isclose(
        explanation["total_score"],
        explanation["collaborative_score"] + explanation["content_score"],
    )
