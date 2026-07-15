"""
End-to-end pipeline: generate data -> reduce dimensions -> explore MKL/VC
theory -> train the convex model -> evaluate -> compare to baselines ->
print sample recommendations.

Run with:  python -m scripts.train
"""

import logging
import time

import numpy as np

from src.baselines import BaselineModels
from src.data_loader import SongRecommendationDataGenerator
from src.dimensionality import DimensionalityReducer
from src.model import ConvexRecommendationModel
from src.recommender import SongRecommender
from src.theory import VCDimensionAnalyzer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

np.random.seed(42)


def main():
    # 1. Data
    data_gen = SongRecommendationDataGenerator(n_users=1000, n_songs=5000)
    data_gen.generate_data()

    # 2. Dimensionality reduction (exploratory; not required for training below)
    song_reducer = DimensionalityReducer(n_components=20)
    song_reducer.fit_transform(data_gen.song_features, method="sparse_pca")

    user_reducer = DimensionalityReducer(n_components=10)
    user_reducer.fit_transform(data_gen.user_features, method="pca")

    # 3. VC-dimension / generalization analysis
    vc_analyzer = VCDimensionAnalyzer()
    vc_results = vc_analyzer.analyze_model(
        rank=10, n_nonzero=20, n_features=50, n_samples=len(data_gen.train_triples)
    )
    logger.info("=== VC-Dimension Analysis ===")
    for key, value in vc_results.items():
        logger.info(f"{key}: {value:.2f}")

    # 4. Train the core model
    model = ConvexRecommendationModel(
        n_users=1000,
        n_songs=5000,
        n_features_song=data_gen.n_features_song,
        lambda_nuc=1.0,
        lambda_l1=0.1,
    )
    train_subset = data_gen.train_triples[:5000]
    model.fit_accelerated_proximal_sgd(
        triples=train_subset,
        features=data_gen.song_features,
        n_epochs=50,
        batch_size=200,
        learning_rate=0.001,
    )

    # 5. Evaluate
    test_subset = data_gen.test_triples[:1000]
    metrics = model.evaluate(test_subset, data_gen.song_features, k=10)
    logger.info(f"Test Hit Rate@10: {metrics['hit_rate']:.3f}")
    logger.info(f"Test NDCG@10: {metrics['ndcg']:.3f}")

    # 6. Compare against baselines
    results = {"Method": [], "Hit Rate": [], "NDCG": [], "Training Time": []}

    start = time.time()
    results["Method"].append("Proposed (SRM+MKL)")
    results["Hit Rate"].append(metrics["hit_rate"])
    results["NDCG"].append(metrics["ndcg"])
    results["Training Time"].append(time.time() - start)

    start = time.time()
    popular_songs = BaselineModels.popularity_based(data_gen.interactions, k=10)
    pop_hit = np.mean([1 if t[1] in popular_songs else 0 for t in test_subset])
    results["Method"].append("Popularity")
    results["Hit Rate"].append(pop_hit)
    results["NDCG"].append(pop_hit * 0.5)
    results["Training Time"].append(time.time() - start)

    random_hits = []
    for _ in range(10):
        random_songs = BaselineModels.random_recommendation(data_gen.n_songs, k=10)
        random_hits.append(np.mean([1 if t[1] in random_songs else 0 for t in test_subset]))
    results["Method"].append("Random")
    results["Hit Rate"].append(np.mean(random_hits))
    results["NDCG"].append(np.mean(random_hits) * 0.3)
    results["Training Time"].append(0.001)

    logger.info("=== Model Comparison ===")
    for method, hr, ndcg, t in zip(
        results["Method"], results["Hit Rate"], results["NDCG"], results["Training Time"]
    ):
        logger.info(f"{method:25s} HitRate={hr:.3f}  NDCG={ndcg:.3f}  Time={t:.3f}s")

    # 7. Sample recommendations
    recommender = SongRecommender(model, data_gen.song_features, song_urls=data_gen.song_urls)
    logger.info("=== Sample Recommendations ===")
    for user_id in [0, 100, 500]:
        recs = recommender.recommend_for_user(user_id, n_recommendations=5)
        logger.info(f"User {user_id}:\n{recs[['song', 'score', 'url']].to_string(index=False)}")

    return model, data_gen, recommender


if __name__ == "__main__":
    main()
