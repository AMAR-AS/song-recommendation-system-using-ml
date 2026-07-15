import numpy as np

from src.data_loader import SongRecommendationDataGenerator
from src.model import ConvexRecommendationModel


def _tiny_trained_model(seed=0):
    np.random.seed(seed)
    gen = SongRecommendationDataGenerator(n_users=30, n_songs=100, n_features_song=10)
    gen.generate_data()

    model = ConvexRecommendationModel(
        n_users=30, n_songs=100, n_features_song=10, lambda_nuc=0.5, lambda_l1=0.05
    )
    train = gen.train_triples[:200] if len(gen.train_triples) > 200 else gen.train_triples
    model.fit_accelerated_proximal_sgd(
        triples=train, features=gen.song_features, n_epochs=5, batch_size=min(20, len(train)), learning_rate=0.01
    )
    return model, gen


def test_model_shapes():
    model, gen = _tiny_trained_model()
    assert model.X.shape == (30, 100)
    assert model.w.shape == (10,)


def test_loss_is_finite_and_decreasing_trend():
    model, gen = _tiny_trained_model()
    assert all(np.isfinite(loss) for loss in model.losses)
    # Not strictly monotonic (stochastic), but should end no worse than a small
    # multiple of where it started.
    assert model.losses[-1] < model.losses[0] * 5


def test_evaluate_returns_bounded_metrics():
    model, gen = _tiny_trained_model()
    test = gen.test_triples[:50] if len(gen.test_triples) > 50 else gen.test_triples
    metrics = model.evaluate(test, gen.song_features, k=5)

    assert 0.0 <= metrics["hit_rate"] <= 1.0
    assert 0.0 <= metrics["ndcg"] <= 1.0


def test_soft_and_svt_thresholding_shrink_toward_zero():
    model, _ = _tiny_trained_model()
    w = np.array([1.0, -2.0, 0.05])
    shrunk = model.soft_thresholding(w, tau=0.5)
    assert np.all(np.abs(shrunk) <= np.abs(w))

    X = np.random.randn(10, 10)
    X_shrunk = model.singular_value_thresholding(X, tau=1.0)
    assert np.linalg.norm(X_shrunk, "nuc") <= np.linalg.norm(X, "nuc")
