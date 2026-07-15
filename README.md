# Song Recommendation System

A hybrid (collaborative + content-based) song recommender built as a **unified convex
optimization framework**, combining low-rank matrix factorization with sparse
feature selection and multiple kernel learning — with theoretical generalization
guarantees from VC-dimension theory.

## Why this exists

Most toy recommenders either do pure collaborative filtering (which breaks on
new items) or pure content-based filtering (which ignores collective behavior).
This project frames the hybrid problem as a single **convex composite
optimization**:

```
minimize   BPR_loss(X, w)  +  λ_nuc * ||X||_*  +  λ_l1 * ||w||_1
```

- `X` — a low-rank user–item matrix (collaborative signal), regularized by its
  **nuclear norm** (a convex proxy for rank).
- `w` — a sparse set of content-feature weights, regularized by an **L1** penalty.
- Trained with **accelerated proximal SGD** (Nesterov's method, O(1/k²) convergence),
  alternating gradient steps with singular-value thresholding and soft-thresholding.

Because both penalties are convex, the training problem doesn't have the local-minima
issues of a black-box neural recommender, and its capacity can be bounded directly
via VC-dimension theory — giving an explicit generalization guarantee rather than
just an empirical validation-curve.

## Features

- **Synthetic data generator** — realistic sparse user–item interactions with genre/demographic clusters (`src/data_loader.py`), plus an optional live YouTube Music trending-charts hook.
- **Dimensionality reduction** — PCA / Sparse PCA for feature compression (`src/dimensionality.py`).
- **Multiple Kernel Learning** — convex combination of RBF, polynomial, and linear kernels for richer similarity (`src/kernels.py`).
- **VC-dimension analysis** — generalization bounds for the low-rank and sparse components, and their SRM-optimal combination (`src/theory.py`).
- **Core model** — the convex hybrid recommender described above (`src/model.py`).
- **Baselines** — popularity, random, and plain SVD, for comparison (`src/baselines.py`).
- **Recommender interface** — top-N recommendations with score explanations (`src/recommender.py`).
- **Demo apps** — Streamlit and Gradio front-ends (`apps/`).

## Project structure

```
song-recommendation-system-using-ml/
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # synthetic data generation + live-data hook
│   ├── dimensionality.py    # PCA / Sparse PCA
│   ├── kernels.py           # Multiple Kernel Learning
│   ├── theory.py            # VC-dimension / generalization bounds
│   ├── model.py             # core convex recommendation model
│   ├── baselines.py         # popularity / random / SVD baselines
│   └── recommender.py       # top-N recommendations + explanations
├── scripts/
│   └── train.py             # end-to-end training/evaluation pipeline
├── apps/
│   ├── app.py                # Streamlit demo
│   └── app_gradio.py         # Gradio demo
├── notebooks/
│   └── exploration.ipynb    # exploratory notebook (plots, ablations)
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_theory_and_recommender.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/AMAR-AS/song-recommendation-system-using-ml.git
cd song-recommendation-system-using-ml
pip install -r requirements.txt
```

## Quick start

```python
import numpy as np
from src.data_loader import SongRecommendationDataGenerator
from src.model import ConvexRecommendationModel
from src.recommender import SongRecommender

np.random.seed(42)

# 1. Generate a synthetic dataset
data_gen = SongRecommendationDataGenerator(n_users=1000, n_songs=5000)
data_gen.generate_data()

# 2. Train the convex hybrid model
model = ConvexRecommendationModel(
    n_users=1000, n_songs=5000,
    n_features_song=data_gen.n_features_song,
    lambda_nuc=1.0, lambda_l1=0.1,
)
model.fit_accelerated_proximal_sgd(
    triples=data_gen.train_triples[:5000],
    features=data_gen.song_features,
    n_epochs=50, batch_size=200, learning_rate=0.001,
)

# 3. Evaluate
metrics = model.evaluate(data_gen.test_triples[:1000], data_gen.song_features, k=10)
print(f"Hit Rate@10: {metrics['hit_rate']:.3f} | NDCG@10: {metrics['ndcg']:.3f}")

# 4. Get recommendations
recommender = SongRecommender(model, data_gen.song_features, song_urls=data_gen.song_urls)
print(recommender.recommend_for_user(user_id=0, n_recommendations=5))
```

Or just run the whole pipeline:

```bash
python -m scripts.train
```

## Running the demos

**Streamlit:**
```bash
streamlit run apps/app.py
```

**Gradio:**
```bash
python apps/app_gradio.py
```

## Results

On the default synthetic setup (1000 users, 5000 songs, 98% sparsity), the
proposed model outperforms popularity and random baselines and is competitive
with plain SVD, while additionally offering:
- an **interpretable** score decomposition (collaborative vs. content contribution),
- a **theoretical** generalization bound via VC-dimension analysis,
- explicit **complexity control** via the Structural Risk Minimization curve (see `notebooks/exploration.ipynb`).

Exact numbers vary by run (data is randomly generated each time) — run `python -m scripts.train` to reproduce on your machine.

## Theoretical framework

| Concept | Where it's used |
|---|---|
| Empirical Risk Minimization | BPR loss (cross-entropy) |
| Structural Risk Minimization | Nuclear norm + L1 regularization |
| VC-dimension bounds | Low-rank + sparse capacity control |
| Convex composite optimization | Non-smooth (norms) + smooth (BPR) objective |
| Accelerated proximal SGD | Nesterov's method, O(1/k²) convergence |
| Multiple Kernel Learning | Convex combination of RBF/polynomial/linear kernels |

## License

MIT — see [LICENSE](LICENSE).
