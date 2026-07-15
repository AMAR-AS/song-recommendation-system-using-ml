"""
song-recommendation-system-using-ml
------------------------------------
A hybrid (collaborative + content-based) song recommender built on a
convex composite optimization framework: Nuclear Norm (low-rank) +
L1 (sparse) regularization, trained with accelerated proximal SGD.
"""

from .data_loader import SongRecommendationDataGenerator, fetch_live_trending_data
from .dimensionality import DimensionalityReducer
from .kernels import MultipleKernelLearning
from .theory import VCDimensionAnalyzer
from .model import ConvexRecommendationModel
from .baselines import BaselineModels
from .recommender import SongRecommender

__all__ = [
    "SongRecommendationDataGenerator",
    "fetch_live_trending_data",
    "DimensionalityReducer",
    "MultipleKernelLearning",
    "VCDimensionAnalyzer",
    "ConvexRecommendationModel",
    "BaselineModels",
    "SongRecommender",
]

__version__ = "0.1.0"
