"""
Streamlit demo for the song recommendation system.

Run with:
    streamlit run apps/app.py
"""

import numpy as np
import streamlit as st

from src.data_loader import SongRecommendationDataGenerator
from src.model import ConvexRecommendationModel
from src.recommender import SongRecommender

np.random.seed(42)


@st.cache_resource
def load_trained_system():
    """Generate data and train a model once, cached across reruns."""
    data_gen = SongRecommendationDataGenerator(n_users=1000, n_songs=5000)
    data_gen.generate_data()

    model = ConvexRecommendationModel(
        n_users=1000,
        n_songs=5000,
        n_features_song=data_gen.n_features_song,
        lambda_nuc=1.0,
        lambda_l1=0.1,
    )
    model.fit_accelerated_proximal_sgd(
        triples=data_gen.train_triples[:5000],
        features=data_gen.song_features,
        n_epochs=50,
        batch_size=200,
        learning_rate=0.001,
    )

    recommender = SongRecommender(model, data_gen.song_features, song_urls=data_gen.song_urls)
    return data_gen, recommender


data_gen, recommender = load_trained_system()

st.title("Song Recommendation System")
st.write("Get personalized song recommendations based on your user ID.")

user_id_input = st.number_input(
    "Enter User ID (0 to {}):".format(data_gen.n_users - 1),
    min_value=0,
    max_value=data_gen.n_users - 1,
    value=0,
)

if st.button("Get Recommendations"):
    st.subheader(f"Recommendations for User {user_id_input}")
    recommendations = recommender.recommend_for_user(user_id_input, n_recommendations=10)
    st.dataframe(recommendations[["song", "score", "url"]])

    st.subheader("Explanation for Top Recommendation")
    if not recommendations.empty:
        top_rec_song_id = recommendations.iloc[0]["song_id"]
        explanation = recommender.explain_recommendation(user_id_input, top_rec_song_id)
        st.write(f"**Song: {recommendations.iloc[0]['song']}**")
        st.write(f"- Collaborative Score: {explanation['collaborative_score']:.3f}")
        st.write(f"- Content Score: {explanation['content_score']:.3f}")
        st.write(f"- Total Score: {explanation['total_score']:.3f}")
        st.write("**Top Contributing Features (Index, Value):**")
        for feature_idx, contribution in explanation["top_features"][:5]:
            st.write(f"  - Feature {feature_idx}: {contribution:.3f}")
    else:
        st.write("No recommendations available for explanation.")
