"""
Gradio demo for the song recommendation system, with a media-dashboard-style
HTML result panel.

Run with:
    python apps/app_gradio.py
"""

import gradio as gr
import numpy as np

from src.data_loader import SongRecommendationDataGenerator
from src.model import ConvexRecommendationModel
from src.recommender import SongRecommender

np.random.seed(42)

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

DASHBOARD_CSS = """
<style>
    .media-app {
        font-family: 'Outfit', sans-serif;
        background-color: #0a0a0a;
        color: #ffffff;
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.6);
        position: relative;
        overflow: hidden;
    }
    .app-header {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 25px;
        background: linear-gradient(90deg, #1DB954, #FF0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 20px;
        margin-bottom: 40px;
    }
    .song-card {
        background: #181818;
        border-radius: 12px;
        padding: 16px;
        transition: all 0.3s ease;
        cursor: pointer;
        position: relative;
    }
    .song-card:hover {
        background: #282828;
        transform: translateY(-8px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.4);
    }
    .album-art {
        width: 100%;
        aspect-ratio: 1;
        border-radius: 8px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 40px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
        position: relative;
    }
    .play-overlay {
        position: absolute;
        bottom: 10px;
        right: 10px;
        width: 45px;
        height: 45px;
        background: #1DB954;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0;
        transform: translateY(10px);
        transition: all 0.3s ease;
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }
    .song-card:hover .play-overlay {
        opacity: 1;
        transform: translateY(0);
    }
    .song-title {
        font-weight: 700;
        font-size: 16px;
        margin: 0 0 6px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .song-score {
        font-size: 13px;
        color: #b3b3b3;
        margin: 0;
    }
    .yt-insight-panel {
        background: rgba(255, 255, 255, 0.05);
        border-left: 4px solid #FF0000;
        padding: 20px;
        border-radius: 0 12px 12px 0;
        margin-top: 20px;
    }
    .insight-title {
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .feature-tag {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin: 4px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .now-playing {
        margin-top: 30px;
        padding: 15px 25px;
        background: rgba(29, 185, 84, 0.1);
        border: 1px solid rgba(29, 185, 84, 0.3);
        border-radius: 50px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        backdrop-filter: blur(10px);
    }
    .error-msg {
        color: #ff4444;
        background: rgba(255,0,0,0.1);
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
    }
</style>
"""


def generate_immersive_dashboard(user_id: int):
    try:
        if not (0 <= user_id < data_gen.n_users):
            return "<div class='error-msg'>Please enter a valid User ID.</div>"

        recommendations_df = recommender.recommend_for_user(user_id, n_recommendations=5)
        if recommendations_df.empty:
            return "<div class='error-msg'>No recommendations found.</div>"

        html = f"""
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;500;700&display=swap" rel="stylesheet">
        {DASHBOARD_CSS}
        <div class="media-app">
            <div class="app-header">Made for User {user_id}</div>
            <div class="grid-container">
        """

        for _, row in recommendations_df.iterrows():
            seed = sum(ord(c) for c in row["song"])
            color1 = f"hsl({seed % 360}, 70%, 40%)"
            color2 = f"hsl({(seed + 40) % 360}, 80%, 20%)"

            html += f"""
                <div class="song-card">
                    <div class="album-art" style="background: linear-gradient(135deg, {color1}, {color2});">
                        &#127925;
                        <div class="play-overlay">&#9654;</div>
                    </div>
                    <h3 class="song-title">{row['song']}</h3>
                    <p class="song-score">Match Score: {(row['score'] * 100):.1f}%</p>
                </div>
            """

        html += "</div>"  # close grid

        top_song = recommendations_df.iloc[0]
        explanation = recommender.explain_recommendation(user_id, top_song["song_id"])

        html += f"""
            <div class="yt-insight-panel">
                <div class="insight-title">
                    <span style="font-size: 22px;">&#129504;</span> Behind the Algorithm: {top_song['song']}
                </div>
                <div style="display: flex; gap: 20px; margin-bottom: 15px; color: #b3b3b3; font-size: 14px;">
                    <div><strong>Collab Filter:</strong> {explanation['collaborative_score']:.3f}</div>
                    <div><strong>Content Map:</strong> {explanation['content_score']:.3f}</div>
                </div>
                <div>
                    <div style="font-size: 13px; color: #b3b3b3; margin-bottom: 8px;">Top Audio Signatures Detected:</div>
        """

        for feature_idx, contribution in explanation["top_features"][:4]:
            html += f'<span class="feature-tag">Feature {feature_idx} ({contribution:.2f})</span>'

        html += """
                </div>
            </div>

            <div class="now-playing">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <div style="width: 10px; height: 10px; background: #1DB954; border-radius: 50%; box-shadow: 0 0 10px #1DB954;"></div>
                    <span style="font-weight: 500;">System Ready - Awaiting Playback</span>
                </div>
                <div style="color: #b3b3b3; font-size: 14px;">HQ Audio</div>
            </div>
        </div>
        """
        return html

    except Exception as e:
        return f"<div class='error-msg'>System Error: {str(e)}</div>"


with gr.Blocks(theme=gr.themes.Base()) as app:
    gr.HTML("<style>.gradio-container { max-width: 1200px !important; }</style>")

    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### Control Center")
            user_input = gr.Number(label="User ID", value=0, precision=0)
            generate_btn = gr.Button("Generate Mix", variant="primary")

        with gr.Column(scale=4):
            output_dashboard = gr.HTML(
                value="<div style='text-align:center; padding: 50px; color:#666;'>"
                "Enter a User ID to generate your custom dashboard.</div>"
            )

    generate_btn.click(
        fn=generate_immersive_dashboard,
        inputs=[user_input],
        outputs=[output_dashboard],
    )

if __name__ == "__main__":
    app.launch()
