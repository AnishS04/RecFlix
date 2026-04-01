import sys
sys.path.append("src")
import os
import torch
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

from data_preprocessing import preprocess
from matrix_factorization import MFModel
from recommender import build_popularity_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RecFlix",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --red: #E50914;
    --dark: #0d0d0d;
    --card: #1a1a1a;
    --border: #2a2a2a;
    --text: #e0e0e0;
    --muted: #888;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--dark);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background-color: #111; border-right: 1px solid var(--border); }

h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem !important;
    letter-spacing: 4px;
    color: var(--red) !important;
    margin-bottom: 0 !important;
}

.subtitle {
    color: var(--muted);
    font-size: 0.9rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.rec-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color 0.2s;
}

.rec-card:hover { border-color: var(--red); }

.rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    color: var(--red);
    min-width: 2rem;
    text-align: center;
}

.movie-title {
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text);
}

.movie-score {
    margin-left: auto;
    font-size: 0.8rem;
    color: var(--muted);
    white-space: nowrap;
}

.score-bar-bg {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 4px;
    width: 100%;
}

.score-bar-fill {
    height: 3px;
    background: var(--red);
    border-radius: 2px;
}

.stat-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.stat-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2rem;
    color: var(--red);
}

.stat-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.stButton > button {
    background-color: var(--red) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 2rem !important;
    width: 100%;
}

.stButton > button:hover {
    background-color: #ff0a16 !important;
}

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background-color: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
}

.stNumberInput > div > div > input {
    background-color: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

.stSlider > div { color: var(--text) !important; }
.stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
.stTabs [aria-selected="true"] { color: var(--red) !important; border-bottom-color: var(--red) !important; }

div[data-testid="stMarkdownContainer"] p { color: var(--text); }

.section-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 0.5rem;
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Data & model loading ───────────────────────────────────────────────────────
MODEL_PATH = "mf_model.pt"
REPO_ID = "AnishS04/recflix-mf-model"

@st.cache_resource(show_spinner="Loading data...")
def load_data():
    train_df, test_df = preprocess()
    movies_df = pd.read_csv("data/ml-latest/movies.csv", usecols=["movieId", "title"])

    user_map = {u: i for i, u in enumerate(train_df["userId"].unique())}
    movie_map = {m: i for i, m in enumerate(train_df["movieId"].unique())}

    train_df["user_idx"] = train_df["userId"].map(user_map)
    train_df["movie_idx"] = train_df["movieId"].map(movie_map)

    pop_model = build_popularity_model(train_df, min_ratings=100)

    return train_df, movies_df, user_map, movie_map, pop_model


@st.cache_resource(show_spinner="Loading model...")
def load_model(n_users, n_movies):
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Hugging Face..."):
            hf_hub_download(repo_id=REPO_ID, filename="mf_model.pt", local_dir=".")
    model = MFModel(n_users=n_users, n_movies=n_movies, k=50)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def get_mf_recommendations(model, user_id, user_map, movie_map, train_df, movies_df, top_n=10):
    if user_id not in user_map:
        return None
    user_idx = user_map[user_id]
    seen = set(train_df[train_df["userId"] == user_id]["movieId"])
    candidates = [m for m in movie_map.keys() if m not in seen]

    movie_indices = torch.tensor([movie_map[m] for m in candidates])
    user_tensor = torch.tensor([user_idx] * len(movie_indices))

    with torch.no_grad():
        scores = model(user_tensor, movie_indices).numpy()

    recs = pd.DataFrame({"movieId": candidates, "score": scores})
    recs = recs.sort_values("score", ascending=False).head(top_n)
    recs = recs.merge(movies_df[["movieId", "title"]], on="movieId")
    return recs[["title", "score"]].reset_index(drop=True)


def get_similar_movies(movie_title, model, movie_map, movies_df, top_n=10):
    match = movies_df[movies_df["title"].str.contains(movie_title, case=False, na=False)]
    if match.empty:
        return None, None

    movie_id = match.iloc[0]["movieId"]
    found_title = match.iloc[0]["title"]

    if movie_id not in movie_map:
        return None, found_title

    movie_idx = movie_map[movie_id]
    movie_vec = model.movie_emb.weight[movie_idx].detach()

    all_vecs = model.movie_emb.weight.detach()
    sims = torch.nn.functional.cosine_similarity(movie_vec.unsqueeze(0), all_vecs)
    top_indices = sims.argsort(descending=True)[1:top_n + 1]

    inv_movie_map = {v: k for k, v in movie_map.items()}
    top_movie_ids = [inv_movie_map[i.item()] for i in top_indices]
    top_scores = [sims[i].item() for i in top_indices]

    recs = pd.DataFrame({"movieId": top_movie_ids, "score": top_scores})
    recs = recs.merge(movies_df[["movieId", "title"]], on="movieId")
    return recs[["title", "score"]].reset_index(drop=True), found_title


def render_recommendations(recs):
    for i, row in recs.iterrows():
        score_pct = int(((row["score"] - 0.5) / 4.5) * 100)
        st.markdown(f"""
        <div class="rec-card">
            <div class="rank">{i + 1}</div>
            <div style="flex: 1">
                <div class="movie-title">{row['title']}</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width: {score_pct}%"></div>
                </div>
            </div>
            <div class="movie-score">{f'{row["score"]:.2f}' if row["score"] <= 5 else f'{row["score"]:.2f}'}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Main app ───────────────────────────────────────────────────────────────────
st.markdown('<h1>RECFLIX</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Personalized Movie Recommendations · 32M Ratings · 86K Movies</div>', unsafe_allow_html=True)

train_df, movies_df, user_map, movie_map, pop_model = load_data()
model = load_model(n_users=len(user_map), n_movies=len(movie_map))

# ── Stats row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-box"><div class="stat-number">32M</div><div class="stat-label">Ratings</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-box"><div class="stat-number">{len(user_map):,}</div><div class="stat-label">Users</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-box"><div class="stat-number">{len(movie_map):,}</div><div class="stat-label">Movies</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-box"><div class="stat-number">0.85</div><div class="stat-label">RMSE</div></div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Recommend by User ID", "Find Similar Movies"])

with tab1:
    st.markdown('<div class="section-label">Enter a User ID to get personalized recommendations</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        user_id = st.number_input("User ID", min_value=1, max_value=max(user_map.keys()), value=1, step=1, label_visibility="collapsed")
    with col2:
        top_n = st.slider("Results", 5, 20, 10, label_visibility="collapsed")

    if st.button("Get Recommendations", key="user_btn"):
        recs = get_mf_recommendations(model, user_id, user_map, movie_map, train_df, movies_df, top_n)
        if recs is not None:
            st.markdown(f'<div class="section-label" style="margin-top:1rem">Top {top_n} picks for User {user_id}</div>', unsafe_allow_html=True)
            render_recommendations(recs)
        else:
            st.warning("User ID not found in training data. Try a different ID.")

with tab2:
    st.markdown('<div class="section-label">Search by a movie you liked to find similar titles</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        movie_query = st.text_input("Movie title", placeholder="e.g. The Dark Knight", label_visibility="collapsed")
    with col2:
        top_n_movie = st.slider("Results", 5, 20, 10, key="movie_slider", label_visibility="collapsed")

    if st.button("Find Similar Movies", key="movie_btn"):
        if movie_query:
            recs, found_title = get_similar_movies(movie_query, model, movie_map, movies_df, top_n_movie)
            if recs is not None:
                st.markdown(f'<div class="section-label" style="margin-top:1rem">Movies similar to "{found_title}"</div>', unsafe_allow_html=True)
                render_recommendations(recs)
            else:
                st.warning(f'Could not find "{movie_query}" in the dataset. Try a different title.')
        else:
            st.warning("Please enter a movie title.")