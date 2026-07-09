import os
import sys
import torch
import pandas as pd
import streamlit as st

# Allow importing from the src/ folder when running from project root
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_preprocessing import preprocess
from matrix_factorization import MFModel


# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="RecFlix", page_icon="🎬", layout="wide")

MODEL_PATH = "mf_model.pt"
MOVIES_PATH = "data/ml-latest/movies.csv"
HF_REPO_ID = "AnishS04/recflix-mf-model"


# ── Styling ────────────────────────────────────────────────────────────────────
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

h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem !important;
    letter-spacing: 4px;
    color: var(--red) !important;
    margin-bottom: 0 !important;
}

.subtitle {
    color: var(--muted);
    font-size: 0.85rem;
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
}

.rec-card:hover { border-color: var(--red); }

.rank {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    color: var(--red);
    min-width: 2rem;
    text-align: center;
}

.movie-title { font-size: 0.95rem; font-weight: 500; color: var(--text); }
.movie-score { margin-left: auto; font-size: 0.85rem; color: var(--muted); white-space: nowrap; }

.score-bar-bg {
    height: 3px; background: var(--border); border-radius: 2px;
    margin-top: 5px; width: 100%;
}
.score-bar-fill { height: 3px; background: var(--red); border-radius: 2px; }

.stat-box {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem; text-align: center;
}
.stat-number { font-family: 'Bebas Neue', sans-serif; font-size: 2rem; color: var(--red); }
.stat-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; }

.stButton > button {
    background-color: var(--red) !important; color: white !important;
    border: none !important; border-radius: 4px !important;
    font-weight: 500 !important; width: 100%;
}
.stButton > button:hover { background-color: #ff0a16 !important; }

.stTabs [aria-selected="true"] { color: var(--red) !important; }

.section-label {
    font-size: 0.75rem; text-transform: uppercase; letter-spacing: 2px;
    color: var(--muted); margin-bottom: 0.5rem;
}

hr.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Loading (cached so it only runs once) ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading 32M ratings — this takes a minute on first run...")
def load_data():
    train_df, _ = preprocess()
    movies_df = pd.read_csv(MOVIES_PATH, usecols=["movieId", "title"])

    user_map = {u: i for i, u in enumerate(train_df["userId"].unique())}
    movie_map = {m: i for i, m in enumerate(train_df["movieId"].unique())}

    # Only keep the columns the app actually needs, to save memory
    train_slim = train_df[["userId", "movieId"]]

    # Titles for movies the model actually knows about (for the dropdown)
    known_movies = movies_df[movies_df["movieId"].isin(movie_map.keys())]

    return train_slim, movies_df, known_movies, user_map, movie_map


@st.cache_resource(show_spinner="Loading model...")
def load_model(n_users, n_movies):
    if not os.path.exists(MODEL_PATH):
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=HF_REPO_ID, filename="mf_model.pt", local_dir=".")

    model = MFModel(n_users=n_users, n_movies=n_movies, k=50)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


# ── Recommendation logic ───────────────────────────────────────────────────────
def recommend_for_user(model, user_id, user_map, movie_map, train_df, movies_df, top_n):
    """Predict ratings for every movie this user hasn't seen, return the top N."""
    user_idx = user_map[user_id]
    seen = set(train_df.loc[train_df["userId"] == user_id, "movieId"])
    candidates = [m for m in movie_map if m not in seen]

    movie_indices = torch.tensor([movie_map[m] for m in candidates])
    user_tensor = torch.full((len(candidates),), user_idx, dtype=torch.long)

    with torch.no_grad():
        scores = model(user_tensor, movie_indices).numpy()

    recs = pd.DataFrame({"movieId": candidates, "score": scores})
    recs = recs.nlargest(top_n, "score")
    recs = recs.merge(movies_df, on="movieId")
    return recs[["title", "score"]].reset_index(drop=True)


def find_similar_movies(model, movie_id, movie_map, movies_df, top_n):
    """Find movies whose learned embeddings point in a similar direction."""
    movie_idx = movie_map[movie_id]
    all_vecs = model.movie_emb.weight.detach()
    target_vec = all_vecs[movie_idx].unsqueeze(0)

    sims = torch.nn.functional.cosine_similarity(target_vec, all_vecs)
    top_indices = sims.argsort(descending=True)[1 : top_n + 1]  # skip itself

    idx_to_movie = {v: k for k, v in movie_map.items()}
    recs = pd.DataFrame({
        "movieId": [idx_to_movie[i.item()] for i in top_indices],
        "score": [sims[i].item() for i in top_indices],
    })
    recs = recs.merge(movies_df, on="movieId")
    return recs[["title", "score"]].reset_index(drop=True)


def fit_new_user(model, rated_movie_ids, ratings, movie_map, steps=300, lr=0.1, reg=0.05):
    """
    "Fold-in": learn an embedding for a brand-new user from a handful of ratings.
    Movie embeddings and biases stay frozen — we only solve for this user's vector.
    """
    idxs = torch.tensor([movie_map[m] for m in rated_movie_ids])
    targets = torch.tensor(ratings, dtype=torch.float32)

    # Frozen: these were learned during training
    movie_vecs = model.movie_emb.weight[idxs].detach()
    movie_b = model.movie_bias.weight[idxs].squeeze(1).detach()
    global_b = model.global_bias.detach()

    # The only parameters we're learning
    k = model.user_emb.weight.shape[1]
    u = torch.zeros(k, requires_grad=True)
    b_u = torch.zeros(1, requires_grad=True)

    opt = torch.optim.Adam([u, b_u], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        preds = (movie_vecs @ u) + b_u + movie_b + global_b
        # MSE on their ratings, plus L2 regularization to prevent overfitting
        # on what may be only a few data points
        loss = ((preds - targets) ** 2).mean() + reg * (u ** 2).sum()
        loss.backward()
        opt.step()

    return u.detach(), b_u.detach()


def recommend_new_user(model, u, b_u, seen_ids, movie_map, movies_df, top_n):
    """Score every unseen movie against the freshly-learned user vector."""
    candidates = [m for m in movie_map if m not in seen_ids]
    idxs = torch.tensor([movie_map[m] for m in candidates])

    with torch.no_grad():
        scores = (
            (model.movie_emb.weight[idxs] @ u)
            + b_u
            + model.movie_bias.weight[idxs].squeeze(1)
            + model.global_bias
        ).clamp(0.5, 5.0).numpy()

    recs = pd.DataFrame({"movieId": candidates, "score": scores})
    recs = recs.nlargest(top_n, "score").merge(movies_df, on="movieId")
    return recs[["title", "score"]].reset_index(drop=True)


# ── Rendering ──────────────────────────────────────────────────────────────────
def render_cards(recs, score_type="rating"):
    for i, row in recs.iterrows():
        if score_type == "rating":
            pct = int(((row["score"] - 0.5) / 4.5) * 100)
            label = f"{row['score']:.2f} / 5"
        else:  # cosine similarity, ranges roughly 0 to 1
            pct = int(max(row["score"], 0) * 100)
            label = f"{row['score'] * 100:.0f}% match"

        st.markdown(f"""
        <div class="rec-card">
            <div class="rank">{i + 1}</div>
            <div style="flex:1">
                <div class="movie-title">{row['title']}</div>
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>
            </div>
            <div class="movie-score">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ── App ────────────────────────────────────────────────────────────────────────
st.markdown("<h1>RECFLIX</h1>", unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Matrix Factorization Recommender · 32M Ratings</div>',
    unsafe_allow_html=True,
)

if not os.path.exists(MOVIES_PATH):
    st.error(
        f"Could not find `{MOVIES_PATH}`.\n\n"
        "Run `python src/download_data.py` first, and launch this app from the project root:\n\n"
        "`streamlit run app/app.py`"
    )
    st.stop()

train_df, movies_df, known_movies, user_map, movie_map = load_data()
model = load_model(len(user_map), len(movie_map))

valid_user_ids = sorted(user_map.keys())

# Stats
c1, c2, c3, c4 = st.columns(4)
c1.markdown('<div class="stat-box"><div class="stat-number">32M</div><div class="stat-label">Ratings</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat-box"><div class="stat-number">{len(user_map):,}</div><div class="stat-label">Users</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="stat-box"><div class="stat-number">{len(movie_map):,}</div><div class="stat-label">Movies</div></div>', unsafe_allow_html=True)
c4.markdown('<div class="stat-box"><div class="stat-number">0.85</div><div class="stat-label">Test RMSE</div></div>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Recommend by User", "Find Similar Movies", "Rate & Recommend"])

# ── Tab 1: personalized recommendations ────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-label">Pick a user to see what the model predicts they\'d rate highest</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        user_id = st.selectbox(
            "User ID",
            options=valid_user_ids,
            index=0,
            label_visibility="collapsed",
        )
    with col2:
        top_n = st.number_input("Results", 5, 25, 10, label_visibility="collapsed")

    if st.button("Get Recommendations", key="user_btn"):
        with st.spinner("Scoring movies..."):
            recs = recommend_for_user(model, user_id, user_map, movie_map, train_df, movies_df, top_n)
        n_rated = (train_df["userId"] == user_id).sum()
        st.markdown(
            f'<div class="section-label" style="margin-top:1rem">Top {top_n} for User {user_id} · '
            f'{n_rated} movies already rated</div>',
            unsafe_allow_html=True,
        )
        render_cards(recs, score_type="rating")

# ── Tab 2: similar movies ──────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-label">Pick a movie to find others with similar learned embeddings</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_title = st.selectbox(
            "Movie",
            options=known_movies["title"].tolist(),
            index=0,
            label_visibility="collapsed",
        )
    with col2:
        top_n_movie = st.number_input("Results", 5, 25, 10, key="movie_n", label_visibility="collapsed")

    if st.button("Find Similar Movies", key="movie_btn"):
        movie_id = known_movies.loc[known_movies["title"] == selected_title, "movieId"].iloc[0]
        with st.spinner("Comparing embeddings..."):
            recs = find_similar_movies(model, movie_id, movie_map, movies_df, top_n_movie)
        st.markdown(
            f'<div class="section-label" style="margin-top:1rem">Similar to "{selected_title}"</div>',
            unsafe_allow_html=True,
        )
        render_cards(recs, score_type="similarity")


# ── Tab 3: cold-start via fold-in ──────────────────────────────────────────────
with tab3:
    st.markdown(
        '<div class="section-label">Rate a few movies you\'ve seen and the model will '
        'learn your taste vector on the fly</div>',
        unsafe_allow_html=True,
    )

    picks = st.multiselect(
        "Movies you've seen",
        options=known_movies["title"].tolist(),
        max_selections=10,
        label_visibility="collapsed",
        placeholder="Search and select at least 3 movies...",
    )

    user_ratings = {}
    if picks:
        st.markdown('<div class="section-label" style="margin-top:1rem">Your ratings</div>', unsafe_allow_html=True)
        for title in picks:
            user_ratings[title] = st.slider(title, 0.5, 5.0, 3.5, step=0.5, key=f"rate_{title}")

    col1, col2 = st.columns([3, 1])
    with col2:
        top_n_new = st.number_input("Results", 5, 25, 10, key="new_n", label_visibility="collapsed")

    if st.button("Recommend For Me", key="new_btn"):
        if len(picks) < 3:
            st.warning("Please rate at least 3 movies — fewer than that and the learned vector is mostly noise.")
        else:
            title_to_id = dict(zip(known_movies["title"], known_movies["movieId"]))
            rated_ids = [title_to_id[t] for t in picks]
            ratings = [user_ratings[t] for t in picks]

            with st.spinner("Learning your taste vector..."):
                u, b_u = fit_new_user(model, rated_ids, ratings, movie_map)
                recs = recommend_new_user(model, u, b_u, set(rated_ids), movie_map, movies_df, top_n_new)

            st.markdown(
                f'<div class="section-label" style="margin-top:1rem">Top {top_n_new} picks based on your '
                f'{len(picks)} ratings</div>',
                unsafe_allow_html=True,
            )
            render_cards(recs, score_type="rating")

            if len(picks) < 5:
                st.caption(
                    "Note: with only a few ratings the learned vector is noisy. "
                    "Rate more movies for sharper recommendations."
                )