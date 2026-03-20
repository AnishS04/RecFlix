import os
import torch
import pandas as pd

from data_preprocessing import preprocess
from recommender import build_popularity_model
from matrix_factorization import MFModel, train_model
from evaluation import evaluate_rmse, evaluate_popularity_rmse


MODEL_PATH = "mf_model.pt"

if __name__ == "__main__":
    train_df, test_df = preprocess()

    # --- Popularity baseline ---
    pop_model = build_popularity_model(train_df, min_ratings=100)

    print("Evaluating popularity baseline...")
    pop_rmse = evaluate_popularity_rmse(pop_model, test_df)

    # --- Matrix Factorization ---
    user_map = {u: i for i, u in enumerate(train_df["userId"].unique())}
    movie_map = {m: i for i, m in enumerate(train_df["movieId"].unique())}

    train_df["user_idx"] = train_df.userId.map(user_map)
    train_df["movie_idx"] = train_df.movieId.map(movie_map)

    test_df["user_idx"] = test_df.userId.map(user_map)
    test_df["movie_idx"] = test_df.movieId.map(movie_map)

    model = MFModel(
        n_users=len(user_map),
        n_movies=len(movie_map),
        k=50
    )

    if os.path.exists(MODEL_PATH):
        # Load saved model — no retraining needed
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded saved model from '{MODEL_PATH}'")
    else:
        # Train and save so future runs are instant
        print("No saved model found — training from scratch...")
        train_model(model, train_df, epochs=25, lr=0.005, sample_size=3_000_000)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to '{MODEL_PATH}'")

    print("\nEvaluating MF model...")
    mf_rmse = evaluate_rmse(model, test_df)

    print("\n--- Model Comparison ---")
    print(f"Popularity Baseline RMSE : {pop_rmse:.4f}")
    print(f"Matrix Factorization RMSE: {mf_rmse:.4f}")
    print(f"MF improvement           : {pop_rmse - mf_rmse:+.4f}")