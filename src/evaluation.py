import torch 
import numpy as np
import pandas as pd


def evaluate_rmse(model, test_df):
    model.eval()

    valid_test = test_df[
        test_df["user_idx"].notna() & test_df["movie_idx"].notna()
    ].copy()

    n_dropped = len(test_df) - len(valid_test)
    if n_dropped > 0:
        print(f"  [evaluate_rmse] Dropped {n_dropped:,} cold-start rows "
              f"({n_dropped / len(test_df):.1%} of test set)")

    valid_test["user_idx"] = valid_test["user_idx"].astype(int)
    valid_test["movie_idx"] = valid_test["movie_idx"].astype(int)

    users = torch.tensor(valid_test["user_idx"].values, dtype=torch.long)
    movies = torch.tensor(valid_test["movie_idx"].values, dtype=torch.long)
    true_ratings = torch.tensor(valid_test["rating"].values, dtype=torch.float32)

    with torch.no_grad():
        preds = model(users, movies)
        mse = torch.mean((preds - true_ratings) ** 2)
        rmse = torch.sqrt(mse)

    print(f"  [evaluate_rmse] Evaluated on {len(valid_test):,} / {len(test_df):,} rows")
    return rmse.item()


def evaluate_popularity_rmse(pop_model, test_df):
    """
    Evaluate popularity baseline on the full test set.
    Movies not in the popularity model (below min_ratings threshold) are
    filled with the global mean rating so we evaluate on the same rows as MF.
    """
    merged = test_df.merge(
        pop_model[["movieId", "avg_rating"]],
        on="movieId",
        how="left"
    )

    global_mean = pop_model["avg_rating"].mean()
    n_filled = merged["avg_rating"].isna().sum()
    if n_filled > 0:
        print(f"  [evaluate_popularity_rmse] Filled {n_filled:,} unseen movies "
              f"with global mean ({global_mean:.4f})")
    merged["avg_rating"] = merged["avg_rating"].fillna(global_mean)

    mse = np.mean((merged["avg_rating"] - merged["rating"]) ** 2)
    rmse = np.sqrt(mse)

    print(f"  [evaluate_popularity_rmse] Evaluated on {len(merged):,} / {len(test_df):,} rows")
    return rmse