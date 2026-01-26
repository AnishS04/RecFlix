import pandas as pd
from sklearn.model_selection import train_test_split 

def load_ratings(path="data/ml-latest/ratings.csv"):
  dtypes = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32" 
  }
  return pd.read_csv(path, dtype=dtypes, usecols=["userId", "movieId", "rating"])

def filter_sparse_interactions(ratings, min_user_ratings=20, min_movie_ratings=20):
  user_counts = ratings['userId'].value_counts()
  movie_counts = ratings['movieId'].value_counts()

  return ratings[
    ratings["userId"].isin(user_counts[user_counts >= min_user_ratings].index) &
    ratings["movieId"].isin(movie_counts[movie_counts >= min_movie_ratings].index)
  ]

def split_data(ratings, test_size=0.2, random_state=42):
  return train_test_split(
    ratings,
    test_size=test_size,
    random_state=random_state
  )

def preprocess(
    ratings_path="data/ml-latest/ratings.csv",
    min_user_ratings=20,
    min_movie_ratings=20,
    test_size=0.2,
    random_state=42
):
  ratings = load_ratings(ratings_path)
  filtered = filter_sparse_interactions(
    ratings,
    min_user_ratings,
    min_movie_ratings
  )
  train_df, test_df = split_data(
    filtered,
    test_size=test_size,
    random_state=random_state
  )
  return train_df, test_df

if __name__ == "__main__":
  train_df, test_df = preprocess()
  print(f"Train ratings: {len(train_df):,}")
  print(f"Test ratings: {len(test_df):,}")


