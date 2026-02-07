import pandas as pd

def build_popularity_model(ratings_df, min_ratings=50):

  movie_stats = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"]).rename(columns={"mean": "avg_rating", "count": "rating_count"})
  
   # Filter movies with at least min_ratings
  movie_stats = movie_stats[movie_stats["rating_count"] >= min_ratings]
  
  movie_stats = movie_stats.sort_values(by="avg_rating", ascending=False)

  return movie_stats.reset_index()

def get_top_n(popularity_df, n=10):
  return popularity_df.head(n)

def load_movies(path="data/ml-latest/movies.csv"):
  return pd.read_csv(path, usecols=["movieId", "title"])

def add_movie_titles(popularity_df, movies_df):
  return popularity_df.merge(
    movies_df,
    on="movieId",
    how="left"
  )

def recommend_for_user(user_id, train_df, pop_model, movies_df, n=10):
  # movies user already rated
  seen_movies = set(
    train_df.loc[train_df["userId"] == user_id, "movieId"]
  )

  # remove seen movies from popularity model
  recs = pop_model[~pop_model["movieId"].isin(seen_movies)]

  # join titles
  recs = recs.merge(
    movies_df[["movieId", "title"]],
    on="movieId",
      how="left"
  )

  return recs[["title", "avg_rating", "rating_count"]].head(n)

if __name__ == "__main__":
  from data_preprocessing import preprocess
  import pandas as pd

  train_df, test_df = preprocess()

  movies = pd.read_csv("data/ml-latest/movies.csv")

  pop_model = build_popularity_model(train_df, min_ratings=100)

  for user_id in [1, 50, 500]:
    print(f"\nTop recommendations for user {user_id}:")
    print(recommend_for_user(user_id, train_df, pop_model, movies, n=10))


