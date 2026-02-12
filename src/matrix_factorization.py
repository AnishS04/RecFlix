import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from data_preprocessing import preprocess

class RatingsDataset(Dataset):
  def __init__(self, ratings_df):
    self.users = torch.tensor(ratings_df["user_idx"].values, dtype=torch.long)
    self.movies = torch.tensor(ratings_df["movie_idx"].values, dtype=torch.long)
    self.ratings = torch.tensor(ratings_df["rating"].values, dtype=torch.float32)

  def __len__(self):
    return len(self.ratings)

  def __getitem__(self, idx):
    return self.users[idx], self.movies[idx], self.ratings[idx]
  
class MFModel(nn.Module):
  def __init__(self, n_users, n_movies, k=50):
    super().__init__()
    self.user_emb = nn.Embedding(n_users, k)
    self.movieemb = nn.Embedding(n_movies, k)

  def forward(self, users, movies):
    user_vecs = self.user_emb(users)
    movie_vecs = self.movieemb(movies)
    return (user_vecs * movie_vecs).sum(dim=1)
  

def train_model(model, loader, epochs=3):
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
  loss_fn = nn.MSELoss()

  for epoch in range(epochs):
    total_loss = 0
    for users, movies, ratings in loader:
      
      preds = model(users, movies)
      loss = loss_fn(preds, ratings)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")
  
def recommend_mf(model, user_id, user_map, movie_map, train_df, movies_df, top_n=10):
  model.eval()

  # convert raw userId → index
  if user_id not in user_map:
    print("User not in training set")
    return None

  user_idx = user_map[user_id]

  # movies user already rated
  seen = set(train_df[train_df.userId == user_id].movieId)

  # candidate movies
  candidates = [m for m in movie_map.keys() if m not in seen]

  movie_indices = torch.tensor([movie_map[m] for m in candidates])

  user_tensor = torch.tensor([user_idx] * len(movie_indices))

  with torch.no_grad():
    scores = model(user_tensor, movie_indices)

  recs = pd.DataFrame({
      "movieId": candidates,
      "score": scores.numpy()
  })

  recs = recs.sort_values("score", ascending=False).head(top_n)

  # join titles
  recs = recs.merge(movies_df[["movieId", "title"]], on="movieId")

  return recs[["title", "score"]]

  
if __name__ == "__main__":
  train_df, test_df = preprocess()

  # create user and movie indices
  user_map = {u: i for i, u in enumerate(train_df["userId"].unique())}
  movie_map = {m: i for i, m in enumerate(train_df["movieId"].unique())}

  # map to indices
  train_df["user_idx"] = train_df.userId.map(user_map)
  train_df["movie_idx"] = train_df.movieId.map(movie_map)

  # training data (25M rows) too large so get a sample of 1M rows
  sample_df = train_df.sample(1_000_000, random_state=42)
  dataset = RatingsDataset(sample_df)
  loader = DataLoader(dataset, batch_size=8192, shuffle=True)

  model = MFModel(
    n_users=len(user_map),
    n_movies=len(movie_map),
    k=50
  )

  train_model(model, loader, epochs=5)

  movies_df = pd.read_csv("data/ml-latest/movies.csv")

  print("\nMF Recommendations for user 1:")
  print(recommend_mf(model, 1, user_map, movie_map, train_df, movies_df))

  print("\nUser 1 highest rated movies:")
  print(
    train_df[train_df.userId == 1]
    .sort_values("rating", ascending=False)
    .merge(movies_df, on="movieId")[["title","rating"]]
    .head(10)
)

