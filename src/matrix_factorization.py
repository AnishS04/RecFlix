import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from data_preprocessing import preprocess
from evaluation import evaluate_rmse


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
        self.movie_emb = nn.Embedding(n_movies, k)

        self.user_bias = nn.Embedding(n_users, 1)
        self.movie_bias = nn.Embedding(n_movies, 1)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

        # Global bias anchors predictions to the mean rating (~3.5 for MovieLens)
        self.global_bias = nn.Parameter(torch.tensor(3.5))

    def forward(self, users, movies):
        user_vecs = self.user_emb(users)
        movie_vecs = self.movie_emb(movies)

        dot = (user_vecs * movie_vecs).sum(dim=1)

        u_bias = self.user_bias(users).squeeze(1)
        m_bias = self.movie_bias(movies).squeeze(1)

        out = dot + u_bias + m_bias + self.global_bias
        return torch.clamp(out, 0.5, 5.0)


def train_model(model, train_df, epochs=25, lr=0.005, sample_size=3_000_000):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        # Fresh sample every epoch so the model sees more of the dataset over time
        sample = train_df.sample(sample_size, random_state=epoch)
        dataset = RatingsDataset(sample)
        loader = DataLoader(dataset, batch_size=32768, shuffle=True)

        total_loss = 0
        for users, movies, ratings in loader:
            optimizer.zero_grad()
            preds = model(users, movies)
            loss = loss_fn(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")


def recommend_mf(model, user_id, user_map, movie_map, train_df, movies_df, top_n=10):
    model.eval()

    if user_id not in user_map:
        print("User not in training set")
        return None

    user_idx = user_map[user_id]
    seen = set(train_df[train_df.userId == user_id].movieId)
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
    recs = recs.merge(movies_df[["movieId", "title"]], on="movieId")
    return recs[["title", "score"]]


if __name__ == "__main__":
    train_df, test_df = preprocess()

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

    train_model(model, train_df, epochs=25, sample_size=3_000_000)

    rmse = evaluate_rmse(model, test_df)
    print(f"\nMF RMSE on test set: {rmse:.4f}")