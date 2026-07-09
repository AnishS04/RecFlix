# RecFlix

A movie recommender system built on the MovieLens dataset, comparing a matrix factorization model against a popularity baseline across 32 million ratings.

![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange)
![Streamlit](https://img.shields.io/badge/streamlit-1.x-red)

---

## Results

Both models were evaluated on the same held-out test set of **6,444,332 ratings** — no rows dropped from either.

| Model | Test RMSE |
|---|---|
| Popularity baseline | 0.9593 |
| **Matrix factorization** | **0.8493** |
| Improvement | **+0.1101 (11.5%)** |

Training loss converged from 5.28 to 0.70 over 25 epochs.

---

## What it does

The model learns a 50-dimensional embedding for every user and every movie, plus per-user, per-movie, and global bias terms. A predicted rating is the dot product of a user vector and a movie vector, offset by those biases:

```
prediction = dot(user_vec, movie_vec) + user_bias + movie_bias + global_bias
```

Nothing about genre, cast, or plot is ever fed to the model. Everything it knows about a movie is inferred from who rated it and how.

The Streamlit app exposes three ways to use this:

- **Recommend by user** — pick an existing user, score every movie they haven't rated, return the highest predictions.
- **Find similar movies** — pick a movie, rank all others by cosine similarity of their learned embeddings. Movies that get rated similarly by similar people end up pointing in similar directions.
- **Rate & recommend** — a new user rates a handful of movies. The app freezes every learned movie embedding and solves for just that person's taste vector via gradient descent, then recommends against it. This technique is called *fold-in*, and it's the standard way production systems handle new users without retraining.

---

## Dataset

[MovieLens `ml-latest`](https://grouplens.org/datasets/movielens/) — 32M ratings, 86K movies, 330K users.

After filtering out users and movies with fewer than 20 ratings (sparse interactions add noise without enough signal to learn from), the model trains on **204,443 users** and **23,426 movies**.

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset (~1GB)
python src/download_data.py

# 3. Download pre-trained model weights from Hugging Face
python src/download_model.py

# 4. Reproduce the model comparison
python src/model_comparison.py

# 5. Launch the app (run from the project root)
streamlit run app/app.py
```

Pre-trained weights are hosted at [`AnishS04/recflix-mf-model`](https://huggingface.co/AnishS04/recflix-mf-model). `model_comparison.py` loads them automatically if present and only retrains from scratch if they're missing.

---

## Training configuration

| Parameter | Value |
|---|---|
| Latent factors (`k`) | 50 |
| Epochs | 25 |
| Samples per epoch | 3,000,000 (resampled each epoch) |
| Batch size | 32,768 |
| Learning rate | 0.005 (Adam) |
| Weight decay | 1e-5 |

Rather than iterate over all 25M training rows every epoch, each epoch draws a fresh 3M-row sample. Across 25 epochs the model sees roughly 75M examples — broad coverage of the dataset without holding it all in memory at once.

---

## Project structure

```
RecFlix/
├── app/
│   └── app.py                  # Streamlit frontend
├── src/
│   ├── download_data.py        # Fetches MovieLens
│   ├── download_model.py       # Fetches weights from Hugging Face
│   ├── data_preprocessing.py   # Loading, sparsity filter, train/test split
│   ├── matrix_factorization.py # MFModel, RatingsDataset, training loop
│   ├── recommender.py          # Popularity baseline
│   ├── evaluation.py           # RMSE for both models
│   └── model_comparison.py     # Trains and benchmarks both
├── notebooks/
│   └── eda.ipynb
├── requirements.txt
└── README.md
```

---

## Notes on the evaluation

An early version of the popularity baseline scored 0.99 RMSE, which looked competitive. It wasn't. The evaluation merged test ratings against the popularity table and dropped rows where no match was found — silently discarding every movie below the 100-rating threshold. The baseline was being graded only on the popular, easy-to-predict movies while matrix factorization was graded on everything.

Filling unmatched movies with the global mean instead of dropping them puts both models on the same 6.4M test rows. The comparison in the results table above is apples to apples.

---

## Known limitations

**Cold start.** Matrix factorization only knows what it saw during training. It has no embedding for a user or a movie it has never encountered, and no way to construct one from scratch. The *Rate & Recommend* tab works around this on the user side via fold-in, but a movie outside the training set is genuinely out of reach — it would require content features (genre, cast, plot embeddings) and a different model architecture.

**Long-tail bias in top-N.** Because recommendations rank by predicted rating, obscure movies with few ratings can surface near the top: their movie-bias terms haven't been pulled toward the global mean by enough data. This is expected behavior for vanilla MF and is why production systems typically apply a minimum-ratings filter to the candidate pool before ranking.

**Fold-in accuracy scales with input.** A taste vector learned from 3 ratings is mostly regularization. The app enforces a 3-movie minimum and warns below 5.

---

## Possible extensions

- **SVD++** — incorporate implicit feedback (the fact that a user rated a movie at all, independent of the score). Typically worth ~0.02–0.05 RMSE on MovieLens.
- **NeuMF** — replace the dot product with a learned neural interaction function.
- **Hybrid content model** — use genres and tags to generate embeddings for unseen movies, closing the item cold-start gap.