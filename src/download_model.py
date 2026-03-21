"""
Downloads the pre-trained Matrix Factorization model weights from Hugging Face.
Run this script once before running model_comparison.py or the Streamlit app.
"""

import os
from huggingface_hub import hf_hub_download

REPO_ID = "AnishS04/recflix-mf-model"
FILENAME = "mf_model.pt"
SAVE_PATH = "mf_model.pt"

if os.path.exists(SAVE_PATH):
    print(f"Model already exists at '{SAVE_PATH}' — no download needed.")
else:
    print(f"Downloading model from Hugging Face ({REPO_ID})...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir="."
    )
    print(f"Model saved to '{SAVE_PATH}'")
    print("You can now run model_comparison.py or the Streamlit app.")