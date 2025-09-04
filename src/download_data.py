import os
import requests
import zipfile

url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
zip_path = "data/ml-latest.zip"
extract_dir = "data/ml-latest"

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    print("Downloading dataset...")
    r = requests.get(url)
    with open(zip_path, "wb") as f:
        f.write(r.content)
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    print("Done! Dataset ready in 'data/ml-latest/'")
