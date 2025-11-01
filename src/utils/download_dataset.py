# src/utils/download_dataset.py

import kagglehub
import os
import shutil

# Download latest version
path = kagglehub.dataset_download("adityajn105/flickr30k")

print("✅ Path to dataset files:", path)

# Move dataset to data/images/
target_dir = os.path.join("data", "images")
os.makedirs(target_dir, exist_ok=True)

# Copy dataset files
for item in os.listdir(path):
    s = os.path.join(path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print(f"📦 Dataset copied to: {target_dir}")
