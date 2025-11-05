# src/utils/fix_captions_format.py

import pandas as pd
import os

INPUT_FILE = "data/images/captions.txt"   # your current file
OUTPUT_FILE = "data/captions/train_captions.txt"

def clean_captions():
    print(f"📖 Reading caption file: {INPUT_FILE}")

    # Read as CSV — the separator is a comma
    df = pd.read_csv(INPUT_FILE)

    # Check for correct columns
    if "caption" not in df.columns:
        for i in df.columns:
            print(f"Found column: {i}")
        raise KeyError(f"Expected a 'caption' column. Found: {df.columns}")

    captions = df["caption"].astype(str).str.strip().tolist()

    # Save one caption per line
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for cap in captions:
            f.write(cap + "\n")

    print(f"✅ Cleaned and saved {len(captions)} captions to {OUTPUT_FILE}")

if __name__ == "__main__":
    clean_captions()
