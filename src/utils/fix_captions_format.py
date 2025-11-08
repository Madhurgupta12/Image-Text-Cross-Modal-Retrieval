# src/utils/fix_captions_format.py

import pandas as pd
import os

def extract_captions():
    INPUT = "data/images/captions.txt"   # your original file
    OUTPUT = "data/captions/train_captions.txt"

    print(f"📖 Reading raw captions from: {INPUT}")

    df = pd.read_csv(INPUT, header=None, names=["image", "caption"])

    # Clean caption text
    df["caption"] = df["caption"].astype(str).str.strip()
    df["caption"] = df["caption"].str.replace('"', "")

    # Save ONLY captions
    captions = df["caption"].tolist()

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for c in captions:
            f.write(c + "\n")

    print(f"✅ Clean captions saved at: {OUTPUT}")
    print(f"✅ Total captions: {len(captions)}")


if __name__ == "__main__":
    extract_captions()
