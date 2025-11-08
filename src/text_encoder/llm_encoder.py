# src/text_encoder/text_encoder.py

from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import os
from tqdm import tqdm

def load_captions(csv_path):
    """
    Load image-caption pairs from the original flickr30k captions.csv file.
    Expected format:
        image, caption
    """
    df = pd.read_csv(csv_path, header=None, names=["image", "caption"])
    df["caption"] = df["caption"].astype(str).str.strip()
    return df


def encode_captions(df, output_path):
    """
    Encode all captions using SentenceTransformer (MiniLM).
    """
    print("📖 Loaded captions:", len(df))

    captions = df["caption"].tolist()

    # ✅ Load MiniLM (LLM encoder)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"✍️ Encoding {len(captions)} captions into embeddings...")

    embeddings = model.encode(
        captions,
        batch_size=64,
        show_progress_bar=True,
        convert_to_tensor=True
    )

    # Save embeddings + mapping back to image names
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save({
        "captions": captions,
        "image_names": df["image"].tolist(),
        "embeddings": embeddings
    }, output_path)

    print(f"✅ Saved text embeddings to: {output_path}")


if __name__ == "__main__":
    CAPTION_FILE = "data/images/captions.txt"            # your original file
    OUTPUT = "data/processed/text_embeddings/text_embeds.pt"

    df = load_captions(CAPTION_FILE)
    encode_captions(df, OUTPUT)
