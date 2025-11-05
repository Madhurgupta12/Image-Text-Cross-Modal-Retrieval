# src/text_encoder/llm_text_encoder.py

from sentence_transformers import SentenceTransformer
import torch
import os
from tqdm import tqdm

def encode_text(text_file, output_path):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open(text_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Encoding {len(lines)} enriched KG-text sentences...")
    embeddings = model.encode(lines, show_progress_bar=True, convert_to_tensor=True)
    torch.save(embeddings, output_path)
    print(f"✅ Saved KG-based text embeddings to: {output_path}")

if __name__ == "__main__":
    text_path = "data/knowledge_graph/kg_text.txt"
    output_path = "data/processed/text_embeddings/kg_text_embeddings.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    encode_text(text_path, output_path)
