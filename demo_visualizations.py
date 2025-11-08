"""
Demo script to create visualizations of retrieval results.
Uses the trained model to perform searches and visualize results.
"""

import torch
import numpy as np
from pathlib import Path
import pandas as pd

from src.fusion.projection_head import ProjectionModel
from src.text_encoder.llm_encoder import TextEncoder
from src.evaluation.qualitative_results import (
    visualize_text_to_image_retrieval,
    create_retrieval_grid
)
from src.utils.config import (
    PAIRED_EMB_DIR, MODELS_DIR, VISUALIZATIONS_DIR, DATA_DIR
)


def create_demo_visualizations():
    """Create demo visualizations."""
    print("=" * 60)
    print("  Image-Text Retrieval Visualization Demo")
    print("=" * 60)
    
    # Load model
    print("\n[*] Loading trained model...")
    model = ProjectionModel(img_dim=768, text_dim=384, shared_dim=512)
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pt", map_location='cpu'))
    model.eval()
    print("[OK] Model loaded")
    
    # Load paired embeddings
    print("\n[*] Loading embeddings...")
    img_emb_raw = torch.load(PAIRED_EMB_DIR / "images.pt")
    txt_emb_raw = torch.load(PAIRED_EMB_DIR / "texts.pt")
    img_names = torch.load(PAIRED_EMB_DIR / "img_names.pt")
    print(f"[OK] Loaded {len(img_emb_raw)} image embeddings")
    print(f"[OK] Loaded {len(txt_emb_raw)} text embeddings")
    
    # Project to shared space
    print("\n[*] Projecting embeddings to shared space...")
    with torch.no_grad():
        img_emb_projected = model.encode_image(img_emb_raw, normalize=True)
        txt_emb_projected = model.encode_text(txt_emb_raw, normalize=True)
    print(f"[OK] Projected to {img_emb_projected.shape[1]}-dim space")
    
    # Initialize text encoder
    print("\n[*] Initializing text encoder...")
    text_encoder = TextEncoder(model_name='sentence-transformers/all-MiniLM-L6-v2')
    print("[OK] Text encoder ready")
    
    # Ensure visualization directory exists
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Demo queries
    demo_queries = [
        "people on the beach",
        "a child climbing stairs",
        "a black dog running",
        "sunset over mountains",
        "person riding a bicycle"
    ]
    
    print("\n" + "=" * 60)
    print("  Generating Visualizations")
    print("=" * 60)
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n[*] Query {i}: '{query}'")
        
        # Encode and project query
        query_emb = text_encoder.encode([query])
        with torch.no_grad():
            query_proj = model.encode_text(query_emb, normalize=True)
        
        # Compute similarities
        similarities = torch.matmul(query_proj, img_emb_projected.T)
        similarities = similarities.squeeze(0)
        
        # Get top-10 results
        top_scores, top_indices = torch.topk(similarities, k=10)
        
        # Get image paths
        image_paths = [str(DATA_DIR / "images" / "images" / img_names[idx]) for idx in top_indices.numpy()]
        scores = top_scores.numpy().tolist()
        
        # Create visualization
        save_path = VISUALIZATIONS_DIR / f"query_{i}_{query.replace(' ', '_')[:30]}.png"
        visualize_text_to_image_retrieval(
            query_text=query,
            retrieved_image_paths=image_paths,
            similarity_scores=scores,
            save_path=save_path,
            title=f"Text-to-Image Retrieval Results"
        )
        
        print(f"[OK] Saved visualization to: {save_path.name}")
        print(f"     Top score: {scores[0]:.3f}")
    
    # Create a grid of random high-quality matches
    print("\n[*] Creating grid of high-scoring matches...")
    
    # Find top matches across sample queries
    all_similarities = torch.matmul(txt_emb_projected[:100], img_emb_projected.T)
    top_scores, top_indices = torch.topk(all_similarities.flatten(), k=16)
    
    top_image_paths = []
    top_scores_list = []
    
    for score, idx in zip(top_scores.numpy(), top_indices.numpy()):
        img_idx = idx % len(img_emb_projected)
        img_path = str(DATA_DIR / "images" / "images" / img_names[img_idx])
        if img_path not in top_image_paths:
            top_image_paths.append(img_path)
            top_scores_list.append(float(score))
        if len(top_image_paths) >= 16:
            break
    
    grid_save_path = VISUALIZATIONS_DIR / "top_matches_grid.png"
    create_retrieval_grid(
        image_paths=top_image_paths,
        scores=top_scores_list,
        n_cols=4,
        save_path=grid_save_path,
        title="Top Matching Image-Text Pairs"
    )
    print(f"[OK] Saved grid to: {grid_save_path.name}")
    
    print("\n" + "=" * 60)
    print(f"[OK] All visualizations saved to: {VISUALIZATIONS_DIR}")
    print("=" * 60)
    
    return VISUALIZATIONS_DIR


if __name__ == "__main__":
    vis_dir = create_demo_visualizations()
    print(f"\n[OK] Open the folder to view results: {vis_dir}")
