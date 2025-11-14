"""
Full evaluation script for the trained model on complete dataset.
Computes retrieval metrics on all test samples.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from src.fusion.projection_head import ProjectionModel
from src.evaluation.metrics import evaluate_bidirectional_retrieval
from src.utils.config import (
    ModelConfig, DeviceConfig, DataConfig, 
    BEST_MODEL_PATH, PAIRED_EMB_DIR
)


def load_embeddings():
    """Load pre-computed embeddings."""
    print("[1/4] Loading embeddings...")
    
    images_path = PAIRED_EMB_DIR / "images.pt"
    texts_path = PAIRED_EMB_DIR / "texts.pt"
    
    if not images_path.exists() or not texts_path.exists():
        raise FileNotFoundError("Embeddings not found. Run generate_embeddings.py first.")
    
    image_embeddings = torch.load(images_path, map_location='cpu')
    text_embeddings = torch.load(texts_path, map_location='cpu')
    
    print(f"   Image embeddings: {image_embeddings.shape}")
    print(f"   Text embeddings: {text_embeddings.shape}")
    
    return image_embeddings, text_embeddings


def load_model(model_path, device, img_dim=768, txt_dim=384):
    """Load trained projection model."""
    print(f"\n[2/4] Loading model from {model_path}...")
    
    model = ProjectionModel(
        img_dim=img_dim,
        text_dim=txt_dim,
        shared_dim=ModelConfig.SHARED_DIM
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'i2t_recall@1' in checkpoint:
            print(f"   Previous I2T R@1: {checkpoint['i2t_recall@1']:.2%}")
            print(f"   Previous T2I R@1: {checkpoint['t2i_recall@1']:.2%}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def compute_projections_in_batches(model, img_embeddings, txt_embeddings, device, batch_size=512):
    """
    Compute projections in batches to avoid memory issues.
    
    Args:
        model: Projection model
        img_embeddings: Image embeddings [N, D_img]
        txt_embeddings: Text embeddings [N, D_txt]
        device: Device to compute on
        batch_size: Batch size for processing
    
    Returns:
        Tuple of (img_proj, txt_proj) - projected embeddings [N, shared_dim]
    """
    num_samples = img_embeddings.size(0)
    all_img_proj = []
    all_txt_proj = []
    
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Computing projections"):
            img_batch = img_embeddings[i:i+batch_size].to(device)
            txt_batch = txt_embeddings[i:i+batch_size].to(device)
            
            img_proj, txt_proj = model(img_batch, txt_batch, normalize=True)
            
            all_img_proj.append(img_proj.cpu())
            all_txt_proj.append(txt_proj.cpu())
    
    img_proj = torch.cat(all_img_proj, dim=0)
    txt_proj = torch.cat(all_txt_proj, dim=0)
    
    return img_proj, txt_proj


def evaluate_in_chunks(img_proj, txt_proj, chunk_size=5000):
    """
    Evaluate retrieval metrics in chunks to avoid OOM.
    
    For very large datasets, we compute metrics on multiple chunks
    and average the results.
    """
    num_samples = img_proj.size(0)
    
    if num_samples <= chunk_size:
        # Small enough to evaluate all at once
        print(f"\n[4/4] Computing retrieval metrics on all {num_samples} samples...")
        metrics = evaluate_bidirectional_retrieval(
            img_proj, txt_proj, normalize=False
        )
        return metrics
    
    # Evaluate on multiple chunks
    print(f"\n[4/4] Computing retrieval metrics in chunks of {chunk_size}...")
    all_metrics = []
    
    for i in range(0, num_samples, chunk_size):
        end_idx = min(i + chunk_size, num_samples)
        chunk_img = img_proj[i:end_idx]
        chunk_txt = txt_proj[i:end_idx]
        
        print(f"   Evaluating chunk {i//chunk_size + 1}: samples {i}-{end_idx}")
        
        chunk_metrics = evaluate_bidirectional_retrieval(
            chunk_img, chunk_txt, normalize=False
        )
        all_metrics.append(chunk_metrics)
    
    # Average metrics across chunks
    averaged_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        averaged_metrics[key] = sum(values) / len(values)
    
    return averaged_metrics


def main():
    print("="*70)
    print("FULL EVALUATION ON FLICKR30K")
    print("="*70)
    
    device = DeviceConfig.DEVICE
    print(f"\nDevice: {device}")
    
    # Load embeddings
    image_embeddings, text_embeddings = load_embeddings()
    
    # Load model
    model = load_model(BEST_MODEL_PATH, device)
    
    # Compute projections
    print("\n[3/4] Computing projections...")
    img_proj, txt_proj = compute_projections_in_batches(
        model, image_embeddings, text_embeddings, device, batch_size=512
    )
    
    print(f"\n   Projected shapes:")
    print(f"   Images: {img_proj.shape}")
    print(f"   Texts: {txt_proj.shape}")
    
    # Evaluate retrieval
    metrics = evaluate_in_chunks(img_proj, txt_proj, chunk_size=5000)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"\nImage-to-Text Retrieval:")
    print(f"   Recall@1:  {metrics['i2t_recall@1']:.2%}")
    print(f"   Recall@5:  {metrics['i2t_recall@5']:.2%}")
    print(f"   Recall@10: {metrics['i2t_recall@10']:.2%}")
    
    print(f"\nText-to-Image Retrieval:")
    print(f"   Recall@1:  {metrics['t2i_recall@1']:.2%}")
    print(f"   Recall@5:  {metrics['t2i_recall@5']:.2%}")
    print(f"   Recall@10: {metrics['t2i_recall@10']:.2%}")
    
    print(f"\nMean Recall:")
    print(f"   R@1:  {(metrics['i2t_recall@1'] + metrics['t2i_recall@1']) / 2:.2%}")
    print(f"   R@5:  {(metrics['i2t_recall@5'] + metrics['t2i_recall@5']) / 2:.2%}")
    print(f"   R@10: {(metrics['i2t_recall@10'] + metrics['t2i_recall@10']) / 2:.2%}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Evaluation complete!")
    print("="*70)


if __name__ == "__main__":
    main()
