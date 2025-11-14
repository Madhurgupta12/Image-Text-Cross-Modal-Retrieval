"""
Pair image and text embeddings for training.
This script loads pre-computed embeddings and pairs them (1 image -> 5 captions average).
"""

import torch
import os
from pathlib import Path
from typing import Dict, List, Tuple

from src.utils.config import IMAGE_EMB_DIR, TEXT_EMB_DIR, PAIRED_EMB_DIR


def load_embeddings(
    img_path: str,
    txt_path: str
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Load image and text embeddings from disk.
    
    Args:
        img_path: Path to image embeddings
        txt_path: Path to text embeddings
    
    Returns:
        Tuple of (image_embeddings_dict, text_embeddings_dict)
    """
    print("📥 Loading image embeddings...")
    img_embeds = torch.load(img_path, weights_only=False)
    
    print("📥 Loading text embeddings...")
    txt_embeds = torch.load(txt_path, weights_only=False)
    
    return img_embeds, txt_embeds


def pair_embeddings(
    img_embeds: Dict,
    txt_embeds: Dict,
    captions_per_image: int = 5
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Pair image embeddings with their corresponding text embeddings.
    
    Args:
        img_embeds: Dictionary of image embeddings {filename: vector}
        txt_embeds: Dictionary with 'embeddings', 'captions', 'image_names'
        captions_per_image: Number of captions per image (default: 5 for Flickr30k)
    
    Returns:
        Tuple of (paired_images, paired_texts, image_names)
    """
    # Extract image vectors and filenames
    if isinstance(img_embeds, dict):
        img_names = list(img_embeds.keys())
        img_vectors = [
            torch.tensor(v).float() if not isinstance(v, torch.Tensor) else v.float()
            for v in img_embeds.values()
        ]
    else:
        raise ValueError("❌ Expected image embeddings as dict {filename: vector}")
    
    # Extract caption embeddings
    if isinstance(txt_embeds, dict) and "embeddings" in txt_embeds:
        txt_vectors = txt_embeds["embeddings"]
        if not isinstance(txt_vectors, torch.Tensor):
            txt_vectors = torch.tensor(txt_vectors).float()
        else:
            txt_vectors = txt_vectors.float()
    else:
        raise ValueError(
            "❌ Expected text_embeds in format {captions, image_names, embeddings}"
        )
    
    # Pair images with their caption averages
    paired_img = []
    paired_txt = []
    valid_img_names = []
    
    print(f"🔗 Pairing images with their {captions_per_image} captions...")
    
    for i in range(len(img_vectors)):
        img_vec = img_vectors[i]
        
        start = i * captions_per_image
        end = start + captions_per_image
        
        # Check if we have enough captions
        if end > len(txt_vectors):
            print(f"⚠️  Warning: Not enough captions for image {i} ({img_names[i]}), skipping...")
            break
        
        # Average the captions for this image
        cap_group = txt_vectors[start:end]
        avg_caption = torch.mean(cap_group, dim=0)
        
        paired_img.append(img_vec)
        paired_txt.append(avg_caption)
        valid_img_names.append(img_names[i])
    
    # Stack into tensors
    paired_img = torch.stack(paired_img)
    paired_txt = torch.stack(paired_txt)
    
    print(f"✅ Successfully paired {len(paired_img)} images with their captions")
    
    return paired_img, paired_txt, valid_img_names


def save_paired_embeddings(
    paired_img: torch.Tensor,
    paired_txt: torch.Tensor,
    img_names: List[str],
    output_dir: str
):
    """
    Save paired embeddings to disk.
    
    Args:
        paired_img: Paired image embeddings
        paired_txt: Paired text embeddings
        img_names: Image filenames
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    img_path = os.path.join(output_dir, "images.pt")
    txt_path = os.path.join(output_dir, "texts.pt")
    names_path = os.path.join(output_dir, "img_names.pt")
    
    torch.save(paired_img, img_path)
    torch.save(paired_txt, txt_path)
    torch.save(img_names, names_path)
    
    print(f"\n✅ Paired embeddings saved!")
    print(f"🖼️  Images:    {img_path}   shape={paired_img.shape}")
    print(f"✍️  Text:      {txt_path}   shape={paired_txt.shape}")
    print(f"📛 Filenames: {names_path}")


def main():
    """Main function to pair embeddings."""
    # Define paths
    img_path = IMAGE_EMB_DIR / "vit_embeddings.pt"
    txt_path = TEXT_EMB_DIR / "text_embeddings.pt"
    
    # Check if files exist
    if not img_path.exists():
        raise FileNotFoundError(f"❌ Image embeddings not found: {img_path}")
    if not txt_path.exists():
        raise FileNotFoundError(f"❌ Text embeddings not found: {txt_path}")
    
    # Load embeddings
    img_embeds, txt_embeds = load_embeddings(str(img_path), str(txt_path))
    
    # Pair embeddings
    paired_img, paired_txt, img_names = pair_embeddings(img_embeds, txt_embeds)
    
    # Save paired embeddings
    save_paired_embeddings(paired_img, paired_txt, img_names, str(PAIRED_EMB_DIR))


if __name__ == "__main__":
    main()
