"""
Dataset loader for Image-Text Cross-Modal Retrieval.
Supports Flickr30k, COCO, and custom datasets with proper pairing and augmentation.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

# Only import torchvision for type hints to avoid slow loading
if TYPE_CHECKING:
    import torchvision.transforms as transforms

from src.utils.config import (
    DataConfig, IMAGES_DIR, CAPTIONS_FILE, DeviceConfig
)


class ImageTextDataset(Dataset):
    """
    Dataset for paired image-text data with support for multiple captions per image.
    """
    
    def __init__(
        self,
        image_dir: str,
        captions_file: str,
        split: str = "train",
        transform=None,  # Changed from transforms.Compose to avoid import
        max_text_length: int = DataConfig.MAX_TEXT_LENGTH,
    ):
        """
        Args:
            image_dir: Directory containing images
            captions_file: Path to CSV file with image-caption pairs
            split: 'train', 'val', or 'test'
            transform: Image transformations
            max_text_length: Maximum text sequence length
        """
        self.image_dir = Path(image_dir)
        self.split = split
        self.max_text_length = max_text_length
        
        # Load captions
        self.data = self._load_captions(captions_file)
        
        # Apply data split
        self._apply_split()
        
        # Set transforms
        self.transform = transform if transform else self._get_default_transform()
        
        print(f"📊 Loaded {len(self.data)} {split} samples")
    
    def _load_captions(self, captions_file: str) -> pd.DataFrame:
        """Load and parse captions file"""
        if not os.path.exists(captions_file):
            raise FileNotFoundError(f"Captions file not found: {captions_file}")
        
        # Try different formats
        try:
            # Format: image,caption (Flickr30k)
            df = pd.read_csv(captions_file, sep="|", header=0)
            if len(df.columns) < 2:
                df = pd.read_csv(captions_file)
            
            # Ensure we have image and caption columns
            if 'image' not in df.columns and df.columns[0]:
                df.columns = ['image', 'caption'] + list(df.columns[2:])
            
            # Clean data
            df['caption'] = df['caption'].astype(str).str.strip()
            df['image'] = df['image'].astype(str).str.strip()
            
            # Remove invalid entries
            df = df[df['caption'].str.len() > 0]
            df = df[df['image'].str.len() > 0]
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error loading captions file: {e}")
    
    def _apply_split(self):
        """Split data into train/val/test"""
        # Get unique images
        unique_images = self.data['image'].unique()
        n_images = len(unique_images)
        
        # Set random seed for reproducibility
        np.random.seed(DeviceConfig.SEED)
        indices = np.random.permutation(n_images)
        
        # Calculate split indices
        train_size = int(n_images * DataConfig.TRAIN_SPLIT)
        val_size = int(n_images * DataConfig.VAL_SPLIT)
        
        if self.split == "train":
            split_images = unique_images[indices[:train_size]]
        elif self.split == "val":
            split_images = unique_images[indices[train_size:train_size + val_size]]
        elif self.split == "test":
            split_images = unique_images[indices[train_size + val_size:]]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        # Filter data
        self.data = self.data[self.data['image'].isin(split_images)].reset_index(drop=True)
    
    def _get_default_transform(self):
        """Get default image transformations"""
        # Import here to avoid loading torchvision unless needed
        import torchvision.transforms as transforms
        
        if self.split == "train" and DataConfig.USE_AUGMENTATION:
            return transforms.Compose([
                transforms.Resize((DataConfig.IMAGE_SIZE, DataConfig.IMAGE_SIZE)),
                transforms.RandomCrop(DataConfig.IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ) if DataConfig.COLOR_JITTER else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=DataConfig.NORMALIZE_MEAN,
                    std=DataConfig.NORMALIZE_STD
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((DataConfig.IMAGE_SIZE, DataConfig.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=DataConfig.NORMALIZE_MEAN,
                    std=DataConfig.NORMALIZE_STD
                )
            ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str]:
        """
        Returns:
            image: Transformed image tensor [C, H, W]
            caption: Text caption
            image_id: Image filename
        """
        row = self.data.iloc[idx]
        image_id = row['image']
        caption = row['caption']
        
        # Load image
        image_path = self.image_dir / image_id
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"⚠️ Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            image = torch.zeros(3, DataConfig.IMAGE_SIZE, DataConfig.IMAGE_SIZE)
        
        # Process caption
        if DataConfig.LOWERCASE:
            caption = caption.lower()
        
        return image, caption, image_id


class PairedEmbeddingDataset(Dataset):
    """
    Dataset for pre-computed image and text embeddings.
    Used for training the projection head.
    """
    
    def __init__(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_ids: Optional[List[str]] = None
    ):
        """
        Args:
            image_embeddings: Precomputed image embeddings [N, D_img]
            text_embeddings: Precomputed text embeddings [N, D_text]
            image_ids: Optional list of image identifiers
        """
        assert len(image_embeddings) == len(text_embeddings), \
            "Image and text embeddings must have the same length"
        
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
        self.image_ids = image_ids if image_ids else list(range(len(image_embeddings)))
    
    def __len__(self) -> int:
        return len(self.image_embeddings)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            image_emb: Image embedding [D_img]
            text_emb: Text embedding [D_text]
            image_id: Image identifier
        """
        return (
            self.image_embeddings[idx],
            self.text_embeddings[idx],
            self.image_ids[idx]
        )


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = DataConfig.NUM_WORKERS,
    pin_memory: bool = DataConfig.PIN_MEMORY,
) -> DataLoader:
    """
    Create a DataLoader with appropriate settings.
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and DeviceConfig.USE_CUDA,
        prefetch_factor=DataConfig.PREFETCH_FACTOR if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )


def load_paired_embeddings(
    image_emb_path: str,
    text_emb_path: str,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Load pre-computed paired embeddings.
    
    Args:
        image_emb_path: Path to image embeddings
        text_emb_path: Path to text embeddings
        normalize: Whether to L2-normalize embeddings
    
    Returns:
        Tuple of (image_embeddings, text_embeddings, image_ids)
    """
    print(f"📥 Loading paired embeddings...")
    
    # Load image embeddings
    img_data = torch.load(image_emb_path, map_location='cpu', weights_only=False)
    if isinstance(img_data, dict):
        image_ids = list(img_data.keys())
        image_embeddings = torch.stack([
            torch.tensor(v) if not isinstance(v, torch.Tensor) else v
            for v in img_data.values()
        ]).float()
    else:
        image_embeddings = img_data.float()
        image_ids = list(range(len(image_embeddings)))
    
    # Load text embeddings
    txt_data = torch.load(text_emb_path, map_location='cpu', weights_only=False)
    if isinstance(txt_data, dict) and 'embeddings' in txt_data:
        text_embeddings = txt_data['embeddings']
        if not isinstance(text_embeddings, torch.Tensor):
            text_embeddings = torch.tensor(text_embeddings)
        text_embeddings = text_embeddings.float()
    else:
        text_embeddings = txt_data.float()
    
    # Normalize if requested
    if normalize:
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = torch.nn.functional.normalize(text_embeddings, p=2, dim=-1)
    
    print(f"✅ Loaded embeddings:")
    print(f"   🖼️  Images: {image_embeddings.shape}")
    print(f"   ✍️  Text: {text_embeddings.shape}")
    
    return image_embeddings, text_embeddings, image_ids


# Example usage and testing
if __name__ == "__main__":
    # Test dataset loading
    try:
        dataset = ImageTextDataset(
            image_dir=str(IMAGES_DIR),
            captions_file=str(CAPTIONS_FILE),
            split="train"
        )
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(dataset)}")
        
        # Test dataloader
        dataloader = get_dataloader(dataset, batch_size=32, shuffle=True)
        
        # Get a batch
        images, captions, image_ids = next(iter(dataloader))
        print(f"\n📦 Batch shape:")
        print(f"   Images: {images.shape}")
        print(f"   Captions: {len(captions)}")
        print(f"   Image IDs: {len(image_ids)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
