"""
Vision Transformer (ViT) encoder for image feature extraction.
Supports multiple ViT variants with batch processing and GPU acceleration.
"""

from transformers import ViTImageProcessor, ViTModel
import torch
import torch.nn as nn
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from src.utils.config import (
    ModelConfig, DeviceConfig, IMAGE_EMB_PATH, IMAGES_DIR
)


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder wrapper with configurable model variants.
    """
    
    def __init__(
        self,
        model_name: str = ModelConfig.VIT_MODEL_NAME,
        device: Optional[torch.device] = None,
        pooling: str = "mean"  # Options: mean, cls, max
    ):
        """
        Args:
            model_name: HuggingFace model name for ViT
            device: Device to load model on
            pooling: Pooling strategy for sequence of patch embeddings
        """
        super().__init__()
        
        self.model_name = model_name
        self.device = device or DeviceConfig.DEVICE
        self.pooling = pooling
        
        print(f"🔧 Loading ViT model: {model_name}")
        
        # Load processor and model
        try:
            self.processor = ViTImageProcessor.from_pretrained(model_name)
            self.model = ViTModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get output dimension
            self.output_dim = self.model.config.hidden_size
            
            print(f"✅ ViT model loaded successfully")
            print(f"   Output dimension: {self.output_dim}")
            print(f"   Device: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ViT model: {e}")
    
    def _pool_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence of patch embeddings into single vector.
        
        Args:
            embeddings: Patch embeddings [batch_size, seq_len, hidden_dim]
        
        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        if self.pooling == "mean":
            return embeddings.mean(dim=1)
        elif self.pooling == "cls":
            return embeddings[:, 0, :]  # CLS token
        elif self.pooling == "max":
            return embeddings.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    @torch.no_grad()
    def encode(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Encode images into embeddings.
        
        Args:
            images: List of PIL images or tensor of images
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize embeddings
            show_progress: Show progress bar
        
        Returns:
            Image embeddings [N, D]
        """
        self.model.eval()
        embeddings = []
        
        # Process in batches
        if isinstance(images, list):
            iterator = range(0, len(images), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Encoding images")
            
            for i in iterator:
                batch = images[i:i + batch_size]
                
                # Preprocess
                inputs = self.processor(
                    images=batch,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Pool embeddings
                batch_embeddings = self._pool_embeddings(outputs.last_hidden_state)
                embeddings.append(batch_embeddings.cpu())
            
            embeddings = torch.cat(embeddings, dim=0)
        
        else:  # Tensor input
            # Already preprocessed
            with torch.no_grad():
                outputs = self.model(pixel_values=images.to(self.device))
                embeddings = self._pool_embeddings(outputs.last_hidden_state).cpu()
        
        # Normalize
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        return embeddings
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for use in end-to-end training.
        
        Args:
            pixel_values: Preprocessed images [batch_size, 3, H, W]
        
        Returns:
            Image embeddings [batch_size, D]
        """
        outputs = self.model(pixel_values=pixel_values)
        return self._pool_embeddings(outputs.last_hidden_state)


def encode_image_folder(
    image_folder: Union[str, Path],
    output_path: Union[str, Path],
    model_name: str = ModelConfig.VIT_MODEL_NAME,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    pooling: str = "mean",
    normalize: bool = True,
    save_format: str = "dict"  # Options: dict, tensor
) -> Dict[str, np.ndarray]:
    """
    Encode all images in a folder and save embeddings.
    
    Args:
        image_folder: Path to folder containing images
        output_path: Path to save embeddings
        model_name: ViT model name
        batch_size: Batch size for processing
        device: Device to use
        pooling: Pooling strategy
        normalize: Whether to normalize embeddings
        save_format: Format to save ('dict' or 'tensor')
    
    Returns:
        Dictionary mapping image filenames to embeddings
    """
    image_folder = Path(image_folder)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize encoder
    encoder = ViTEncoder(model_name=model_name, device=device, pooling=pooling)
    
    # Get image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [
        f for f in sorted(image_folder.iterdir())
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")
    
    print(f"📸 Found {len(image_files)} images")
    
    # Process images in batches
    embeddings_dict = {}
    failed_images = []
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="Encoding batches"):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        batch_names = []
        
        # Load images
        for img_file in batch_files:
            try:
                img = Image.open(img_file).convert('RGB')
                batch_images.append(img)
                batch_names.append(img_file.name)
            except Exception as e:
                print(f"⚠️ Failed to load {img_file}: {e}")
                failed_images.append(img_file.name)
        
        if not batch_images:
            continue
        
        # Encode batch
        try:
            batch_embeddings = encoder.encode(
                batch_images,
                batch_size=len(batch_images),
                normalize=normalize,
                show_progress=False
            )
            
            # Store embeddings
            for name, emb in zip(batch_names, batch_embeddings):
                embeddings_dict[name] = emb.numpy()
        
        except Exception as e:
            print(f"⚠️ Failed to encode batch: {e}")
            failed_images.extend(batch_names)
    
    # Save embeddings
    if save_format == "dict":
        torch.save(embeddings_dict, output_path)
    elif save_format == "tensor":
        # Convert to tensor with aligned order
        names = sorted(embeddings_dict.keys())
        embeddings_tensor = torch.stack([
            torch.from_numpy(embeddings_dict[name]) for name in names
        ])
        torch.save({
            'embeddings': embeddings_tensor,
            'image_names': names
        }, output_path)
    
    print(f"\n✅ Saved embeddings to: {output_path}")
    print(f"   Total images encoded: {len(embeddings_dict)}")
    if failed_images:
        print(f"   ⚠️ Failed images: {len(failed_images)}")
    
    return embeddings_dict


def encode_single_image(
    image_path: Union[str, Path],
    model_name: str = ModelConfig.VIT_MODEL_NAME,
    device: Optional[torch.device] = None,
    normalize: bool = True
) -> torch.Tensor:
    """
    Encode a single image.
    
    Args:
        image_path: Path to image
        model_name: ViT model name
        device: Device to use
        normalize: Whether to normalize
    
    Returns:
        Image embedding [D]
    """
    encoder = ViTEncoder(model_name=model_name, device=device)
    
    image = Image.open(image_path).convert('RGB')
    embedding = encoder.encode([image], batch_size=1, normalize=normalize)
    
    return embedding.squeeze(0)


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Encode images using ViT")
    parser.add_argument(
        "--image_folder",
        type=str,
        default=str(IMAGES_DIR),
        help="Path to image folder"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(IMAGE_EMB_PATH),
        help="Path to save embeddings"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=ModelConfig.VIT_MODEL_NAME,
        help="ViT model name"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls", "max"],
        help="Pooling strategy"
    )
    
    args = parser.parse_args()
    
    # Encode images
    try:
        embeddings = encode_image_folder(
            image_folder=args.image_folder,
            output_path=args.output_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
            pooling=args.pooling,
            normalize=True
        )
        
        print(f"\n🎉 Successfully encoded {len(embeddings)} images!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
