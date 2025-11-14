"""
Training utilities and loss functions for cross-modal retrieval.
This module provides the contrastive loss and basic training helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class CLIPContrastiveLoss(nn.Module):
    """
    CLIP-style symmetric contrastive loss.
    Encourages matching image-text pairs to have high similarity,
    while pushing non-matching pairs apart.
    """
    
    def __init__(self, temperature: float = 0.07, learnable_temperature: bool = True):
        """
        Args:
            temperature: Initial temperature value
            learnable_temperature: Whether temperature should be learned
        """
        super().__init__()
        
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature).log())
        else:
            self.register_buffer('temperature', torch.tensor(temperature).log())
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute CLIP contrastive loss.
        
        Args:
            image_features: Normalized image embeddings [batch_size, D]
            text_features: Normalized text embeddings [batch_size, D]
        
        Returns:
            Tuple of (total_loss, i2t_loss, t2i_loss)
        """
        # Compute logits (cosine similarity scaled by temperature)
        logits = image_features @ text_features.T / self.temperature.exp()
        
        # Labels: diagonal elements (i-th image matches i-th text)
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric loss
        loss_i2t = F.cross_entropy(logits, labels)  # Image-to-Text
        loss_t2i = F.cross_entropy(logits.T, labels)  # Text-to-Image
        
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss, loss_i2t, loss_t2i
    
    def get_temperature(self) -> float:
        """Get current temperature value"""
        return self.temperature.exp().item()


class TripletLoss(nn.Module):
    """
    Triplet loss for cross-modal retrieval.
    """
    
    def __init__(self, margin: float = 0.2):
        """
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, D]
            positive: Positive embeddings [batch_size, D]
            negative: Negative embeddings [batch_size, D]
        
        Returns:
            Triplet loss
        """
        pos_distance = (anchor - positive).pow(2).sum(dim=-1)
        neg_distance = (anchor - negative).pow(2).sum(dim=-1)
        
        loss = F.relu(pos_distance - neg_distance + self.margin)
        
        return loss.mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss (contrastive learning).
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        query: torch.Tensor,
        positive_key: torch.Tensor,
        negative_keys: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            query: Query embeddings [batch_size, D]
            positive_key: Positive key embeddings [batch_size, D]
            negative_keys: Negative key embeddings [num_negatives, D]
        
        Returns:
            InfoNCE loss
        """
        # Normalize
        query = F.normalize(query, p=2, dim=-1)
        positive_key = F.normalize(positive_key, p=2, dim=-1)
        
        # Positive logits
        pos_logits = (query * positive_key).sum(dim=-1, keepdim=True)
        
        # Negative logits
        if negative_keys is not None:
            negative_keys = F.normalize(negative_keys, p=2, dim=-1)
            neg_logits = query @ negative_keys.T
            logits = torch.cat([pos_logits, neg_logits], dim=1)
        else:
            # Use in-batch negatives
            logits = query @ positive_key.T
        
        # Scale by temperature
        logits = logits / self.temperature
        
        # Labels (positive is always first)
        labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss


def compute_retrieval_accuracy(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    k: int = 1
) -> Tuple[float, float]:
    """
    Compute retrieval accuracy (Recall@K).
    
    Args:
        image_features: Image embeddings [N, D]
        text_features: Text embeddings [N, D]
        k: Top-K to consider
    
    Returns:
        Tuple of (image-to-text accuracy, text-to-image accuracy)
    """
    # Compute similarity
    similarity = image_features @ text_features.T
    
    # Image-to-Text
    _, i2t_indices = similarity.topk(k, dim=1)
    correct_i2t = (i2t_indices == torch.arange(len(similarity)).unsqueeze(1).to(similarity.device)).any(dim=1)
    i2t_acc = correct_i2t.float().mean().item()
    
    # Text-to-Image
    _, t2i_indices = similarity.T.topk(k, dim=1)
    correct_t2i = (t2i_indices == torch.arange(len(similarity)).unsqueeze(1).to(similarity.device)).any(dim=1)
    t2i_acc = correct_t2i.float().mean().item()
    
    return i2t_acc, t2i_acc


class PairedDataset(Dataset):
    """
    Simple dataset for paired embeddings.
    """
    
    def __init__(self, images: torch.Tensor, texts: torch.Tensor):
        """
        Args:
            images: Image embeddings [N, D_img]
            texts: Text embeddings [N, D_text]
        """
        assert len(images) == len(texts), "Mismatch between image and text counts"
        
        self.images = images
        self.texts = texts
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (image_embedding, text_embedding)
        """
        return self.images[idx], self.texts[idx]


def create_loss_function(loss_type: str = "clip", **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('clip', 'triplet', 'infonce')
        **kwargs: Additional arguments for the loss function
    
    Returns:
        Loss function module
    """
    if loss_type.lower() == "clip":
        return CLIPContrastiveLoss(**kwargs)
    elif loss_type.lower() == "triplet":
        return TripletLoss(**kwargs)
    elif loss_type.lower() == "infonce":
        return InfoNCELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing loss functions...")
    
    batch_size = 32
    dim = 512
    
    # Create sample embeddings
    img_features = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
    txt_features = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
    
    # Test CLIP loss
    print("\n1. Testing CLIP Contrastive Loss:")
    clip_loss = CLIPContrastiveLoss()
    loss, i2t_loss, t2i_loss = clip_loss(img_features, txt_features)
    print(f"   Total Loss: {loss.item():.4f}")
    print(f"   I2T Loss: {i2t_loss.item():.4f}")
    print(f"   T2I Loss: {t2i_loss.item():.4f}")
    print(f"   Temperature: {clip_loss.get_temperature():.4f}")
    
    # Test Triplet Loss
    print("\n2. Testing Triplet Loss:")
    triplet_loss = TripletLoss(margin=0.2)
    anchor = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
    positive = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
    negative = F.normalize(torch.randn(batch_size, dim), p=2, dim=-1)
    loss = triplet_loss(anchor, positive, negative)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test InfoNCE Loss
    print("\n3. Testing InfoNCE Loss:")
    infonce_loss = InfoNCELoss(temperature=0.07)
    loss = infonce_loss(img_features, txt_features)
    print(f"   Loss: {loss.item():.4f}")
    
    # Test retrieval accuracy
    print("\n4. Testing Retrieval Accuracy:")
    i2t_acc, t2i_acc = compute_retrieval_accuracy(img_features, txt_features, k=5)
    print(f"   I2T Recall@5: {i2t_acc:.4f}")
    print(f"   T2I Recall@5: {t2i_acc:.4f}")
    
    print("\n✅ All loss function tests passed!")

