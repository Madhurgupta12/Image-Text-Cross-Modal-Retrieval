# srs/fusion/projection_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionModel(nn.Module):
    """
    Maps image embeddings and text embeddings into a shared latent space.
    This is the core module for unified embeddings.
    """

    def __init__(self, img_dim=768, text_dim=384, hidden_dim=512):
        super().__init__()

        # Linear layer that maps ViT image embeddings → shared space
        self.image_proj = nn.Linear(img_dim, hidden_dim)

        # Linear layer that maps MiniLM text embeddings → shared space
        self.text_proj = nn.Linear(text_dim, hidden_dim)

    def forward(self, img_embeddings, text_embeddings):
        """
        Inputs:
            img_embeddings:  [batch_size, img_dim]
            text_embeddings: [batch_size, text_dim]

        Outputs:
            img_projected:   [batch_size, hidden_dim]
            text_projected:  [batch_size, hidden_dim]
        """

        # Apply linear layers + L2-normalize
        img_out = F.normalize(self.image_proj(img_embeddings), dim=-1)
        txt_out = F.normalize(self.text_proj(text_embeddings), dim=-1)

        return img_out, txt_out
