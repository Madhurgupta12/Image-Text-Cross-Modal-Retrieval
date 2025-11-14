"""
Projection head for mapping image and text embeddings into a shared latent space.
Supports multi-layer projections, residual connections, and various normalization techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from src.utils.config import ModelConfig


class MLPProjection(nn.Module):
    """
    Multi-layer perceptron projection with configurable architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_batch_norm: bool = True,
        use_dropout: bool = True,
        dropout_rate: float = 0.1,
        activation: str = "relu",
        residual: bool = False
    ):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output embedding dimension
            use_batch_norm: Whether to use batch normalization
            use_dropout: Whether to use dropout
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'gelu', 'tanh')
            residual: Whether to use residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = residual and (input_dim == output_dim)
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            # Dropout
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, input_dim]
        
        Returns:
            Projected embeddings [batch_size, output_dim]
        """
        out = self.projection(x)
        
        # Residual connection
        if self.residual:
            out = out + x
        
        return out


class ProjectionModel(nn.Module):
    """
    Dual projection model for image and text embeddings.
    Maps both modalities into a shared latent space for cross-modal retrieval.
    """
    
    def __init__(
        self,
        img_dim: int = ModelConfig.VIT_OUTPUT_DIM,
        text_dim: int = ModelConfig.TEXT_OUTPUT_DIM,
        shared_dim: int = ModelConfig.SHARED_DIM,
        hidden_dims: Optional[List[int]] = None,
        use_batch_norm: bool = ModelConfig.USE_BATCH_NORM,
        use_dropout: bool = ModelConfig.USE_DROPOUT,
        dropout_rate: float = ModelConfig.DROPOUT_RATE,
        activation: str = ModelConfig.ACTIVATION,
        residual: bool = False
    ):
        """
        Args:
            img_dim: Image embedding dimension (from ViT)
            text_dim: Text embedding dimension (from LLM)
            shared_dim: Shared latent space dimension
            hidden_dims: Hidden layer dimensions for projection
            use_batch_norm: Whether to use batch normalization
            use_dropout: Whether to use dropout
            dropout_rate: Dropout probability
            activation: Activation function
            residual: Whether to use residual connections
        """
        super().__init__()
        
        self.img_dim = img_dim
        self.text_dim = text_dim
        self.shared_dim = shared_dim
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = ModelConfig.PROJECTION_HIDDEN_DIMS
        
        # Image projection network
        self.image_proj = MLPProjection(
            input_dim=img_dim,
            hidden_dims=hidden_dims,
            output_dim=shared_dim,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            residual=residual
        )
        
        # Text projection network
        self.text_proj = MLPProjection(
            input_dim=text_dim,
            hidden_dims=hidden_dims,
            output_dim=shared_dim,
            use_batch_norm=use_batch_norm,
            use_dropout=use_dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            residual=residual
        )
        
        # Learnable temperature for contrastive learning
        if ModelConfig.LEARNABLE_TEMPERATURE:
            self.temperature = nn.Parameter(
                torch.tensor(ModelConfig.INITIAL_TEMPERATURE).log()
            )
        else:
            self.register_buffer(
                'temperature',
                torch.tensor(ModelConfig.INITIAL_TEMPERATURE).log()
            )
    
    def forward(
        self,
        img_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        normalize: bool = True
    ) -> tuple:
        """
        Project image and text embeddings into shared space.
        
        Args:
            img_embeddings: Image embeddings [batch_size, img_dim]
            text_embeddings: Text embeddings [batch_size, text_dim]
            normalize: Whether to L2-normalize the projections
        
        Returns:
            Tuple of (projected_images, projected_texts)
                projected_images: [batch_size, shared_dim]
                projected_texts: [batch_size, shared_dim]
        """
        # Project to shared space
        img_proj = self.image_proj(img_embeddings)
        text_proj = self.text_proj(text_embeddings)
        
        # Normalize
        if normalize:
            img_proj = F.normalize(img_proj, p=2, dim=-1)
            text_proj = F.normalize(text_proj, p=2, dim=-1)
        
        return img_proj, text_proj
    
    def encode_image(self, img_embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode image embeddings only"""
        img_proj = self.image_proj(img_embeddings)
        if normalize:
            img_proj = F.normalize(img_proj, p=2, dim=-1)
        return img_proj
    
    def encode_text(self, text_embeddings: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode text embeddings only"""
        text_proj = self.text_proj(text_embeddings)
        if normalize:
            text_proj = F.normalize(text_proj, p=2, dim=-1)
        return text_proj
    
    def get_temperature(self) -> float:
        """Get current temperature value"""
        return self.temperature.exp().item()
    
    def compute_similarity(
        self,
        img_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity matrix between images and texts.
        
        Args:
            img_embeddings: Image embeddings [N, img_dim]
            text_embeddings: Text embeddings [M, text_dim]
        
        Returns:
            Similarity matrix [N, M]
        """
        # Project
        img_proj = self.encode_image(img_embeddings, normalize=True)
        text_proj = self.encode_text(text_embeddings, normalize=True)
        
        # Compute cosine similarity
        similarity = img_proj @ text_proj.T
        
        # Scale by temperature
        similarity = similarity / self.temperature.exp()
        
        return similarity


class SimpleProjectionHead(nn.Module):
    """
    Simple single-layer projection head for baseline comparison.
    """
    
    def __init__(
        self,
        img_dim: int = ModelConfig.VIT_OUTPUT_DIM,
        text_dim: int = ModelConfig.TEXT_OUTPUT_DIM,
        shared_dim: int = ModelConfig.SHARED_DIM
    ):
        """
        Args:
            img_dim: Image embedding dimension
            text_dim: Text embedding dimension
            shared_dim: Shared latent space dimension
        """
        super().__init__()
        
        self.img_dim = img_dim
        self.text_dim = text_dim
        self.shared_dim = shared_dim
        
        # Single linear layers
        self.image_proj = nn.Linear(img_dim, shared_dim)
        self.text_proj = nn.Linear(text_dim, shared_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.image_proj.weight)
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.zeros_(self.image_proj.bias)
        nn.init.zeros_(self.text_proj.bias)
    
    def forward(
        self,
        img_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        normalize: bool = True
    ) -> tuple:
        """
        Project image and text embeddings into shared space.
        
        Args:
            img_embeddings: Image embeddings [batch_size, img_dim]
            text_embeddings: Text embeddings [batch_size, text_dim]
            normalize: Whether to L2-normalize the projections
        
        Returns:
            Tuple of (projected_images, projected_texts)
        """
        img_proj = self.image_proj(img_embeddings)
        text_proj = self.text_proj(text_embeddings)
        
        if normalize:
            img_proj = F.normalize(img_proj, p=2, dim=-1)
            text_proj = F.normalize(text_proj, p=2, dim=-1)
        
        return img_proj, text_proj


# Factory function
def create_projection_model(model_type: str = "mlp", **kwargs) -> nn.Module:
    """
    Create a projection model.
    
    Args:
        model_type: Type of projection model ('simple', 'mlp')
        **kwargs: Additional arguments for the model
    
    Returns:
        Projection model instance
    """
    if model_type == "simple":
        return SimpleProjectionHead(**kwargs)
    elif model_type == "mlp":
        return ProjectionModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test projection models
    print("Testing projection models...")
    
    # Create sample data
    batch_size = 32
    img_emb = torch.randn(batch_size, ModelConfig.VIT_OUTPUT_DIM)
    text_emb = torch.randn(batch_size, ModelConfig.TEXT_OUTPUT_DIM)
    
    # Test MLP projection
    print("\n1. Testing MLP Projection Model:")
    model = ProjectionModel()
    img_proj, text_proj = model(img_emb, text_emb)
    print(f"   Image projection: {img_proj.shape}")
    print(f"   Text projection: {text_proj.shape}")
    print(f"   Temperature: {model.get_temperature():.4f}")
    
    # Test similarity computation
    similarity = model.compute_similarity(img_emb, text_emb)
    print(f"   Similarity matrix: {similarity.shape}")
    
    # Test simple projection
    print("\n2. Testing Simple Projection Model:")
    simple_model = SimpleProjectionHead()
    img_proj, text_proj = simple_model(img_emb, text_emb)
    print(f"   Image projection: {img_proj.shape}")
    print(f"   Text projection: {text_proj.shape}")
    
    print("\n✅ All tests passed!")
