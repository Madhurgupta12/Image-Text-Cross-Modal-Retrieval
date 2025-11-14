"""
Embedding visualization using t-SNE and UMAP.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Tuple
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from src.utils.config import VISUALIZATIONS_DIR, EvalConfig


def visualize_embeddings(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    method: str = 'tsne',
    n_samples: Optional[int] = None,
    save_path: Optional[Path] = None,
    title: str = "Embedding Visualization",
    figsize: Tuple[int, int] = (12, 8),
    random_state: int = 42
):
    """
    Visualize image and text embeddings in 2D space.
    
    Args:
        image_embeddings: Image embeddings [N, D]
        text_embeddings: Text embeddings [N, D]
        method: Dimensionality reduction method ('tsne' or 'umap')
        n_samples: Number of samples to visualize (None for all)
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
        random_state: Random seed
    """
    # Convert to numpy
    if isinstance(image_embeddings, torch.Tensor):
        image_embeddings = image_embeddings.cpu().numpy()
    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.cpu().numpy()
    
    # Sample if needed
    if n_samples and n_samples < len(image_embeddings):
        indices = np.random.choice(len(image_embeddings), n_samples, replace=False)
        image_embeddings = image_embeddings[indices]
        text_embeddings = text_embeddings[indices]
    
    # Combine embeddings
    all_embeddings = np.vstack([image_embeddings, text_embeddings])
    labels = ['Image'] * len(image_embeddings) + ['Text'] * len(text_embeddings)
    
    # Reduce dimensions
    print(f"📊 Reducing dimensions using {method.upper()}...")
    
    if method.lower() == 'tsne':
        reducer = TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=min(EvalConfig.TSNE_PERPLEXITY, len(all_embeddings) - 1)
        )
        embeddings_2d = reducer.fit_transform(all_embeddings)
    
    elif method.lower() == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")
        
        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=min(EvalConfig.UMAP_N_NEIGHBORS, len(all_embeddings) - 1)
        )
        embeddings_2d = reducer.fit_transform(all_embeddings)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Split back into image and text
    img_2d = embeddings_2d[:len(image_embeddings)]
    txt_2d = embeddings_2d[len(image_embeddings):]
    
    # Plot
    plt.scatter(img_2d[:, 0], img_2d[:, 1], 
                c='blue', label='Images', alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    plt.scatter(txt_2d[:, 0], txt_2d[:, 1], 
                c='red', label='Texts', alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    # Draw connections between matched pairs
    for i in range(min(100, len(img_2d))):  # Limit connections for clarity
        plt.plot([img_2d[i, 0], txt_2d[i, 0]], 
                [img_2d[i, 1], txt_2d[i, 1]], 
                'gray', alpha=0.2, linewidth=0.5)
    
    plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save or show
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_similarity_matrix(
    similarity_matrix: torch.Tensor,
    save_path: Optional[Path] = None,
    title: str = "Similarity Matrix",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis'
):
    """
    Plot similarity matrix heatmap.
    
    Args:
        similarity_matrix: Similarity matrix [N, M]
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
        cmap: Colormap
    """
    if isinstance(similarity_matrix, torch.Tensor):
        similarity_matrix = similarity_matrix.cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    sns.heatmap(
        similarity_matrix,
        cmap=cmap,
        cbar=True,
        square=False,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Similarity Score'}
    )
    
    plt.xlabel('Text Embeddings', fontsize=12)
    plt.ylabel('Image Embeddings', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Similarity matrix saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_embedding_distribution(
    embeddings: torch.Tensor,
    save_path: Optional[Path] = None,
    title: str = "Embedding Distribution",
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot embedding distribution statistics.
    
    Args:
        embeddings: Embeddings [N, D]
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Norm distribution
    norms = np.linalg.norm(embeddings, axis=1)
    axes[0].hist(norms, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('L2 Norm', fontsize=10)
    axes[0].set_ylabel('Frequency', fontsize=10)
    axes[0].set_title('Embedding Norms', fontsize=12)
    axes[0].grid(alpha=0.3)
    
    # Dimension-wise mean
    dim_means = embeddings.mean(axis=0)
    axes[1].plot(dim_means, color='green', linewidth=2)
    axes[1].set_xlabel('Dimension', fontsize=10)
    axes[1].set_ylabel('Mean Value', fontsize=10)
    axes[1].set_title('Dimension-wise Mean', fontsize=12)
    axes[1].grid(alpha=0.3)
    
    # Dimension-wise variance
    dim_vars = embeddings.var(axis=0)
    axes[2].plot(dim_vars, color='red', linewidth=2)
    axes[2].set_xlabel('Dimension', fontsize=10)
    axes[2].set_ylabel('Variance', fontsize=10)
    axes[2].set_title('Dimension-wise Variance', fontsize=12)
    axes[2].grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Distribution plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# Example usage
if __name__ == "__main__":
    print("🎨 Testing visualization functions...")
    
    # Create sample embeddings
    n_samples = 500
    dim = 512
    
    img_emb = torch.randn(n_samples, dim)
    txt_emb = torch.randn(n_samples, dim)
    
    # Normalize
    img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=-1)
    txt_emb = torch.nn.functional.normalize(txt_emb, p=2, dim=-1)
    
    # Test t-SNE visualization
    print("\n1. Testing t-SNE visualization...")
    visualize_embeddings(
        img_emb, txt_emb,
        method='tsne',
        n_samples=200,
        save_path=VISUALIZATIONS_DIR / "test_tsne.png"
    )
    
    # Test UMAP visualization
    if UMAP_AVAILABLE:
        print("\n2. Testing UMAP visualization...")
        visualize_embeddings(
            img_emb, txt_emb,
            method='umap',
            n_samples=200,
            save_path=VISUALIZATIONS_DIR / "test_umap.png"
        )
    
    # Test similarity matrix plot
    print("\n3. Testing similarity matrix plot...")
    similarity = img_emb[:50] @ txt_emb[:50].T
    plot_similarity_matrix(
        similarity,
        save_path=VISUALIZATIONS_DIR / "test_similarity.png"
    )
    
    # Test embedding distribution
    print("\n4. Testing embedding distribution plot...")
    plot_embedding_distribution(
        img_emb,
        save_path=VISUALIZATIONS_DIR / "test_distribution.png"
    )
    
    print("\n✅ All visualization tests passed!")
