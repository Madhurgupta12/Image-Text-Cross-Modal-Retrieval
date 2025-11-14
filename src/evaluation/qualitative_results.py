"""
Qualitative results visualization for image-text retrieval.
Show retrieval examples with images and captions.
"""

import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from src.utils.config import VISUALIZATIONS_DIR, RESULTS_DIR


def visualize_retrieval_results(
    query_image_path: str,
    retrieved_image_paths: List[str],
    retrieved_captions: List[str],
    similarity_scores: List[float],
    save_path: Optional[Path] = None,
    title: str = "Image-to-Text Retrieval Results",
    figsize: Tuple[int, int] = (15, 10)
):
    """
    Visualize image-to-text retrieval results.
    
    Args:
        query_image_path: Path to query image
        retrieved_image_paths: Paths to retrieved images
        retrieved_captions: Retrieved captions
        similarity_scores: Similarity scores
        save_path: Path to save visualization
        title: Plot title
        figsize: Figure size
    """
    n_results = len(retrieved_captions)
    n_cols = min(3, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols + 1  # +1 for query
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_results > 1 else [axes]
    
    # Load and display query image
    query_img = Image.open(query_image_path).convert('RGB')
    axes[0].imshow(query_img)
    axes[0].set_title('Query Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Hide remaining axes in first row
    for i in range(1, n_cols):
        axes[i].axis('off')
    
    # Display retrieved results
    for i, (img_path, caption, score) in enumerate(zip(retrieved_image_paths, retrieved_captions, similarity_scores)):
        ax_idx = n_cols + i
        if ax_idx >= len(axes):
            break
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
            axes[ax_idx].imshow(img)
        except (IOError, FileNotFoundError) as e:
            # Show placeholder if image can't be loaded
            axes[ax_idx].text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
        # Add caption and score
        title_text = f'Rank {i+1} (Score: {score:.3f})\n{caption[:80]}...' if len(caption) > 80 else f'Rank {i+1} (Score: {score:.3f})\n{caption}'
        axes[ax_idx].set_title(title_text, fontsize=10, wrap=True)
        axes[ax_idx].axis('off')
    
    # Hide unused axes
    for i in range(n_cols + len(retrieved_captions), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Retrieval results saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_text_to_image_retrieval(
    query_text: str,
    retrieved_image_paths: List[str],
    similarity_scores: List[float],
    save_path: Optional[Path] = None,
    title: str = "Text-to-Image Retrieval Results",
    figsize: Tuple[int, int] = (15, 8)
):
    """
    Visualize text-to-image retrieval results.
    
    Args:
        query_text: Query text
        retrieved_image_paths: Paths to retrieved images
        similarity_scores: Similarity scores
        save_path: Path to save visualization
        title: Plot title
        figsize: Figure size
    """
    n_results = len(retrieved_image_paths)
    n_cols = min(5, n_results)
    n_rows = (n_results + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_results == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Display retrieved images
    for i, (img_path, score) in enumerate(zip(retrieved_image_paths, similarity_scores)):
        if i >= len(axes):
            break
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
        except (IOError, FileNotFoundError) as e:
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
        axes[i].set_title(f'Rank {i+1}\nScore: {score:.3f}', fontsize=10)
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(len(retrieved_image_paths), len(axes)):
        axes[i].axis('off')
    
    # Add query text as title
    query_display = query_text if len(query_text) < 100 else query_text[:100] + '...'
    plt.suptitle(f'{title}\n\nQuery: "{query_display}"', fontsize=12, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Retrieval results saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_retrieval_grid(
    image_paths: List[str],
    captions: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    n_cols: int = 4,
    save_path: Optional[Path] = None,
    title: str = "Retrieval Results",
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Create a grid of images with optional captions and scores.
    
    Args:
        image_paths: List of image paths
        captions: Optional captions for each image
        scores: Optional similarity scores
        n_cols: Number of columns in grid
        save_path: Path to save visualization
        title: Plot title
        figsize: Figure size (auto-calculated if None)
    """
    n_images = len(image_paths)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, img_path in enumerate(image_paths):
        if i >= len(axes):
            break
        
        # Load and display image
        try:
            img = Image.open(img_path).convert('RGB')
            axes[i].imshow(img)
        except (IOError, FileNotFoundError) as e:
            axes[i].text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
        # Add title with caption and/or score
        title_parts = []
        if scores is not None and i < len(scores):
            title_parts.append(f'Score: {scores[i]:.3f}')
        if captions is not None and i < len(captions):
            caption_short = captions[i][:50] + '...' if len(captions[i]) > 50 else captions[i]
            title_parts.append(caption_short)
        
        if title_parts:
            axes[i].set_title('\n'.join(title_parts), fontsize=9)
        
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Grid saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_rank_distribution(
    ranks: List[int],
    save_path: Optional[Path] = None,
    title: str = "Rank Distribution of Correct Matches",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot distribution of ranks where correct matches were found.
    
    Args:
        ranks: List of ranks (1-indexed)
        save_path: Path to save plot
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create histogram
    plt.hist(ranks, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add statistics
    median_rank = np.median(ranks)
    mean_rank = np.mean(ranks)
    
    plt.axvline(median_rank, color='red', linestyle='--', linewidth=2, label=f'Median: {median_rank:.1f}')
    plt.axvline(mean_rank, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_rank:.1f}')
    
    plt.xlabel('Rank', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Rank distribution saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# Example usage
if __name__ == "__main__":
    print("🎨 Testing qualitative results visualization...")
    
    # Note: This is a demonstration. In practice, you would use real image paths
    print("⚠️ This module requires actual image files to display results.")
    print("   Use the functions in your inference pipeline with real data.")
    
    print("\n✅ Qualitative results module loaded successfully!")
