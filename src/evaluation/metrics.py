"""
Comprehensive evaluation metrics for image-text cross-modal retrieval.
Includes Recall@K, Mean Average Precision (mAP), Mean Reciprocal Rank (MRR), and more.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

from src.utils.config import EvalConfig


def compute_similarity_matrix(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    normalize: bool = True,
    batch_size: int = 1000
) -> torch.Tensor:
    """
    Compute pairwise similarity matrix with memory-efficient batching.
    
    Args:
        query_embeddings: Query embeddings [N, D]
        gallery_embeddings: Gallery embeddings [M, D]
        normalize: Whether to L2-normalize embeddings
        batch_size: Batch size for computing similarities (to avoid OOM)
    
    Returns:
        Similarity matrix [N, M]
    """
    if normalize:
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        gallery_embeddings = F.normalize(gallery_embeddings, p=2, dim=-1)
    
    N = query_embeddings.size(0)
    M = gallery_embeddings.size(0)
    
    # If dataset is small, compute directly
    if N * M < 10_000_000:  # Less than 10M elements (~40MB for float32)
        return query_embeddings @ gallery_embeddings.T
    
    # Otherwise, compute in batches to avoid OOM
    similarity_matrix = torch.zeros(N, M, dtype=query_embeddings.dtype)
    
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        batch_similarities = query_embeddings[i:end_i] @ gallery_embeddings.T
        similarity_matrix[i:end_i] = batch_similarities
    
    return similarity_matrix


def recall_at_k(
    similarity_matrix: torch.Tensor,
    k: int = 1,
    return_ranks: bool = False
) -> Tuple[float, Optional[torch.Tensor]]:
    """
    Compute Recall@K metric.
    
    Args:
        similarity_matrix: Similarity matrix [N, M] where N=M for cross-modal retrieval
        k: Top-K to consider
        return_ranks: Whether to return ranks of correct matches
    
    Returns:
        Recall@K score and optionally ranks
    """
    N = similarity_matrix.size(0)
    
    # Get top-k indices
    _, indices = similarity_matrix.topk(k, dim=1)
    
    # Ground truth: diagonal elements (i-th query matches i-th gallery item)
    correct = torch.arange(N, device=similarity_matrix.device).unsqueeze(1)
    
    # Check if correct match is in top-k
    matches = (indices == correct).any(dim=1)
    recall = matches.float().mean().item()
    
    if return_ranks:
        # Get rank of correct match for each query
        _, sorted_indices = similarity_matrix.sort(dim=1, descending=True)
        ranks = (sorted_indices == correct).nonzero(as_tuple=True)[1]
        return recall, ranks
    
    return recall, None


def mean_average_precision(
    similarity_matrix: torch.Tensor
) -> float:
    """
    Compute Mean Average Precision (mAP).
    
    Args:
        similarity_matrix: Similarity matrix [N, M]
    
    Returns:
        mAP score
    """
    N = similarity_matrix.size(0)
    
    # Convert to numpy for sklearn
    sim_np = similarity_matrix.cpu().numpy()
    
    # Ground truth: one-hot encoded diagonal
    gt = np.eye(N)
    
    # Compute AP for each query
    aps = []
    for i in range(N):
        ap = average_precision_score(gt[i], sim_np[i])
        aps.append(ap)
    
    return np.mean(aps)


def mean_reciprocal_rank(
    similarity_matrix: torch.Tensor
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        similarity_matrix: Similarity matrix [N, M]
    
    Returns:
        MRR score
    """
    N = similarity_matrix.size(0)
    
    # Get sorted indices
    _, sorted_indices = similarity_matrix.sort(dim=1, descending=True)
    
    # Ground truth: diagonal
    correct = torch.arange(N, device=similarity_matrix.device).unsqueeze(1)
    
    # Find rank of correct match (1-indexed)
    ranks = (sorted_indices == correct).nonzero(as_tuple=True)[1].float() + 1
    
    # Compute reciprocal ranks
    reciprocal_ranks = 1.0 / ranks
    
    return reciprocal_ranks.mean().item()


def median_rank(
    similarity_matrix: torch.Tensor
) -> float:
    """
    Compute median rank of correct matches.
    
    Args:
        similarity_matrix: Similarity matrix [N, M]
    
    Returns:
        Median rank
    """
    N = similarity_matrix.size(0)
    
    # Get sorted indices
    _, sorted_indices = similarity_matrix.sort(dim=1, descending=True)
    
    # Ground truth: diagonal
    correct = torch.arange(N, device=similarity_matrix.device).unsqueeze(1)
    
    # Find rank of correct match (1-indexed)
    ranks = (sorted_indices == correct).nonzero(as_tuple=True)[1].float() + 1
    
    return ranks.median().item()


def evaluate_retrieval(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    k_values: List[int] = EvalConfig.RECALL_AT_K,
    compute_map: bool = EvalConfig.COMPUTE_MAP,
    compute_mrr: bool = EvalConfig.COMPUTE_MRR,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Comprehensive retrieval evaluation.
    
    Args:
        query_embeddings: Query embeddings [N, D]
        gallery_embeddings: Gallery embeddings [M, D]
        k_values: List of K values for Recall@K
        compute_map: Whether to compute mAP
        compute_mrr: Whether to compute MRR
        normalize: Whether to normalize embeddings
    
    Returns:
        Dictionary of metrics
    """
    # Compute similarity matrix
    similarity = compute_similarity_matrix(
        query_embeddings, gallery_embeddings, normalize=normalize
    )
    
    metrics = {}
    
    # Recall@K
    for k in k_values:
        recall, _ = recall_at_k(similarity, k=k)
        metrics[f'recall@{k}'] = recall
    
    # Mean Average Precision
    if compute_map:
        metrics['mAP'] = mean_average_precision(similarity)
    
    # Mean Reciprocal Rank
    if compute_mrr:
        metrics['MRR'] = mean_reciprocal_rank(similarity)
    
    # Median Rank
    metrics['median_rank'] = median_rank(similarity)
    
    return metrics


def evaluate_bidirectional_retrieval(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    k_values: List[int] = EvalConfig.RECALL_AT_K,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Evaluate both image-to-text and text-to-image retrieval.
    
    Args:
        image_embeddings: Image embeddings [N, D]
        text_embeddings: Text embeddings [N, D]
        k_values: List of K values for Recall@K
        normalize: Whether to normalize embeddings
    
    Returns:
        Dictionary of metrics for both directions
    """
    metrics = {}
    
    # Image-to-Text retrieval
    i2t_metrics = evaluate_retrieval(
        image_embeddings, text_embeddings, k_values=k_values, normalize=normalize
    )
    for key, value in i2t_metrics.items():
        metrics[f'i2t_{key}'] = value
    
    # Text-to-Image retrieval
    t2i_metrics = evaluate_retrieval(
        text_embeddings, image_embeddings, k_values=k_values, normalize=normalize
    )
    for key, value in t2i_metrics.items():
        metrics[f't2i_{key}'] = value
    
    # Average metrics
    for key in i2t_metrics.keys():
        metrics[f'avg_{key}'] = (i2t_metrics[key] + t2i_metrics[key]) / 2
    
    return metrics


def compute_embedding_statistics(
    embeddings: torch.Tensor
) -> Dict[str, float]:
    """
    Compute statistics about embeddings quality.
    
    Args:
        embeddings: Embeddings [N, D]
    
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Norm statistics
    norms = embeddings.norm(dim=-1)
    stats['mean_norm'] = norms.mean().item()
    stats['std_norm'] = norms.std().item()
    stats['min_norm'] = norms.min().item()
    stats['max_norm'] = norms.max().item()
    
    # Pairwise similarity statistics
    similarity = embeddings @ embeddings.T
    
    # Remove diagonal (self-similarity)
    mask = ~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
    off_diagonal = similarity[mask]
    
    stats['mean_similarity'] = off_diagonal.mean().item()
    stats['std_similarity'] = off_diagonal.std().item()
    stats['min_similarity'] = off_diagonal.min().item()
    stats['max_similarity'] = off_diagonal.max().item()
    
    # Dimension utilization (variance per dimension)
    dim_variance = embeddings.var(dim=0)
    stats['mean_dim_variance'] = dim_variance.mean().item()
    stats['std_dim_variance'] = dim_variance.std().item()
    
    return stats


def compute_alignment_uniformity(
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    normalize: bool = True,
    temperature: float = 2.0
) -> Tuple[float, float]:
    """
    Compute alignment and uniformity metrics from
    "Understanding Contrastive Representation Learning through Alignment and Uniformity"
    
    Args:
        image_embeddings: Image embeddings [N, D]
        text_embeddings: Text embeddings [N, D]
        normalize: Whether to normalize embeddings
        temperature: Temperature parameter
    
    Returns:
        Tuple of (alignment, uniformity)
    """
    if normalize:
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
    
    # Alignment: how well matched pairs align
    alignment = (image_embeddings - text_embeddings).norm(dim=-1).pow(2).mean()
    
    # Uniformity: how uniformly distributed embeddings are
    def uniformity(embeddings):
        sim_matrix = embeddings @ embeddings.T
        # Remove diagonal
        n = embeddings.size(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
        similarities = sim_matrix[mask]
        return torch.exp(-temperature * similarities.pow(2)).mean().log()
    
    img_uniformity = uniformity(image_embeddings)
    text_uniformity = uniformity(text_embeddings)
    avg_uniformity = (img_uniformity + text_uniformity) / 2
    
    return alignment.item(), avg_uniformity.item()


def print_metrics(metrics: Dict[str, float], title: str = "Evaluation Metrics"):
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics
    """
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:8.4f}")
        else:
            print(f"{key:30s}: {value}")
    
    print("=" * 60 + "\n")


# Example usage and testing
if __name__ == "__main__":
    print("Testing evaluation metrics...")
    
    # Create sample embeddings
    N = 100
    D = 512
    
    image_emb = torch.randn(N, D)
    text_emb = torch.randn(N, D)
    
    # Normalize
    image_emb = F.normalize(image_emb, p=2, dim=-1)
    text_emb = F.normalize(text_emb, p=2, dim=-1)
    
    # Evaluate
    print("\n1. Testing bidirectional retrieval...")
    metrics = evaluate_bidirectional_retrieval(image_emb, text_emb, k_values=[1, 5, 10])
    print_metrics(metrics, "Retrieval Metrics")
    
    # Embedding statistics
    print("\n2. Testing embedding statistics...")
    stats = compute_embedding_statistics(image_emb)
    print_metrics(stats, "Embedding Statistics")
    
    # Alignment and uniformity
    print("\n3. Testing alignment and uniformity...")
    alignment, uniformity = compute_alignment_uniformity(image_emb, text_emb)
    print(f"Alignment: {alignment:.4f}")
    print(f"Uniformity: {uniformity:.4f}")
    
    print("\n✅ All metric tests passed!")
