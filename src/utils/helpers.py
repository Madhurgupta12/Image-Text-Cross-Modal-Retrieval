"""
Helper utilities for the image-text retrieval system.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
from datetime import datetime
import shutil


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    save_path: Path,
    scheduler: Optional[Any] = None,
    is_best: bool = False
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Training/validation metrics
        save_path: Path to save checkpoint
        scheduler: Learning rate scheduler
        is_best: Whether this is the best model so far
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.parent / f"best_{save_path.name}"
        shutil.copy(save_path, best_path)
        print(f"✅ Best model saved to: {best_path}")
    
    print(f"✅ Checkpoint saved to: {save_path}")


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        device: Device to load to
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device or 'cpu')
    
    # Load model state - handle both formats (direct state_dict or wrapped)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Checkpoint is the state dict itself
        model.load_state_dict(checkpoint)
        # Return a minimal checkpoint dict
        checkpoint = {'epoch': 0, 'metrics': {}}
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"✅ Checkpoint loaded from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Metrics: {checkpoint.get('metrics', {})}")
    
    return checkpoint


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters
    
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def format_number(num: int) -> str:
    """Format large numbers with K, M, B suffixes"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def print_model_info(model: nn.Module):
    """Print model architecture and parameter information"""
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print("=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Total parameters: {format_number(total_params)} ({total_params:,})")
    print(f"Trainable parameters: {format_number(trainable_params)} ({trainable_params:,})")
    print(f"Non-trainable parameters: {format_number(total_params - trainable_params)}")
    print("=" * 60)


def get_device_info():
    """Print device information"""
    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA available: No")
    
    print(f"PyTorch version: {torch.__version__}")
    print("=" * 60)


def compute_cosine_similarity(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings [N, D]
        embeddings2: Second set of embeddings [M, D]
        normalize: Whether to normalize embeddings
    
    Returns:
        Similarity matrix [N, M]
    """
    if normalize:
        embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=-1)
        embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=-1)
    
    return embeddings1 @ embeddings2.T


def batch_process(
    data: List[Any],
    process_fn: callable,
    batch_size: int = 32,
    desc: str = "Processing"
) -> List[Any]:
    """
    Process data in batches.
    
    Args:
        data: List of data to process
        process_fn: Function to process each batch
        batch_size: Batch size
        desc: Description for progress bar
    
    Returns:
        List of processed results
    """
    from tqdm import tqdm
    
    results = []
    for i in tqdm(range(0, len(data), batch_size), desc=desc):
        batch = data[i:i + batch_size]
        batch_result = process_fn(batch)
        results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
    
    return results


def save_json(data: Dict, filepath: Path):
    """Save dictionary to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Saved JSON to: {filepath}")


def load_json(filepath: Path) -> Dict:
    """Load dictionary from JSON file"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data


def save_pickle(data: Any, filepath: Path):
    """Save data to pickle file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Saved pickle to: {filepath}")


def load_pickle(filepath: Path) -> Any:
    """Load data from pickle file"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    return data


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def freeze_model(model: nn.Module):
    """Freeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module):
    """Unfreeze all model parameters"""
    for param in model.parameters():
        param.requires_grad = True


# Example usage
if __name__ == "__main__":
    print("Testing helper utilities...")
    
    # Test device info
    get_device_info()
    
    # Test average meter
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"\nAverage Meter: {meter.avg:.2f}")
    
    # Test early stopping
    early_stop = EarlyStopping(patience=3)
    losses = [0.5, 0.4, 0.45, 0.46, 0.47, 0.48]
    for i, loss in enumerate(losses):
        if early_stop(loss):
            print(f"Early stopping triggered at epoch {i}")
            break
    
    print("\n✅ Helper utilities test passed!")
