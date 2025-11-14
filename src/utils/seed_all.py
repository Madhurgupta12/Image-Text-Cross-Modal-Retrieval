"""
Reproducibility utilities for setting random seeds across all libraries.
"""

import random
import numpy as np
import torch
import os


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: Enable deterministic algorithms (slower but reproducible)
        benchmark: Enable cudnn.benchmark for faster training (may reduce reproducibility)
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # cuDNN settings
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = benchmark
    
    # Environment variables for additional libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # TensorFlow (if used)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"[OK] Random seed set to: {seed}")
    print(f"   Deterministic: {deterministic}")
    print(f"   Benchmark: {benchmark}")


def worker_init_fn(worker_id: int):
    """
    Worker initialization function for DataLoader to ensure reproducibility.
    
    Args:
        worker_id: Worker ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
