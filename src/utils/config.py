"""
Configuration module for Image-Text Cross-Modal Retrieval System.
Contains all paths, hyperparameters, and model configurations.
"""

import os
import torch
from pathlib import Path

# =============================
# Base Paths
# =============================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Dataset Paths
DATASET_PATH = os.getenv("DATASET_PATH", str(DATA_DIR / "raw"))
IMAGES_DIR = DATA_DIR / "images" / "images"
CAPTIONS_FILE = DATA_DIR / "images" / "results.csv"

# Processed Paths
PROCESSED_DIR = DATA_DIR / "processed"
CAPTION_TXT_PATH = DATA_DIR / "captions" / "train_captions.txt"

# Embedding Paths
IMAGE_EMB_DIR = PROCESSED_DIR / "image_embeddings"
TEXT_EMB_DIR = PROCESSED_DIR / "text_embeddings"
IMAGE_EMB_PATH = IMAGE_EMB_DIR / "vit_embeddings.pt"
TEXT_EMB_PATH = TEXT_EMB_DIR / "text_embeddings.pt"
PAIRED_EMB_DIR = DATA_DIR / "paired_embeddings"

# Model Paths
MODELS_DIR = DATA_DIR / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"
PROJECTION_MODEL_PATH = MODELS_DIR / "projection_model.pt"

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_DIR / "results"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
LOGS_DIR = OUTPUT_DIR / "logs"

# =============================
# Model Configuration
# =============================
class ModelConfig:
    """Model architecture configurations"""
    
    # Vision Transformer
    VIT_MODEL_NAME = "google/vit-base-patch16-224"  # Options: base, large, huge
    VIT_OUTPUT_DIM = 768
    
    # Text Encoder (LLM)
    TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    # Alternative options:
    # - "sentence-transformers/all-mpnet-base-v2"  # 768 dim
    # - "bert-base-uncased"  # 768 dim
    # - "sentence-transformers/all-MiniLM-L12-v2"  # 384 dim
    TEXT_OUTPUT_DIM = 384
    
    # Projection Head
    SHARED_DIM = 512  # Shared embedding space dimension
    PROJECTION_HIDDEN_DIMS = [512]  # Multi-layer projection
    USE_BATCH_NORM = True
    USE_DROPOUT = True
    DROPOUT_RATE = 0.1
    ACTIVATION = "relu"  # Options: relu, gelu, tanh
    
    # Temperature for contrastive learning
    INITIAL_TEMPERATURE = 0.07
    LEARNABLE_TEMPERATURE = True

# =============================
# Training Configuration
# =============================
class TrainingConfig:
    """Training hyperparameters"""
    
    # Training settings
    BATCH_SIZE = 128
    NUM_EPOCHS = 100  # Increased from 50 for more training
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    WARMUP_EPOCHS = 5
    
    # Optimizer
    OPTIMIZER = "adamw"  # Options: adam, adamw, sgd
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    
    # Learning Rate Scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "cosine"  # Options: cosine, step, plateau
    LR_DECAY_RATE = 0.1
    LR_DECAY_EPOCHS = [30, 40]
    MIN_LR = 1e-6
    
    # Loss
    LOSS_TYPE = "clip"  # Options: clip, triplet, infonce
    MARGIN = 0.2  # For triplet loss
    
    # Regularization
    GRAD_CLIP = 1.0
    LABEL_SMOOTHING = 0.0
    
    # Early Stopping
    USE_EARLY_STOPPING = True
    PATIENCE = 20  # Increased from 10 for more patience
    MIN_DELTA = 1e-4
    
    # Checkpointing
    SAVE_EVERY_N_EPOCHS = 5
    KEEP_LAST_N_CHECKPOINTS = 3
    
    # Validation
    VAL_EVERY_N_EPOCHS = 1
    VAL_BATCH_SIZE = 256

# =============================
# Data Configuration
# =============================
class DataConfig:
    """Data processing configurations"""
    
    # Dataset
    DATASET_NAME = "flickr30k"  # Options: flickr30k, coco, custom
    NUM_CAPTIONS_PER_IMAGE = 5
    
    # Data Splits
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # Image Processing
    IMAGE_SIZE = 224
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Data Augmentation
    USE_AUGMENTATION = True
    RANDOM_CROP = True
    RANDOM_FLIP = True
    COLOR_JITTER = True
    
    # Text Processing
    MAX_TEXT_LENGTH = 77
    LOWERCASE = True
    REMOVE_PUNCTUATION = False
    
    # DataLoader
    NUM_WORKERS = 4
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2

# =============================
# Evaluation Configuration
# =============================
class EvalConfig:
    """Evaluation metrics configurations"""
    
    # Retrieval Metrics
    RECALL_AT_K = [1, 5, 10, 20, 50]
    COMPUTE_MAP = True
    COMPUTE_MRR = True
    
    # Embedding Analysis
    COMPUTE_EMBEDDING_STATS = True
    VISUALIZE_EMBEDDINGS = True
    TSNE_PERPLEXITY = 30
    UMAP_N_NEIGHBORS = 15
    
    # Qualitative Results
    NUM_QUALITATIVE_SAMPLES = 20
    SHOW_TOP_K_RESULTS = 5

# =============================
# Device Configuration
# =============================
class DeviceConfig:
    """Hardware and device configurations"""
    
    # Device selection
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
    
    # Multi-GPU
    USE_MULTI_GPU = torch.cuda.device_count() > 1
    GPU_IDS = list(range(torch.cuda.device_count()))
    
    # Mixed Precision Training
    USE_AMP = True  # Automatic Mixed Precision
    
    # Reproducibility
    SEED = 42
    DETERMINISTIC = True
    BENCHMARK = True

# =============================
# Logging Configuration
# =============================
class LoggingConfig:
    """Logging and monitoring configurations"""
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_TO_FILE = True
    LOG_TO_CONSOLE = True
    
    # Wandb / TensorBoard
    USE_WANDB = False
    USE_TENSORBOARD = True
    WANDB_PROJECT = "image-text-retrieval"
    WANDB_ENTITY = None
    
    # Logging Frequency
    LOG_EVERY_N_STEPS = 10
    LOG_IMAGES_EVERY_N_EPOCHS = 5
    
    # Verbose
    VERBOSE = True

# =============================
# Helper Functions
# =============================
def create_directories():
    """Create all necessary directories"""
    directories = [
        DATA_DIR, PROCESSED_DIR, IMAGE_EMB_DIR, TEXT_EMB_DIR,
        PAIRED_EMB_DIR, MODELS_DIR, CHECKPOINT_DIR, OUTPUT_DIR,
        RESULTS_DIR, VISUALIZATIONS_DIR, LOGS_DIR
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_config_dict():
    """Get all configurations as a dictionary"""
    config = {
        "model": {k: v for k, v in ModelConfig.__dict__.items() if not k.startswith("_")},
        "training": {k: v for k, v in TrainingConfig.__dict__.items() if not k.startswith("_")},
        "data": {k: v for k, v in DataConfig.__dict__.items() if not k.startswith("_")},
        "eval": {k: v for k, v in EvalConfig.__dict__.items() if not k.startswith("_")},
        "device": {k: str(v) if isinstance(v, torch.device) else v 
                   for k, v in DeviceConfig.__dict__.items() if not k.startswith("_")},
        "logging": {k: v for k, v in LoggingConfig.__dict__.items() if not k.startswith("_")},
    }
    return config

def print_config():
    """Print all configurations"""
    config = get_config_dict()
    print("=" * 60)
    print("Configuration Settings")
    print("=" * 60)
    for category, settings in config.items():
        print(f"\n{category.upper()}:")
        for key, value in settings.items():
            print(f"  {key}: {value}")
    print("=" * 60)

# Initialize directories on import
create_directories()
