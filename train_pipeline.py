"""
Comprehensive training pipeline for image-text cross-modal retrieval.
Includes full training loop, validation, checkpointing, and logging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
from typing import Optional, Dict, Tuple

from src.fusion.projection_head import ProjectionModel
from src.fusion.retrieval import CLIPContrastiveLoss
from src.image_encoder.dataset_loader import load_paired_embeddings, PairedEmbeddingDataset, get_dataloader
from src.evaluation.metrics import evaluate_bidirectional_retrieval, print_metrics
from src.utils.config import (
    TrainingConfig, ModelConfig, DeviceConfig, DataConfig, LoggingConfig,
    PAIRED_EMB_DIR, CHECKPOINT_DIR, BEST_MODEL_PATH
)
from src.utils.seed_all import set_seed
from src.utils.logging_utils import Logger, MetricsLogger, setup_tensorboard
from src.utils.helpers import (
    save_checkpoint, load_checkpoint, AverageMeter, EarlyStopping,
    get_learning_rate, print_model_info, count_parameters
)


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer based on configuration"""
    if config.OPTIMIZER.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=config.BETAS,
            eps=config.EPS,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            betas=config.BETAS,
            eps=config.EPS,
            weight_decay=config.WEIGHT_DECAY
        )
    elif config.OPTIMIZER.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.LEARNING_RATE,
            momentum=0.9,
            weight_decay=config.WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    """Create learning rate scheduler"""
    if not config.USE_SCHEDULER:
        return None
    
    if config.SCHEDULER_TYPE == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS,
            eta_min=config.MIN_LR
        )
    elif config.SCHEDULER_TYPE == "step":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.LR_DECAY_EPOCHS,
            gamma=config.LR_DECAY_RATE
        )
    elif config.SCHEDULER_TYPE == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.LR_DECAY_RATE,
            patience=5,
            min_lr=config.MIN_LR
        )
    else:
        return None


def train_epoch(
    model: ProjectionModel,
    dataloader: DataLoader,
    criterion: CLIPContrastiveLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: Logger,
    use_amp: bool = DeviceConfig.USE_AMP
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    loss_meter = AverageMeter()
    
    # For mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (img_emb, txt_emb, _) in enumerate(pbar):
        img_emb = img_emb.to(device)
        txt_emb = txt_emb.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                img_proj, txt_proj = model(img_emb, txt_emb, normalize=True)
                loss, loss_i2t, loss_t2i = criterion(img_proj, txt_proj)
        else:
            img_proj, txt_proj = model(img_emb, txt_emb, normalize=True)
            loss, loss_i2t, loss_t2i = criterion(img_proj, txt_proj)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            if TrainingConfig.GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if TrainingConfig.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.GRAD_CLIP)
            optimizer.step()
        
        # Update metrics
        loss_meter.update(loss.item(), img_emb.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'lr': f'{get_learning_rate(optimizer):.6f}',
            'temp': f'{model.get_temperature():.4f}'
        })
    
    metrics = {
        'loss': loss_meter.avg,
        'learning_rate': get_learning_rate(optimizer),
        'temperature': model.get_temperature()
    }
    
    return metrics


@torch.no_grad()
def validate(
    model: ProjectionModel,
    dataloader: DataLoader,
    criterion: CLIPContrastiveLoss,
    device: torch.device,
    compute_retrieval_metrics: bool = True,
    max_samples_for_retrieval: int = 5000  # Limit samples to avoid OOM
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        max_samples_for_retrieval: Maximum number of samples to use for retrieval metrics (to avoid OOM)
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    loss_meter = AverageMeter()
    
    all_img_proj = []
    all_txt_proj = []
    
    for img_emb, txt_emb, _ in tqdm(dataloader, desc="Validating"):
        img_emb = img_emb.to(device)
        txt_emb = txt_emb.to(device)
        
        # Forward pass
        img_proj, txt_proj = model(img_emb, txt_emb, normalize=True)
        loss, loss_i2t, loss_t2i = criterion(img_proj, txt_proj)
        
        loss_meter.update(loss.item(), img_emb.size(0))
        
        # Store projections for retrieval metrics (limited to avoid OOM)
        if compute_retrieval_metrics and len(all_img_proj) * img_proj.size(0) < max_samples_for_retrieval:
            all_img_proj.append(img_proj.cpu())
            all_txt_proj.append(txt_proj.cpu())
    
    metrics = {'loss': loss_meter.avg}
    
    # Compute retrieval metrics on subset
    if compute_retrieval_metrics and len(all_img_proj) > 0:
        all_img_proj = torch.cat(all_img_proj, dim=0)
        all_txt_proj = torch.cat(all_txt_proj, dim=0)
        
        print(f"\n   [INFO] Computing retrieval metrics on {all_img_proj.size(0)} samples (subset to avoid OOM)")
        
        retrieval_metrics = evaluate_bidirectional_retrieval(
            all_img_proj, all_txt_proj, normalize=False  # Already normalized
        )
        metrics.update(retrieval_metrics)
    
    return metrics


def train(
    model: ProjectionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = TrainingConfig.NUM_EPOCHS,
    device: torch.device = DeviceConfig.DEVICE,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    resume_from: Optional[str] = None
):
    """
    Main training function.
    
    Args:
        model: Projection model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        resume_from: Path to checkpoint to resume from
    """
    # Setup
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger = Logger("training")
    metrics_logger = MetricsLogger(checkpoint_dir)
    
    # Model info
    print_model_info(model)
    logger.info(f"Total parameters: {count_parameters(model):,}")
    
    # Move model to device
    model = model.to(device)
    
    # Multi-GPU
    if DeviceConfig.USE_MULTI_GPU and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Criterion and optimizer
    criterion = CLIPContrastiveLoss(
        temperature=ModelConfig.INITIAL_TEMPERATURE,
        learnable_temperature=ModelConfig.LEARNABLE_TEMPERATURE
    )
    criterion = criterion.to(device)
    optimizer = create_optimizer(model, TrainingConfig)
    scheduler = create_scheduler(optimizer, TrainingConfig)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=TrainingConfig.PATIENCE,
        min_delta=TrainingConfig.MIN_DELTA,
        mode='min'
    ) if TrainingConfig.USE_EARLY_STOPPING else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_from and os.path.exists(resume_from):
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['metrics'].get('val_loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # TensorBoard
    writer = setup_tensorboard(checkpoint_dir / "tensorboard") if LoggingConfig.USE_TENSORBOARD else None
    
    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch+1, logger
        )
        
        logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        
        # Validate
        if (epoch + 1) % TrainingConfig.VAL_EVERY_N_EPOCHS == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            
            # Print retrieval metrics
            if 'i2t_recall@1' in val_metrics:
                logger.info(f"I2T R@1: {val_metrics['i2t_recall@1']:.4f}")
                logger.info(f"T2I R@1: {val_metrics['t2i_recall@1']:.4f}")
        else:
            val_metrics = None
        
        # Log metrics
        metrics_logger.log_epoch(epoch+1, train_metrics, val_metrics)
        
        # TensorBoard logging
        if writer is not None:
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            if val_metrics:
                writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
                if 'i2t_recall@1' in val_metrics:
                    writer.add_scalar('Retrieval/i2t_recall@1', val_metrics['i2t_recall@1'], epoch)
                    writer.add_scalar('Retrieval/t2i_recall@1', val_metrics['t2i_recall@1'], epoch)
        
        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'] if val_metrics else train_metrics['loss'])
            else:
                scheduler.step()
        
        # Save checkpoint
        is_best = False
        if val_metrics and val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            is_best = True
        
        if (epoch + 1) % TrainingConfig.SAVE_EVERY_N_EPOCHS == 0 or is_best:
            save_checkpoint(
                model.module if hasattr(model, 'module') else model,
                optimizer,
                epoch,
                {'train': train_metrics, 'val': val_metrics} if val_metrics else {'train': train_metrics},
                checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt",
                scheduler,
                is_best
            )
        
        # Early stopping
        if early_stopping and val_metrics:
            if early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Save final model
    torch.save(
        (model.module if hasattr(model, 'module') else model).state_dict(),
        BEST_MODEL_PATH
    )
    logger.info(f"Training complete! Best model saved to: {BEST_MODEL_PATH}")
    
    # Save metrics
    metrics_logger.save()
    
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(DeviceConfig.SEED, DeviceConfig.DETERMINISTIC, DeviceConfig.BENCHMARK)
    
    print("[*] Starting training pipeline...")
    
    # Load paired embeddings
    img_embeddings, txt_embeddings, img_ids = load_paired_embeddings(
        PAIRED_EMB_DIR / "images.pt",
        PAIRED_EMB_DIR / "texts.pt",
        normalize=True
    )
    
    # Split into train/val
    n = len(img_embeddings)
    train_size = int(n * 0.9)
    
    train_dataset = PairedEmbeddingDataset(
        img_embeddings[:train_size],
        txt_embeddings[:train_size],
        img_ids[:train_size]
    )
    
    val_dataset = PairedEmbeddingDataset(
        img_embeddings[train_size:],
        txt_embeddings[train_size:],
        img_ids[train_size:]
    )
    
    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset,
        batch_size=TrainingConfig.BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=TrainingConfig.VAL_BATCH_SIZE,
        shuffle=False
    )
    
    print(f"[*] Dataset sizes:")
    print(f"   Train: {len(train_dataset)}")
    print(f"   Val: {len(val_dataset)}")
    
    # Create model
    model = ProjectionModel(
        img_dim=img_embeddings.size(1),
        text_dim=txt_embeddings.size(1),
        shared_dim=ModelConfig.SHARED_DIM
    )
    
    # Train
    train(model, train_loader, val_loader)
    
    print("\n[OK] Training pipeline complete!")
