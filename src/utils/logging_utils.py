"""
Logging utilities for training and evaluation.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

from src.utils.config import LOGS_DIR, LoggingConfig


class Logger:
    """
    Custom logger with file and console output.
    """
    
    def __init__(
        self,
        name: str = "image-text-retrieval",
        log_dir: Optional[Path] = None,
        log_to_file: bool = LoggingConfig.LOG_TO_FILE,
        log_to_console: bool = LoggingConfig.LOG_TO_CONSOLE,
        log_level: str = LoggingConfig.LOG_LEVEL
    ):
        """
        Args:
            name: Logger name
            log_dir: Directory to save log files
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            log_level: Logging level
        """
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = self.log_dir / f"{name}_{timestamp}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"Logging to file: {log_file}")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)
    
    def log_dict(self, data: Dict[str, Any], title: str = ""):
        """Log dictionary in a formatted way"""
        if title:
            self.info(f"{title}:")
        for key, value in data.items():
            self.info(f"  {key}: {value}")


class MetricsLogger:
    """
    Logger for tracking training and evaluation metrics.
    """
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Args:
            save_dir: Directory to save metrics
        """
        self.save_dir = Path(save_dir) if save_dir else LOGS_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'train': [],
            'val': [],
            'test': []
        }
        
        self.current_epoch = 0
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Optional[Dict[str, float]] = None,
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        self.current_epoch = epoch
        
        if train_metrics:
            train_metrics['epoch'] = epoch
            self.metrics['train'].append(train_metrics)
        
        if val_metrics:
            val_metrics['epoch'] = epoch
            self.metrics['val'].append(val_metrics)
    
    def log_test(self, test_metrics: Dict[str, float]):
        """Log test metrics"""
        self.metrics['test'].append(test_metrics)
    
    def save(self, filename: str = "metrics.json"):
        """Save metrics to JSON file"""
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✅ Metrics saved to: {save_path}")
    
    def load(self, filename: str = "metrics.json"):
        """Load metrics from JSON file"""
        load_path = self.save_dir / filename
        if load_path.exists():
            with open(load_path, 'r') as f:
                self.metrics = json.load(f)
            print(f"✅ Metrics loaded from: {load_path}")
        else:
            print(f"⚠️ Metrics file not found: {load_path}")
    
    def get_best_epoch(self, metric_name: str = 'loss', split: str = 'val', mode: str = 'min') -> int:
        """
        Get the epoch with best metric value.
        
        Args:
            metric_name: Name of the metric
            split: Data split ('train', 'val', 'test')
            mode: 'min' for metrics to minimize, 'max' for metrics to maximize
        
        Returns:
            Best epoch number
        """
        if not self.metrics[split]:
            return 0
        
        values = [(m['epoch'], m.get(metric_name, float('inf' if mode == 'min' else '-inf'))) 
                  for m in self.metrics[split]]
        
        if mode == 'min':
            best_epoch = min(values, key=lambda x: x[1])[0]
        else:
            best_epoch = max(values, key=lambda x: x[1])[0]
        
        return best_epoch


def setup_tensorboard(log_dir: Optional[Path] = None):
    """
    Setup TensorBoard writer.
    
    Args:
        log_dir: Directory for TensorBoard logs
    
    Returns:
        SummaryWriter instance
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        
        log_dir = Path(log_dir) if log_dir else LOGS_DIR / "tensorboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter(log_dir / f"run_{timestamp}")
        
        print(f"✅ TensorBoard logging to: {log_dir}")
        print(f"   Run: tensorboard --logdir={log_dir}")
        
        return writer
    
    except ImportError:
        print("⚠️ TensorBoard not available. Install with: pip install tensorboard")
        return None


def setup_wandb(project_name: str = LoggingConfig.WANDB_PROJECT, config: Optional[Dict] = None):
    """
    Setup Weights & Biases logging.
    
    Args:
        project_name: W&B project name
        config: Configuration dictionary
    
    Returns:
        wandb run object
    """
    try:
        import wandb
        
        run = wandb.init(
            project=project_name,
            entity=LoggingConfig.WANDB_ENTITY,
            config=config
        )
        
        print(f"✅ W&B logging initialized")
        print(f"   Project: {project_name}")
        print(f"   Run URL: {run.get_url()}")
        
        return run
    
    except ImportError:
        print("⚠️ W&B not available. Install with: pip install wandb")
        return None


# Example usage
if __name__ == "__main__":
    # Test logger
    logger = Logger("test")
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Test metrics logger
    metrics_logger = MetricsLogger()
    metrics_logger.log_epoch(
        epoch=1,
        train_metrics={'loss': 0.5, 'accuracy': 0.8},
        val_metrics={'loss': 0.6, 'accuracy': 0.75}
    )
    metrics_logger.save()
    
    print("\n✅ Logging utilities test passed!")
