"""
Logging Utilities
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = 'gcn_pruning',
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (if None, only console logging)
        level: Logging level
        console: Whether to add console handler
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


class MetricsLogger:
    """Simple metrics logger for tracking training progress."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.metrics = []
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            # Write header
            with open(log_file, 'w') as f:
                f.write("epoch,train_loss,train_acc,val_loss,val_acc,test_acc\n")
    
    def log(self, epoch: int, **metrics):
        """Log metrics for an epoch."""
        entry = {'epoch': epoch, **metrics}
        self.metrics.append(entry)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                values = [str(entry.get(k, '')) for k in 
                         ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_acc']]
                f.write(','.join(values) + '\n')
    
    def get_best(self, metric: str = 'val_acc', mode: str = 'max'):
        """Get best metric value and corresponding epoch."""
        if not self.metrics:
            return None, None
        
        if mode == 'max':
            best_entry = max(self.metrics, key=lambda x: x.get(metric, float('-inf')))
        else:
            best_entry = min(self.metrics, key=lambda x: x.get(metric, float('inf')))
        
        return best_entry.get(metric), best_entry.get('epoch')


def log_environment_info(logger: logging.Logger):
    """Log system and environment information."""
    import torch
    import platform
    
    logger.info("=" * 60)
    logger.info("Environment Information")
    logger.info("=" * 60)
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    logger.info(f"Platform: {platform.platform()}")
    logger.info("=" * 60)
