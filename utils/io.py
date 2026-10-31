"""
I/O Utilities for Checkpoint Management
"""
import os
import json
import torch
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional


def get_git_hash() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def save_checkpoint(
    save_path: str,
    model_state: Dict[str, Any],
    config: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    epoch: Optional[int] = None,
    optimizer_state: Optional[Dict[str, Any]] = None,
    include_git_hash: bool = True
):
    """
    Save model checkpoint with all metadata.
    
    Args:
        save_path: Path to save checkpoint
        model_state: Model state dict
        config: Configuration dictionary
        metrics: Dictionary of metrics (e.g., {'val_acc': 0.85, 'test_acc': 0.83})
        epoch: Current epoch number
        optimizer_state: Optimizer state dict
        include_git_hash: Whether to include git commit hash
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model_state,
        'config': config,
    }
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if optimizer_state is not None:
        checkpoint['optimizer_state_dict'] = optimizer_state
    
    if include_git_hash:
        git_hash = get_git_hash()
        if git_hash:
            checkpoint['git_hash'] = git_hash
    
    torch.save(checkpoint, save_path)
    print(f"[I/O] Checkpoint saved to {save_path}")
    
    # Also save config as JSON for easy inspection
    config_path = save_path.replace('.pth', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_checkpoint(
    load_path: str,
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load checkpoint from file.
    
    Args:
        load_path: Path to checkpoint file
        map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0')
    
    Returns:
        Dictionary containing all saved information
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint not found at {load_path}")
    
    checkpoint = torch.load(load_path, map_location=map_location)
    print(f"[I/O] Checkpoint loaded from {load_path}")
    
    # Print summary
    if 'epoch' in checkpoint:
        print(f"  - Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"  - Metrics: {checkpoint['metrics']}")
    if 'git_hash' in checkpoint:
        print(f"  - Git hash: {checkpoint['git_hash'][:8]}...")
    
    return checkpoint


def save_results(
    save_path: str,
    results: Dict[str, Any],
    format: str = 'json'
):
    """
    Save experiment results to file.
    
    Args:
        save_path: Path to save results
        results: Results dictionary
        format: File format ('json' or 'txt')
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'txt':
        with open(save_path, 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
    
    print(f"[I/O] Results saved to {save_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config
