"""
Random Seed Utilities for Reproducibility
"""
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seed for reproducibility across numpy, torch, and cuda.
    
    Args:
        seed: Random seed value
        deterministic: If True, set cudnn to deterministic mode (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    print(f"[Seed] Set random seed to {seed} (deterministic={deterministic})")


def get_seed_state():
    """Get current random state for all RNGs."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }


def set_seed_state(state_dict):
    """Restore random state from saved state dict."""
    random.setstate(state_dict['python'])
    np.random.set_state(state_dict['numpy'])
    torch.set_rng_state(state_dict['torch'])
    if torch.cuda.is_available() and state_dict['cuda'] is not None:
        torch.cuda.set_rng_state_all(state_dict['cuda'])
