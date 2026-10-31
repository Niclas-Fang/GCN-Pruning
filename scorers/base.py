"""
Base Scorer Interface for Unified Pruning Score Computation
"""
import torch
import numpy as np
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Callable
from abc import ABC, abstractmethod


class PruningScorer(ABC):
    """
    Abstract base class for pruning scorers.
    
    Provides unified interface for scoring nodes, edges, or channels
    under conditional pruning scenarios.
    """
    
    def __init__(self, unit_type: str, method: str):
        """
        Initialize scorer.
        
        Args:
            unit_type: Type of unit to score ('node', 'edge', 'channel')
            method: Scoring method name
        """
        assert unit_type in ['node', 'edge', 'channel'], \
            f"Invalid unit_type: {unit_type}"
        
        self.unit_type = unit_type
        self.method = method
    
    @abstractmethod
    def compute_scores(
        self,
        model,
        data,
        base_masks: Dict[str, Any],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute importance scores.
        
        Args:
            model: GCN model (can be MaskedModelWrapper)
            data: Graph data object
            base_masks: Dictionary containing:
                - 'node_mask': [num_nodes] boolean tensor
                - 'edge_mask': [num_edges] boolean tensor  
                - 'channel_masks': dict {layer_name: [hidden_dim] tensor}
            **kwargs: Additional arguments
        
        Returns:
            Dictionary with scores and metadata
        """
        pass
    
    def score(
        self,
        model,
        data,
        base_masks: Optional[Dict[str, Any]] = None,
        save_path: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main scoring interface with timing and saving.
        
        Args:
            model: Model to evaluate
            data: Graph data
            base_masks: Base masks for conditional evaluation
            save_path: Path to save scores (optional)
            seed: Random seed (optional)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary containing scores and metadata
        """
        # Default base masks (no pruning)
        if base_masks is None:
            base_masks = self._create_default_masks(model, data)
        
        # Validate base masks
        self._validate_base_masks(base_masks, model, data)
        
        # Time the scoring
        start_time = time.time()
        
        # Compute scores
        scores_dict = self.compute_scores(model, data, base_masks, **kwargs)
        
        elapsed_time = time.time() - start_time
        
        # Add metadata
        result = {
            'unit_type': self.unit_type,
            'method': self.method,
            'scores': scores_dict,
            'base_masks': base_masks,
            'elapsed_time': elapsed_time,
            'seed': seed,
            'config': kwargs
        }
        
        # Save if path provided
        if save_path is not None:
            self._save_scores(result, save_path)
        
        return result
    
    def _create_default_masks(self, model, data) -> Dict[str, Any]:
        """Create default masks (all ones - no pruning)."""
        base_masks = {
            'node_mask': torch.ones(data.num_nodes, dtype=torch.bool),
            'edge_mask': torch.ones(data.edge_index.size(1), dtype=torch.bool),
            'channel_masks': {}
        }
        
        # Get channel masks if model is MaskedModelWrapper
        if hasattr(model, 'layer_names'):
            for layer_name in model.layer_names:
                mask = model.get_layer_mask(layer_name)
                base_masks['channel_masks'][layer_name] = mask.clone()
        elif hasattr(model, 'list_prunable_layers'):
            # Base model - create default masks
            for layer_name, layer in model.list_prunable_layers():
                if hasattr(layer, 'out_features'):
                    dim = layer.out_features
                elif hasattr(layer, 'weight'):
                    dim = layer.weight.size(0)
                else:
                    continue
                base_masks['channel_masks'][layer_name] = torch.ones(dim)
        
        return base_masks
    
    def _validate_base_masks(self, base_masks: Dict[str, Any], model, data):
        """Validate base masks have correct shapes."""
        assert 'node_mask' in base_masks, "Missing node_mask"
        assert 'edge_mask' in base_masks, "Missing edge_mask"
        assert 'channel_masks' in base_masks, "Missing channel_masks"
        
        assert base_masks['node_mask'].size(0) == data.num_nodes, \
            f"node_mask size mismatch: {base_masks['node_mask'].size(0)} vs {data.num_nodes}"
        
        assert base_masks['edge_mask'].size(0) == data.edge_index.size(1), \
            f"edge_mask size mismatch"
    
    def _save_scores(self, result: Dict[str, Any], save_path: str):
        """Save scores to pickle file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        
        print(f"[Scorer] Scores saved to {save_path}")
    
    @staticmethod
    def load_scores(load_path: str) -> Dict[str, Any]:
        """Load scores from pickle file."""
        with open(load_path, 'rb') as f:
            result = pickle.load(f)
        return result


class SimpleScorer(PruningScorer):
    """
    Simple scorer that doesn't require model evaluation.
    
    Used for baseline methods like random, degree, magnitude.
    """
    
    def __init__(self, unit_type: str, method: str, score_fn: Callable):
        """
        Initialize simple scorer.
        
        Args:
            unit_type: Type of unit ('node', 'edge', 'channel')
            method: Method name
            score_fn: Function that computes scores
        """
        super().__init__(unit_type, method)
        self.score_fn = score_fn
    
    def compute_scores(
        self,
        model,
        data,
        base_masks: Dict[str, Any],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute scores using the score function."""
        return self.score_fn(model, data, base_masks, **kwargs)


def score(
    unit_type: str,
    method: str,
    model,
    data,
    base_masks: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Unified scoring interface.
    
    Args:
        unit_type: 'node', 'edge', or 'channel'
        method: Scoring method name
        model: Model to evaluate
        data: Graph data
        base_masks: Base masks for conditional evaluation
        save_path: Path to save scores
        seed: Random seed
        **kwargs: Method-specific arguments
    
    Returns:
        Dictionary with scores and metadata
    """
    # Import method implementations
    from .node_scorers import get_node_scorer
    from .edge_scorers import get_edge_scorer
    from .channel_scorers import get_channel_scorer
    
    # Get appropriate scorer
    if unit_type == 'node':
        scorer = get_node_scorer(method)
    elif unit_type == 'edge':
        scorer = get_edge_scorer(method)
    elif unit_type == 'channel':
        scorer = get_channel_scorer(method)
    else:
        raise ValueError(f"Unknown unit_type: {unit_type}")
    
    # Compute and return scores
    return scorer.score(model, data, base_masks, save_path, seed, **kwargs)
