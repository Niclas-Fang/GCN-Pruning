"""
Node Scoring Methods
"""
import torch
from .base import SimpleScorer


def degree_score(model, data, base_masks, **kwargs):
    """
    Score nodes by degree (number of neighbors).
    
    Higher degree = more important.
    """
    # Get edge index with base masks applied
    edge_mask = base_masks['edge_mask']
    edge_index = data.edge_index[:, edge_mask]
    
    # Remove self-loops for degree counting
    is_self_loop = edge_index[0] == edge_index[1]
    edge_index_no_loops = edge_index[:, ~is_self_loop]
    
    # Count degree
    src = edge_index_no_loops[0]
    degree = torch.bincount(src, minlength=data.num_nodes).float()
    
    return {'node_scores': degree}


def random_score(model, data, base_masks, **kwargs):
    """
    Score nodes randomly.
    
    Baseline method for comparison.
    """
    seed = kwargs.get('seed', None)
    if seed is not None:
        torch.manual_seed(seed)
    
    scores = torch.rand(data.num_nodes)
    return {'node_scores': scores}


def get_node_scorer(method: str):
    """Get node scorer by method name."""
    if method == 'degree':
        return SimpleScorer('node', 'degree', degree_score)
    elif method == 'random':
        return SimpleScorer('node', 'random', random_score)
    else:
        raise ValueError(f"Unknown node scoring method: {method}")
