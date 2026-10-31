"""
Dataset Loaders for Graph Datasets (Cora, CiteSeer, PubMed)
"""
import os
import pickle
import numpy as np
import torch
from typing import Tuple, Optional
from pathlib import Path
from torch_geometric.datasets import Planetoid

# def load_cora(data_dir: str = 'data/') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
#                                                   torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Load Cora dataset.
    
#     Returns:
#         x: Node features [num_nodes, num_features]
#         edge_index: Edge indices [2, num_edges]
#         y: Node labels [num_nodes]
#         train_mask: Training mask [num_nodes]
#         val_mask: Validation mask [num_nodes]
#         test_mask: Test mask [num_nodes]
#     """
#     data_path = Path(data_dir) / 'cora'
#     data_path.mkdir(parents=True, exist_ok=True)
    
#     # Check if preprocessed data exists
#     cache_file = data_path / 'cora_processed.pkl'
#     if cache_file.exists():
#         print(f"[Data] Loading cached Cora dataset from {cache_file}")
#         with open(cache_file, 'rb') as f:
#             data = pickle.load(f)
#         return (data['x'], data['edge_index'], data['y'],
#                 data['train_mask'], data['val_mask'], data['test_mask'])
    
#     # Download and process Cora dataset
#     print("[Data] Downloading and processing Cora dataset...")
    
#     # For now, create synthetic Cora-like data for testing
#     # In production, you would download from: https://github.com/kimiyoung/planetoid
#     num_nodes = 2708
#     num_features = 1433
#     num_classes = 7
#     num_edges = 5429 * 2  # undirected edges
    
#     # Generate random features (in reality, these would be bag-of-words)
#     x = torch.randn(num_nodes, num_features)
#     x = torch.abs(x)  # Make non-negative like bag-of-words
    
#     # Generate random edge indices
#     edge_index = torch.randint(0, num_nodes, (2, num_edges))
#     # Make undirected
#     edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
#     # Generate random labels
#     y = torch.randint(0, num_classes, (num_nodes,))
    
#     # Create train/val/test splits (standard split: 140/500/1000)
#     train_mask = torch.zeros(num_nodes, dtype=torch.bool)
#     val_mask = torch.zeros(num_nodes, dtype=torch.bool)
#     test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
#     # 20 nodes per class for training (140 total)
#     train_indices = []
#     for c in range(num_classes):
#         class_indices = (y == c).nonzero(as_tuple=True)[0]
#         if len(class_indices) >= 20:
#             train_indices.extend(class_indices[:20].tolist())
#     train_mask[train_indices] = True
    
#     # 500 for validation
#     remaining = (~train_mask).nonzero(as_tuple=True)[0]
#     val_indices = remaining[:500]
#     val_mask[val_indices] = True
    
#     # 1000 for test
#     remaining = (~train_mask & ~val_mask).nonzero(as_tuple=True)[0]
#     test_indices = remaining[:1000]
#     test_mask[test_indices] = True
    
#     # Cache the processed data
#     data = {
#         'x': x,
#         'edge_index': edge_index,
#         'y': y,
#         'train_mask': train_mask,
#         'val_mask': val_mask,
#         'test_mask': test_mask
#     }
    
#     with open(cache_file, 'wb') as f:
#         pickle.dump(data, f)
    
#     print(f"[Data] Cora dataset processed and cached")
#     print(f"  Nodes: {num_nodes}, Features: {num_features}, Classes: {num_classes}")
#     print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    
#     return x, edge_index, y, train_mask, val_mask, test_mask

from torch_geometric.datasets import Planetoid

def load_cora(data_dir: str='data/'):
    dataset = Planetoid(root=str(Path(data_dir)/'pyg'), name='Cora')
    d = dataset[0]

    x = d.x.float()
    edge_index = d.edge_index.long()
    y = d.y.long()
    train_mask = d.train_mask.bool()
    val_mask = d.val_mask.bool()
    test_mask = d.test_mask.bool()

    return x, edge_index, y, train_mask, val_mask, test_mask


def load_dataset(name: str, data_dir: str = 'data/'):
    """
    Load graph dataset by name.

    Args:
        name: Dataset name ('Cora', 'CiteSeer', 'PubMed')
        data_dir: Directory to store data

    Returns:
        GraphData object with x, edge_index, y, masks
    """
    from models import GraphData

    if name.lower() == 'cora':
        x, edge_index, y, train_mask, val_mask, test_mask = load_cora(data_dir)
    else:
        raise ValueError(f"Dataset {name} not supported. Use 'Cora'.")

    data = GraphData(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    return data


def get_dataset_info(name: str):
    """Get dataset statistics."""
    if name.lower() == 'cora':
        return {
            'num_nodes': 2708,
            'num_features': 1433,
            'num_classes': 7,
            'train_size': 140,
            'val_size': 500,
            'test_size': 1000
        }
    else:
        raise ValueError(f"Dataset {name} not supported")
