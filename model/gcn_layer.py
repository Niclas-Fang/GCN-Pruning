"""
Custom GCN Layer Implementation (without PyTorch Geometric dependency)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GCNConv(nn.Module):
    """
    Graph Convolutional Layer.
    
    Implements: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    where A is the adjacency matrix with self-loops.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            bias: Whether to use bias
        """
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_features]
            adj_norm: Normalized adjacency matrix [num_nodes, num_nodes]
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        # Linear transformation: H * W
        support = torch.mm(x, self.weight)
        
        # Graph convolution: A * (H * W)
        output = torch.spmm(adj_norm, support)
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.in_features}, {self.out_features})'


def normalize_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Normalize adjacency matrix: D^(-1/2) A D^(-1/2)
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
    
    Returns:
        Normalized adjacency matrix (sparse) [num_nodes, num_nodes]
    """
    # Add self-loops
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index_with_loops = torch.cat([edge_index, loop_index], dim=1)
    
    # Create adjacency matrix with self-loops
    edge_weight = torch.ones(edge_index_with_loops.size(1), 
                            dtype=torch.float, 
                            device=edge_index.device)
    
    adj = torch.sparse_coo_tensor(
        edge_index_with_loops,
        edge_weight,
        (num_nodes, num_nodes)
    )
    
    # Compute degree matrix
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    
    # D^(-1/2)
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    
    # Normalize: D^(-1/2) A D^(-1/2)
    indices = edge_index_with_loops
    values = edge_weight * d_inv_sqrt[indices[0]] * d_inv_sqrt[indices[1]]
    
    adj_norm = torch.sparse_coo_tensor(
        indices,
        values,
        (num_nodes, num_nodes)
    )
    
    return adj_norm


class GraphData:
    """Simple graph data structure (similar to PyG Data)."""
    
    def __init__(self, x, edge_index, y, train_mask, val_mask, test_mask):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_nodes = x.size(0)
        self.num_features = x.size(1)
        self.num_classes = int(y.max().item()) + 1
