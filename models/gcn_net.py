"""
Multi-layer GCN Network for Node Classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union
from .gcn_layer import GCNConv


class GCNNet(nn.Module):
    """
    Multi-layer Graph Convolutional Network.
    
    Architecture:
        input_proj -> (GCN + ReLU + Dropout)* -> (MLP + ReLU + Dropout)* -> output_head
    
    Layers:
        - input_proj: Input projection layer (NOT prunable)
        - gcn_layers: Stack of GCN layers (prunable)
        - mlp_layers: Stack of MLP layers (prunable)
        - output_head: Output classification layer (NOT prunable)
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_gcn_layers: int = 3,
        num_mlp_layers: int = 1,
        dropout: float = 0.5
    ):
        """
        Initialize GCN network.
        
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes
            num_gcn_layers: Number of GCN layers (>= 3)
            num_mlp_layers: Number of MLP layers (>= 1)
            dropout: Dropout rate
        """
        super(GCNNet, self).__init__()
        
        assert num_gcn_layers >= 1, "num_gcn_layers must be >= 1"
        assert num_mlp_layers >= 0, "num_mlp_layers must be >= 0"
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_gcn_layers = num_gcn_layers
        self.num_mlp_layers = num_mlp_layers
        self.dropout = dropout
        
        # Input projection layer (NOT prunable)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # GCN layers (prunable)
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) 
            for _ in range(num_gcn_layers)
        ])
        
        # MLP layers (prunable)
        self.mlp_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_mlp_layers)
        ])
        
        # Output head (NOT prunable)
        self.output_head = nn.Linear(hidden_dim, num_classes)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize all parameters."""
        self.input_proj.reset_parameters()
        for layer in self.gcn_layers:
            layer.reset_parameters()
        for layer in self.mlp_layers:
            layer.reset_parameters()
        self.output_head.reset_parameters()

    def forward(
        self, 
        x: torch.Tensor, 
        adj_norm: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, in_dim]
            adj_norm: Normalized adjacency matrix [num_nodes, num_nodes] (sparse)
            return_embeddings: If True, return (logits, embeddings)
        
        Returns:
            If return_embeddings=False: logits [num_nodes, num_classes]
            If return_embeddings=True: (logits, embeddings) where embeddings is [num_nodes, hidden_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # GCN layers
        for gcn_layer in self.gcn_layers:
            h = gcn_layer(h, adj_norm)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # MLP layers
        for mlp_layer in self.mlp_layers:
            h = mlp_layer(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Save embeddings before output head
        embeddings = h
        
        # Output head
        logits = self.output_head(h)
        
        if return_embeddings:
            return logits, embeddings
        else:
            return logits
    
    def list_prunable_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        List all prunable layers (excludes input_proj and output_head).
        
        Returns:
            List of (layer_name, layer_module) tuples for prunable layers
        """
        prunable = []
        
        # GCN layers are prunable
        for i, layer in enumerate(self.gcn_layers):
            prunable.append((f'gcn_layers.{i}', layer))
        
        # MLP layers are prunable
        for i, layer in enumerate(self.mlp_layers):
            prunable.append((f'mlp_layers.{i}', layer))
        
        return prunable
    
    def get_prunable_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """
        Get all prunable parameters (weights from prunable layers).
        
        Returns:
            List of (param_name, parameter) tuples
        """
        prunable_params = []
        
        for layer_name, layer in self.list_prunable_layers():
            for param_name, param in layer.named_parameters():
                full_name = f"{layer_name}.{param_name}"
                prunable_params.append((full_name, param))
        
        return prunable_params
    
    def count_parameters(self, only_prunable: bool = False) -> int:
        """
        Count total number of parameters.
        
        Args:
            only_prunable: If True, count only prunable parameters
        
        Returns:
            Number of parameters
        """
        if only_prunable:
            return sum(p.numel() for _, p in self.get_prunable_parameters())
        else:
            return sum(p.numel() for p in self.parameters())
    
    def __repr__(self):
        """String representation."""
        lines = [
            f"GCNNet(",
            f"  in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, num_classes={self.num_classes}",
            f"  num_gcn_layers={self.num_gcn_layers}, num_mlp_layers={self.num_mlp_layers}",
            f"  dropout={self.dropout}",
            f"  total_params={self.count_parameters():,}",
            f"  prunable_params={self.count_parameters(only_prunable=True):,}",
            f")"
        ]
        return '\n'.join(lines)


def create_gcn_model(
    num_features: int,
    num_classes: int,
    hidden_dim: int = 64,
    num_gcn_layers: int = 3,
    num_mlp_layers: int = 1,
    dropout: float = 0.5
) -> GCNNet:
    """
    Factory function to create GCN model.
    
    Args:
        num_features: Number of input features
        num_classes: Number of output classes
        hidden_dim: Hidden dimension
        num_gcn_layers: Number of GCN layers
        num_mlp_layers: Number of MLP layers
        dropout: Dropout rate
    
    Returns:
        GCNNet model
    """
    model = GCNNet(
        in_dim=num_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_gcn_layers=num_gcn_layers,
        num_mlp_layers=num_mlp_layers,
        dropout=dropout
    )
    return model
