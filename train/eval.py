"""
Evaluation Utilities for GCN Training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def evaluate(
    model: nn.Module,
    data,
    mask: torch.Tensor,
    adj_norm: torch.Tensor,
    criterion: nn.Module = None
) -> Tuple[float, float]:
    """
    Evaluate model on a given split.
    
    Args:
        model: GCN model
        data: Graph data object with x, y attributes
        mask: Boolean mask for nodes to evaluate [num_nodes]
        adj_norm: Normalized adjacency matrix
        criterion: Loss function (default: CrossEntropyLoss)
    
    Returns:
        (loss, accuracy) tuple
    """
    model.eval()
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        # Forward pass
        logits = model(data.x, adj_norm)
        
        # Get predictions and targets for masked nodes
        logits_masked = logits[mask]
        targets_masked = data.y[mask]
        
        # Compute loss
        loss = criterion(logits_masked, targets_masked)
        
        # Compute accuracy
        pred = logits_masked.argmax(dim=1)
        correct = (pred == targets_masked).sum().item()
        accuracy = correct / mask.sum().item()
    
    return loss.item(), accuracy


def evaluate_all_splits(
    model: nn.Module,
    data,
    adj_norm: torch.Tensor,
    criterion: nn.Module = None
) -> dict:
    """
    Evaluate model on all splits (train, val, test).
    
    Args:
        model: GCN model
        data: Graph data object
        adj_norm: Normalized adjacency matrix
        criterion: Loss function
    
    Returns:
        Dictionary with metrics for all splits
    """
    results = {}
    
    # Train split
    train_loss, train_acc = evaluate(model, data, data.train_mask, adj_norm, criterion)
    results['train_loss'] = train_loss
    results['train_acc'] = train_acc
    
    # Validation split
    val_loss, val_acc = evaluate(model, data, data.val_mask, adj_norm, criterion)
    results['val_loss'] = val_loss
    results['val_acc'] = val_acc
    
    # Test split
    test_loss, test_acc = evaluate(model, data, data.test_mask, adj_norm, criterion)
    results['test_loss'] = test_loss
    results['test_acc'] = test_acc
    
    return results


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.
    
    Args:
        logits: Model predictions [num_samples, num_classes]
        targets: Ground truth labels [num_samples]
    
    Returns:
        Accuracy as a float
    """
    pred = logits.argmax(dim=1)
    correct = (pred == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy
