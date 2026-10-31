"""
Graph Masking Utilities for Edge and Node Pruning
"""
import torch
from typing import Tuple, Optional
import warnings


def to_undirected(edge_index: torch.Tensor, num_nodes: Optional[int] = None) -> torch.Tensor:
    """
    Convert directed edge_index to undirected by adding reverse edges.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes (optional)
    
    Returns:
        Undirected edge indices [2, num_edges * 2]
    """
    # Add reverse edges
    edge_index_reversed = edge_index.flip(0)
    edge_index_undirected = torch.cat([edge_index, edge_index_reversed], dim=1)
    
    # Remove duplicates
    edge_index_undirected = torch.unique(edge_index_undirected, dim=1)
    
    return edge_index_undirected


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Add self-loops to edge_index.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
    
    Returns:
        Edge indices with self-loops [2, num_edges + num_nodes]
    """
    # Create self-loop edges
    loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    
    # Concatenate with existing edges
    edge_index_with_loops = torch.cat([edge_index, loop_index], dim=1)
    
    # Remove duplicates (in case self-loops already exist)
    edge_index_with_loops = torch.unique(edge_index_with_loops, dim=1)
    
    return edge_index_with_loops


def remove_self_loops(edge_index: torch.Tensor) -> torch.Tensor:
    """
    Remove self-loops from edge_index.
    
    Args:
        edge_index: Edge indices [2, num_edges]
    
    Returns:
        Edge indices without self-loops
    """
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def apply_edge_mask(
    data,
    edge_mask: torch.Tensor,
    ensure_undirected: bool = True,
    add_self_loops_flag: bool = True
) -> torch.Tensor:
    """
    Apply edge mask to prune edges from the graph.
    
    Args:
        data: Graph data object with edge_index attribute
        edge_mask: Boolean mask [num_edges] indicating which edges to keep
        ensure_undirected: If True, ensure the graph remains undirected
        add_self_loops_flag: If True, add self-loops after masking
    
    Returns:
        New edge_index after applying mask
    """
    # Get original edge_index
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # Apply mask
    edge_index_masked = edge_index[:, edge_mask]
    
    # Ensure undirected if requested
    if ensure_undirected:
        edge_index_masked = to_undirected(edge_index_masked, num_nodes)
    
    # Add self-loops if requested
    if add_self_loops_flag:
        edge_index_masked = add_self_loops(edge_index_masked, num_nodes)
    
    return edge_index_masked


def apply_node_mask(
    data,
    node_mask: torch.Tensor,
    ensure_undirected: bool = True,
    preserve_self_loops: bool = True
) -> torch.Tensor:
    """
    Apply node mask to prune nodes from the graph.
    
    Semantics:
    - Pruned nodes (node_mask[i] = False) are not aggregated by other nodes
      → Remove edges connecting to pruned nodes
    - Pruned nodes still need to make predictions using only their own features
      → Preserve self-loops for pruned nodes
    
    Args:
        data: Graph data object with edge_index attribute
        node_mask: Boolean mask [num_nodes] indicating which nodes to keep
        ensure_undirected: If True, ensure graph remains undirected
        preserve_self_loops: If True, preserve self-loops for all nodes
    
    Returns:
        New edge_index after applying node mask
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # Remove existing self-loops first (we'll add them back later if needed)
    edge_index_no_loops = remove_self_loops(edge_index)
    
    # Create edge mask: keep edges where BOTH endpoints are kept nodes
    src, dst = edge_index_no_loops[0], edge_index_no_loops[1]
    edge_mask = node_mask[src] & node_mask[dst]
    
    # Apply edge mask
    edge_index_masked = edge_index_no_loops[:, edge_mask]
    
    # Ensure undirected if requested
    if ensure_undirected:
        edge_index_masked = to_undirected(edge_index_masked, num_nodes)
    
    # Add self-loops
    if preserve_self_loops:
        # Add self-loops for ALL nodes (including pruned ones)
        # This ensures pruned nodes can still use their own features
        edge_index_masked = add_self_loops(edge_index_masked, num_nodes)
    
    return edge_index_masked


def get_graph_statistics(edge_index: torch.Tensor, num_nodes: int) -> dict:
    """
    Compute graph statistics.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes
    
    Returns:
        Dictionary with statistics
    """
    # Remove self-loops for counting
    edge_index_no_loops = remove_self_loops(edge_index)
    
    # Count edges
    num_edges = edge_index.size(1)
    num_edges_no_loops = edge_index_no_loops.size(1)
    num_self_loops = num_edges - num_edges_no_loops
    
    # Count degree (excluding self-loops)
    src = edge_index_no_loops[0]
    degree = torch.bincount(src, minlength=num_nodes)
    
    # Find isolated nodes (degree = 0, excluding self-loops)
    isolated_nodes = (degree == 0).sum().item()
    
    # Average degree
    avg_degree = degree.float().mean().item()
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_edges_no_loops': num_edges_no_loops,
        'num_self_loops': num_self_loops,
        'num_isolated_nodes': isolated_nodes,
        'avg_degree': avg_degree,
        'max_degree': degree.max().item(),
        'min_degree': degree.min().item(),
    }


def print_graph_statistics(edge_index: torch.Tensor, num_nodes: int, prefix: str = ""):
    """
    Print graph statistics in a readable format.
    
    Args:
        edge_index: Edge indices
        num_nodes: Number of nodes
        prefix: Prefix for print statements
    """
    stats = get_graph_statistics(edge_index, num_nodes)
    
    print(f"{prefix}Graph Statistics:")
    print(f"{prefix}  Nodes: {stats['num_nodes']}")
    print(f"{prefix}  Total edges: {stats['num_edges']}")
    print(f"{prefix}  Edges (no self-loops): {stats['num_edges_no_loops']}")
    print(f"{prefix}  Self-loops: {stats['num_self_loops']}")
    print(f"{prefix}  Isolated nodes: {stats['num_isolated_nodes']}")
    print(f"{prefix}  Avg degree: {stats['avg_degree']:.2f}")
    print(f"{prefix}  Degree range: [{stats['min_degree']}, {stats['max_degree']}]")


def compare_graphs(
    edge_index_before: torch.Tensor,
    edge_index_after: torch.Tensor,
    num_nodes: int
):
    """
    Compare two graphs and print statistics.
    
    Args:
        edge_index_before: Edge indices before masking
        edge_index_after: Edge indices after masking
        num_nodes: Number of nodes
    """
    print("=" * 60)
    print("Graph Comparison: Before vs After Masking")
    print("=" * 60)
    
    print("\nBEFORE Masking:")
    print_graph_statistics(edge_index_before, num_nodes, prefix="  ")
    
    print("\nAFTER Masking:")
    print_graph_statistics(edge_index_after, num_nodes, prefix="  ")
    
    # Compute differences
    stats_before = get_graph_statistics(edge_index_before, num_nodes)
    stats_after = get_graph_statistics(edge_index_after, num_nodes)
    
    edge_reduction = stats_before['num_edges'] - stats_after['num_edges']
    edge_reduction_pct = (edge_reduction / stats_before['num_edges'] * 100) if stats_before['num_edges'] > 0 else 0
    
    isolated_change = stats_after['num_isolated_nodes'] - stats_before['num_isolated_nodes']
    
    print("\nChanges:")
    print(f"  Edges removed: {edge_reduction} ({edge_reduction_pct:.1f}%)")
    print(f"  Isolated nodes change: {isolated_change:+d}")
    print("=" * 60)
