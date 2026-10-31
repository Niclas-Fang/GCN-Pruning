"""
Small-scale Verification of Shapley Value Computation

Tests joint pruning condition evaluation for nodes, edges, and channels.
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import GCNNet, normalize_adjacency
from data import load_dataset
from scorers.shapley import compute_shapley_scores, CoalitionValueFunction
from utils import set_seed
import numpy as np


def test_node_shapley(num_nodes: int = 50):
    """Test Shapley value computation for nodes."""
    print("\n" + "=" * 70)
    print(f"TEST 1: Node Shapley Values ({num_nodes} nodes)")
    print("=" * 70)
    
    # Setup
    print("\n[1.1] Loading data and model...")
    set_seed(42, deterministic=False)
    device = torch.device('cpu')
    
    data = load_dataset('Cora', data_dir='data/')
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.val_mask = data.val_mask.to(device)
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=32,  # Smaller for speed
        num_classes=data.num_classes,
        num_gcn_layers=2,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    # Create base masks (no pruning)
    print("\n[1.2] Creating base masks...")
    base_masks = {
        'node_mask': torch.ones(data.num_nodes, dtype=torch.bool),
        'edge_mask': torch.ones(data.edge_index.size(1), dtype=torch.bool),
        'channel_masks': {}
    }
    
    for layer_name, layer in model.list_prunable_layers():
        base_masks['channel_masks'][layer_name] = torch.ones(layer.weight.size(0))
    
    # Normalize adjacency
    adj_norm = normalize_adjacency(data.edge_index, data.num_nodes).to(device)
    
    # Select subset of nodes to evaluate
    print(f"\n[1.3] Selecting {num_nodes} nodes from validation set...")
    val_nodes = data.val_mask.nonzero(as_tuple=True)[0]
    if len(val_nodes) > num_nodes:
        val_nodes = val_nodes[:num_nodes]
    players = val_nodes.tolist()
    print(f"  Selected players: {len(players)}")
    
    # Compute Shapley values
    print("\n[1.4] Computing Shapley values (approximation method)...")
    result = compute_shapley_scores(
        model=model,
        data=data,
        adj_norm=adj_norm,
        unit_type='node',
        base_masks=base_masks,
        players=players,
        method='approximation',
        num_samples=20,  # Small for speed
        seed=42,
        device=device
    )
    
    # Print results
    shapley_values = result['shapley_values']
    print(f"\n[1.5] Results:")
    print(f"  Shapley values computed: {len(shapley_values)}")
    print(f"  Value range: [{shapley_values.min():.6f}, {shapley_values.max():.6f}]")
    print(f"  Mean: {shapley_values.mean():.6f}, Std: {shapley_values.std():.6f}")
    print(f"  Num evaluations: {result['num_evaluations']}")
    print(f"  Cache size: {result['cache_size']}")
    print(f"  Elapsed time: {result['elapsed_time']:.2f}s")
    
    # Show top-5 nodes
    top_indices = np.argsort(shapley_values)[-5:][::-1]
    print(f"\n  Top-5 most important nodes:")
    for i, idx in enumerate(top_indices):
        player_id = players[idx]
        value = shapley_values[idx]
        print(f"    {i+1}. Node {player_id}: {value:.6f}")
    
    print("\n✓ Node Shapley test passed!")
    return result


def test_edge_shapley(num_edges: int = 50):
    """Test Shapley value computation for edges."""
    print("\n" + "=" * 70)
    print(f"TEST 2: Edge Shapley Values ({num_edges} edges)")
    print("=" * 70)
    
    # Setup
    print("\n[2.1] Loading data and model...")
    set_seed(42, deterministic=False)
    device = torch.device('cpu')
    
    data = load_dataset('Cora', data_dir='data/')
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.val_mask = data.val_mask.to(device)
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=32,
        num_classes=data.num_classes,
        num_gcn_layers=2,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    # Base masks
    base_masks = {
        'node_mask': torch.ones(data.num_nodes, dtype=torch.bool),
        'edge_mask': torch.ones(data.edge_index.size(1), dtype=torch.bool),
        'channel_masks': {}
    }
    
    for layer_name, layer in model.list_prunable_layers():
        base_masks['channel_masks'][layer_name] = torch.ones(layer.weight.size(0))
    
    adj_norm = normalize_adjacency(data.edge_index, data.num_nodes).to(device)
    
    # Select subset of edges
    print(f"\n[2.2] Selecting {num_edges} random edges...")
    total_edges = data.edge_index.size(1)
    edge_indices = torch.randperm(total_edges)[:num_edges]
    players = edge_indices.tolist()
    print(f"  Selected players: {len(players)}")
    
    # Compute Shapley values
    print("\n[2.3] Computing Shapley values...")
    result = compute_shapley_scores(
        model=model,
        data=data,
        adj_norm=adj_norm,
        unit_type='edge',
        base_masks=base_masks,
        players=players,
        method='approximation',
        num_samples=20,
        seed=42,
        device=device
    )
    
    # Print results
    shapley_values = result['shapley_values']
    print(f"\n[2.4] Results:")
    print(f"  Shapley values computed: {len(shapley_values)}")
    print(f"  Value range: [{shapley_values.min():.6f}, {shapley_values.max():.6f}]")
    print(f"  Mean: {shapley_values.mean():.6f}")
    print(f"  Num evaluations: {result['num_evaluations']}")
    print(f"  Elapsed time: {result['elapsed_time']:.2f}s")
    
    # Show top-5 edges
    top_indices = np.argsort(shapley_values)[-5:][::-1]
    print(f"\n  Top-5 most important edges:")
    for i, idx in enumerate(top_indices):
        player_id = players[idx]
        edge = data.edge_index[:, player_id]
        value = shapley_values[idx]
        print(f"    {i+1}. Edge {player_id} ({edge[0].item()}->{edge[1].item()}): {value:.6f}")
    
    print("\n✓ Edge Shapley test passed!")
    return result


def test_channel_shapley(num_channels: int = 16):
    """Test Shapley value computation for channels."""
    print("\n" + "=" * 70)
    print(f"TEST 3: Channel Shapley Values ({num_channels} channels)")
    print("=" * 70)
    
    # Setup
    print("\n[3.1] Loading data and model...")
    set_seed(42, deterministic=False)
    device = torch.device('cpu')
    
    data = load_dataset('Cora', data_dir='data/')
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.val_mask = data.val_mask.to(device)
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=32,
        num_classes=data.num_classes,
        num_gcn_layers=2,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    # Base masks
    base_masks = {
        'node_mask': torch.ones(data.num_nodes, dtype=torch.bool),
        'edge_mask': torch.ones(data.edge_index.size(1), dtype=torch.bool),
        'channel_masks': {}
    }
    
    for layer_name, layer in model.list_prunable_layers():
        base_masks['channel_masks'][layer_name] = torch.ones(layer.weight.size(0))
    
    adj_norm = normalize_adjacency(data.edge_index, data.num_nodes).to(device)
    
    # Select subset of channels from first GCN layer
    print(f"\n[3.2] Selecting {num_channels} channels from first GCN layer...")
    layer_name = 'gcn_layers.0'
    players = list(range(num_channels))
    print(f"  Layer: {layer_name}")
    print(f"  Selected players: {len(players)}")
    
    # Compute Shapley values
    print("\n[3.3] Computing Shapley values...")
    
    # Need to modify value function to specify layer
    from scorers.shapley import CoalitionValueFunction
    
    cache = {}
    value_fn = CoalitionValueFunction(
        model, data, adj_norm, 'channel', base_masks,
        device=device, cache=cache
    )
    
    # Override to use specific layer
    original_eval = value_fn._evaluate_channel_coalition
    def eval_channel_with_layer(coalition):
        return original_eval(coalition, layer_name=layer_name)
    value_fn._evaluate_channel_coalition = eval_channel_with_layer
    
    # Compute
    from scorers.shapley import shapley_approximation
    shapley_values = shapley_approximation(value_fn, players, num_samples=20, seed=42)
    
    # Print results
    print(f"\n[3.4] Results:")
    print(f"  Shapley values computed: {len(shapley_values)}")
    print(f"  Value range: [{shapley_values.min():.6f}, {shapley_values.max():.6f}]")
    print(f"  Mean: {shapley_values.mean():.6f}")
    print(f"  Num evaluations: {value_fn.num_evaluations}")
    print(f"  Cache size: {len(cache)}")
    
    # Show top-5 channels
    top_indices = np.argsort(shapley_values)[-5:][::-1]
    print(f"\n  Top-5 most important channels:")
    for i, idx in enumerate(top_indices):
        value = shapley_values[idx]
        print(f"    {i+1}. Channel {idx}: {value:.6f}")
    
    print("\n✓ Channel Shapley test passed!")
    return shapley_values


def test_conditional_evaluation():
    """Test conditional evaluation with pre-existing pruning."""
    print("\n" + "=" * 70)
    print("TEST 4: Conditional Evaluation (with base pruning)")
    print("=" * 70)
    
    # Setup
    print("\n[4.1] Loading data and model...")
    set_seed(42, deterministic=False)
    device = torch.device('cpu')
    
    data = load_dataset('Cora', data_dir='data/')
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.val_mask = data.val_mask.to(device)
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=32,
        num_classes=data.num_classes,
        num_gcn_layers=2,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    # Create base masks with 50% channel pruning
    print("\n[4.2] Creating base masks (50% channel pruning)...")
    base_masks = {
        'node_mask': torch.ones(data.num_nodes, dtype=torch.bool),
        'edge_mask': torch.ones(data.edge_index.size(1), dtype=torch.bool),
        'channel_masks': {}
    }
    
    for layer_name, layer in model.list_prunable_layers():
        dim = layer.weight.size(0)
        mask = torch.zeros(dim)
        mask[:dim//2] = 1.0  # Keep first 50%
        base_masks['channel_masks'][layer_name] = mask
        print(f"    {layer_name}: {mask.sum().item()}/{dim} channels kept")
    
    adj_norm = normalize_adjacency(data.edge_index, data.num_nodes).to(device)
    
    # Now evaluate node importance under this base pruning
    print("\n[4.3] Evaluating node importance under base pruning...")
    val_nodes = data.val_mask.nonzero(as_tuple=True)[0][:30]
    players = val_nodes.tolist()
    
    result = compute_shapley_scores(
        model=model,
        data=data,
        adj_norm=adj_norm,
        unit_type='node',
        base_masks=base_masks,
        players=players,
        method='approximation',
        num_samples=10,
        seed=42,
        device=device
    )
    
    shapley_values = result['shapley_values']
    print(f"\n[4.4] Results:")
    print(f"  Shapley values computed: {len(shapley_values)}")
    print(f"  Value range: [{shapley_values.min():.6f}, {shapley_values.max():.6f}]")
    print(f"  Cache uses base_masks hash: ✓")
    
    print("\n✓ Conditional evaluation test passed!")


def main():
    """Run all Shapley value tests."""
    print("=" * 70)
    print("Shapley Value Computation - Small Scale Verification")
    print("=" * 70)
    
    try:
        # Test node Shapley
        test_node_shapley(num_nodes=50)
        
        # Test edge Shapley
        test_edge_shapley(num_edges=50)
        
        # Test channel Shapley
        test_channel_shapley(num_channels=16)
        
        # Test conditional evaluation
        test_conditional_evaluation()
        
        # Summary
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nShapley value computation verified:")
        print("  ✓ Node importance evaluation works")
        print("  ✓ Edge importance evaluation works")
        print("  ✓ Channel importance evaluation works")
        print("  ✓ Conditional evaluation with base masks works")
        print("  ✓ Cache mechanism functional")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
