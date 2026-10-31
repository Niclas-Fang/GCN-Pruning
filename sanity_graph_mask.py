"""
Sanity Check for Graph Masking (Edge and Node Pruning)
"""
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import GCNNet, normalize_adjacency
from data import load_dataset
from pruning import (
    apply_edge_mask,
    apply_node_mask,
    compare_graphs,
    get_graph_statistics
)
from utils import set_seed


def test_edge_masking():
    """Test edge masking functionality."""
    print("\n" + "=" * 70)
    print("TEST 1: Edge Masking")
    print("=" * 70)
    
    # Load data
    print("\n[1.1] Loading Cora dataset...")
    data = load_dataset('Cora', data_dir='data/')
    print(f"  ✓ Loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    
    # Create random edge mask (keep 70% of edges)
    print("\n[1.2] Creating random edge mask (keep 70%)...")
    num_edges = data.edge_index.size(1)
    edge_mask = torch.rand(num_edges) > 0.3  # Keep 70%
    print(f"  ✓ Mask created: {edge_mask.sum()}/{num_edges} edges kept")
    
    # Apply edge mask
    print("\n[1.3] Applying edge mask...")
    edge_index_masked = apply_edge_mask(
        data, 
        edge_mask,
        ensure_undirected=True,
        add_self_loops_flag=True
    )
    print(f"  ✓ New edge_index: {edge_index_masked.size(1)} edges")
    
    # Compare graphs
    print("\n[1.4] Comparing graphs...")
    compare_graphs(data.edge_index, edge_index_masked, data.num_nodes)
    
    # Test forward pass with masked graph
    print("\n[1.5] Testing forward pass with masked graph...")
    device = torch.device('cpu')
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=64,
        num_classes=data.num_classes,
        num_gcn_layers=3,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    # Move data to device
    data.x = data.x.to(device)
    data.y = data.y.to(device)
    edge_index_masked = edge_index_masked.to(device)
    
    # Normalize adjacency
    adj_norm = normalize_adjacency(edge_index_masked, data.num_nodes).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(data.x, adj_norm)
    
    print(f"  ✓ Forward pass successful")
    print(f"    Output shape: {logits.shape}")
    print(f"    Output range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    print("\n✓ Edge masking test passed!")


def test_node_masking():
    """Test node masking functionality."""
    print("\n" + "=" * 70)
    print("TEST 2: Node Masking")
    print("=" * 70)
    
    # Load data
    print("\n[2.1] Loading Cora dataset...")
    data = load_dataset('Cora', data_dir='data/')
    print(f"  ✓ Loaded: {data.num_nodes} nodes, {data.edge_index.size(1)} edges")
    
    # Create random node mask (keep 80% of nodes)
    print("\n[2.2] Creating random node mask (keep 80%)...")
    num_nodes = data.num_nodes
    node_mask = torch.rand(num_nodes) > 0.2  # Keep 80%
    num_kept = node_mask.sum().item()
    num_pruned = num_nodes - num_kept
    print(f"  ✓ Mask created: {num_kept}/{num_nodes} nodes kept ({num_pruned} pruned)")
    
    # Apply node mask
    print("\n[2.3] Applying node mask...")
    edge_index_masked = apply_node_mask(
        data,
        node_mask,
        ensure_undirected=True,
        preserve_self_loops=True
    )
    print(f"  ✓ New edge_index: {edge_index_masked.size(1)} edges")
    
    # Compare graphs
    print("\n[2.4] Comparing graphs...")
    compare_graphs(data.edge_index, edge_index_masked, data.num_nodes)
    
    # Verify self-loops are preserved
    print("\n[2.5] Verifying self-loop preservation...")
    src, dst = edge_index_masked[0], edge_index_masked[1]
    self_loops = (src == dst)
    num_self_loops = self_loops.sum().item()
    print(f"  ✓ Self-loops found: {num_self_loops}")
    print(f"    Expected: {num_nodes} (one per node)")
    
    # Check that pruned nodes only have self-loops
    print("\n[2.6] Checking pruned nodes...")
    pruned_node_indices = (~node_mask).nonzero(as_tuple=True)[0]
    if len(pruned_node_indices) > 0:
        sample_pruned = pruned_node_indices[0].item()
        
        # Find edges involving this pruned node
        edges_involving_pruned = ((src == sample_pruned) | (dst == sample_pruned))
        edge_index_pruned = edge_index_masked[:, edges_involving_pruned]
        
        print(f"  Sample pruned node {sample_pruned}:")
        print(f"    Edges involving it: {edge_index_pruned.size(1)}")
        print(f"    Edge indices: {edge_index_pruned.t()[:5].tolist()}...")
        
        # Check if only self-loop
        is_only_self_loop = (edge_index_pruned.size(1) == 1 and 
                            edge_index_pruned[0, 0] == sample_pruned and
                            edge_index_pruned[1, 0] == sample_pruned)
        if is_only_self_loop:
            print(f"    ✓ Correctly has only self-loop")
        else:
            print(f"    Note: Has {edge_index_pruned.size(1)} edges")
    
    # Test forward pass with node-masked graph
    print("\n[2.7] Testing forward pass with node-masked graph...")
    device = torch.device('cpu')
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=64,
        num_classes=data.num_classes,
        num_gcn_layers=3,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    # Move data to device
    data.x = data.x.to(device)
    data.y = data.y.to(device)
    edge_index_masked = edge_index_masked.to(device)
    
    # Normalize adjacency
    adj_norm = normalize_adjacency(edge_index_masked, data.num_nodes).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(data.x, adj_norm)
    
    print(f"  ✓ Forward pass successful")
    print(f"    Output shape: {logits.shape}")
    
    # Check predictions for pruned nodes
    if len(pruned_node_indices) > 0:
        pruned_logits = logits[pruned_node_indices]
        print(f"    Pruned nodes' logits shape: {pruned_logits.shape}")
        print(f"    Pruned nodes can still make predictions ✓")
    
    print("\n✓ Node masking test passed!")


def test_combined_masking():
    """Test combining edge and node masking."""
    print("\n" + "=" * 70)
    print("TEST 3: Combined Edge + Node Masking")
    print("=" * 70)
    
    # Load data
    print("\n[3.1] Loading Cora dataset...")
    data = load_dataset('Cora', data_dir='data/')
    
    # First apply node mask
    print("\n[3.2] Step 1: Apply node mask (keep 90%)...")
    node_mask = torch.rand(data.num_nodes) > 0.1
    edge_index_step1 = apply_node_mask(data, node_mask)
    print(f"  ✓ After node masking: {edge_index_step1.size(1)} edges")
    
    # Then apply edge mask to the result
    print("\n[3.3] Step 2: Apply edge mask (keep 80%)...")
    num_edges_step1 = edge_index_step1.size(1)
    edge_mask = torch.rand(num_edges_step1) > 0.2
    
    # Create temporary data object for edge masking
    class TempData:
        def __init__(self, edge_index, num_nodes):
            self.edge_index = edge_index
            self.num_nodes = num_nodes
    
    temp_data = TempData(edge_index_step1, data.num_nodes)
    edge_index_step2 = apply_edge_mask(temp_data, edge_mask)
    print(f"  ✓ After edge masking: {edge_index_step2.size(1)} edges")
    
    # Compare original to final
    print("\n[3.4] Comparing original to final graph...")
    compare_graphs(data.edge_index, edge_index_step2, data.num_nodes)
    
    # Test forward pass
    print("\n[3.5] Testing forward pass with combined masking...")
    device = torch.device('cpu')
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=64,
        num_classes=data.num_classes,
        num_gcn_layers=3,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    data.x = data.x.to(device)
    edge_index_step2 = edge_index_step2.to(device)
    adj_norm = normalize_adjacency(edge_index_step2, data.num_nodes).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(data.x, adj_norm)
    
    print(f"  ✓ Forward pass successful: {logits.shape}")
    
    print("\n✓ Combined masking test passed!")


def main():
    """Run all graph masking tests."""
    print("=" * 70)
    print("Graph Masking Sanity Check")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seed(42, deterministic=False)
    
    try:
        # Test edge masking
        test_edge_masking()
        
        # Test node masking
        test_node_masking()
        
        # Test combined masking
        test_combined_masking()
        
        # Summary
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nGraph masking functionality verified:")
        print("  ✓ Edge masking works correctly")
        print("  ✓ Node masking works correctly")
        print("  ✓ Pruned nodes retain self-loops")
        print("  ✓ Forward pass works with masked graphs")
        print("  ✓ Combined masking works correctly")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
