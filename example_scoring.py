"""
Example Usage of Unified Scoring System
"""
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import GCNNet, normalize_adjacency
from data import load_dataset
from scorers import score
from utils import set_seed


# def example_basic_scoring():
#     """Example: Basic scoring with different methods."""
#     print("=" * 70)
#     print("Example 1: Basic Scoring")
#     print("=" * 70)
    
#     # Setup
#     set_seed(42)
#     device = torch.device('cpu')
    
#     data = load_dataset('Cora')
#     data.x = data.x.to(device)
#     data.edge_index = data.edge_index.to(device)
    
#     model = GCNNet(
#         in_dim=data.num_features,
#         hidden_dim=64,
#         num_classes=data.num_classes,
#         num_gcn_layers=3,
#         num_mlp_layers=1,
#         dropout=0.5
#     ).to(device)
    
#     # Node scoring: degree
#     print("\n[1] Node Scoring - Degree")
#     result = score(
#         unit_type='edge',
#         method='feature_l2',
#         model=model,
#         data=data,
#         seed=42
#     )
#     edge_scores = result['scores']['edge_scores']
#     print(f"  Scores computed: {edge_scores.shape}")
#     print(f"  Score range: [{edge_scores.min():.2f}, {edge_scores.max():.2f}]")
    
#     # Edge scoring: random
#     print("\n[2] Edge Scoring - Random")
#     result = score(
#         unit_type='edge',
#         method='sensitivity',
#         model=model,
#         data=data,
#         seed=42
#     )
#     edge_scores = result['scores']['edge_scores']
#     print(f"  Scores computed: {edge_scores.shape}")
    
#     # Channel scoring: magnitude
#     print("\n[3] Channel Scoring - Magnitude")
#     result = score(
#         unit_type='channel',
#         method='activation_l2',
#         model=model,
#         data=data
#     )
#     channel_scores = result['scores']
#     print(f"  Layers scored: {list(channel_scores.keys())}")
#     for layer_name, scores in channel_scores.items():
#         print(f"    {layer_name}: {scores.shape}, range [{scores.min():.4f}, {scores.max():.4f}]")


# def example_conditional_scoring():
#     """Example: Scoring with base masks (conditional evaluation)."""
#     print("\n" + "=" * 70)
#     print("Example 2: Conditional Scoring")
#     print("=" * 70)
    
#     # Setup
#     set_seed(42)
#     device = torch.device('cpu')
    
#     data = load_dataset('Cora')
#     data.x = data.x.to(device)
#     data.edge_index = data.edge_index.to(device)
    
#     model = GCNNet(
#         in_dim=data.num_features,
#         hidden_dim=64,
#         num_classes=data.num_classes,
#         num_gcn_layers=3,
#         num_mlp_layers=1,
#         dropout=0.5
#     ).to(device)
    
#     # Create base masks with some pruning
#     print("\n[1] Creating base masks (50% edge pruning)...")
#     base_masks = {
#         'node_mask': torch.ones(data.num_nodes, dtype=torch.bool),
#         'edge_mask': torch.rand(data.edge_index.size(1)) > 0.5,  # Keep 50%
#         'channel_masks': {}
#     }
    
#     for layer_name, layer in model.list_prunable_layers():
#         base_masks['channel_masks'][layer_name] = torch.ones(layer.weight.size(0))
    
#     print(f"  Edges kept: {base_masks['edge_mask'].sum()}/{base_masks['edge_mask'].size(0)}")
    
#     # Score nodes under this edge pruning
#     print("\n[2] Scoring nodes with edge pruning...")
#     result = score(
#         unit_type='edge',
#         method='grad',
#         model=model,
#         data=data,
#         base_masks=base_masks
#     )
    
#     edge_scores = result['scores']['edge_scores']
#     print(f"  Edge scores computed: {edge_scores.shape}")
#     print(f"  (These scores reflect importance under the pruned graph)")


# def example_saving_loading():
#     """Example: Saving and loading scores."""
#     print("\n" + "=" * 70)
#     print("Example 3: Saving and Loading Scores")
#     print("=" * 70)
    
#     # Setup
#     set_seed(42)
#     device = torch.device('cpu')
    
#     data = load_dataset('Cora')
#     data.x = data.x.to(device)
#     data.edge_index = data.edge_index.to(device)
    
#     model = GCNNet(
#         in_dim=data.num_features,
#         hidden_dim=64,
#         num_classes=data.num_classes,
#         num_gcn_layers=3,
#         num_mlp_layers=1,
#         dropout=0.5
#     ).to(device)
    
#     # Compute and save scores
#     save_path = 'outputs/test_scores/node_degree.pkl'
#     print(f"\n[1] Computing and saving scores to {save_path}...")
    
#     result = score(
#         unit_type='edge',
#         method='taylor',
#         model=model,
#         data=data,
#         save_path=save_path,
#         seed=42
#     )
    
#     print(f"  Scores saved!")
#     print(f"  Elapsed time: {result['elapsed_time']:.4f}s")
    
#     # Load scores
#     print(f"\n[2] Loading scores from {save_path}...")
#     from scorers.base import PruningScorer
    
#     loaded_result = PruningScorer.load_scores(save_path)
    
#     print(f"  Loaded successfully!")
#     print(f"  Unit type: {loaded_result['unit_type']}")
#     print(f"  Method: {loaded_result['method']}")
#     print(f"  Seed: {loaded_result['seed']}")
    
#     # Verify scores match
#     original_scores = result['scores']['edge_scores']
#     loaded_scores = loaded_result['scores']['edge_scores']
    
#     if torch.equal(original_scores, loaded_scores):
#         print(f"  ✓ Scores match!")
#     else:
#         print(f"  ✗ Scores don't match!")


def example_multiple_methods():
    """Example: Compare multiple scoring methods."""
    print("\n" + "=" * 70)
    print("Example 4: Comparing Multiple Methods")
    print("=" * 70)
    
    # Setup
    set_seed(42)
    device = torch.device('cpu')
    
    data = load_dataset('Cora')
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=64,
        num_classes=data.num_classes,
        num_gcn_layers=3,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    from pruning import MaskedModelWrapper

    base_model = model  # 你训练好的 GCNNet
    model_for_channel = MaskedModelWrapper(base_model).to(device)
    model_for_channel.eval()

    print("\n[1] Computing node scores with multiple methods...")
    
    methods = [
        "random",
        "magnitude",
        "activation_l2",
        "grad",
        "taylor",
        "sensitivity",
    ]
    results = {}
    
    for method in methods:
        print(f"\n  Method: {method}")
        result = score(
            unit_type='channel',
            method=method,
            model=model_for_channel,
            data=data,
            seed=42
        )
        results[method] = result['scores']
        
    
    # Compare top-10 channels
    print(f"\n[2] Comparing top-10 channels:")
    for method in methods:
        scores = results[method]
        top10 = torch.topk(scores, 10).indices.tolist()
        print(f"  {method:10s}: {top10[:5]}...")


def main():
    """Run all examples."""
    print("=" * 70)
    print("Scoring System Usage Examples")
    print("=" * 70)
    
    try:
        # Basic scoring
        # example_basic_scoring()
        
        # # Conditional scoring
        # example_conditional_scoring()
        
        # # Saving/loading
        # example_saving_loading()
        
        # Multiple methods
        example_multiple_methods()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Example failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
