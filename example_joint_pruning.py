"""
Example: Run Joint Pruning Experiments

Demonstrates different configurations and usage patterns.
"""
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import GCNNet
from data import load_dataset
from run_joint_pruning import (
    JointPruningRunner,
    create_keep_ratio_grid,
    create_keep_ratio_sequence
)
from utils import set_seed


def example_1_simple():
    """Example 1: Simple joint pruning with degree + magnitude."""
    print("=" * 70)
    print("Example 1: Simple Joint Pruning (Degree + Magnitude)")
    print("=" * 70)
    
    # Setup
    set_seed(42)
    device = torch.device('cpu')
    
    # Load data
    data = load_dataset('Cora')
    
    # Create model
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=64,
        num_classes=data.num_classes,
        num_gcn_layers=3,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    # Create runner
    runner = JointPruningRunner(
        model=model,
        data=data,
        graph_unit='edge',
        device=device,
        seed=42
    )
    
    # Define keep ratios (graph, model)
    keep_ratios = [
        (1.0, 1.0),  # No pruning
        (0.9, 0.9),  # 10% pruning
        (0.8, 0.8),  # 20% pruning
        (0.7, 0.7),  # 30% pruning
    ]
    
    # Run experiment
    results = runner.run_experiment(
        graph_scoring_method='degree',
        model_scoring_method='magnitude',
        keep_ratios=keep_ratios,
        channel_strategy='global',
        output_dir='outputs/example1_simple'
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("Results Summary:")
    print("=" * 70)
    for r in results:
        print(f"Keep Ratios: G={r['graph_keep_ratio']:.2f}, M={r['model_keep_ratio']:.2f}")
        print(f"  Sparsity: G={r['graph_sparsity']:.2%}, M={r['model_sparsity']:.2%}")
        print(f"  Test Acc: {r['test_acc']:.4f}")
        print()


def example_2_grid_search():
    """Example 2: Grid search over keep ratios."""
    print("\n" + "=" * 70)
    print("Example 2: Grid Search (Random baselines)")
    print("=" * 70)
    
    # Setup
    set_seed(42)
    device = torch.device('cpu')
    
    data = load_dataset('Cora')
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=32,  # Smaller for speed
        num_classes=data.num_classes,
        num_gcn_layers=2,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    runner = JointPruningRunner(
        model=model,
        data=data,
        graph_unit='edge',  # Edge pruning
        device=device,
        seed=42
    )
    
    # Create 2D grid
    graph_ratios = [1.0, 0.8, 0.6]
    model_ratios = [1.0, 0.8, 0.6]
    keep_ratios = create_keep_ratio_grid(graph_ratios, model_ratios)
    
    print(f"Grid: {len(graph_ratios)} × {len(model_ratios)} = {len(keep_ratios)} configs")
    
    # Run
    results = runner.run_experiment(
        graph_scoring_method='random',
        model_scoring_method='random',
        keep_ratios=keep_ratios,
        channel_strategy='layerwise',
        output_dir='outputs/example2_grid'
    )
    
    # Find best configuration
    best = max(results, key=lambda x: x['test_acc'])
    print("\n" + "=" * 70)
    print("Best Configuration:")
    print(f"  Graph ratio: {best['graph_keep_ratio']:.2f}")
    print(f"  Model ratio: {best['model_keep_ratio']:.2f}")
    print(f"  Test accuracy: {best['test_acc']:.4f}")
    print("=" * 70)


def example_3_compare_strategies():
    """Example 3: Compare global vs layerwise channel pruning."""
    print("\n" + "=" * 70)
    print("Example 3: Compare Channel Pruning Strategies")
    print("=" * 70)
    
    # Setup
    set_seed(42)
    device = torch.device('cpu')
    
    data = load_dataset('Cora')
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=64,
        num_classes=data.num_classes,
        num_gcn_layers=3,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    # Keep ratios
    keep_ratios = [(1.0, 0.8), (1.0, 0.6), (1.0, 0.4)]
    
    results_all = {}
    
    for strategy in ['global', 'layerwise']:
        print(f"\n--- Strategy: {strategy} ---")
        
        runner = JointPruningRunner(
            model=model,
            data=data,
            graph_unit='node',
            device=device,
            seed=42
        )
        
        results = runner.run_experiment(
            graph_scoring_method='degree',
            model_scoring_method='magnitude',
            keep_ratios=keep_ratios,
            channel_strategy=strategy,
            output_dir=f'outputs/example3_{strategy}'
        )
        
        results_all[strategy] = results
    
    # Compare
    print("\n" + "=" * 70)
    print("Strategy Comparison:")
    print("=" * 70)
    print(f"{'Ratio':>6s} | {'Global Acc':>10s} | {'Layerwise Acc':>14s}")
    print("-" * 70)
    
    for i in range(len(keep_ratios)):
        ratio = keep_ratios[i][1]
        global_acc = results_all['global'][i]['test_acc']
        layer_acc = results_all['layerwise'][i]['test_acc']
        print(f"{ratio:>6.2f} | {global_acc:>10.4f} | {layer_acc:>14.4f}")


def example_4_node_vs_edge():
    """Example 4: Compare node vs edge pruning."""
    print("\n" + "=" * 70)
    print("Example 4: Node vs Edge Pruning")
    print("=" * 70)
    
    # Setup
    set_seed(42)
    device = torch.device('cpu')
    
    data = load_dataset('Cora')
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=64,
        num_classes=data.num_classes,
        num_gcn_layers=3,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    keep_ratios = [(0.9, 1.0), (0.8, 1.0), (0.7, 1.0)]
    
    results_all = {}
    
    for graph_unit in ['node', 'edge']:
        print(f"\n--- Graph unit: {graph_unit} ---")
        
        runner = JointPruningRunner(
            model=model,
            data=data,
            graph_unit=graph_unit,
            device=device,
            seed=42
        )
        
        scoring_method = 'degree' if graph_unit == 'node' else 'random'
        
        results = runner.run_experiment(
            graph_scoring_method=scoring_method,
            model_scoring_method='magnitude',
            keep_ratios=keep_ratios,
            channel_strategy='global',
            output_dir=f'outputs/example4_{graph_unit}'
        )
        
        results_all[graph_unit] = results
    
    # Compare
    print("\n" + "=" * 70)
    print("Node vs Edge Pruning:")
    print("=" * 70)
    print(f"{'Keep Ratio':>11s} | {'Node Acc':>9s} | {'Edge Acc':>9s}")
    print("-" * 70)
    
    for i in range(len(keep_ratios)):
        ratio = keep_ratios[i][0]
        node_acc = results_all['node'][i]['test_acc']
        edge_acc = results_all['edge'][i]['test_acc']
        print(f"{ratio:>11.2f} | {node_acc:>9.4f} | {edge_acc:>9.4f}")


def example_5_custom_ratios():
    """Example 5: Custom keep ratio configurations."""
    print("\n" + "=" * 70)
    print("Example 5: Custom Keep Ratio Configurations")
    print("=" * 70)
    
    # Setup
    set_seed(42)
    device = torch.device('cpu')
    
    data = load_dataset('Cora')
    
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=64,
        num_classes=data.num_classes,
        num_gcn_layers=3,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    runner = JointPruningRunner(
        model=model,
        data=data,
        graph_unit='node',
        device=device,
        seed=42
    )
    
    # Custom configurations
    # Format: (graph_keep_ratio, model_keep_ratio)
    keep_ratios = [
        (1.0, 1.0),  # Baseline
        (0.9, 1.0),  # Prune only graph
        (1.0, 0.9),  # Prune only model
        (0.9, 0.9),  # Prune both equally
        (0.8, 0.95), # More graph pruning
        (0.95, 0.8), # More model pruning
    ]
    
    results = runner.run_experiment(
        graph_scoring_method='degree',
        model_scoring_method='magnitude',
        keep_ratios=keep_ratios,
        channel_strategy='global',
        output_dir='outputs/example5_custom'
    )
    
    # Analyze
    print("\n" + "=" * 70)
    print("Custom Configuration Results:")
    print("=" * 70)
    
    baseline = results[0]['test_acc']
    
    for r in results[1:]:  # Skip baseline
        graph_r = r['graph_keep_ratio']
        model_r = r['model_keep_ratio']
        acc = r['test_acc']
        acc_drop = baseline - acc
        
        print(f"G={graph_r:.2f}, M={model_r:.2f}: Acc={acc:.4f} (drop: {acc_drop:+.4f})")


def main():
    """Run all examples."""
    print("=" * 70)
    print("Joint Pruning Experiments - Examples")
    print("=" * 70)
    
    try:
        # Simple example
        example_1_simple()
        
        # Grid search
        # example_2_grid_search()  # Commented out - takes longer
        
        # Strategy comparison
        # example_3_compare_strategies()  # Commented out - takes longer
        
        # Node vs edge
        # example_4_node_vs_edge()  # Commented out - takes longer
        
        # Custom ratios
        # example_5_custom_ratios()  # Commented out - takes longer
        
        print("\n" + "=" * 70)
        print("✓ Examples complete!")
        print("=" * 70)
        print("\nTo run more examples, uncomment them in the script.")
        print("Results are saved to outputs/example*/ directories.")
        
    except Exception as e:
        print(f"\n✗ Example failed:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
