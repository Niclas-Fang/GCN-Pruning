"""
Example: Channel Coalition Building

Demonstrates channel coalition strategies with detailed output.
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from coalition_builder import build_channel_coalitions


def example_1_uniform_fixed_size():
    """Example 1: Uniform fixed-size groups."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Uniform Fixed Size Grouping")
    print("="*70)
    
    # Test different configurations
    configs = [
        (64, 8),   # 8 groups of 8
        (64, 16),  # 4 groups of 16
        (10, 4),   # 2 groups of 4, 1 group of 2 (uneven)
        (100, 32), # 3 groups of 32, 1 group of 4
    ]
    
    for num_channels, group_size in configs:
        result = build_channel_coalitions(
            'uniform_fixed_size',
            num_channels=num_channels,
            group_size=group_size
        )
        
        print(f"\nConfig: {num_channels} channels, group_size={group_size}")
        print(f"  Groups: {len(result.coalitions)}")
        print(f"  Sizes: {[len(c) for c in result.coalitions]}")
        
        # Validate
        result.validate()
        print(f"  ✓ Validation passed")


def example_2_layerwise():
    """Example 2: Layer-wise grouping."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Layer-wise Grouping")
    print("="*70)
    
    # Simulate 3-layer GCN
    layer_channel_counts = [64, 64, 32]
    layer_names = ['gcn_layer_0', 'gcn_layer_1', 'mlp_layer_0']
    
    result = build_channel_coalitions(
        'layerwise',
        layer_channel_counts=layer_channel_counts,
        layer_names=layer_names
    )
    
    print(result.summary())
    
    print("\nLayer details:")
    for meta in result.group_metadata:
        print(f"  {meta['name']}: channels {result.coalitions[meta['group_id']][0]} - "
              f"{result.coalitions[meta['group_id']][-1]} ({meta['size']} channels)")
    
    # Validate
    result.validate()
    print("\n✓ Validation passed")


def example_3_random():
    """Example 3: Random grouping."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Random Grouping")
    print("="*70)
    
    num_channels = 64
    num_groups = 8
    
    # Same seed should give same result
    result1 = build_channel_coalitions(
        'random',
        num_channels=num_channels,
        num_groups=num_groups,
        seed=42
    )
    
    result2 = build_channel_coalitions(
        'random',
        num_channels=num_channels,
        num_groups=num_groups,
        seed=42
    )
    
    print(f"Random grouping: {num_channels} channels → {num_groups} groups")
    print(f"Group sizes: {[len(c) for c in result1.coalitions]}")
    
    # Check reproducibility
    assert result1.coalitions == result2.coalitions, "Seeds should give same result"
    print("✓ Reproducibility verified (same seed → same groups)")
    
    # Different seed
    result3 = build_channel_coalitions(
        'random',
        num_channels=num_channels,
        num_groups=num_groups,
        seed=123
    )
    
    assert result1.coalitions != result3.coalitions, "Different seeds should differ"
    print("✓ Different seed gives different grouping")


def example_4_functional():
    """Example 4: Functional grouping based on activations."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Functional Grouping (Activation-based)")
    print("="*70)
    
    # Simulate activation patterns
    num_samples = 100
    num_channels = 32
    
    # Create synthetic activations with 4 functional groups
    activations = torch.zeros(num_samples, num_channels)
    
    # Group 0: channels 0-7 (high activation pattern 1)
    activations[:, 0:8] = torch.randn(num_samples, 8) + 2.0
    
    # Group 1: channels 8-15 (low activation)
    activations[:, 8:16] = torch.randn(num_samples, 8) - 1.0
    
    # Group 2: channels 16-23 (oscillating)
    for i in range(16, 24):
        activations[:, i] = torch.sin(torch.linspace(0, 10, num_samples)) + torch.randn(num_samples) * 0.1
    
    # Group 3: channels 24-31 (sparse)
    activations[:, 24:32] = torch.randn(num_samples, 8) * 0.1
    
    # Build functional coalitions
    result = build_channel_coalitions(
        'functional',
        activations=activations,
        num_groups=4,
        method='kmeans'
    )
    
    print(result.summary())
    
    print("\nFunctional groups found:")
    for i, coalition in enumerate(result.coalitions):
        print(f"  Group {i}: {len(coalition)} channels")
        print(f"    Channels: {coalition}")
    
    # Validate
    result.validate()
    print("\n✓ Validation passed")


def compare_strategies():
    """Compare different strategies on same channels."""
    print("\n" + "="*70)
    print("COMPARISON: Different Strategies")
    print("="*70)
    
    num_channels = 64
    
    strategies = {
        'uniform_8': ('uniform_fixed_size', {'group_size': 8}),
        'uniform_16': ('uniform_fixed_size', {'group_size': 16}),
        'layerwise': ('layerwise', {'layer_channel_counts': [32, 32]}),
        'random_4': ('random', {'num_groups': 4}),
    }
    
    print(f"Comparing strategies on {num_channels} channels:\n")
    print(f"{'Strategy':<20s} | {'Groups':>6s} | {'Sizes':>30s}")
    print("-" * 70)
    
    for name, (strategy, kwargs) in strategies.items():
        if strategy == 'uniform_fixed_size':
            result = build_channel_coalitions(strategy, num_channels=num_channels, **kwargs)
        else:
            result = build_channel_coalitions(strategy, num_channels=128, **kwargs)
        
        sizes_str = str([len(c) for c in result.coalitions])
        print(f"{name:<20s} | {len(result.coalitions):>6d} | {sizes_str:>30s}")


def validate_all_strategies():
    """Validate all channel coalition strategies."""
    print("\n" + "="*70)
    print("VALIDATION: All Channel Strategies")
    print("="*70)
    
    print("Testing all strategies...\n")
    
    # Test 1: Uniform fixed size
    try:
        result = build_channel_coalitions('uniform_fixed_size', num_channels=64, group_size=8)
        result.validate()
        print(f"  ✓ uniform_fixed_size: {len(result.coalitions)} groups")
    except Exception as e:
        print(f"  ✗ uniform_fixed_size: {e}")
    
    # Test 2: Layerwise
    try:
        result = build_channel_coalitions('layerwise', layer_channel_counts=[64, 64, 32])
        result.validate()
        print(f"  ✓ layerwise:          {len(result.coalitions)} groups")
    except Exception as e:
        print(f"  ✗ layerwise: {e}")
    
    # Test 3: Random
    try:
        result = build_channel_coalitions('random', num_channels=128, num_groups=8)
        result.validate()
        print(f"  ✓ random:             {len(result.coalitions)} groups")
    except Exception as e:
        print(f"  ✗ random: {e}")
    
    # Test 4: Functional (requires activations)
    try:
        activations = torch.randn(50, 32)
        result = build_channel_coalitions('functional', activations=activations, num_groups=4)
        result.validate()
        print(f"  ✓ functional:         {len(result.coalitions)} groups")
    except Exception as e:
        print(f"  ✗ functional: {e}")


def main():
    """Run all examples."""
    print("="*70)
    print("Channel Coalition Building Examples")
    print("="*70)
    
    try:
        example_1_uniform_fixed_size()
        example_2_layerwise()
        example_3_random()
        example_4_functional()
        
        compare_strategies()
        validate_all_strategies()
        
        print("\n" + "="*70)
        print("✓ All channel examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
