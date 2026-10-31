"""
Example: Edge Coalition Building

Demonstrates all 5 edge coalition strategies with detailed output.
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from coalition_builder import build_edge_coalitions


def create_example_graph():
    """
    Create a small example graph for demonstration.
    
    Structure:
    - 10 nodes
    - Community structure: {0,1,2,3} and {4,5,6,7,8,9}
    - Some triangles, bridges, etc.
    """
    edges = [
        # Community 0: densely connected
        (0, 1), (1, 0),
        (0, 2), (2, 0),
        (1, 2), (2, 1),
        (1, 3), (3, 1),
        (2, 3), (3, 2),
        
        # Bridge
        (3, 4), (4, 3),
        
        # Community 1: densely connected
        (4, 5), (5, 4),
        (4, 6), (6, 4),
        (5, 6), (6, 5),
        (5, 7), (7, 5),
        (6, 7), (7, 6),
        (7, 8), (8, 7),
        (8, 9), (9, 8),
        (7, 9), (9, 7),
    ]
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, num_edges]
    num_nodes = 10
    
    return edge_index, num_nodes


def example_1_source_node():
    """Example 1: Group by source node."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Source Node Grouping")
    print("="*70)
    
    edge_index, num_nodes = create_example_graph()
    
    result = build_edge_coalitions(
        strategy='source_node',
        edge_index=edge_index,
        num_nodes=num_nodes
    )
    
    print(result.summary())
    
    print("\nDetailed groups:")
    for i, (coalition, meta) in enumerate(zip(result.coalitions[:5], result.group_metadata[:5])):
        print(f"  {meta['name']}: edges {coalition}")


def example_2_target_node():
    """Example 2: Group by target node."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Target Node Grouping")
    print("="*70)
    
    edge_index, num_nodes = create_example_graph()
    
    result = build_edge_coalitions(
        strategy='target_node',
        edge_index=edge_index,
        num_nodes=num_nodes
    )
    
    print(result.summary())


def example_3_community():
    """Example 3: Group by community structure."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Community-based Grouping")
    print("="*70)
    
    edge_index, num_nodes = create_example_graph()
    
    # Method 1: Auto-detect communities
    result = build_edge_coalitions(
        strategy='community',
        edge_index=edge_index,
        num_nodes=num_nodes,
        method='louvain'
    )
    
    print(result.summary())
    
    print("\nGroup details:")
    for meta in result.group_metadata:
        print(f"  {meta['name']}: {meta['group_types']} ({meta['size']} edges)")
    
    # Method 2: Provide pre-computed communities
    print("\n--- With Pre-computed Communities ---")
    node_communities = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    
    result2 = build_edge_coalitions(
        strategy='community',
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_communities=node_communities
    )
    
    print(result2.summary())


def example_4_motif_bridge():
    """Example 4: Group by bridge edges."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Motif-based Grouping (Bridge)")
    print("="*70)
    
    edge_index, num_nodes = create_example_graph()
    
    result = build_edge_coalitions(
        strategy='motif_bridge',
        edge_index=edge_index,
        num_nodes=num_nodes
    )
    
    print(result.summary())
    
    print("\nBridge analysis:")
    for meta in result.group_metadata:
        print(f"  {meta['name']}: {meta['size']} edges")


def example_5_motif_triangle():
    """Example 5: Group by triangle participation."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Motif-based Grouping (Triangle)")
    print("="*70)
    
    edge_index, num_nodes = create_example_graph()
    
    result = build_edge_coalitions(
        strategy='motif_triangle',
        edge_index=edge_index,
        num_nodes=num_nodes
    )
    
    print(result.summary())
    
    print("\nTriangle participation:")
    for meta in result.group_metadata:
        print(f"  {meta['name']}: {meta['size']} edges")


def example_6_motif_clustering():
    """Example 6: Group by local clustering coefficient."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Motif-based Grouping (Clustering)")
    print("="*70)
    
    edge_index, num_nodes = create_example_graph()
    
    result = build_edge_coalitions(
        strategy='motif_local_clustering',
        edge_index=edge_index,
        num_nodes=num_nodes
    )
    
    print(result.summary())


def example_7_from_node_groups():
    """Example 7: Map from node groups."""
    print("\n" + "="*70)
    print("EXAMPLE 7: From Node Groups")
    print("="*70)
    
    edge_index, num_nodes = create_example_graph()
    
    # Define node groups (e.g., by degree)
    node_to_group = {
        0: 0, 1: 0, 2: 0, 3: 0,  # Group 0: community 0
        4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1  # Group 1: community 1
    }
    
    result = build_edge_coalitions(
        strategy='from_node_groups',
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_to_group=node_to_group,
        num_node_groups=2
    )
    
    print(result.summary())
    
    print("\nEdge group types:")
    for meta in result.group_metadata:
        print(f"  {meta['name']}: {meta['group_types']} ({meta['size']} edges)")


def validate_all_strategies():
    """Validate that all strategies work correctly."""
    print("\n" + "="*70)
    print("VALIDATION: All Strategies")
    print("="*70)
    
    edge_index, num_nodes = create_example_graph()
    
    strategies = [
        'source_node',
        'target_node',
        'community',
        'motif_bridge',
        'motif_triangle',
        'motif_local_clustering',
    ]
    
    print(f"Testing {len(strategies)} strategies on graph with {edge_index.size(1)} edges...")
    
    for strategy in strategies:
        try:
            result = build_edge_coalitions(strategy, edge_index, num_nodes)
            result.validate()
            print(f"  ✓ {strategy:25s}: {len(result.coalitions):2d} groups, "
                  f"{len(result.player_to_group):3d} edges")
        except Exception as e:
            print(f"  ✗ {strategy:25s}: {e}")
    
    # Also test from_node_groups
    try:
        node_to_group = {i: i % 3 for i in range(num_nodes)}
        result = build_edge_coalitions(
            'from_node_groups', edge_index, num_nodes,
            node_to_group=node_to_group, num_node_groups=3
        )
        result.validate()
        print(f"  ✓ {'from_node_groups':25s}: {len(result.coalitions):2d} groups, "
              f"{len(result.player_to_group):3d} edges")
    except Exception as e:
        print(f"  ✗ {'from_node_groups':25s}: {e}")


def main():
    """Run all examples."""
    print("="*70)
    print("Edge Coalition Building Examples")
    print("="*70)
    
    try:
        example_1_source_node()
        example_2_target_node()
        example_3_community()
        example_4_motif_bridge()
        example_5_motif_triangle()
        example_6_motif_clustering()
        example_7_from_node_groups()
        
        validate_all_strategies()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
