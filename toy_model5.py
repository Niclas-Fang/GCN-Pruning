"""
Toy Edge Owen-Shapley Demo - Pure Interface Reuse

Demonstrates edge coalition grouping + Owen-Shapley on toy graph.
Uses ONLY existing interfaces from:
  - coalition_builder/edge_builders.py
  - scorers/shapley.py
  
No reimplementation - pure composition of existing modules.
"""
import sys
import math
from itertools import combinations
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import existing modules
from coalition_builder.base import CoalitionResult
from coalition_builder.edge_builders import build_edge_coalitions
from scorers.shapley import CoalitionValueFunction, owen_shapley_two_stage


# ============================================
# Toy Graph Construction
# ============================================

@dataclass
class ToyData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_nodes: int


def build_toy_edge_pruning_graph(device="cpu", make_undirected=True):
    """Build toy graph (from your specification)."""
    x = torch.tensor([
        # [1.0, -1.0],  # v0 center
        # [1.0, 0.0],  # v1 relay
        # [1.0, 0.0],  # v2 relay
        # [1.0, 1.0],  # v3 positive source
        # [1.0, 1.0],  # v4 positive source
        # [-1.0, -1.0],  # v5 negative/noisy node
        # [1.0, -2.0],  # v6 negative/noisy node
        [0.0, 0.0],  # v0 center
        [0.0, 1.0],  # v1 relay
        [0.0, 1.0],  # v2 relay
        [1.0, 0.0],  # v3 positive source
        [1.0, 0.0],  # v4 positive source
        [0.0, 0.0],  # v5 positive source
        [0.0, 0.0],  # v6 positive source
        [0.0, 0.0],  # v7 negative/noisy node
    ], dtype=torch.float, device=device)

    y = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0], dtype=torch.long, device=device)

    undirected_edges = [
        (0, 1), (0, 2), (1, 3), (2, 4),(3, 4),
        (0, 5), (5, 6), (6, 7),
    ]

    edge_list = []
    edge_name_map = []

    if make_undirected:
        for u, v in undirected_edges:
            edge_list.append([u, v])
            edge_name_map.append(f"({u},{v})")
            edge_list.append([v, u])
            edge_name_map.append(f"({v},{u})")
    else:
        for u, v in undirected_edges:
            edge_list.append([u, v])
            edge_name_map.append(f"({u},{v})")

    edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    num_nodes = x.size(0)

    train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    data = ToyData(
        x=x, edge_index=edge_index, y=y,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        num_nodes=num_nodes
    )

    return data, edge_name_map


# ============================================
# Directed → Undirected Folding
# ============================================

def build_undirected_mapping(edge_index: torch.Tensor) -> Dict:
    """
    Map directed edges to undirected edge groups.
    
    Returns:
        {
            (u, v): [directed_edge_id1, directed_edge_id2, ...],
            ...
        }
    where (u, v) is canonical: u < v
    """
    mapping = defaultdict(list)
    
    for e_id in range(edge_index.size(1)):
        u = int(edge_index[0, e_id])
        v = int(edge_index[1, e_id])
        
        # Canonical key
        key = (min(u, v), max(u, v))
        mapping[key].append(e_id)
    
    return dict(mapping)


def fold_to_undirected(
    directed_scores: np.ndarray,
    edge_index: torch.Tensor
) -> Dict:
    """
    Fold directed edge scores to undirected edges.
    
    Args:
        directed_scores: [num_directed_edges] scores
        edge_index: [2, num_directed_edges]
    
    Returns:
        {
            (u, v): avg_score,
            ...
        }
    """
    undirected_map = build_undirected_mapping(edge_index)
    
    undirected_scores = {}
    for (u, v), edge_ids in undirected_map.items():
        # Average scores of all directed instances
        scores = [directed_scores[e_id] for e_id in edge_ids]
        undirected_scores[(u, v)] = np.mean(scores)
    
    return undirected_scores

# ============================================
# Exact Shapley (Self-written)
# ============================================

def exact_shapley_values(
    value_fn,
    players: List[int]
) -> np.ndarray:
    """
    Compute exact Shapley values for all players.

    Args:
        value_fn: callable coalition value function, accepts List[int]
        players: list of player ids

    Returns:
        np.ndarray of shape [num_players], aligned with `players`
    """
    n = len(players)
    if n == 0:
        return np.array([], dtype=float)

    factorial = math.factorial
    n_fact = factorial(n)

    shapley = np.zeros(n, dtype=float)

    # Map index -> player
    for i, player in enumerate(players):
        other_players = [p for p in players if p != player]

        phi_i = 0.0

        # Enumerate all subsets S ⊆ N \ {i}
        for s in range(n):
            for subset in combinations(other_players, s):
                subset = list(subset)

                weight = factorial(s) * factorial(n - s - 1) / n_fact

                v_s = value_fn(subset)
                v_s_with_i = value_fn(subset + [player])

                phi_i += weight * (v_s_with_i - v_s)

        shapley[i] = phi_i

    return shapley


# # 统一版本
# class ToyGCNValueFunction:
#     """
#     Simplified value function for toy graph.

#     Value: v(S) = -classification_loss
#     where classification uses parameter-free 2-step neighborhood mean propagation
#     on the selected directed edges in S.
#     """

#     def __init__(self, data, device='cpu'):
#         self.data = data
#         self.device = device

#         # Keep the original classifier head and external interface
#         self.classifier = nn.Linear(2, 2).to(device)
#         nn.init.xavier_uniform_(self.classifier.weight)

#         self.cache = {}
#         self.num_calls = 0

#     def _mean_aggregate(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
#         """
#         Pure incoming-neighbor mean aggregation.
#         No residual/self addition, no extra scaling.
#         """
#         out = torch.zeros_like(features)

#         if edge_index.size(1) == 0:
#             return out

#         for node in range(self.data.num_nodes):
#             incoming = edge_index[1] == node
#             if incoming.any():
#                 neighbor_ids = edge_index[0][incoming]
#                 out[node] = features[neighbor_ids].mean(dim=0)

#         return out

#     def __call__(self, edge_coalition: List[int]) -> float:
#         """Evaluate coalition of directed edges."""
#         coalition_key = tuple(sorted(edge_coalition))
#         if coalition_key in self.cache:
#             return self.cache[coalition_key]

#         self.num_calls += 1

#         edge_mask = torch.zeros(
#             self.data.edge_index.size(1),
#             dtype=torch.bool,
#             device=self.data.edge_index.device
#         )
#         edge_mask[edge_coalition] = True

#         masked_edge_index = self.data.edge_index[:, edge_mask]

#         # Pure 2-step mean propagation
#         h1 = self._mean_aggregate(self.data.x, masked_edge_index)
#         h2 = self._mean_aggregate(h1, masked_edge_index)

#         logits = self.classifier(h2)
#         loss = F.cross_entropy(
#             logits[self.data.val_mask],
#             self.data.y[self.data.val_mask]
#         )

#         value = -loss.item()
#         self.cache[coalition_key] = value
#         return value

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ToyGCNValueFunction:
    """
    Simplified value function for toy graph.

    Value: v(S) = -classification_loss
    where classification uses parameter-free 2-step neighborhood mean propagation
    on the selected directed edges in S.
    """

    def __init__(self, data, device='cpu'):
        self.data = data
        self.device = device

        # Keep the original classifier head and external interface
        self.classifier = nn.Linear(2, 2).to(device)
        nn.init.xavier_uniform_(self.classifier.weight)

        self.cache = {}
        self.num_calls = 0

    def _mean_aggregate(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Neighborhood mean aggregation.

        For each node i, collect features of all its incoming neighbors j
        (i.e. edges j -> i in edge_index), and compute their mean:

            h_i' = mean({ h_j | (j -> i) in E })

        Notes:
        - No self-loop is added automatically
        - No residual/self feature addition
        - No learnable transformation inside propagation
        - If a node has no incoming neighbors, its output is set to zero
        """
        out = torch.zeros_like(features)

        if edge_index.size(1) == 0:
            return out

        for node in range(self.data.num_nodes):
            incoming_mask = (edge_index[1] == node)

            if incoming_mask.any():
                neighbor_ids = edge_index[0][incoming_mask]
                neighbor_features = features[neighbor_ids]
                out[node] = neighbor_features.mean(dim=0)

        return out

    def propagate_features(self, edge_coalition: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function for inspection/debugging:
        return 1-step and 2-step propagated node features.
        This does not change the original external interface.
        """
        edge_mask = torch.zeros(
            self.data.edge_index.size(1),
            dtype=torch.bool,
            device=self.data.edge_index.device
        )
        edge_mask[edge_coalition] = True

        masked_edge_index = self.data.edge_index[:, edge_mask]

        h1 = self._mean_aggregate(self.data.x, masked_edge_index)
        h2 = self._mean_aggregate(h1, masked_edge_index)

        return h1, h2

    def __call__(self, edge_coalition: List[int]) -> float:
        """Evaluate coalition of directed edges."""
        coalition_key = tuple(sorted(edge_coalition))
        if coalition_key in self.cache:
            return self.cache[coalition_key]

        self.num_calls += 1

        edge_mask = torch.zeros(
            self.data.edge_index.size(1),
            dtype=torch.bool,
            device=self.data.edge_index.device
        )
        edge_mask[edge_coalition] = True

        masked_edge_index = self.data.edge_index[:, edge_mask]

        # 2-step neighborhood mean propagation
        h1 = self._mean_aggregate(self.data.x, masked_edge_index)
        h2 = self._mean_aggregate(h1, masked_edge_index)

        logits = self.classifier(h2)
        loss = F.cross_entropy(
            logits[self.data.val_mask],
            self.data.y[self.data.val_mask]
        )

        value = -loss.item()
        self.cache[coalition_key] = value
        return value


# ============================================
# Visualization (Text-based)
# ============================================

def visualize_coalition_text(
    coalition_result: CoalitionResult,
    edge_name_map: List[str]
):
    """Text-based coalition visualization."""
    print(f"\n{'='*70}")
    print(f"Coalition Strategy: {coalition_result.strategy}")
    print(f"{'='*70}")
    
    for meta in coalition_result.group_metadata:
        g_id = meta['group_id']
        name = meta['name']
        size = meta['size']
        
        edge_ids = coalition_result.coalitions[g_id]
        edge_names = [edge_name_map[e_id] for e_id in edge_ids[:5]]  # First 5
        
        print(f"\n  Group {g_id}: {name} ({size} edges)")
        print(f"    Edges: {', '.join(edge_names)}")
        if size > 5:
            print(f"    ... and {size - 5} more")


def visualize_scores_text(
    undirected_scores: Dict,
    strategy: str
):
    """Text-based score visualization."""
    print(f"\n{'='*70}")
    print(f"Undirected Edge Scores: {strategy}")
    print(f"{'='*70}")
    
    # Sort by score
    sorted_edges = sorted(undirected_scores.items(), 
                         key=lambda x: x[1], reverse=True)
    
    print(f"\n  {'Edge':<12s} {'Score':>10s}")
    print(f"  {'-'*22}")
    
    for (u, v), score in sorted_edges:
        print(f"  ({u},{v}){' '*8} {score:>10.4f}")
    
    print(f"\n  Mean:  {np.mean(list(undirected_scores.values())):.4f}")
    print(f"  Std:   {np.std(list(undirected_scores.values())):.4f}")


# ============================================
# Main Demo Function
# ============================================

def run_strategy(
    strategy_name: str,
    data: ToyData,
    edge_name_map: List[str],
    value_fn: ToyGCNValueFunction,
    device: str = 'cpu',
    **strategy_kwargs
):
    """
    Run one edge grouping strategy end-to-end.
    
    Steps:
    1. Build coalitions using existing builder
    2. Visualize coalition grouping
    3. Compute Owen-Shapley scores using existing scorer
    3b. Compute Exact Shapley scores using self-written function
    4. Fold to undirected edges
    5. Visualize scores
    """
    print(f"\n\n{'#'*70}")
    print(f"# STRATEGY: {strategy_name}")
    print(f"{'#'*70}")
    
    # ============================================
    # Step 1: Build Coalitions (EXISTING INTERFACE)
    # ============================================
    print(f"\n[Step 1] Building coalitions with existing builder...")
    
    try:
        coalition_result = build_edge_coalitions(
            strategy=strategy_name,
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            **strategy_kwargs
        )
        
        print(f"  ✓ Built {len(coalition_result.coalitions)} coalitions")
        print(f"  ✓ Covering {len(coalition_result.player_to_group)} directed edges")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return None
    
    # ============================================
    # Step 2: Visualize Coalition Grouping
    # ============================================
    print(f"\n[Step 2] Visualizing coalition grouping...")
    visualize_coalition_text(coalition_result, edge_name_map)
    
    # ============================================
    # Step 3: Compute Owen-Shapley (EXISTING INTERFACE)
    # ============================================
    print(f"\n[Step 3] Computing Owen-Shapley scores with existing scorer...")
    
    players = list(coalition_result.player_to_group.keys())
    
    print(f"  Players: {len(players)} directed edges")
    print(f"  Coalitions: {len(coalition_result.coalitions)} groups")
    print(f"  Running Owen-Shapley...")
    
    calls_before_owen = value_fn.num_calls
    owen_values = owen_shapley_two_stage(
        value_fn=value_fn,
        players=players,
        coalitions=coalition_result.coalitions,
        num_samples_q=5,
        num_samples_p=5,
        K=5,
        seed=456
    )
    calls_after_owen = value_fn.num_calls
    
    print(f"  ✓ Computed {len(owen_values)} Owen values")
    print(f"  ✓ Value function called {calls_after_owen - calls_before_owen} times "
          f"(cumulative: {value_fn.num_calls})")
    
    # ============================================
    # Step 3b: Compute Exact Shapley (SELF-WRITTEN)
    # ============================================
    print(f"\n[Step 3b] Computing Exact Shapley scores with self-written function...")
    print(f"  Running exact Shapley...")
    
    calls_before_exact = value_fn.num_calls
    exact_values = exact_shapley_values(
        value_fn=value_fn,
        players=players
    )
    calls_after_exact = value_fn.num_calls
    
    print(f"  ✓ Computed {len(exact_values)} exact Shapley values")
    print(f"  ✓ Value function called {calls_after_exact - calls_before_exact} times "
          f"(cumulative: {value_fn.num_calls})")
    
    # ============================================
    # Step 4: Fold Owen-Shapley to Undirected Edges
    # ============================================
    print(f"\n[Step 4] Folding Owen-Shapley to 7 undirected edges...")
    
    player_to_owen_score = {players[i]: owen_values[i] for i in range(len(players))}
    directed_owen_scores = np.zeros(data.edge_index.size(1))
    for player, score in player_to_owen_score.items():
        directed_owen_scores[player] = score
    
    undirected_owen_scores = fold_to_undirected(directed_owen_scores, data.edge_index)
    
    print(f"  ✓ Folded to {len(undirected_owen_scores)} undirected edges")
    
    # ============================================
    # Step 4b: Fold Exact Shapley to Undirected Edges
    # ============================================
    print(f"\n[Step 4b] Folding Exact Shapley to 7 undirected edges...")
    
    player_to_exact_score = {players[i]: exact_values[i] for i in range(len(players))}
    directed_exact_scores = np.zeros(data.edge_index.size(1))
    for player, score in player_to_exact_score.items():
        directed_exact_scores[player] = score
    
    undirected_exact_scores = fold_to_undirected(directed_exact_scores, data.edge_index)
    
    print(f"  ✓ Folded to {len(undirected_exact_scores)} undirected edges")
    
    # ============================================
    # Step 5: Visualize Owen-Shapley Scores
    # ============================================
    print(f"\n[Step 5] Visualizing Owen-Shapley undirected edge scores...")
    visualize_scores_text(undirected_owen_scores, f"{strategy_name} (Owen)")
    
    # ============================================
    # Step 5b: Visualize Exact Shapley Scores
    # ============================================
    print(f"\n[Step 5b] Visualizing Exact Shapley undirected edge scores...")
    visualize_scores_text(undirected_exact_scores, f"{strategy_name} (Exact)")
    
    return {
        'coalition_result': coalition_result,
        'owen_values': owen_values,
        'exact_values': exact_values,
        'undirected_scores': undirected_owen_scores,          # 保持原键不变，兼容原 summary
        'undirected_exact_scores': undirected_exact_scores,   # 新增 exact 结果
        'players': players
    }


def main():
    """Main entry point - test all edge grouping strategies."""
    
    print("="*70)
    print("Toy Edge Owen-Shapley Demo")
    print("Pure Interface Reuse - No Reimplementation")
    print("="*70)
    
    device = 'cpu'
    
    # ============================================
    # Setup
    # ============================================
    print("\n[Setup] Building toy graph...")
    data, edge_name_map = build_toy_edge_pruning_graph(device, make_undirected=True)
    
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Directed edges: {data.edge_index.size(1)}")
    print(f"  Undirected edges: 7")
    
    print("\n[Setup] Creating value function...")
    value_fn = ToyGCNValueFunction(data, device)
    
    # Test value function
    all_edges = list(range(data.edge_index.size(1)))
    v_full = value_fn(all_edges)
    v_empty = value_fn([])
    print(f"  v(all edges): {v_full:.4f}")
    print(f"  v(empty):     {v_empty:.4f}")
    
    # ============================================
    # Test All Available Strategies
    # ============================================
    
    strategies_to_test = [
        ('source_node', {}),
        ('target_node', {}),
        ('community', {'method': 'louvain'}),
        ('motif_bridge', {}),
        ('motif_triangle', {}),
        ('motif_local_clustering', {}),
    ]
    
    results = {}
    
    for strategy_name, kwargs in strategies_to_test:
        try:
            result = run_strategy(
                strategy_name, data, edge_name_map, value_fn,
                device=device, **kwargs
            )
            if result is not None:
                results[strategy_name] = result
        except Exception as e:
            print(f"\n✗ Strategy '{strategy_name}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================
    # Summary
    # ============================================
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n✓ Tested {len(results)} strategies successfully")
    
    for strategy_name, result in results.items():
        undirected_scores = result['undirected_scores']
        mean_score = np.mean(list(undirected_scores.values()))
        
        print(f"\n  {strategy_name:20s}: mean score = {mean_score:.4f}")
        
        # Show top edge
        top_edge = max(undirected_scores.items(), key=lambda x: x[1])
        print(f"    Top edge: {top_edge[0]} = {top_edge[1]:.4f}")
    
    # Compare strategies
    if len(results) > 1:
        print("\n[Cross-Strategy Comparison]")
        
        # Get common undirected edges
        all_edges_set = set()
        for result in results.values():
            all_edges_set.update(result['undirected_scores'].keys())
        
        print(f"\n  {'Edge':<12s}", end='')
        for strategy in results.keys():
            print(f" {strategy[:10]:>10s}", end='')
        print()
        
        print(f"  {'-'*12}", end='')
        for _ in results.keys():
            print(f" {'-'*10}", end='')
        print()
        
        for edge in sorted(all_edges_set):
            print(f"  {edge!s:<12s}", end='')
            for strategy, result in results.items():
                score = result['undirected_scores'].get(edge, 0.0)
                print(f" {score:>10.4f}", end='')
            print()
    
    print("\n" + "="*70)
    print("✓ Demo Complete!")
    print("="*70)
    
    print(f"\nKey Achievements:")
    print(f"  1. ✓ Reused existing coalition_builder.edge_builders")
    print(f"  2. ✓ Reused existing scorers.shapley.owen_shapley_two_stage")
    print(f"  3. ✓ Tested {len(results)} edge grouping strategies")
    print(f"  4. ✓ Folded {data.edge_index.size(1)} directed → 7 undirected edges")
    print(f"  5. ✓ Visualized coalitions + scores for each strategy")
    
    return results


if __name__ == '__main__':
    results = main()
