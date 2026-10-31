# toy_edge_graph_demo.py
"""
全局的边剪枝重要性演示。
沿用scorers接口。
Toy edge-pruning graph demo for node classification.

Global version:
the computed edge importance is w.r.t. the WHOLE-GRAPH node classification
performance (on all nodes in val_mask), rather than only node v0.

This version reuses the project's scoring interface:

    from scorers import score
    from scorers.shapley import compute_shapley_scores

And additionally provides:
    - exact Shapley computation for the toy graph
"""

from collections import defaultdict
from dataclasses import dataclass
import itertools
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scorers import score
from scorers.shapley import compute_shapley_scores


# =========================================================
# Minimal Data Container
# =========================================================

@dataclass
class ToyData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_nodes: int

    @property
    def num_features(self):
        return self.x.size(1)

    @property
    def num_classes(self):
        return int(self.y.max().item()) + 1


# =========================================================
# Graph Construction
# =========================================================

def build_toy_edge_pruning_graph(device="cpu", make_undirected=True):
    """
    Build the toy graph.

    Graph:
          v3            v4
           |             |
          v1             v2
            \           /
                 v0
               /    \
             v5 ---- v6

    Labels:
      v0,v1,v2,v3,v4 -> class 0
      v5,v6          -> class 1

    Features:
      dim 0 = evidence for class 0
      dim 1 = evidence for class 1

    IMPORTANT:
    In this GLOBAL version, all nodes are included in val_mask,
    so edge importance reflects whole-graph classification performance.
    """
    x = torch.tensor([
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

    # GLOBAL evaluation on all nodes
    train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    data = ToyData(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
    )

    return data, edge_name_map


def make_full_base_masks(data, device=None):
    if device is None:
        device = data.edge_index.device

    return {
        "node_mask": torch.ones(data.num_nodes, dtype=torch.bool, device=device),
        "edge_mask": torch.ones(data.edge_index.size(1), dtype=torch.bool, device=device),
        "channel_masks": {},
    }


def apply_edge_mask_local(data, edge_mask: torch.Tensor) -> torch.Tensor:
    """
    Local toy replacement for pruning.apply_edge_mask.
    Keeps only edges where edge_mask == True.
    """
    if edge_mask.dtype != torch.bool:
        raise TypeError(f"edge_mask must be bool, got {edge_mask.dtype}")
    return data.edge_index[:, edge_mask]


# =========================================================
# Adjacency utility for the toy model
# =========================================================

def normalize_adjacency_weighted(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor = None,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Weighted normalized adjacency:
        D^{-1/2}(A + I)D^{-1/2}
    """
    device = edge_index.device
    m = edge_index.size(1)

    if edge_weight is None:
        edge_weight = torch.ones(m, device=device, dtype=torch.float)

    if add_self_loops:
        loop = torch.arange(num_nodes, device=device, dtype=torch.long)
        loop_index = loop.unsqueeze(0).repeat(2, 1)
        loop_weight = torch.ones(num_nodes, device=device, dtype=edge_weight.dtype)

        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    adj = torch.sparse_coo_tensor(
        edge_index, edge_weight, (num_nodes, num_nodes)
    ).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    d_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)

    idx = adj.indices()
    val = adj.values()
    val_norm = val * d_inv_sqrt[idx[0]] * d_inv_sqrt[idx[1]]

    adj_norm = torch.sparse_coo_tensor(
        idx, val_norm, (num_nodes, num_nodes)
    ).coalesce()

    return adj_norm


def normalize_adjacency_local(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Local toy replacement for models.normalize_adjacency.
    Uses unit weights.
    """
    return normalize_adjacency_weighted(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weight=None,
        add_self_loops=True
    )


# =========================================================
# Toy propagation model
# =========================================================

# class TwoStepMeanClassifier(nn.Module):
#     """
#     Parameter-free 2-step neighborhood mean propagation.

#     Compatible with scorer calls of the form:
#         logits = model(x, adj_norm)
#     or
#         logits, emb = model(x, adj_norm, return_embeddings=True)
#     """

#     def __init__(self):
#         super().__init__()

#     def forward(self, x, adj_norm, return_embeddings=False):
#         h1 = torch.sparse.mm(adj_norm, x)
#         h2 = torch.sparse.mm(adj_norm, h1)
#         logits = h2

#         if return_embeddings:
#             return logits, h2
#         return logits

# # 统一版本
# class TwoStepMeanClassifier(nn.Module):
#     """
#     Parameter-free 2-step neighborhood mean propagation.

#     Compatible with scorer calls of the form:
#         logits = model(x, adj_mean)
#     or
#         logits, emb = model(x, adj_mean, return_embeddings=True)

#     Assumption:
#         adj_mean is a sparse mean-aggregation matrix, e.g. row-normalized adjacency,
#         without extra residual/self-addition outside the matrix itself.
#     """

#     def __init__(self):
#         super().__init__()

#     def forward(self, x, adj_mean, return_embeddings=False):
#         h1 = torch.sparse.mm(adj_mean, x)
#         h2 = torch.sparse.mm(adj_mean, h1)
#         logits = h2

#         if return_embeddings:
#             return logits, h2
#         return logits


class TwoStepMeanClassifier(nn.Module):
    """
    Parameter-free 2-step neighborhood mean propagation with self included.

    Compatible with scorer calls of the form:
        logits = model(x, adj_mean)
    or
        logits, emb = model(x, adj_mean, return_embeddings=True)

    Here we interpret adjacency structure from adj_mean, and for each node
    aggregate the features of:
        neighbors + itself
    using simple mean.
    """

    def __init__(self, in_dim=2, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x, adj_mean, return_embeddings=False):
        h1 = torch.sparse.mm(adj_mean, x)
        h2 = torch.sparse.mm(adj_mean, h1)
        logits = self.classifier(h2)

        if return_embeddings:
            return logits, h2
        return logits



# =========================================================
# Printing utilities
# =========================================================

def print_graph_summary(data, edge_name_map):
    print("=" * 70)
    print("Toy Graph Summary (GLOBAL evaluation)")
    print("=" * 70)
    print(f"num_nodes    : {data.num_nodes}")
    print(f"num_features : {data.num_features}")
    print(f"num_classes  : {data.num_classes}")
    print(f"num_edges    : {data.edge_index.size(1)} (directed representation)")
    print(f"train nodes  : {int(data.train_mask.sum().item())}")
    print(f"val nodes    : {int(data.val_mask.sum().item())}")
    print(f"test nodes   : {int(data.test_mask.sum().item())}")

    print("\nNode features / labels:")
    for i in range(data.num_nodes):
        print(f"  v{i}: x={data.x[i].tolist()}, y={int(data.y[i].item())}")

    print("\nDirected edges:")
    for i, name in enumerate(edge_name_map):
        print(f"  {i:2d}: {name}")


def inspect_baseline_prediction(model, data, add_self_loops=True):
    """
    Inspect baseline prediction on the WHOLE graph.
    """
    adj_norm = normalize_adjacency_weighted(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        edge_weight=None,
        add_self_loops=add_self_loops
    )

    logits = model(data.x, adj_norm)
    probs = torch.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1)

    val_logits = logits[data.val_mask]
    val_targets = data.y[data.val_mask]
    loss_global = F.cross_entropy(val_logits, val_targets).item()
    acc_global = (pred[data.val_mask] == val_targets).float().mean().item()

    print("\n" + "=" * 70)
    print("Baseline Prediction (GLOBAL)")
    print("=" * 70)
    for i in range(data.num_nodes):
        print(
            f"v{i}: logits={logits[i].tolist()} "
            f"probs={probs[i].tolist()} "
            f"pred={int(pred[i].item())} "
            f"label={int(data.y[i].item())}"
        )

    print(f"\nGlobal val loss: {loss_global:.6f}")
    print(f"Global val acc : {acc_global:.6f}")


def print_directed_scores(edge_scores, edge_name_map, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for i, s in enumerate(edge_scores):
        print(f"{i:2d}  edge {edge_name_map[i]:>6s}  score = {float(s): .6f}")


def aggregate_directed_scores_to_undirected(edge_index, edge_scores):
    bucket = defaultdict(list)

    edge_index_cpu = edge_index.detach().cpu()
    edge_scores_cpu = edge_scores.detach().cpu()

    for k in range(edge_index_cpu.size(1)):
        u = int(edge_index_cpu[0, k])
        v = int(edge_index_cpu[1, k])
        key = (u, v) if u < v else (v, u)
        bucket[key].append(float(edge_scores_cpu[k]))

    out = {}
    for key, vals in bucket.items():
        out[key] = sum(vals) / len(vals)
    return out


def print_undirected_scores(data, edge_scores, title):
    agg = aggregate_directed_scores_to_undirected(data.edge_index, edge_scores)
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for key, val in sorted(agg.items()):
        print(f"edge {key}: {val:.6f}")


# =========================================================
# Exact Shapley Support (LOCAL TO THIS TOY FILE)
# =========================================================

class ExactToyEdgeValueFunction:
    """
    Cached value function for exact/global edge-Shapley on the toy graph.

    Coalition semantics:
        coalition = list of directed edge indices to KEEP

    Value:
        v(S) = - global validation loss
    """
    def __init__(self, model, data, device=None, add_self_loops=True, criterion=None):
        self.model = model
        self.data = data
        self.device = device if device is not None else data.x.device
        self.add_self_loops = add_self_loops
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.cache = {}
        self.num_evaluations = 0

    def __call__(self, coalition):
        key = tuple(sorted(coalition))
        if key in self.cache:
            return self.cache[key]

        num_edges = self.data.edge_index.size(1)
        edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=self.device)
        if len(coalition) > 0:
            edge_mask[coalition] = True

        edge_index_sub = self.data.edge_index[:, edge_mask]

        adj_norm = normalize_adjacency_local(
            edge_index=edge_index_sub,
            num_nodes=self.data.num_nodes
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(self.data.x, adj_norm)
            val_logits = logits[self.data.val_mask]
            val_targets = self.data.y[self.data.val_mask]
            loss = self.criterion(val_logits, val_targets).item()

        value = -loss
        self.cache[key] = value
        self.num_evaluations += 1
        return value


def compute_exact_toy_edge_shapley(
    model,
    data,
    device=None,
    add_self_loops=True,
):
    """
    Exact Shapley value for edge importance on the toy graph.

    Uses the definition:
        phi_i = sum_{S subseteq N\\{i}} |S|!(n-|S|-1)! / n! * [v(S∪{i}) - v(S)]

    Returns:
        dict compatible with print_shapley_scores(...)
    """
    if device is None:
        device = data.x.device

    players = list(range(data.edge_index.size(1)))
    n = len(players)

    value_fn = ExactToyEdgeValueFunction(
        model=model,
        data=data,
        device=device,
        add_self_loops=add_self_loops,
    )

    start_time = time.time()

    shapley_values = np.zeros(n, dtype=np.float64)
    factorial_n = math.factorial(n)

    for i, p in enumerate(players):
        others = [q for q in players if q != p]

        for k in range(len(others) + 1):
            for subset in itertools.combinations(others, k):
                S = list(subset)
                weight = (math.factorial(k) * math.factorial(n - k - 1)) / factorial_n
                marginal = value_fn(S + [p]) - value_fn(S)
                shapley_values[i] += weight * marginal

    elapsed_time = time.time() - start_time

    edge_scores = torch.zeros(data.edge_index.size(1), dtype=torch.float32, device=device)
    edge_scores[torch.tensor(players, dtype=torch.long, device=device)] = \
        torch.from_numpy(shapley_values).float().to(device)

    return {
        "edge_scores": edge_scores,
        "shapley_values": shapley_values,
        "players": players,
        "num_evaluations": value_fn.num_evaluations,
        "cache_size": len(value_fn.cache),
        "elapsed_time": elapsed_time,
    }




def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    data, edge_name_map = build_toy_edge_pruning_graph(device=device, make_undirected=True)
    model = TwoStepMeanClassifier().to(device)

    print_graph_summary(data, edge_name_map)
    inspect_baseline_prediction(model, data, add_self_loops=True)

    methods = ["feature_l2", "sensitivity", "grad", "taylor"]
    base_masks = make_full_base_masks(data, device=device)

    for method in methods:
        print("\n" + "#" * 70)
        print(f"Running project scorer: {method}")
        print("#" * 70)

        result = score(
            unit_type="edge",
            method=method,
            model=model,
            data=data,
            base_masks=base_masks,
            device=device,
            add_self_loops=True,
        )

        edge_scores = result["scores"]["edge_scores"]

        print_directed_scores(edge_scores, edge_name_map, f"{method.upper()} Scores (Directed)")
        print_undirected_scores(data, edge_scores, f"{method.upper()} Scores (Undirected Averaged)")

    

if __name__ == "__main__":
    main()