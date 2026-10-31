# toy_edge_graph_demo.py
"""
只针对V0节点的边剪枝重要性演示。且包含了精确的Shapley值计算。
Toy edge-pruning graph demo for node classification.

This version reuses the project's scoring interface:

    from scorers import score

It provides:
1. toy graph construction
2. toy propagation model (two-step neighborhood mean)
3. printing / aggregation utilities
4. local toy Shapley implementations:
   - approximation
   - kernelshap
   - exact

Project scorers are reused for:
   - feature_l2
   - sensitivity
   - grad
   - taylor
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

# =========================================================
# Reuse project scorer interface
# =========================================================
from scorers import score


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
    """
    x = torch.tensor([
        [0.0, 0.0],  # v0 target
        [0.0, 0.0],  # v1 relay
        [0.0, 0.0],  # v2 relay
        [1.0, 0.0],  # v3 positive source
        [1.0, 0.0],  # v4 positive source
        [0.0, 1.0],  # v5 negative/noisy node
        [0.0, 1.0],  # v6 negative/noisy node
    ], dtype=torch.float, device=device)

    y = torch.tensor([
        0,  # v0
        0,  # v1
        0,  # v2
        0,  # v3
        0,  # v4
        1,  # v5
        1,  # v6
    ], dtype=torch.long, device=device)

    undirected_edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 5),
        (0, 6),
        (5, 6),
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
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)

    # Only evaluate the target node v0
    train_mask[0] = True
    val_mask[0] = True
    test_mask[0] = True

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


def normalize_adjacency_local(edge_index: torch.Tensor, num_nodes: int, add_self_loops: bool = True) -> torch.Tensor:
    return normalize_adjacency_weighted(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weight=None,
        add_self_loops=add_self_loops
    )


# =========================================================
# Toy propagation model
# =========================================================

class TwoStepMeanClassifier(nn.Module):
    """
    Parameter-free 2-step neighborhood mean propagation.

    Compatible with scorer calls of the form:
        logits = model(x, adj_norm)
    or
        logits, emb = model(x, adj_norm, return_embeddings=True)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, adj_norm, return_embeddings=False):
        h1 = torch.sparse.mm(adj_norm, x)
        h2 = torch.sparse.mm(adj_norm, h1)
        logits = h2

        if return_embeddings:
            return logits, h2
        return logits


# =========================================================
# Printing utilities
# =========================================================

def print_graph_summary(data, edge_name_map):
    print("=" * 70)
    print("Toy Graph Summary")
    print("=" * 70)
    print(f"num_nodes    : {data.num_nodes}")
    print(f"num_features : {data.num_features}")
    print(f"num_classes  : {data.num_classes}")
    print(f"num_edges    : {data.edge_index.size(1)} (directed representation)")

    print("\nNode features / labels:")
    for i in range(data.num_nodes):
        print(f"  v{i}: x={data.x[i].tolist()}, y={int(data.y[i].item())}")

    print("\nDirected edges:")
    for i, name in enumerate(edge_name_map):
        print(f"  {i:2d}: {name}")


def inspect_baseline_prediction(model, data, add_self_loops=True):
    adj_norm = normalize_adjacency_weighted(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        edge_weight=None,
        add_self_loops=add_self_loops
    )

    logits = model(data.x, adj_norm)
    probs = torch.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1)
    loss_v0 = F.cross_entropy(logits[data.val_mask], data.y[data.val_mask]).item()

    print("\n" + "=" * 70)
    print("Baseline Prediction")
    print("=" * 70)
    for i in range(data.num_nodes):
        print(
            f"v{i}: logits={logits[i].tolist()} "
            f"probs={probs[i].tolist()} "
            f"pred={int(pred[i].item())} "
            f"label={int(data.y[i].item())}"
        )
    print(f"\nTarget-node loss (v0): {loss_v0:.6f}")


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
# Local Toy Shapley Support
# =========================================================

def evaluate_edge_coalition_value(
    model,
    data,
    coalition,
    criterion=None,
    device=None,
    add_self_loops=True,
):
    """
    coalition: list[int], directed edge indices to KEEP
    value: v(S) = - val_loss
    """
    if device is None:
        device = data.x.device
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model.eval()

    num_edges = data.edge_index.size(1)
    edge_mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    if len(coalition) > 0:
        edge_mask[coalition] = True

    edge_index_sub = data.edge_index[:, edge_mask]

    adj_norm = normalize_adjacency_local(
        edge_index=edge_index_sub,
        num_nodes=data.num_nodes,
        add_self_loops=add_self_loops
    ).to(device)

    with torch.no_grad():
        logits = model(data.x, adj_norm)
        val_logits = logits[data.val_mask]
        val_targets = data.y[data.val_mask]
        loss = criterion(val_logits, val_targets).item()

    return -loss


class ToyEdgeValueFunction:
    """
    Cached coalition value function for toy edge-Shapley.
    """

    def __init__(self, model, data, device=None, add_self_loops=True):
        self.model = model
        self.data = data
        self.device = device if device is not None else data.x.device
        self.add_self_loops = add_self_loops
        self.cache = {}
        self.num_evaluations = 0

    def __call__(self, coalition):
        key = tuple(sorted(coalition))
        if key in self.cache:
            return self.cache[key]

        value = evaluate_edge_coalition_value(
            model=self.model,
            data=self.data,
            coalition=coalition,
            device=self.device,
            add_self_loops=self.add_self_loops,
        )
        self.cache[key] = value
        self.num_evaluations += 1
        return value


def toy_shapley_approximation(
    value_fn,
    players,
    num_samples=100,
    seed=42,
):
    """
    Monte Carlo permutation sampling Shapley approximation.
    """
    rng = np.random.RandomState(seed)
    n = len(players)
    phi = np.zeros(n, dtype=np.float64)
    player_to_idx = {p: i for i, p in enumerate(players)}

    for _ in range(num_samples):
        perm = list(players)
        rng.shuffle(perm)

        coalition = []
        prev_value = value_fn(coalition)

        for p in perm:
            coalition.append(p)
            cur_value = value_fn(coalition)
            phi[player_to_idx[p]] += (cur_value - prev_value)
            prev_value = cur_value

    phi /= float(max(1, num_samples))
    return phi


def toy_shapley_kernelshap(
    value_fn,
    players,
    num_samples=200,
    seed=42,
    ridge=1e-6,
):
    """
    KernelSHAP-style approximation for toy edge importance.
    """
    rng = np.random.RandomState(seed)
    n = len(players)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    M = max(2, num_samples)
    Z = np.zeros((M + 2, n), dtype=np.float64)

    for m in range(M):
        k = rng.randint(1, n)  # 1..n-1
        idx = rng.choice(n, size=k, replace=False)
        Z[m, idx] = 1.0

    # anchors
    Z[M, :] = 0.0
    Z[M + 1, :] = 1.0

    y = np.zeros(M + 2, dtype=np.float64)
    for m in range(M + 2):
        S = [players[i] for i in range(n) if Z[m, i] > 0.5]
        y[m] = value_fn(S)

    w = np.zeros(M + 2, dtype=np.float64)
    for m in range(M):
        k = int(Z[m].sum())
        comb = math.comb(n, k)
        w[m] = (n - 1) / (comb * k * (n - k))

    w[M] = 1e3
    w[M + 1] = 1e3

    X = np.concatenate([np.ones((M + 2, 1)), Z], axis=1)
    W = np.diag(w)

    XtW = X.T @ W
    A = XtW @ X + ridge * np.eye(n + 1)
    b = XtW @ y

    beta = np.linalg.solve(A, b)
    phi = beta[1:]
    return phi.astype(np.float64)


def toy_shapley_exact(
    value_fn,
    players,
):
    """
    Exact Shapley value by enumerating all subsets S ⊆ N\\{i}.

    phi_i = sum_{S ⊆ N\\{i}} |S|!(n-|S|-1)! / n! * [v(S∪{i}) - v(S)]
    """
    n = len(players)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    phi = np.zeros(n, dtype=np.float64)
    factorial_n = math.factorial(n)

    for i, p in enumerate(players):
        others = [q for q in players if q != p]

        for k in range(len(others) + 1):
            for subset in itertools.combinations(others, k):
                S = list(subset)
                weight = (math.factorial(k) * math.factorial(n - k - 1)) / factorial_n
                marginal = value_fn(S + [p]) - value_fn(S)
                phi[i] += weight * marginal

    return phi


def compute_toy_edge_shapley(
    model,
    data,
    method="approximation",
    num_samples=100,
    seed=42,
    device=None,
    add_self_loops=True,
):
    """
    Compute edge-level Shapley scores for the toy graph.

    method:
        - approximation
        - kernelshap
        - exact
    """
    if device is None:
        device = data.x.device

    players = list(range(data.edge_index.size(1)))
    value_fn = ToyEdgeValueFunction(
        model=model,
        data=data,
        device=device,
        add_self_loops=add_self_loops,
    )

    start_time = time.time()

    if method == "approximation":
        shapley_values = toy_shapley_approximation(
            value_fn=value_fn,
            players=players,
            num_samples=num_samples,
            seed=seed,
        )
    elif method == "kernelshap":
        shapley_values = toy_shapley_kernelshap(
            value_fn=value_fn,
            players=players,
            num_samples=num_samples,
            seed=seed,
            ridge=1e-6,
        )
    elif method == "exact":
        shapley_values = toy_shapley_exact(
            value_fn=value_fn,
            players=players,
        )
    else:
        raise ValueError(f"Unknown toy shapley method: {method}")

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


def print_shapley_scores(data, edge_name_map, result, title="SHAPLEY Scores"):
    edge_scores = result["edge_scores"]

    print("\n" + "=" * 70)
    print(title + " (Directed)")
    print("=" * 70)
    for i, s in enumerate(edge_scores):
        print(f"{i:2d}  edge {edge_name_map[i]:>6s}  score = {float(s): .6f}")

    print("\n" + "=" * 70)
    print(title + " (Undirected Averaged)")
    print("=" * 70)
    agg = aggregate_directed_scores_to_undirected(data.edge_index, edge_scores)
    for key, val in sorted(agg.items()):
        print(f"edge {key}: {val:.6f}")

    print("\nmeta:")
    print(f"  players         : {len(result['players'])}")
    print(f"  num_evaluations : {result['num_evaluations']}")
    print(f"  cache_size      : {result['cache_size']}")
    print(f"  elapsed_time    : {result['elapsed_time']:.4f}s")


# =========================================================
# Main demo using PROJECT score()
# =========================================================

def main():
    torch.manual_seed(42)
    device = torch.device("cpu")

    data, edge_name_map = build_toy_edge_pruning_graph(device=device, make_undirected=True)
    model = TwoStepMeanClassifier().to(device)

    print_graph_summary(data, edge_name_map)
    inspect_baseline_prediction(model, data, add_self_loops=True)

    methods = ["feature_l2", "sensitivity", "grad", "taylor"]

    for method in methods:
        print("\n" + "#" * 70)
        print(f"Running project scorer: {method}")
        print("#" * 70)

        # If your project score() requires explicit base_masks, switch to Version B.
        result = score(
            unit_type="edge",
            method=method,
            model=model,
            data=data,
            device=device,
            add_self_loops=True,
        )

        # Version B:
        # base_masks = make_full_base_masks(data, device=device)
        # result = score(
        #     unit_type="edge",
        #     method=method,
        #     model=model,
        #     data=data,
        #     base_masks=base_masks,
        #     device=device,
        #     add_self_loops=True,
        # )

        edge_scores = result["scores"]["edge_scores"]

        print_directed_scores(edge_scores, edge_name_map, f"{method.upper()} Scores (Directed)")
        print_undirected_scores(data, edge_scores, f"{method.upper()} Scores (Undirected Averaged)")

    # -----------------------------------------------------
    # Toy Shapley-based edge importance
    # -----------------------------------------------------
    shapley_methods = ["approximation", "kernelshap", "exact"]

    for shap_method in shapley_methods:
        print("\n" + "#" * 70)
        print(f"Running Shapley method: {shap_method}")
        print("#" * 70)

        # exact on 14 directed edges is expensive but still feasible for a toy demo
        # due to caching; if needed, switch make_undirected=False to reduce players.
        shap_result = compute_toy_edge_shapley(
            model=model,
            data=data,
            method=shap_method,
            num_samples=100,
            seed=42,
            device=device,
            add_self_loops=True,
        )

        print_shapley_scores(
            data=data,
            edge_name_map=edge_name_map,
            result=shap_result,
            title=f"SHAPLEY-{shap_method.upper()}"
        )


if __name__ == "__main__":
    main()