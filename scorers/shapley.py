

"""
Two-stage Owen-Shapley and Shapley Approximation for Pruning

Implements joint pruning condition evaluation where:
- Value function: v(S) = -val_loss (evaluated on validation set)
- Conditional evaluation: Base masks control fixed pruning state
"""
import math  
import torch
import torch.nn as nn
import numpy as np
import hashlib
import time
from typing import Dict, List, Any, Optional


def _tensor_bytes(t: torch.Tensor) -> bytes:
    """Stable bytes for hashing (CPU, contiguous, fixed dtype where needed)."""
    if t is None:
        return b""
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(t)}")
    # Use contiguous CPU representation
    arr = t.detach().to("cpu").contiguous().numpy()
    return arr.tobytes()


def hash_masks(base_masks: Dict[str, Any]) -> str:
    """
    Compute hash of base masks for cache key.

    base_masks must contain:
      - 'node_mask': Bool tensor [num_nodes]
      - 'edge_mask': Bool tensor [num_edges]
      - 'channel_masks': Dict[str, Float/Binary tensor]
    """
    parts = []

    parts.append(_tensor_bytes(base_masks["node_mask"]))
    parts.append(_tensor_bytes(base_masks["edge_mask"]))

    channel_masks = base_masks.get("channel_masks", {})
    for key in sorted(channel_masks.keys()):
        parts.append(key.encode("utf-8"))
        parts.append(_tensor_bytes(channel_masks[key]))

    combined = b"".join(parts)
    return hashlib.md5(combined).hexdigest()


class CoalitionValueFunction:
    """
    Value function for coalition evaluation: v(S) = -val_loss.

    unit_type:
      - 'node': coalition contains node ids to KEEP
      - 'edge': coalition contains edge indices (column indices of edge_index) to KEEP
      - 'channel': coalition contains channel indices to KEEP for a specified layer
    """

    def __init__(
        self,
        model,
        data,
        adj_norm,
        unit_type: str,
        base_masks: Dict[str, Any],
        criterion: nn.Module = None,
        device: torch.device = None,
        cache: Optional[Dict] = None,
        target_layer: Optional[str] = None,
    ):
        self.model = model
        self.data = data
        self.adj_norm = adj_norm
        self.unit_type = unit_type
        self.base_masks = base_masks
        self.base_masks_hash = hash_masks(base_masks)
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.device = device if device is not None else torch.device("cpu")
        self.cache = cache if cache is not None else {}
        self.target_layer = target_layer

        self.num_evaluations = 0

        # 只创建一次 masked model，避免在每次 coalition evaluation 时反复初始化
        from pruning import MaskedModelWrapper
        self.masked_model = MaskedModelWrapper(self.model).to(self.device)
        self.masked_model.eval()

    def __call__(self, coalition: List[int]) -> float:
        coalition_tuple = tuple(sorted(coalition))
        cache_key = (self.base_masks_hash, self.unit_type, self.target_layer, coalition_tuple)

        if cache_key in self.cache:
            return self.cache[cache_key]

        if self.unit_type == "node":
            value = self._evaluate_node_coalition(coalition)
        elif self.unit_type == "edge":
            value = self._evaluate_edge_coalition(coalition)
        elif self.unit_type == "channel":
            value = self._evaluate_channel_coalition(coalition)
        else:
            raise ValueError(f"Unknown unit_type: {self.unit_type}")

        self.cache[cache_key] = value
        self.num_evaluations += 1

        return value

    def _apply_base_channel_masks(self):
        """Apply base channel masks to the cached masked model."""
        for layer_name, mask in self.base_masks['channel_masks'].items():
            if layer_name in self.masked_model.layer_names:
                self.masked_model.set_layer_mask(layer_name, mask.to(self.device))

    def _evaluate_node_coalition(self, coalition: List[int]) -> float:
        """Evaluate coalition of nodes."""
        from pruning import apply_node_mask
        from models import normalize_adjacency

        # Create node mask: keep only coalition nodes
        node_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool)
        node_mask[coalition] = True

        # Apply node mask
        new_edge_index = apply_node_mask(
            self.data, node_mask,
            ensure_undirected=True,
            preserve_self_loops=True
        )

        # Normalize adjacency
        adj_norm = normalize_adjacency(
            new_edge_index, self.data.num_nodes
        ).to(self.device)

        # Apply base channel masks to cached masked model
        self._apply_base_channel_masks()

        # Evaluate
        return self._evaluate_loss(self.masked_model, adj_norm)

    def _evaluate_edge_coalition(self, coalition: List[int]) -> float:
        """Evaluate coalition of edges."""
        from pruning import apply_edge_mask
        from models import normalize_adjacency

        # Create edge mask
        edge_mask = torch.zeros(self.data.edge_index.size(1), dtype=torch.bool)
        edge_mask[coalition] = True

        # Apply edge mask
        new_edge_index = apply_edge_mask(
            self.data, edge_mask,
            ensure_undirected=True,
            add_self_loops_flag=True
        )

        # Normalize adjacency
        adj_norm = normalize_adjacency(
            new_edge_index, self.data.num_nodes
        ).to(self.device)

        # Apply base channel masks to cached masked model
        self._apply_base_channel_masks()

        return self._evaluate_loss(self.masked_model, adj_norm)

    def _evaluate_channel_coalition(
        self,
        coalition: List[int],
        layer_name: Optional[str] = None
    ) -> float:
        """Evaluate coalition of channels."""
        # Use base adjacency (no graph pruning)
        adj_norm = self.adj_norm.to(self.device)

        # 优先使用显式传入的 layer_name，否则使用初始化时指定的 target_layer
        target_layer = layer_name if layer_name is not None else self.target_layer
        if target_layer is None:
            raise ValueError("Channel coalition evaluation requires target_layer")

        # Apply channel masks
        for ln, mask in self.base_masks['channel_masks'].items():
            if ln == target_layer:
                # For target layer, use coalition
                new_mask = torch.zeros_like(mask)
                new_mask[coalition] = 1.0
                self.masked_model.set_layer_mask(ln, new_mask.to(self.device))
            else:
                # For other layers, use base mask
                self.masked_model.set_layer_mask(ln, mask.to(self.device))

        return self._evaluate_loss(self.masked_model, adj_norm)

    def _evaluate_loss(self, model, adj_norm) -> float:
        """Evaluate validation loss."""
        model.eval()

        x = self.data.x.to(self.device)
        y = self.data.y.to(self.device)
        val_mask = self.data.val_mask.to(self.device)

        with torch.no_grad():
            logits = model(x, adj_norm)
            loss = self.criterion(logits[val_mask], y[val_mask])

        # Return negative loss (higher is better)
        return -loss.item()


def owen_shapley_two_stage(
    value_fn: CoalitionValueFunction,
    players: List[int],
    coalitions: List[List[int]],
    num_samples_q: int = 10,
    num_samples_p: int = 10,
    K: int = 5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    True Two-Stage Owen-Shapley with Prior Coalitions.
    
    This implements the CORRECT two-stage Owen value computation:
    - Stage 1: Sample coalition indices ~ Bernoulli(q)
    - Stage 2: Sample within target player's coalition ~ Bernoulli(p)
    
    Args:
        value_fn: Coalition value function v(S)
        players: List of all players (must equal union of coalitions)
        coalitions: List of disjoint coalitions forming partition
                    Example: [[0,1], [2,3,4]] means:
                    - Coalition 0: players {0,1}
                    - Coalition 1: players {2,3,4}
        num_samples_q: Number of coalition-level samples (q grid points)
        num_samples_p: Number of intra-coalition samples (p grid points)
        K: Mini-batches per (q, p) pair
        seed: Random seed
    
    Returns:
        np.ndarray of shape [len(players)] aligned with `players` order
    
    Algorithm:
        For each (q, p) grid point:
            For k = 1..K:
                1. Sample S_outer ~ Bernoulli(q) over coalition indices
                2. For each player j:
                    a) Identify j's coalition: B[g_j]
                    b) Sample S_inner ~ Bernoulli(p) within B[g_j]
                    c) Background: S = (∪ coalitions in S_outer) ∪ S_inner \ {j}
                    d) Marginal: v(S ∪ {j}) - v(S)
    
    Key Differences from Standard Shapley:
        - Background set S depends on which coalition player j belongs to
        - Different players can have different background sets in same sample
        - Respects coalition structure through two-stage sampling
    
    Example:
        players = [0,1,2,3,4]
        coalitions = [[0,1], [2,3,4]]
        
        Sample iteration:
        - Stage 1: Sample coalition index 1 → S_outer = {2,3,4}
        - For player j=0 (in coalition 0):
            - Stage 2: Sample {1} within {0,1} → S_inner = {1}
            - Background: S = {2,3,4} ∪ {1} = {1,2,3,4}
            - Marginal: v({0,1,2,3,4}) - v({1,2,3,4})
        - For player j=3 (in coalition 1):
            - Stage 2: Sample {2,4} within {2,3,4} → S_inner = {2,4}
            - Background: S = {2,3,4} ∪ {2,4} \ {3} = {2,4}
            - Marginal: v({2,3,4}) - v({2,4})
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    
    # Validate input
    n = len(players)
    m = len(coalitions)
    
    # Flatten coalitions and verify partition
    flat_coalitions = set()
    for B in coalitions:
        flat_coalitions.update(B)
    assert flat_coalitions == set(players), \
        f"Coalitions must partition players. Got {flat_coalitions} vs {set(players)}"
    
    # Check coalitions are disjoint
    for i in range(m):
        for j in range(i+1, m):
            overlap = set(coalitions[i]) & set(coalitions[j])
            assert len(overlap) == 0, \
                f"Coalitions {i} and {j} overlap: {overlap}"
    
    # Build player -> coalition index mapping
    player_to_coalition = {}
    for g_idx, B in enumerate(coalitions):
        for p in B:
            player_to_coalition[p] = g_idx
    
    # Build player -> array index mapping
    player_to_idx = {p: i for i, p in enumerate(players)}
    
    # Initialize Shapley values
    shapley_values = np.zeros(n, dtype=np.float64)
    
    # Grid for q and p
    total_samples = num_samples_q * num_samples_p * K
    q_values = np.linspace(0, 1, num_samples_q + 1)[:-1]  # exclude 1.0
    p_values = np.linspace(0, 1, num_samples_p + 1)[:-1]
    
    # Two-stage sampling
    for q in q_values:
        for p in p_values:
            for _k in range(K):
                # ============================================
                # STAGE 1: Sample coalition indices ~ Bernoulli(q)
                # ============================================
                S_outer_indices = [g for g in range(m) if rng.rand() < q]
                
                # Convert coalition indices to player set
                S_outer_players = set()
                for g in S_outer_indices:
                    S_outer_players.update(coalitions[g])
                
                # ============================================
                # STAGE 2: For each player j
                # ============================================
                for j in players:
                    g_j = player_to_coalition[j]  # j's coalition index
                    B_j = coalitions[g_j]         # j's coalition members
                    
                    # Sample within j's coalition ~ Bernoulli(p)
                    # Exclude j itself from sampling
                    S_inner = [i for i in B_j if i != j and rng.rand() < p]
                    
                    # ============================================
                    # Background set: S = S_outer ∪ S_inner \ {j}
                    # ============================================
                    S = list(S_outer_players.union(S_inner))
                    if j in S:
                        S.remove(j)
                    
                    # ============================================
                    # Marginal contribution: v(S ∪ {j}) - v(S)
                    # ============================================
                    S_with_j = sorted(set(S + [j]))
                    
                    v_S = value_fn(S)
                    v_S_j = value_fn(S_with_j)
                    
                    marginal = v_S_j - v_S
                    shapley_values[player_to_idx[j]] += marginal
    
    # Normalize by total samples
    shapley_values /= float(max(1, total_samples))
    
    return shapley_values


def shapley_approximation(
    value_fn: CoalitionValueFunction,
    players: List[int],
    num_samples: int = 100,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Monte Carlo permutation sampling for Shapley values.

    Returns an array aligned with `players` order.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random

    n = len(players)
    shapley_values = np.zeros(n, dtype=np.float64)
    player_to_idx = {p: i for i, p in enumerate(players)}

    for _ in range(num_samples):
        perm = list(players)
        rng.shuffle(perm)

        coalition = []
        prev_value = value_fn(coalition)  # IMPORTANT: do NOT use 0.0 baseline

        for p in perm:
            coalition.append(p)
            cur_value = value_fn(coalition)
            marginal = cur_value - prev_value

            shapley_values[player_to_idx[p]] += marginal
            prev_value = cur_value

    shapley_values /= float(max(1, num_samples))
    return shapley_values

def shapley_kernelshap(
    value_fn: CoalitionValueFunction,
    players: List[int],
    num_samples: int = 200,
    seed: Optional[int] = None,
    ridge: float = 1e-6,
) -> np.ndarray:
    """
    KernelSHAP-style approximation using Shapley kernel weighted linear regression.

    We fit: v(S) ≈ w0 + sum_i phi_i * z_i(S)
    where z_i(S)=1 if player i in S else 0.
    Weighted by Shapley kernel:
        w(S) = (n-1) / (C(n, |S|) * |S| * (n-|S|))   for 1<=|S|<=n-1
    We include empty set and full set as anchor points with large weight.

    Returns np.ndarray aligned with `players`.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    n = len(players)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # sample subsets as binary matrix Z: [M, n]
    # also include empty and full
    M = max(2, num_samples)
    Z = np.zeros((M + 2, n), dtype=np.float64)

    # random subsets with sizes in [1, n-1] to avoid kernel singularities
    for m in range(M):
        k = rng.randint(1, n)  # 1..n-1
        idx = rng.choice(n, size=k, replace=False)
        Z[m, idx] = 1.0

    # anchors
    Z[M, :] = 0.0  # empty
    Z[M + 1, :] = 1.0  # full

    # evaluate v(S) for each row
    y = np.zeros(M + 2, dtype=np.float64)
    for m in range(M + 2):
        S = [players[i] for i in range(n) if Z[m, i] > 0.5]
        y[m] = value_fn(S)

    # Shapley kernel weights
    # w = (n-1) / (C(n,k) * k * (n-k))
    # anchors: give them large weight to stabilize
    w = np.zeros(M + 2, dtype=np.float64)
    for m in range(M):
        k = int(Z[m].sum())
        # combinatorial term
        comb = math.comb(n, k)
        w[m] = (n - 1) / (comb * k * (n - k))

    w[M] = 1e3
    w[M + 1] = 1e3

    # weighted ridge regression solve for [w0, phi_1..phi_n]
    X = np.concatenate([np.ones((M + 2, 1)), Z], axis=1)  # [M+2, n+1]
    W = np.diag(w)

    XtW = X.T @ W
    A = XtW @ X
    b = XtW @ y

    # ridge for numerical stability
    A = A + ridge * np.eye(n + 1)

    beta = np.linalg.solve(A, b)
    phi = beta[1:]  # drop intercept
    return phi.astype(np.float64)

def shapley_marginal_sampling(
    value_fn: CoalitionValueFunction,
    players: List[int],
    num_samples: int = 200,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Marginal-contribution sampling:
      For each sample:
        1) Pick random subset size k ~ uniform(0, n)
        2) Pick random subset S of size k
        3) Pick random player j not in S
        4) Compute v(S ∪ {j}) - v(S) → credit to j

    This is a valid (though high-variance) estimator of Shapley values.
    Returns array aligned with `players`.
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    n = len(players)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    shapley_values = np.zeros(n, dtype=np.float64)
    player_to_idx = {p: i for i, p in enumerate(players)}

    for _ in range(num_samples):
        # pick random subset size
        k = rng.randint(0, n)  # 0..n-1
        if k == n:
            # can't add anyone
            continue

        # pick random subset of size k
        S_idx = rng.choice(n, size=k, replace=False)
        S = [players[i] for i in S_idx]

        # pick random player j not in S
        not_in_S = [p for p in players if p not in S]
        if len(not_in_S) == 0:
            continue
        j = rng.choice(not_in_S)

        # marginal
        v_S = value_fn(S)
        v_Sj = value_fn(S + [j])
        marginal = v_Sj - v_S

        shapley_values[player_to_idx[j]] += marginal

    shapley_values /= float(max(1, num_samples))
    return shapley_values


def compute_shapley_scores(
    model,
    data,
    adj_norm,
    unit_type: str,
    base_masks: Dict[str, Any],
    players: List[int],
    method: str = 'approximation',
    coalitions: Optional[List[List[int]]] = None,  # For Owen
    num_samples: int = 100,
    num_samples_q: int = 10,  # For Owen
    num_samples_p: int = 10,  # For Owen
    K: int = 5,               # For Owen
    seed: Optional[int] = None,
    device: torch.device = None,
    target_layer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute Shapley values for pruning units.
    
    Args:
        model: GCN model
        data: Graph data
        adj_norm: Normalized adjacency
        unit_type: 'node', 'edge', or 'channel'
        base_masks: Base masks for conditional evaluation
        players: List of unit indices to evaluate
        method: 'approximation', 'owen', 'kernelshap', 'marginal'
        coalitions: Required for 'owen' method
        num_samples: Number of samples (for approximation/kernelshap/marginal)
        num_samples_q: Coalition-level samples (for owen)
        num_samples_p: Intra-coalition samples (for owen)
        K: Mini-batches per (q,p) (for owen)
        seed: Random seed
        device: Device
        target_layer: Required for channel Owen/Shapley evaluation
    
    Returns:
        Dictionary with Shapley values and metadata
    """
    if device is None:
        device = torch.device('cpu')
    
    # Create value function
    cache = {}
    value_fn = CoalitionValueFunction(
        model,
        data,
        adj_norm,
        unit_type,
        base_masks,
        device=device,
        cache=cache,
        target_layer=target_layer
    )
    
    # Compute Shapley values
    start_time = time.time()
    
    if method == 'owen':
        if coalitions is None:
            raise ValueError("Owen method requires 'coalitions' argument")
        shapley_values = owen_shapley_two_stage(
            value_fn,
            players,
            coalitions,
            num_samples_q=num_samples_q,
            num_samples_p=num_samples_p,
            K=K,
            seed=seed
        )
    elif method == 'approximation':
        shapley_values = shapley_approximation(
            value_fn,
            players,
            num_samples=num_samples,
            seed=seed
        )
    elif method == 'kernelshap':
        shapley_values = shapley_kernelshap(
            value_fn,
            players,
            num_samples=num_samples,
            seed=seed
        )
    elif method == 'marginal':
        shapley_values = shapley_marginal_sampling(
            value_fn,
            players,
            num_samples=num_samples,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    elapsed_time = time.time() - start_time
    
    # Convert to full-size tensor
    if unit_type == 'node':
        full_scores = torch.zeros(data.num_nodes)
        full_scores[players] = torch.from_numpy(shapley_values).float()
    elif unit_type == 'edge':
        full_scores = torch.zeros(data.edge_index.size(1))
        full_scores[players] = torch.from_numpy(shapley_values).float()
    else:  # channel
        # For channel, keep returning raw shapley_values as before
        full_scores = shapley_values
    
    return {
        f'{unit_type}_scores': full_scores,
        'shapley_values': shapley_values,
        'players': players,
        'coalitions': coalitions,
        'method': method,
        'num_evaluations': value_fn.num_evaluations,
        'cache_size': len(cache),
        'elapsed_time': elapsed_time,
        'seed': seed
    }