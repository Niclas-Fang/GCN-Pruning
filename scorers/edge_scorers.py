"""
Edge Scoring Methods (extended)

Supports:
- random
- betweenness (degree-proxy, undirected fix)
- feature_l2
- embedding_l2
- grad
- taylor (grad*input)

All methods return scores aligned with ACTIVE edges defined by base_masks['edge_mask'].
"""

import torch
import torch.nn.functional as F
from .base import SimpleScorer


# =========================================================
# Utilities
# =========================================================

def _get_active_edges(data, base_masks):
    """
    Returns:
      edge_index_active: [2, m_active]
      edge_mask: [m_total] bool
    """
    edge_mask = base_masks["edge_mask"]
    if edge_mask.dtype != torch.bool:
        raise TypeError(f"edge_mask must be bool, got {edge_mask.dtype}")
    edge_index_active = data.edge_index[:, edge_mask]
    return edge_index_active, edge_mask


def normalize_adjacency_weighted(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: torch.Tensor = None,
    add_self_loops: bool = True
) -> torch.Tensor:
    """
    Weighted normalize adjacency: D^{-1/2} (A + I) D^{-1/2}

    Args:
        edge_index: [2, m]
        num_nodes: N
        edge_weight: [m] (can require_grad)
        add_self_loops: whether to add I

    Returns:
        adj_norm (sparse COO): [N, N]
    """
    device = edge_index.device
    m = edge_index.size(1)

    if edge_weight is None:
        edge_weight = torch.ones(m, device=device, dtype=torch.float)

    if add_self_loops:
        loop = torch.arange(num_nodes, device=device, dtype=torch.long)
        loop_index = loop.unsqueeze(0).repeat(2, 1)                 # [2, N]
        loop_weight = torch.ones(num_nodes, device=device, dtype=edge_weight.dtype)
        edge_index = torch.cat([edge_index, loop_index], dim=1)     # [2, m+N]
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)  # [m+N]

    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).coalesce()

    deg = torch.sparse.sum(adj, dim=1).to_dense()                  # [N]
    d_inv_sqrt = deg.clamp(min=1e-12).pow(-0.5)

    idx = adj.indices()
    val = adj.values()
    val_norm = val * d_inv_sqrt[idx[0]] * d_inv_sqrt[idx[1]]

    adj_norm = torch.sparse_coo_tensor(idx, val_norm, (num_nodes, num_nodes)).coalesce()
    return adj_norm


def _safe_zero_grad(model):
    # works for most torch versions
    if hasattr(model, "zero_grad"):
        try:
            model.zero_grad(set_to_none=True)
        except TypeError:
            model.zero_grad()


# =========================================================
# Scorers
# =========================================================

def random_score(model, data, base_masks, **kwargs):
    """
    Score ACTIVE edges randomly.
    Baseline.
    """
    seed = kwargs.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)

    edge_index_active, _ = _get_active_edges(data, base_masks)
    m_active = edge_index_active.size(1)
    scores = torch.rand(m_active, device=edge_index_active.device)
    return {"edge_scores": scores}


def betweenness_score(model, data, base_masks, **kwargs):
    """
    Degree-proxy for betweenness-like importance (fast baseline).

    Fix: for undirected graphs, compute degree using BOTH endpoints (src+dst),
    rather than only src.

    Score definition (proxy):
        s_e = deg(u) * deg(v)

    kwargs:
      - use_coalesce_degree: bool (default False)
          If True, degree is computed on coalesced adjacency counts (dedup + sum),
          which can be more stable when edge_index contains duplicates.
    """
    edge_index_active, _ = _get_active_edges(data, base_masks)
    src, dst = edge_index_active[0], edge_index_active[1]

    # Optionally drop self-loops before degree
    remove_self_loops = kwargs.get("remove_self_loops", True)
    if remove_self_loops:
        keep = src != dst
        src_deg = src[keep]
        dst_deg = dst[keep]
    else:
        src_deg, dst_deg = src, dst

    num_nodes = data.num_nodes
    use_coalesce_degree = kwargs.get("use_coalesce_degree", False)

    if not use_coalesce_degree:
        # Undirected degree count: count both endpoints
        deg = torch.bincount(src_deg, minlength=num_nodes).float()
        deg = deg + torch.bincount(dst_deg, minlength=num_nodes).float()
    else:
        # Coalesce degree: build unweighted adjacency and sum row counts
        # (Still cheap at this scale; more robust to duplicates)
        ones = torch.ones(src_deg.numel(), device=src_deg.device)
        ei = torch.stack([src_deg, dst_deg], dim=0)
        adj = torch.sparse_coo_tensor(ei, ones, (num_nodes, num_nodes)).coalesce()
        # For undirected, also count reverse direction
        ei_rev = torch.stack([dst_deg, src_deg], dim=0)
        adj_rev = torch.sparse_coo_tensor(ei_rev, ones, (num_nodes, num_nodes)).coalesce()
        adj = (adj + adj_rev).coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense().float()

    scores = deg[src] * deg[dst]
    return {"edge_scores": scores}


def feature_l2_score(model, data, base_masks, **kwargs):
    """
    Edge score by L2 distance between endpoint INPUT features:
        s_e = ||x_u - x_v||_2
    """
    edge_index_active, _ = _get_active_edges(data, base_masks)
    x = data.x
    src, dst = edge_index_active[0], edge_index_active[1]
    scores = torch.norm(x[src] - x[dst], p=2, dim=-1)
    return {"edge_scores": scores}


@torch.inference_mode()
def embedding_l2_score(model, data, base_masks, **kwargs):
    """
    Edge score by L2 distance between endpoint EMBEDDINGS:
        s_e = ||z_u - z_v||_2

    Uses your custom GCNNet signature:
        logits, emb = model(x, adj_norm, return_embeddings=True)

    kwargs:
      - device: torch.device or str (optional)
      - add_self_loops: bool (default True)
    """
    device = kwargs.get("device", data.x.device)
    add_self_loops = kwargs.get("add_self_loops", True)

    model.eval()
    x = data.x.to(device)
    edge_index_active, _ = _get_active_edges(data, base_masks)
    edge_index_active = edge_index_active.to(device)

    # embeddings are computed on the (current) base graph structure.
    # Here we use FULL graph edge_index to compute adj_norm for stable embeddings.
    adj_norm = normalize_adjacency_weighted(
        edge_index=data.edge_index.to(device),
        num_nodes=data.num_nodes,
        edge_weight=None,
        add_self_loops=add_self_loops
    )

    logits, z = model(x, adj_norm, return_embeddings=True)  # z: [N, hidden_dim]
    src, dst = edge_index_active[0], edge_index_active[1]
    scores = torch.norm(z[src] - z[dst], p=2, dim=-1)
    return {"edge_scores": scores}


def _grad_gate_scores(model, data, base_masks, variant: str, **kwargs):
    """
    Core: gradient-based edge scoring using differentiable gates g on ACTIVE edges.

    We construct adj_norm(g) from ACTIVE edges only:
        edge_weight = g (requires_grad)
        adj_norm = normalize_adjacency_weighted(edge_index_active, edge_weight=g)

    Loss is computed on validation nodes (data.val_mask).
    Scores returned correspond to ACTIVE edges.

    variant:
      - 'grad'   : |dL/dg|
      - 'taylor' : |g * dL/dg|  (a.k.a. grad*input)
    """
    device = kwargs.get("device", data.x.device)
    add_self_loops = kwargs.get("add_self_loops", True)

    model.eval()
    x = data.x.to(device)
    y = data.y.to(device)
    val_mask = data.val_mask.to(device)

    edge_index_active, _ = _get_active_edges(data, base_masks)
    edge_index_active = edge_index_active.to(device)
    m_active = edge_index_active.size(1)

    # differentiable gates
    g = torch.ones(m_active, device=device, requires_grad=True)

    # build weighted adjacency from ACTIVE edges only
    adj_norm = normalize_adjacency_weighted(
        edge_index=edge_index_active,
        num_nodes=data.num_nodes,
        edge_weight=g,
        add_self_loops=add_self_loops
    )

    _safe_zero_grad(model)
    logits = model(x, adj_norm)
    loss = F.cross_entropy(logits[val_mask], y[val_mask])
    loss.backward()

    grad = g.grad.detach()

    if variant == "grad":
        scores = grad.abs()
    elif variant == "taylor":
        scores = (g.detach() * grad).abs()
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return {"edge_scores": scores}


def grad_score(model, data, base_masks, **kwargs):
    """|dL/dg| gradient score."""
    return _grad_gate_scores(model, data, base_masks, variant="grad", **kwargs)


def taylor_score(model, data, base_masks, **kwargs):
    """|g * dL/dg| (first-order Taylor / grad*input) score."""
    return _grad_gate_scores(model, data, base_masks, variant="taylor", **kwargs)


import torch
import torch.nn.functional as F

@torch.inference_mode()
def sensitivity_score(model, data, base_masks, **kwargs):
    """
    Edge sensitivity via ablation (set a single edge weight to 0 each time).
    Returns scores aligned with ACTIVE edges only.

    score[e] = L_val(adj with edge e ablated) - L_val(adj baseline)

    kwargs:
      - device: torch.device/str (optional)
      - add_self_loops: bool (default True)
      - max_units: int or None (default None)  # limit number of edges to evaluate (for speed)
      - chunk_size: int (default 256)          # not true batching; just progress chunking
      - return_indices: bool (default False)   # if True, also return evaluated edge indices within active set
    """
    device = kwargs.get("device", data.x.device)
    add_self_loops = kwargs.get("add_self_loops", True)
    max_units = kwargs.get("max_units", None)
    chunk_size = kwargs.get("chunk_size", 256)
    return_indices = kwargs.get("return_indices", False)

    model.eval()
    x = data.x.to(device)
    y = data.y.to(device)
    val_mask = data.val_mask.to(device)

    edge_index_active = data.edge_index[:, base_masks["edge_mask"]].to(device)
    m_active = edge_index_active.size(1)

    # baseline: all active edges weight = 1
    g_base = torch.ones(m_active, device=device)
    adj_base = normalize_adjacency_weighted(
        edge_index=edge_index_active,
        num_nodes=data.num_nodes,
        edge_weight=g_base,
        add_self_loops=add_self_loops
    )
    logits_base = model(x, adj_base)
    loss_base = F.cross_entropy(logits_base[val_mask], y[val_mask]).item()

    # decide which edges to evaluate
    if max_units is None or max_units >= m_active:
        eval_ids = torch.arange(m_active, device=device)
    else:
        # deterministic subset: first max_units (you can randomize if you want)
        eval_ids = torch.arange(max_units, device=device)

    scores = torch.empty(eval_ids.numel(), device=device, dtype=torch.float)

    # ablate one edge at a time
    for start in range(0, eval_ids.numel(), chunk_size):
        end = min(start + chunk_size, eval_ids.numel())
        chunk = eval_ids[start:end]

        for t, e_id in enumerate(chunk):
            g = g_base.clone()
            g[int(e_id)] = 0.0  # ablate this edge
            adj = normalize_adjacency_weighted(
                edge_index=edge_index_active,
                num_nodes=data.num_nodes,
                edge_weight=g,
                add_self_loops=add_self_loops
            )
            logits = model(x, adj)
            loss = F.cross_entropy(logits[val_mask], y[val_mask]).item()
            scores[start + t] = loss - loss_base

    out = {"edge_scores": scores}
    if return_indices:
        out["edge_indices_in_active"] = eval_ids.detach().cpu()
    return out

# =========================================================
# Factory
# =========================================================

def get_edge_scorer(method: str):
    """Get edge scorer by method name."""
    method = method.lower()
    if method == "random":
        return SimpleScorer("edge", "random", random_score)
    elif method == "betweenness":
        return SimpleScorer("edge", "betweenness", betweenness_score)
    elif method == "feature_l2":
        return SimpleScorer("edge", "feature_l2", feature_l2_score)
    # elif method == "embedding_l2":
    #     return SimpleScorer("edge", "embedding_l2", embedding_l2_score)
    elif method == "sensitivity":
        return SimpleScorer("edge", "sensitivity", sensitivity_score)
    elif method == "grad":
        return SimpleScorer("edge", "grad", grad_score)
    elif method == "taylor":
        return SimpleScorer("edge", "taylor", taylor_score)
    else:
        raise ValueError(f"Unknown edge scoring method: {method}")