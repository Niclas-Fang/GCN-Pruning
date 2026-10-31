"""
Channel/Neuron Scoring Methods (extended)
"""
import torch
import torch.nn.functional as F
from .base import SimpleScorer


# -----------------------------
# helpers
# -----------------------------
def _get_prunable_layers(model):
    if hasattr(model, "base_model"):
        return model.base_model.list_prunable_layers()
    return model.list_prunable_layers()


def _filter_layers(prunable_layers, layer_name=None):
    if layer_name is None:
        return prunable_layers
    return [(n, l) for (n, l) in prunable_layers if n == layer_name]


def _is_linear_layer(layer):
    return hasattr(layer, "weight") and layer.weight.dim() == 2


def _channel_dim_and_weight_view(layer):
    """
    Return (out_dim, weight_for_channel_norm)
    For nn.Linear: weight is [out, in] -> channel along dim=1? Actually per-output-channel is row => dim=1 norm over in.
    For your GCNConv: weight is [in, out] -> per-output-channel is column => dim=0 norm over in.
    """
    W = layer.weight
    if W.shape[0] == W.shape[1]:
        # ambiguous square; default treat as [out, in] like Linear
        out_dim = W.size(0)
        return out_dim, W, "linear_like"
    # Heuristic: if layer has in_features/out_features fields, use them
    if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
        # your GCNConv: [in, out]
        if W.size(0) == layer.in_features and W.size(1) == layer.out_features:
            return layer.out_features, W, "gcn_like"
        # nn.Linear: [out, in]
        if W.size(0) == layer.out_features and W.size(1) == layer.in_features:
            return layer.out_features, W, "linear_like"
    # fallback: assume [out, in]
    out_dim = W.size(0)
    return out_dim, W, "linear_like"


def _set_layer_mask(model, layer_name, mask_tensor):
    """
    Try to set channel mask on a MaskedModelWrapper-like object.
    Adjust here if your wrapper uses different API.
    """
    if hasattr(model, "set_layer_mask"):
        model.set_layer_mask(layer_name, mask_tensor)
        return
    if hasattr(model, "layer_masks") and isinstance(model.layer_masks, dict):
        model.layer_masks[layer_name] = mask_tensor
        return
    raise ValueError(
        "Model does not support setting layer masks. "
        "To use grad/taylor/embedding_delta channel scorers, "
        "please provide/enable a MaskedModelWrapper with set_layer_mask(...) or layer_masks dict."
    )


def _get_layer_mask(model, layer_name):
    if hasattr(model, "get_layer_mask"):
        return model.get_layer_mask(layer_name)
    if hasattr(model, "layer_masks") and isinstance(model.layer_masks, dict):
        return model.layer_masks[layer_name]
    return None


# -----------------------------
# scorers
# -----------------------------
def magnitude_score(model, data, base_masks, layer_name=None, **kwargs):
    """
    Channel magnitude (L2 norm of weights per output channel).
    Compatible with:
      - nn.Linear: weight [out, in] -> norm over dim=1
      - your GCNConv: weight [in, out] -> norm over dim=0
    """
    scores = {}
    prunable_layers = _filter_layers(_get_prunable_layers(model), layer_name)

    for name, layer in prunable_layers:
        if not hasattr(layer, "weight"):
            continue

        out_dim, W, kind = _channel_dim_and_weight_view(layer)
        if kind == "linear_like":
            # per-output-channel is row
            channel_norms = torch.norm(W, p=2, dim=1)
        else:
            # gcn_like: per-output-channel is column
            channel_norms = torch.norm(W, p=2, dim=0)

        scores[name] = channel_norms.detach().cpu() if kwargs.get("to_cpu", False) else channel_norms

    return scores


def random_score(model, data, base_masks, layer_name=None, **kwargs):
    seed = kwargs.get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)

    scores = {}
    prunable_layers = _filter_layers(_get_prunable_layers(model), layer_name)

    for name, layer in prunable_layers:
        if hasattr(layer, "weight"):
            out_dim, _, _ = _channel_dim_and_weight_view(layer)
            scores[name] = torch.rand(out_dim, device=layer.weight.device)
        elif hasattr(layer, "out_features"):
            scores[name] = torch.rand(layer.out_features)
    return scores


@torch.inference_mode()
def activation_l2_score(model, data, base_masks, layer_name=None, **kwargs):
    """
    Channel score by activation magnitude (aligns conceptually with edge feature-based heuristics).

    Definition (for a layer l, channel c):
      s_{l,c} = ||H^{(l)}_{:,c}||_2     (optionally only on val nodes)

    Requires model to be your GCNNet (or wrapper around it) so we can hook.
    kwargs:
      - use_val_only: bool (default True)
      - device: torch.device/str (optional)
      - adj_norm: optionally pass precomputed adj_norm to avoid recompute
    """
    device = kwargs.get("device", data.x.device)
    use_val_only = kwargs.get("use_val_only", True)

    # Build adj_norm if not provided
    adj_norm = kwargs.get("adj_norm", None)
    if adj_norm is None:
        # assume you have normalize_adjacency(...) somewhere in your project
        from models import normalize_adjacency  # adjust import to your actual location
        adj_norm = normalize_adjacency(data.edge_index.to(device), data.num_nodes)

    x = data.x.to(device)
    val_mask = data.val_mask.to(device)

    # Decide which layers to score (gcn_layers.* and mlp_layers.* typically)
    prunable_layers = _filter_layers(_get_prunable_layers(model), layer_name)
    target_names = set(n for n, _ in prunable_layers)

    handles = []
    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            # out is [N, hidden_dim] for both GCNConv and Linear
            activations[name] = out.detach()
        return hook

    # Register hooks
    # We hook modules by matching names as used in list_prunable_layers()
    # This requires your model/wrapper exposes named_modules consistent with those names.
    for n, _ in prunable_layers:
        # locate module by name
        mod = dict(model.named_modules()).get(n, None)
        if mod is None:
            # fallback: try base_model
            if hasattr(model, "base_model"):
                mod = dict(model.base_model.named_modules()).get(n, None)
        if mod is None:
            raise ValueError(f"Cannot locate module '{n}' via named_modules().")
        handles.append(mod.register_forward_hook(make_hook(n)))

    try:
        model.eval()
        # forward
        if hasattr(model, "base_model"):
            _ = model(x, adj_norm)  # wrapper should forward to base_model
        else:
            _ = model(x, adj_norm)
    finally:
        for h in handles:
            h.remove()

    scores = {}
    for n, _ in prunable_layers:
        H = activations.get(n, None)
        if H is None:
            continue
        if use_val_only:
            H = H[val_mask]
        # L2 norm per channel => norm over node dimension
        # H: [n_sel, hidden_dim]
        scores[n] = torch.norm(H, p=2, dim=0)

    return scores


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def _channel_grad_like(model, data, base_masks, variant="taylor", layer_name=None, **kwargs):
    device = kwargs.get("device", data.x.device)

    # 必须是 MaskedModelWrapper
    if not (hasattr(model, "set_layer_mask") and hasattr(model, "get_layer_mask") and hasattr(model, "_to_key")):
        raise ValueError("channel grad/taylor requires MaskedModelWrapper (with _to_key, layer_masks)")

    adj_norm = kwargs.get("adj_norm", None)
    if adj_norm is None:
        from models import normalize_adjacency
        adj_norm = normalize_adjacency(data.edge_index.to(device), data.num_nodes)

    model.eval()
    x = data.x.to(device)
    y = data.y.to(device)
    val_mask = data.val_mask.to(device)

    prunable_layers = model.base_model.list_prunable_layers() if hasattr(model, "base_model") else model.list_prunable_layers()
    if layer_name is not None:
        prunable_layers = [(n, l) for (n, l) in prunable_layers if n == layer_name]

    # 先把 base_masks 里的 mask 写入 wrapper（作为 baseline），这一步可以用 set_layer_mask（不需要梯度）
    for n, _ in prunable_layers:
        if n in base_masks.get("channel_masks", {}):
            model.set_layer_mask(n, base_masks["channel_masks"][n].to(device))

    # 保存现场并开始逐层求梯度
    saved_params = {}
    scores = {}

    for n, _ in prunable_layers:
        key = model._to_key(n)

        # 1) 取出 wrapper 内部真正参与 forward 的 Parameter
        p = model.layer_masks[key]                      # nn.Parameter
        saved_params[n] = (p.detach().clone(), p.requires_grad)

        # 2) 让它可导
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad.zero_()

        # 3) forward/backward
        model.zero_grad(set_to_none=True)
        logits = model(x, adj_norm)
        loss = F.cross_entropy(logits[val_mask], y[val_mask])
        loss.backward()

        grad = p.grad
        if grad is None:
            raise RuntimeError(
                f"Gradient is None for layer mask '{n}'. "
                "Check that forward uses h = h * mask and mask is the same Parameter."
            )

        if variant == "grad":
            s = grad.detach().abs()
        else:  # taylor
            s = (p.detach() * grad.detach()).abs()

        scores[n] = s

        # 4) 恢复 requires_grad，避免影响后续
        p.requires_grad_(saved_params[n][1])

    # 恢复 mask 数值（可选：一般不需要改数值，这里保险起见恢复一遍）
    for n, _ in prunable_layers:
        key = model._to_key(n)
        val, req = saved_params[n]
        model.layer_masks[key].data.copy_(val)
        model.layer_masks[key].requires_grad_(req)

    return scores


def grad_score(model, data, base_masks, layer_name=None, **kwargs):
    return _channel_grad_like(model, data, base_masks, variant="grad", layer_name=layer_name, **kwargs)

def taylor_score(model, data, base_masks, layer_name=None, **kwargs):
    return _channel_grad_like(model, data, base_masks, variant="taylor", layer_name=layer_name, **kwargs)

@torch.inference_mode()
def embedding_delta_score(model, data, base_masks, layer_name=None, **kwargs):
    """
    Optional: channel score by embedding change when ablating a single channel.

    s_{l,c} = ||Z - Z^{(l,c->0)}||_F

    WARNING: This is expensive (O(#channels) forwards per layer) unless you batch tricks.
    Also requires the model to apply channel masks in forward.

    kwargs:
      - device
      - adj_norm
      - use_val_only: bool (default False)  # whether to compute delta only on val nodes
    """
    device = kwargs.get("device", data.x.device)
    use_val_only = kwargs.get("use_val_only", False)

    adj_norm = kwargs.get("adj_norm", None)
    if adj_norm is None:
        from models import normalize_adjacency  # adjust import
        adj_norm = normalize_adjacency(data.edge_index.to(device), data.num_nodes)

    x = data.x.to(device)
    val_mask = data.val_mask.to(device)

    prunable_layers = _filter_layers(_get_prunable_layers(model), layer_name)

    # baseline embeddings
    logits, Z0 = model(x, adj_norm, return_embeddings=True)
    if use_val_only:
        Z0 = Z0[val_mask]

    scores = {}
    for n, layer in prunable_layers:
        out_dim, _, _ = _channel_dim_and_weight_view(layer)
        base_mask = base_masks["channel_masks"].get(n, torch.ones(out_dim)).to(device)

        layer_scores = torch.zeros(out_dim, device=device)
        for c in range(out_dim):
            g = base_mask.clone()
            g[c] = 0.0  # ablate channel c
            _set_layer_mask(model, n, g)

            _, Z1 = model(x, adj_norm, return_embeddings=True)
            if use_val_only:
                Z1 = Z1[val_mask]

            layer_scores[c] = torch.norm(Z0 - Z1, p="fro")

        scores[n] = layer_scores

        # restore
        _set_layer_mask(model, n, base_mask)

    return scores

# import torch
# import torch.nn.functional as F

# @torch.inference_mode()
# def sensitivity_score(model, data, base_masks, layer_name=None, **kwargs):
#     """
#     Channel sensitivity via ablation:
#       s_{l,c} = L_val(mask channel c off) - L_val(baseline mask)

#     Returns:
#       dict: {layer_name: scores_tensor[out_dim]}  (or only one layer if layer_name provided)

#     kwargs:
#       - device: torch.device/str (optional)
#       - adj_norm: precomputed adj_norm (optional)
#       - max_units: int or None (default None)   # limit channels per layer for speed
#     """
#     device = kwargs.get("device", data.x.device)
#     max_units = kwargs.get("max_units", None)

#     # build adj_norm if not provided
#     adj_norm = kwargs.get("adj_norm", None)
#     if adj_norm is None:
#         from models import normalize_adjacency  # adjust to your actual import path
#         adj_norm = normalize_adjacency(data.edge_index.to(device), data.num_nodes)

#     model.eval()
#     x = data.x.to(device)
#     y = data.y.to(device)
#     val_mask = data.val_mask.to(device)

#     # get prunable layers
#     if hasattr(model, "base_model"):
#         prunable_layers = model.base_model.list_prunable_layers()
#     else:
#         prunable_layers = model.list_prunable_layers()

#     if layer_name is not None:
#         prunable_layers = [(n, l) for (n, l) in prunable_layers if n == layer_name]

#     # baseline loss under base channel masks
#     # (assumes model already uses its internal masks; we will set them explicitly from base_masks)
#     # snapshot current masks to restore
#     saved_masks = {}
#     if hasattr(model, "layer_names"):
#         # wrapper-like
#         for n, _ in prunable_layers:
#             try:
#                 saved_masks[n] = model.get_layer_mask(n).detach().clone()
#             except Exception:
#                 pass

#     # apply base masks
#     for n, layer in prunable_layers:
#         base_m = base_masks["channel_masks"].get(n, None)
#         if base_m is None:
#             # default all-ones if not present
#             if hasattr(layer, "out_features"):
#                 base_m = torch.ones(layer.out_features, device=device)
#             elif hasattr(layer, "weight"):
#                 # infer out_dim for your GCNConv/Linear
#                 W = layer.weight
#                 out_dim = W.size(1) if hasattr(layer, "in_features") and hasattr(layer, "out_features") and W.size(1) == layer.out_features else W.size(0)
#                 base_m = torch.ones(out_dim, device=device)
#             else:
#                 continue
#         base_m = base_m.to(device)
#         _set_layer_mask(model, n, base_m)  # uses your helper

#     logits_base = model(x, adj_norm)
#     loss_base = F.cross_entropy(logits_base[val_mask], y[val_mask]).item()

#     scores = {}
#     for n, layer in prunable_layers:
#         base_m = base_masks["channel_masks"].get(n, None)
#         if base_m is None:
#             continue
#         base_m = base_m.to(device)

#         out_dim = base_m.numel()
#         if max_units is None or max_units >= out_dim:
#             eval_ids = range(out_dim)
#         else:
#             eval_ids = range(max_units)

#         layer_scores = torch.empty(len(list(eval_ids)), device=device, dtype=torch.float)

#         for idx, c in enumerate(eval_ids):
#             m = base_m.clone()
#             m[c] = 0.0  # ablate channel c
#             _set_layer_mask(model, n, m)

#             logits = model(x, adj_norm)
#             loss = F.cross_entropy(logits[val_mask], y[val_mask]).item()
#             layer_scores[idx] = loss - loss_base

#         # restore base mask for this layer
#         _set_layer_mask(model, n, base_m)
#         scores[n] = layer_scores

#     # restore original masks if we saved them
#     for n, m in saved_masks.items():
#         try:
#             _set_layer_mask(model, n, m.to(device))
#         except Exception:
#             pass

#     return scores

import torch
import torch.nn.functional as F
from .base import SimpleScorer

@torch.inference_mode()
def sensitivity_score(model, data, base_masks, layer_name=None, **kwargs):
    """
    Channel sensitivity (ablation) under current base_masks['channel_masks'].

    For each prunable layer l and channel c:
      score[l][c] = L_val(mask c off) - L_val(baseline)

    Requires:
      model is MaskedModelWrapper (or at least has set_layer_mask/get_layer_mask)

    Returns:
      dict {layer_name: Tensor[num_channels]}
    kwargs:
      - device: torch.device/str (optional)
      - adj_norm: optional precomputed adj_norm (to save time)
      - max_units: int or None (default None)  # evaluate first max_units channels per layer
    """
    device = kwargs.get("device", data.x.device)
    max_units = kwargs.get("max_units", None)

    if not hasattr(model, "set_layer_mask") or not hasattr(model, "get_layer_mask"):
        raise ValueError("channel sensitivity requires MaskedModelWrapper with set_layer_mask/get_layer_mask")

    # adj_norm
    adj_norm = kwargs.get("adj_norm", None)
    if adj_norm is None:
        from models import normalize_adjacency  # adjust to your project import
        adj_norm = normalize_adjacency(data.edge_index.to(device), data.num_nodes)

    model.eval()
    x = data.x.to(device)
    y = data.y.to(device)
    val_mask = data.val_mask.to(device)

    # prunable layers (use ORIGINAL names like 'gcn_layers.0')
    prunable_layers = model.base_model.list_prunable_layers() if hasattr(model, "base_model") else model.list_prunable_layers()
    if layer_name is not None:
        prunable_layers = [(n, l) for (n, l) in prunable_layers if n == layer_name]

    # save current masks to restore
    saved_masks = {n: model.get_layer_mask(n).detach().clone().to(device) for n, _ in prunable_layers}

    # apply base channel masks (if provided), otherwise keep current
    for n, _ in prunable_layers:
        if n in base_masks.get("channel_masks", {}):
            model.set_layer_mask(n, base_masks["channel_masks"][n].to(device))

    # baseline loss
    logits0 = model(x, adj_norm)
    loss0 = F.cross_entropy(logits0[val_mask], y[val_mask]).item()

    scores = {}
    for n, layer in prunable_layers:
        base_m = model.get_layer_mask(n).detach().clone().to(device)  # baseline mask actually in effect
        out_dim = base_m.numel()

        num_eval = out_dim if (max_units is None or max_units >= out_dim) else int(max_units)
        s = torch.empty(num_eval, device=device)

        for c in range(num_eval):
            m = base_m.clone()
            m[c] = 0.0
            model.set_layer_mask(n, m)

            logits = model(x, adj_norm)
            loss = F.cross_entropy(logits[val_mask], y[val_mask]).item()
            s[c] = loss - loss0

        # restore baseline mask for this layer
        model.set_layer_mask(n, base_m)
        scores[n] = s

    # restore all saved masks
    for n, m in saved_masks.items():
        model.set_layer_mask(n, m)

    return scores


def get_channel_scorer(method: str):
    """Get channel scorer by method name."""
    method = method.lower()
    if method == "magnitude":
        return SimpleScorer("channel", "magnitude", magnitude_score)
    elif method == "random":
        return SimpleScorer("channel", "random", random_score)
    elif method == "activation_l2":
        return SimpleScorer("channel", "activation_l2", activation_l2_score)
    elif method == "grad":
        return SimpleScorer("channel", "grad", grad_score)
    elif method == "taylor":
        return SimpleScorer("channel", "taylor", taylor_score)
    elif method == "sensitivity":
        return SimpleScorer("channel", "sensitivity", sensitivity_score)
    # elif method == "embedding_delta":
    #     return SimpleScorer("channel", "embedding_delta", embedding_delta_score)
    else:
        raise ValueError(f"Unknown channel scoring method: {method}")