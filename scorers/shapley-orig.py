# """
# Two-stage Owen-Shapley and Shapley Approximation for Pruning

# Implements joint pruning condition evaluation where:
# - Value function: v(S) = -val_loss (evaluated on validation set)
# - Conditional evaluation: Base masks control fixed pruning state
# """
# import math  
# import torch
# import torch.nn as nn
# import numpy as np
# import hashlib
# import time
# from typing import Dict, List, Any, Optional


# def _tensor_bytes(t: torch.Tensor) -> bytes:
#     """Stable bytes for hashing (CPU, contiguous, fixed dtype where needed)."""
#     if t is None:
#         return b""
#     if not isinstance(t, torch.Tensor):
#         raise TypeError(f"Expected torch.Tensor, got {type(t)}")
#     # Use contiguous CPU representation
#     arr = t.detach().to("cpu").contiguous().numpy()
#     return arr.tobytes()


# def hash_masks(base_masks: Dict[str, Any]) -> str:
#     """
#     Compute hash of base masks for cache key.

#     Compatible with toy edge-only setting:
#       - node_mask optional
#       - edge_mask optional
#       - channel_masks optional
#     """
#     parts = []

#     node_mask = base_masks.get("node_mask", None)
#     edge_mask = base_masks.get("edge_mask", None)

#     parts.append(_tensor_bytes(node_mask))
#     parts.append(_tensor_bytes(edge_mask))

#     channel_masks = base_masks.get("channel_masks", {})
#     for key in sorted(channel_masks.keys()):
#         parts.append(key.encode("utf-8"))
#         parts.append(_tensor_bytes(channel_masks[key]))

#     combined = b"".join(parts)
#     return hashlib.md5(combined).hexdigest()

# class CoalitionValueFunction:
#     def __init__(
#         self,
#         model,
#         data,
#         adj_norm,
#         unit_type: str,
#         base_masks: Dict[str, Any],
#         criterion: nn.Module = None,
#         device: torch.device = None,
#         cache: Optional[Dict] = None,
#         normalize_fn=None,
#         apply_edge_mask_fn=None,
#         masked_wrapper_cls=None,
#     ):
#         self.model = model
#         self.data = data
#         self.adj_norm = adj_norm
#         self.unit_type = unit_type
#         self.base_masks = base_masks
#         self.base_masks_hash = hash_masks(base_masks)
#         self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
#         self.device = device if device is not None else torch.device("cpu")
#         self.cache = cache if cache is not None else {}

#         # NEW
#         self.normalize_fn = normalize_fn
#         self.apply_edge_mask_fn = apply_edge_mask_fn
#         self.masked_wrapper_cls = masked_wrapper_cls

#         self.num_evaluations = 0

#     def __call__(self, coalition: List[int]) -> float:
#         coalition_tuple = tuple(sorted(coalition))
#         cache_key = (self.base_masks_hash, self.unit_type, coalition_tuple)

#         if cache_key in self.cache:
#             return self.cache[cache_key]

#         if self.unit_type == "node":
#             value = self._evaluate_node_coalition(coalition)
#         elif self.unit_type == "edge":
#             value = self._evaluate_edge_coalition(coalition)
#         elif self.unit_type == "channel":
#             value = self._evaluate_channel_coalition(coalition)
#         else:
#             raise ValueError(f"Unknown unit_type: {self.unit_type}")

#         self.cache[cache_key] = value
#         self.num_evaluations += 1
#         return value

#     def _evaluate_node_coalition(self, coalition: List[int]) -> float:
#         """Keep only nodes in coalition (AND base node_mask)."""
#         from pruning import apply_node_mask
#         from models import normalize_adjacency

#         # base node mask AND coalition mask
#         node_mask = self.base_masks["node_mask"].clone()
#         coalition_mask = torch.zeros(self.data.num_nodes, dtype=torch.bool, device=node_mask.device)
#         if len(coalition) > 0:
#             coalition_mask[coalition] = True
#         node_mask = node_mask & coalition_mask

#         class TempData:
#             def __init__(self, edge_index, num_nodes):
#                 self.edge_index = edge_index
#                 self.num_nodes = num_nodes

#         temp_data = TempData(self.data.edge_index, self.data.num_nodes)
#         new_edge_index = apply_node_mask(temp_data, node_mask)

#         adj_norm = normalize_adjacency(new_edge_index, self.data.num_nodes).to(self.device)
#         return self._evaluate_loss(adj_norm, self.base_masks["channel_masks"])

#     def _evaluate_edge_coalition(self, coalition: List[int]) -> float:
#         """Keep only edges in coalition (AND base edge_mask)."""
#         edge_mask = torch.zeros(
#             self.data.edge_index.size(1),
#             dtype=torch.bool,
#             device=self.base_masks["edge_mask"].device
#         )
#         if len(coalition) > 0:
#             edge_mask[coalition] = True

#         edge_mask = self.base_masks["edge_mask"] & edge_mask

#         class TempData:
#             def __init__(self, edge_index, num_nodes):
#                 self.edge_index = edge_index
#                 self.num_nodes = num_nodes

#         temp_data = TempData(self.data.edge_index, self.data.num_nodes)

#         # prefer injected local function for toy demo
#         if self.apply_edge_mask_fn is not None:
#             new_edge_index = self.apply_edge_mask_fn(temp_data, edge_mask)
#         else:
#             from pruning import apply_edge_mask
#             new_edge_index = apply_edge_mask(temp_data, edge_mask)

#         if self.normalize_fn is not None:
#             adj_norm = self.normalize_fn(new_edge_index, self.data.num_nodes).to(self.device)
#         else:
#             from models import normalize_adjacency
#             adj_norm = normalize_adjacency(new_edge_index, self.data.num_nodes).to(self.device)

#         return self._evaluate_loss(adj_norm, self.base_masks.get("channel_masks", {}))


#     def _evaluate_channel_coalition(self, coalition: List[int], layer_name: str = None) -> float:
#         """Keep only channels in coalition for a layer (AND base channel mask)."""
#         if layer_name is None:
#             layer_name = list(self.base_masks["channel_masks"].keys())[0]

#         original_mask = self.base_masks["channel_masks"][layer_name]
#         channel_mask = torch.zeros_like(original_mask)
#         if len(coalition) > 0:
#             channel_mask[coalition] = 1.0

#         channel_mask = original_mask * channel_mask

#         new_channel_masks = dict(self.base_masks["channel_masks"])
#         new_channel_masks[layer_name] = channel_mask

#         return self._evaluate_loss(self.adj_norm, new_channel_masks)

#     def _evaluate_loss(self, adj_norm: torch.Tensor, channel_masks: Dict[str, torch.Tensor]) -> float:
#         """
#         Evaluate validation loss with given adj_norm and channel_masks.
#         Returns -loss.
#         """
#         self.model.eval()

#         use_wrapper = len(channel_masks) > 0

#         if use_wrapper:
#             if self.masked_wrapper_cls is not None:
#                 MaskedModelWrapper = self.masked_wrapper_cls
#             else:
#                 from pruning import MaskedModelWrapper

#             if isinstance(self.model, MaskedModelWrapper):
#                 masked_model = self.model
#             else:
#                 masked_model = MaskedModelWrapper(self.model).to(self.device)

#             for layer_name, mask in channel_masks.items():
#                 if hasattr(masked_model, "layer_names") and layer_name in masked_model.layer_names:
#                     masked_model.set_layer_mask(layer_name, mask.to(self.device))

#             eval_model = masked_model
#         else:
#             eval_model = self.model

#         with torch.no_grad():
#             logits = eval_model(self.data.x, adj_norm)
#             val_logits = logits[self.data.val_mask]
#             val_targets = self.data.y[self.data.val_mask]
#             loss = self.criterion(val_logits, val_targets).item()

#         return -loss


# def owen_shapley_two_stage(
#     value_fn: CoalitionValueFunction,
#     players: List[int],
#     num_samples_q: int = 10,
#     num_samples_p: int = 10,
#     K: int = 5,
#     C_R: float = 1.0,
#     C_S: float = 1.0,
#     seed: Optional[int] = None,
# ) -> np.ndarray:
#     """
#     NOTE: Your original version had indexing bugs (used player id as array index).
#     This version returns an array aligned with `players` order.
#     """
#     rng = np.random.RandomState(seed) if seed is not None else np.random

#     n = len(players)
#     shapley_values = np.zeros(n, dtype=np.float64)
#     player_to_idx = {p: i for i, p in enumerate(players)}

#     # If你确实想按论文那样采样 q/p 的网格，可以保留；这里仍沿用你的网格写法
#     q_values = np.arange(0, 1, 1.0 / (max(1e-12, (C_R * C_S * K))))
#     p_values = np.arange(0, 1, 1.0 / (max(1e-12, (C_S * K))))

#     for q in q_values:
#         for p in p_values:
#             e = np.zeros(n, dtype=np.float64)

#             for _k in range(1, K + 1):
#                 I_q = [i for i in players if rng.rand() < q]
#                 I_p = [i for i in players if rng.rand() < p]

#                 base_coalition = list(set(I_q + I_p))
#                 v2 = value_fn(base_coalition)  # reuse for speed (still consistent)

#                 for j in players:
#                     coalition1 = list(set(base_coalition + [j]))
#                     v1 = value_fn(coalition1)
#                     h = v1 - v2
#                     e[player_to_idx[j]] += h

#             shapley_values += e

#     shapley_values /= float(max(1, (C_R * C_S * K)))
#     return shapley_values


# def shapley_approximation(
#     value_fn: CoalitionValueFunction,
#     players: List[int],
#     num_samples: int = 100,
#     seed: Optional[int] = None,
# ) -> np.ndarray:
#     """
#     Monte Carlo permutation sampling for Shapley values.

#     Returns an array aligned with `players` order.
#     """
#     rng = np.random.RandomState(seed) if seed is not None else np.random

#     n = len(players)
#     shapley_values = np.zeros(n, dtype=np.float64)
#     player_to_idx = {p: i for i, p in enumerate(players)}

#     for _ in range(num_samples):
#         perm = list(players)
#         rng.shuffle(perm)

#         coalition = []
#         prev_value = value_fn(coalition)  # IMPORTANT: do NOT use 0.0 baseline

#         for p in perm:
#             coalition.append(p)
#             cur_value = value_fn(coalition)
#             marginal = cur_value - prev_value

#             shapley_values[player_to_idx[p]] += marginal
#             prev_value = cur_value

#     shapley_values /= float(max(1, num_samples))
#     return shapley_values

# def shapley_kernelshap(
#     value_fn: CoalitionValueFunction,
#     players: List[int],
#     num_samples: int = 200,
#     seed: Optional[int] = None,
#     ridge: float = 1e-6,
# ) -> np.ndarray:
#     """
#     KernelSHAP-style approximation using Shapley kernel weighted linear regression.

#     We fit: v(S) ≈ w0 + sum_i phi_i * z_i(S)
#     where z_i(S)=1 if player i in S else 0.
#     Weighted by Shapley kernel:
#         w(S) = (n-1) / (C(n, |S|) * |S| * (n-|S|))   for 1<=|S|<=n-1
#     We include empty set and full set as anchor points with large weight.

#     Returns np.ndarray aligned with `players`.
#     """
#     rng = np.random.RandomState(seed) if seed is not None else np.random
#     n = len(players)
#     if n == 0:
#         return np.zeros(0, dtype=np.float64)

#     # sample subsets as binary matrix Z: [M, n]
#     # also include empty and full
#     M = max(2, num_samples)
#     Z = np.zeros((M + 2, n), dtype=np.float64)

#     # random subsets with sizes in [1, n-1] to avoid kernel singularities
#     for m in range(M):
#         k = rng.randint(1, n)  # 1..n-1
#         idx = rng.choice(n, size=k, replace=False)
#         Z[m, idx] = 1.0

#     # anchors
#     Z[M, :] = 0.0  # empty
#     Z[M + 1, :] = 1.0  # full

#     # evaluate v(S) for each row
#     y = np.zeros(M + 2, dtype=np.float64)
#     for m in range(M + 2):
#         S = [players[i] for i in range(n) if Z[m, i] > 0.5]
#         y[m] = value_fn(S)

#     # Shapley kernel weights
#     # w = (n-1) / (C(n,k) * k * (n-k))
#     # anchors: give them large weight to stabilize
#     w = np.zeros(M + 2, dtype=np.float64)
#     for m in range(M):
#         k = int(Z[m].sum())
#         # combinatorial term
#         comb = math.comb(n, k)
#         w[m] = (n - 1) / (comb * k * (n - k))

#     w[M] = 1e3
#     w[M + 1] = 1e3

#     # weighted ridge regression solve for [w0, phi_1..phi_n]
#     X = np.concatenate([np.ones((M + 2, 1)), Z], axis=1)  # [M+2, n+1]
#     W = np.diag(w)

#     XtW = X.T @ W
#     A = XtW @ X
#     b = XtW @ y

#     # ridge for numerical stability
#     A = A + ridge * np.eye(n + 1)

#     beta = np.linalg.solve(A, b)
#     phi = beta[1:]  # drop intercept
#     return phi.astype(np.float64)

# def shapley_marginal_sampling(
#     value_fn: CoalitionValueFunction,
#     players: List[int],
#     num_samples: int = 200,
#     seed: Optional[int] = None,
# ) -> np.ndarray:
#     """
#     Marginal contribution sampling (size-weighted, unbiased).

#     For each sample:
#       - pick target player i uniformly
#       - pick subset size k uniformly from {0,...,n-1}
#       - sample S uniformly among subsets of size k from N\{i}
#       - accumulate marginal: v(S ∪ {i}) - v(S) with weight 1 / n

#     This yields an unbiased estimator of Shapley values when k is uniform over sizes.
#     (It matches Shapley definition which averages uniformly over subset sizes.)

#     Returns np.ndarray aligned with `players`.
#     """
#     rng = np.random.RandomState(seed) if seed is not None else np.random
#     n = len(players)
#     if n == 0:
#         return np.zeros(0, dtype=np.float64)

#     phi = np.zeros(n, dtype=np.float64)

#     for _ in range(num_samples):
#         i = rng.randint(0, n)  # target index in players list
#         # choose k in [0, n-1]
#         k = rng.randint(0, n)
#         # sample subset from remaining indices
#         rest = [j for j in range(n) if j != i]
#         if k > 0:
#             chosen = rng.choice(rest, size=k, replace=False)
#             S = [players[j] for j in chosen]
#         else:
#             S = []

#         vS = value_fn(S)
#         vS_i = value_fn(S + [players[i]])
#         phi[i] += (vS_i - vS)

#     # average; note: each i sampled ~ num_samples/n times, so divide by num_samples_i expectation
#     # unbiased scaling: we sampled i uniformly, so E[phi[i]] = shapley_i * num_samples / n
#     phi = (phi / float(num_samples)) * float(n)
#     return phi.astype(np.float64)

# def compute_shapley_scores(
#     model,
#     data,
#     adj_norm,
#     unit_type: str,
#     base_masks: Dict[str, Any],
#     players: List[int],
#     method: str = "approximation",
#     num_samples: int = 100,
#     seed: Optional[int] = None,
#     device: torch.device = None,
#     **kwargs,
# ) -> Dict[str, Any]:
#     """
#     Returns:
#       - result['shapley_values']: np.ndarray, aligned with `players`
#       - result[f'{unit_type}_scores']: torch.Tensor, indexed by original id space
#     """
#     if device is None:
#         device = torch.device("cpu")

#     cache = {}
#     value_fn = CoalitionValueFunction(
#         model=model,
#         data=data,
#         adj_norm=adj_norm,
#         unit_type=unit_type,
#         base_masks=base_masks,
#         device=device,
#         cache=cache,
#         normalize_fn=kwargs.get("normalize_fn", None),
#         apply_edge_mask_fn=kwargs.get("apply_edge_mask_fn", None),
#         masked_wrapper_cls=kwargs.get("masked_wrapper_cls", None),
#     )

#     start_time = time.time()

#     if method == "owen":
#         shapley_values = owen_shapley_two_stage(
#             value_fn=value_fn,
#             players=players,
#             num_samples_q=kwargs.get("num_samples_q", max(1, num_samples // 10)),
#             num_samples_p=kwargs.get("num_samples_p", max(1, num_samples // 10)),
#             K=kwargs.get("K", 5),
#             C_R=kwargs.get("C_R", 1.0),
#             C_S=kwargs.get("C_S", 1.0),
#             seed=seed,
#         )
#     elif method == "approximation":
#         shapley_values = shapley_approximation(
#             value_fn=value_fn,
#             players=players,
#             num_samples=num_samples,
#             seed=seed,
#         )
#     elif method == "kernelshap":
#         shapley_values = shapley_kernelshap(
#             value_fn=value_fn,
#             players=players,
#             num_samples=num_samples,
#             seed=seed,
#             ridge=kwargs.get("ridge", 1e-6),
#         )
#     elif method == "marginal":
#         shapley_values = shapley_marginal_sampling(
#             value_fn=value_fn,
#             players=players,
#             num_samples=num_samples,
#             seed=seed,
#         )
#     else:
#         raise ValueError(f"Unknown method: {method}")

#     elapsed_time = time.time() - start_time

#     # Put back into original-id space
#     # - node: size = num_nodes
#     # - edge: size = num_edges
#     # - channel: depends; here we keep max(players)+1 for the chosen player list
#     if unit_type == "node":
#         scores = torch.zeros(data.num_nodes, dtype=torch.float32)
#     elif unit_type == "edge":
#         scores = torch.zeros(data.edge_index.size(1), dtype=torch.float32)
#     else:
#         scores = torch.zeros((max(players) + 1) if players else 0, dtype=torch.float32)

#     if len(players) > 0:
#         scores[torch.tensor(players, dtype=torch.long)] = torch.from_numpy(shapley_values).float()

#     return {
#         f"{unit_type}_scores": scores,
#         "shapley_values": shapley_values,
#         "players": players,
#         "num_evaluations": value_fn.num_evaluations,
#         "cache_size": len(cache),
#         "elapsed_time": elapsed_time,
#     }
