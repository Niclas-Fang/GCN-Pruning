"""
Model Masking Wrapper for Channel/Neuron Pruning
(Version: fixed ParameterDict key + correct out_dim for GCNConv)
"""
import torch
import torch.nn as nn
from typing import Dict


class MaskedModelWrapper(nn.Module):
    """
    Wrapper that applies channel/neuron masks to a base GCN model.

    - Detects prunable layers from base_model.list_prunable_layers()
    - Maintains a channel mask for each prunable layer's output dimension
    - Does NOT mask input_proj / output_head (consistent with your design)
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()

        self.base_model = base_model

        if not hasattr(base_model, "list_prunable_layers"):
            raise ValueError("base_model must have list_prunable_layers() method")

        prunable_layers = base_model.list_prunable_layers()

        # ParameterDict keys cannot contain '.', so we map names:
        #   'gcn_layers.0' -> 'gcn_layers__0'
        self.layer_masks = nn.ParameterDict()
        self.layer_names = []  # keep ORIGINAL names with dots for user-facing API

        for layer_name, layer in prunable_layers:
            out_dim = self._infer_out_dim(layer, layer_name)

            mask = torch.ones(out_dim, dtype=torch.float32)
            key = self._to_key(layer_name)
            self.layer_masks[key] = nn.Parameter(mask, requires_grad=False)
            self.layer_names.append(layer_name)

        print(f"[MaskedModelWrapper] Initialized with {len(self.layer_names)} prunable layers")
        for name in self.layer_names:
            k = self._to_key(name)
            print(f"  - {name}: mask shape {tuple(self.layer_masks[k].shape)}")

    # -----------------------------
    # helpers
    # -----------------------------
    @staticmethod
    def _to_key(layer_name: str) -> str:
        """Convert layer name to a ParameterDict-safe key."""
        return layer_name.replace(".", "__")

    @staticmethod
    def _from_key(key: str) -> str:
        """Convert back to original layer name (mainly for debugging)."""
        return key.replace("__", ".")

    @staticmethod
    def _infer_out_dim(layer: nn.Module, layer_name: str) -> int:
        """
        Infer output dimension of a prunable layer.
        - nn.Linear: out_features
        - conv-like: out_channels
        - custom layers with weight:
            * Your GCNConv has weight shape [in_features, out_features] -> out_dim = size(1)
            * Some layers may use [out_features, in_features] -> out_dim = size(0)
          We choose a robust rule:
            - If layer has out_features/out_channels -> use that
            - Else if weight is 2D:
                - Prefer interpreting as [in, out] when that matches common GCNConv style
                - Fallback to [out, in] if needed
        """
        if hasattr(layer, "out_features"):
            return int(layer.out_features)

        if hasattr(layer, "out_channels"):
            return int(layer.out_channels)

        if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor):
            w = layer.weight
            if w.dim() == 2:
                # Heuristic:
                # - For your GCNConv: [in, out] -> out is dim 1
                # - For nn.Linear weight (not used here because it has out_features), it's [out, in]
                # We'll treat as [in, out] if dim1 is reasonably the hidden/out dimension.
                # In your project, prunable layers output hidden_dim, so size(1) is correct.
                return int(w.size(1))
            else:
                # Fallback
                return int(w.size(0))

        raise ValueError(f"Cannot determine output dimension for layer {layer_name}")

    # -----------------------------
    # forward
    # -----------------------------
    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor, return_embeddings: bool = False):
        model = self.base_model

        # Input projection (NOT masked)
        h = model.input_proj(x)
        h = torch.relu(h)
        h = torch.dropout(h, p=model.dropout, train=self.training)

        # GCN / Transformer layers (MASKED)
        for i, gcn_layer in enumerate(model.gcn_layers):
            layer_name = f"gcn_layers.{i}"
            h = gcn_layer(h, adj_norm)

            key = self._to_key(layer_name)
            if key in self.layer_masks:
                mask = self.layer_masks[key].to(h.device)
                h = h * mask.view(1, -1)

            h = torch.relu(h)
            h = torch.dropout(h, p=model.dropout, train=self.training)

        # MLP layers (MASKED)
        for i, mlp_layer in enumerate(model.mlp_layers):
            layer_name = f"mlp_layers.{i}"
            h = mlp_layer(h)

            key = self._to_key(layer_name)
            if key in self.layer_masks:
                mask = self.layer_masks[key].to(h.device)
                h = h * mask.view(1, -1)

            h = torch.relu(h)
            h = torch.dropout(h, p=model.dropout, train=self.training)

        embeddings = h
        logits = model.output_head(h)

        if return_embeddings:
            return logits, embeddings
        return logits

    # -----------------------------
    # mask APIs (use ORIGINAL names)
    # -----------------------------
    def set_layer_mask(self, layer_name: str, mask: torch.Tensor):
        key = self._to_key(layer_name)
        if key not in self.layer_masks:
            raise ValueError(f"Layer {layer_name} not found in prunable layers")

        expected_shape = self.layer_masks[key].shape
        if mask.shape != expected_shape:
            raise ValueError(f"Mask shape {tuple(mask.shape)} does not match expected shape {tuple(expected_shape)}")

        self.layer_masks[key].data = mask.float()

    def set_all_masks(self, masks: Dict[str, torch.Tensor]):
        for layer_name, mask in masks.items():
            self.set_layer_mask(layer_name, mask)

    def get_layer_mask(self, layer_name: str) -> torch.Tensor:
        key = self._to_key(layer_name)
        if key not in self.layer_masks:
            raise ValueError(f"Layer {layer_name} not found")
        return self.layer_masks[key]

    def get_all_masks(self) -> Dict[str, torch.Tensor]:
        return {name: self.get_layer_mask(name).clone() for name in self.layer_names}

    def reset_masks(self):
        for layer_name in self.layer_names:
            self.layer_masks[self._to_key(layer_name)].data.fill_(1.0)

    # -----------------------------
    # sparsity
    # -----------------------------
    def get_layer_sparsity(self, layer_name: str) -> float:
        mask = self.get_layer_mask(layer_name)
        return 1.0 - mask.float().mean().item()

    def get_global_sparsity(self) -> float:
        total = 0
        pruned = 0
        for layer_name in self.layer_names:
            mask = self.get_layer_mask(layer_name)
            total += mask.numel()
            pruned += (mask == 0).sum().item()
        return pruned / total if total > 0 else 0.0

    def get_sparsity_statistics(self) -> Dict[str, float]:
        stats = {name: self.get_layer_sparsity(name) for name in self.layer_names}
        stats["global"] = self.get_global_sparsity()
        return stats

    # -----------------------------
    # export / import (use ORIGINAL names)
    # -----------------------------
    def export_masks(self) -> Dict[str, torch.Tensor]:
        """
        Export masks as {original_layer_name: mask_tensor_cpu}.
        This keeps compatibility with your sanity tests and future checkpoints.
        """
        return {name: self.get_layer_mask(name).detach().cpu().clone() for name in self.layer_names}

    def import_masks(self, masks: Dict[str, torch.Tensor]):
        for layer_name, mask in masks.items():
            if layer_name in self.layer_names:
                self.set_layer_mask(layer_name, mask)
            else:
                print(f"Warning: Layer {layer_name} not found, skipping")

    # -----------------------------
    # parameter counting (approx; safe for your use)
    # -----------------------------
    def count_parameters(self, only_active: bool = False) -> int:
        if not only_active:
            return sum(p.numel() for p in self.base_model.parameters())

        total = 0

        # Non-prunable layers (input_proj, output_head)
        for name, param in self.base_model.named_parameters():
            if "input_proj" in name or "output_head" in name:
                total += param.numel()

        # Prunable layers: count active output channels * incoming dim (approx)
        prunable_layer_map = dict(self.base_model.list_prunable_layers())

        for layer_name in self.layer_names:
            layer = prunable_layer_map[layer_name]
            mask = self.get_layer_mask(layer_name)
            active_channels = int(mask.sum().item())

            if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor):
                w = layer.weight
                if w.dim() == 2:
                    # For your GCNConv: [in, out] -> in_dim = size(0)
                    # For Linear-like: [out, in] -> in_dim = size(1)
                    # We prefer treating GCNConv as [in, out].
                    in_dim = int(w.size(0))
                    total += active_channels * in_dim

                if hasattr(layer, "bias") and layer.bias is not None:
                    total += active_channels

        return total

    def print_sparsity_summary(self):
        print("=" * 60)
        print("Sparsity Summary")
        print("=" * 60)

        stats = self.get_sparsity_statistics()

        for layer_name in self.layer_names:
            sparsity = stats[layer_name]
            mask = self.get_layer_mask(layer_name)
            active = int(mask.sum().item())
            total = int(mask.numel())
            print(f"  {layer_name:20s}: {sparsity:6.2%} pruned ({active}/{total} active)")

        print("-" * 60)
        print(f"  {'Global':20s}: {stats['global']:6.2%} pruned")

        total_params = self.count_parameters(only_active=False)
        active_params = self.count_parameters(only_active=True)
        print(f"\nParameters: {active_params:,} / {total_params:,} active")
        print("=" * 60)
