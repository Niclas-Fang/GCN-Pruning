"""
Joint Pruning Experiment Runner

Supports joint graph-model pruning with various scoring methods and keep ratios.
"""
import sys
import json
import csv
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import GCNNet, normalize_adjacency
from data import load_dataset
from pruning import apply_node_mask, apply_edge_mask, MaskedModelWrapper
from scorers import score
from scorers.shapley import compute_shapley_scores
from train import evaluate
from utils import set_seed, setup_logger, save_results


class JointPruningRunner:
    """
    Runner for joint graph-model pruning experiments.
    
    Supports:
    - Graph pruning: node or edge removal
    - Model pruning: channel removal (multiple layers)
    - Various scoring methods
    - Multiple keep ratios (grid search)
    """
    
    def __init__(
        self,
        model,
        data,
        graph_unit: str = 'node',  # 'node' or 'edge'
        model_unit: str = 'channel',
        device: torch.device = None,
        seed: int = 42
    ):
        """
        Initialize runner.
        
        Args:
            model: GCN model
            data: Graph data
            graph_unit: 'node' or 'edge'
            model_unit: 'channel' (for now)
            device: Device
            seed: Random seed
        """
        assert graph_unit in ['node', 'edge'], f"Invalid graph_unit: {graph_unit}"
        assert model_unit == 'channel', f"Only 'channel' supported for model_unit"
        
        self.model = model
        self.data = data
        self.graph_unit = graph_unit
        self.model_unit = model_unit
        self.device = device if device is not None else torch.device('cpu')
        self.seed = seed
        
        # Move data to device
        self.data.x = self.data.x.to(self.device)
        self.data.edge_index = self.data.edge_index.to(self.device)
        self.data.y = self.data.y.to(self.device)
        self.data.train_mask = self.data.train_mask.to(self.device)
        self.data.val_mask = self.data.val_mask.to(self.device)
        self.data.test_mask = self.data.test_mask.to(self.device)
        
        # Normalize original adjacency
        self.adj_norm_original = normalize_adjacency(
            self.data.edge_index, self.data.num_nodes
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Cache for scores
        self.graph_scores = None
        self.model_scores = None
    
    # 在 JointPruningRunner 里加一个小工具函数（为了grad
    def _infer_out_dim(self, layer) -> int:
        # 与你的 MaskedModelWrapper 口径一致
        if hasattr(layer, "out_features"):
            return int(layer.out_features)
        if hasattr(layer, "out_channels"):
            return int(layer.out_channels)
        if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor) and layer.weight.dim() == 2:
            # your GCNConv: [in, out] -> out is dim 1
            return int(layer.weight.size(1))
        raise ValueError("Cannot infer out_dim for layer")

    def compute_graph_scores(
        self,
        method: str,
        save_path: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute or load graph unit scores.

        Args:
            method: Scoring method ('degree', 'random', 'owen', 'approximation')
            save_path: Path to save/load scores
            **kwargs: Method-specific arguments

        Returns:
            Scores tensor
        """
        if save_path is not None and Path(save_path).exists():
            print(f"[Runner] Loading graph scores from {save_path}")
            from scorers.base import PruningScorer
            result = PruningScorer.load_scores(save_path)
            scores_key = f'{self.graph_unit}_scores'
            return result['scores'][scores_key]

        print(f"[Runner] Computing {self.graph_unit} scores with method '{method}'...")

        if method in ['owen', 'approximation']:
            # 1) 外部优先：如果已经传了 players，就直接用
            players = kwargs.get('players', None)

            # 2) 否则按原逻辑内部选 players
            if players is None:
                if self.graph_unit == 'node':
                    val_nodes = self.data.val_mask.nonzero(as_tuple=True)[0]
                    num_players = min(kwargs.get('num_players', 100), len(val_nodes))
                    players = val_nodes[:num_players].tolist()
                else:  # edge
                    total_edges = self.data.edge_index.size(1)
                    num_players = min(kwargs.get('num_players', 100), total_edges)
                    edge_indices = torch.randperm(total_edges)[:num_players]
                    players = edge_indices.tolist()

            base_masks = {
                'node_mask': torch.ones(self.data.num_nodes, dtype=torch.bool, device=self.device),
                'edge_mask': torch.ones(self.data.edge_index.size(1), dtype=torch.bool, device=self.device),
                'channel_masks': {}
            }

            for layer_name, layer in self.model.list_prunable_layers():
                out_dim = self._infer_out_dim(layer)
                base_masks['channel_masks'][layer_name] = torch.ones(out_dim, device=self.device)

            result = compute_shapley_scores(
                model=self.model,
                data=self.data,
                adj_norm=self.adj_norm_original,
                unit_type=self.graph_unit,
                base_masks=base_masks,
                players=players,
                method=method,
                coalitions=kwargs.get('coalitions', None),
                num_samples=kwargs.get('num_samples', 50),
                num_samples_q=kwargs.get('num_samples_q', 10),
                num_samples_p=kwargs.get('num_samples_p', 10),
                K=kwargs.get('K', 5),
                seed=self.seed,
                device=self.device
            )

            scores = result[f'{self.graph_unit}_scores']

        else:
            result = score(
                unit_type=self.graph_unit,
                method=method,
                model=self.model,
                data=self.data,
                save_path=save_path,
                seed=self.seed,
                **kwargs
            )

            scores_key = f'{self.graph_unit}_scores'
            scores = result['scores'][scores_key]

        if save_path is not None:
            result_to_save = {
                'unit_type': self.graph_unit,
                'method': method,
                'scores': {f'{self.graph_unit}_scores': scores},
                'seed': self.seed,
                'config': kwargs
            }

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(result_to_save, f)
            print(f"[Runner] Saved graph scores to {save_path}")

        self.graph_scores = scores
        return scores
    
    def compute_model_scores(
        self,
        method: str,
        save_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute or load model (channel) scores.

        Args:
            method: Scoring method ('magnitude', 'random', 'owen', 'approximation')
            save_path: Path to save/load scores
            **kwargs: Method-specific arguments

        Returns:
            Dictionary mapping layer names to scores
        """
        if save_path is not None and Path(save_path).exists():
            print(f"[Runner] Loading model scores from {save_path}")
            from scorers.base import PruningScorer
            result = PruningScorer.load_scores(save_path)
            return result['scores']

        print(f"[Runner] Computing channel scores with method '{method}'...")

        mask_required_methods = {'grad', 'taylor', 'sensitivity', 'embedding_delta'}
        optional_wrapper_methods = {'activation_l2'}

        # -------------------------
        # A. 非 Shapley 方法：保持你原逻辑
        # -------------------------
        if method in ['magnitude', 'random', 'activation_l2', 'grad', 'taylor', 'sensitivity', 'embedding_delta']:
            model_for_scoring = self.model
            if method in mask_required_methods or method in optional_wrapper_methods:
                model_for_scoring = MaskedModelWrapper(self.model).to(self.device)
                model_for_scoring.eval()

            result = score(
                unit_type='channel',
                method=method,
                model=model_for_scoring,
                data=self.data,
                save_path=save_path,
                seed=self.seed,
                **kwargs
            )
            scores = result['scores']

        # -------------------------
        # B. Shapley / Owen 方法
        # -------------------------
        else:
            scores = {}

            # 可选：只计算某一层（推荐给你现在的 owen_32 / owen_64 用）
            target_layer = kwargs.get('target_layer', None)

            # 外部可传入固定 players / coalitions
            external_players = kwargs.get('players', None)
            external_coalitions = kwargs.get('coalitions', None)

            layers = list(self.model.list_prunable_layers())

            print("Prunable layers:", [ln for ln, _ in layers])# 打印所有可剪枝层，方便调试

            if target_layer is not None:
                layers = [(ln, l) for ln, l in layers if ln == target_layer]
                if len(layers) == 0:
                    raise ValueError(f"target_layer '{target_layer}' not found in prunable layers")

            base_masks = {
                'node_mask': torch.ones(self.data.num_nodes, dtype=torch.bool, device=self.device),
                'edge_mask': torch.ones(self.data.edge_index.size(1), dtype=torch.bool, device=self.device),
                'channel_masks': {}
            }
            for ln, l in self.model.list_prunable_layers():
                out_dim = self._infer_out_dim(l)
                base_masks['channel_masks'][ln] = torch.ones(out_dim, device=self.device)

            for layer_name, layer in layers:
                print(f"  Computing scores for {layer_name}...")

                num_channels = self._infer_out_dim(layer)

                # 1) 外部优先：若已传 players，直接用
                if external_players is not None:
                    players = external_players
                else:
                    num_players = min(kwargs.get('num_players', 16), num_channels)
                    players = list(range(num_players))

                result = compute_shapley_scores(
                    model=self.model,
                    data=self.data,
                    adj_norm=self.adj_norm_original,
                    unit_type='channel',
                    base_masks=base_masks,
                    players=players,
                    method=method,
                    coalitions=external_coalitions,
                    num_samples=kwargs.get('num_samples', 20),
                    num_samples_q=kwargs.get('num_samples_q', 10),
                    num_samples_p=kwargs.get('num_samples_p', 10),
                    K=kwargs.get('K', 5),
                    seed=self.seed,
                    device=self.device,
                    target_layer=layer_name
                )

                # channel 情况下，这里取返回的 shapley_values，再手工映射到目标层完整 tensor
                shapley_values = result['shapley_values']

                layer_scores = torch.zeros(num_channels, device=self.device)
                layer_scores[players] = torch.from_numpy(shapley_values).float().to(self.device)
                scores[layer_name] = layer_scores

            # 对未参与计算的层补零，避免后续 create_channel_masks 出错
            all_layer_names = [ln for ln, _ in self.model.list_prunable_layers()]
            for ln, l in self.model.list_prunable_layers():
                if ln not in scores:
                    out_dim = self._infer_out_dim(l)
                    scores[ln] = torch.zeros(out_dim, device=self.device)

        if save_path is not None:
            result_to_save = {
                'unit_type': 'channel',
                'method': method,
                'scores': scores,
                'seed': self.seed,
                'config': kwargs
            }

            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(result_to_save, f)
            print(f"[Runner] Saved model scores to {save_path}")

        self.model_scores = scores
        return scores

    def create_graph_mask(
        self,
        scores: torch.Tensor,
        keep_ratio: float
    ) -> torch.Tensor:
        """
        Create graph mask from scores.
        
        Args:
            scores: Importance scores
            keep_ratio: Fraction to keep (0-1)
        
        Returns:
            Boolean mask
        """
        num_units = scores.size(0)
        num_keep = int(num_units * keep_ratio)

        if num_keep == 0:
            num_keep = 1  # Keep at least one

        # Get top-k indices
        _, top_indices = torch.topk(scores, num_keep)

        # Create mask on the same device as the graph
        mask = torch.zeros(num_units, dtype=torch.bool)
        mask[top_indices] = True

        return mask
    
    def create_channel_masks(
        self,
        scores: Dict[str, torch.Tensor],
        keep_ratio: float,
        strategy: str = 'global'
    ) -> Dict[str, torch.Tensor]:
        """
        Create channel masks from scores.
        
        Args:
            scores: Dictionary mapping layer names to scores
            keep_ratio: Fraction to keep
            strategy: 'global' or 'layerwise'
                - global: Rank all channels globally, keep top-k
                - layerwise: Keep top-k% per layer
        
        Returns:
            Dictionary mapping layer names to masks
        """
        masks = {}
        
        if strategy == 'global':
            # Concatenate all scores
            all_scores = []
            layer_sizes = []
            layer_order = []
            
            for layer_name in sorted(scores.keys()):
                all_scores.append(scores[layer_name])
                layer_sizes.append(scores[layer_name].size(0))
                layer_order.append(layer_name)

            all_scores = torch.cat(all_scores)
            
            # Get top-k globally
            num_total = all_scores.size(0)
            num_keep = int(num_total * keep_ratio)
            if num_keep == 0:
                num_keep = 1
            
            _, top_indices = torch.topk(all_scores, num_keep)
            
            # Distribute back to layers
            global_mask = torch.zeros(num_total, dtype=torch.bool)
            global_mask[top_indices] = True
            
            offset = 0
            for layer_name, size in zip(layer_order, layer_sizes):
                layer_mask = global_mask[offset:offset+size].float()
                masks[layer_name] = layer_mask
                offset += size
        
        elif strategy == 'layerwise':
            # Keep top-k% per layer
            for layer_name, layer_scores in scores.items():
                num_channels = layer_scores.size(0)
                num_keep = int(num_channels * keep_ratio)
                if num_keep == 0:
                    num_keep = 1

                _, top_indices = torch.topk(layer_scores, num_keep)

                mask = torch.zeros(num_channels)
                mask[top_indices] = 1.0
                
                masks[layer_name] = mask
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return masks
    
    def evaluate_joint_pruning(
        self,
        graph_mask: Optional[torch.Tensor],
        channel_masks: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate model with joint graph-model pruning.
        
        Args:
            graph_mask: Node or edge mask (None = no pruning)
            channel_masks: Channel masks per layer
        
        Returns:
            Dictionary with metrics
        """
        # Apply graph mask
        if graph_mask is not None:
            if self.graph_unit == 'node':
                new_edge_index = apply_node_mask(
                    self.data, graph_mask,
                    ensure_undirected=True,
                    preserve_self_loops=True
                )
            else:  # edge
                new_edge_index = apply_edge_mask(
                    self.data, graph_mask,
                    ensure_undirected=True,
                    add_self_loops_flag=True
                )
            
            # Normalize adjacency
            adj_norm = normalize_adjacency(
                new_edge_index, self.data.num_nodes
            ).to(self.device)
        else:
            adj_norm = self.adj_norm_original
        
        # Create masked model
        masked_model = MaskedModelWrapper(self.model).to(self.device)
        
        # Apply channel masks
        for layer_name, mask in channel_masks.items():
            if layer_name in masked_model.layer_names:
                masked_model.set_layer_mask(layer_name, mask.to(self.device))
        
        # Evaluate
        masked_model.eval()
        
        val_loss, val_acc = evaluate(
            masked_model, self.data, self.data.val_mask, adj_norm, self.criterion
        )
        
        test_loss, test_acc = evaluate(
            masked_model, self.data, self.data.test_mask, adj_norm, self.criterion
        )
        
        # Compute sparsities
        if graph_mask is not None:
            graph_sparsity = 1.0 - (graph_mask.sum().item() / graph_mask.size(0))
        else:
            graph_sparsity = 0.0
        
        model_sparsity = masked_model.get_global_sparsity()
        
        return {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'graph_sparsity': graph_sparsity,
            'model_sparsity': model_sparsity
        }
    
    def run_experiment(
        self,
        graph_scoring_method: str,
        model_scoring_method: str,
        keep_ratios: List[Tuple[float, float]],  # [(graph_ratio, model_ratio), ...]
        channel_strategy: str = 'global',
        output_dir: str = 'outputs/joint_pruning',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Run joint pruning experiment.
        
        Args:
            graph_scoring_method: Method for graph scoring
            model_scoring_method: Method for model scoring
            keep_ratios: List of (graph_keep_ratio, model_keep_ratio) tuples
            channel_strategy: 'global' or 'layerwise'
            output_dir: Output directory
            **kwargs: Additional arguments
        
        Returns:
            List of result dictionaries
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("Joint Pruning Experiment")
        print("=" * 70)
        print(f"Graph unit: {self.graph_unit}")
        print(f"Graph scoring: {graph_scoring_method}")
        print(f"Model scoring: {model_scoring_method}")
        print(f"Channel strategy: {channel_strategy}")
        print(f"Keep ratios: {len(keep_ratios)} configurations")
        print("=" * 70)
        
        # Compute scores
        graph_scores_path = output_dir / f'scores_{self.graph_unit}_{graph_scoring_method}.pkl'
        model_scores_path = output_dir / f'scores_channel_{model_scoring_method}.pkl'
        
        graph_scores = self.compute_graph_scores(
            graph_scoring_method,
            save_path=str(graph_scores_path),
            **kwargs.get('graph_scoring_kwargs', {})
        )
        
        model_scores = self.compute_model_scores(
            model_scoring_method,
            save_path=str(model_scores_path),
            **kwargs.get('model_scoring_kwargs', {})
        )
        
        # Run experiments for each keep ratio
        results = []
        
        for i, (graph_ratio, model_ratio) in enumerate(keep_ratios):
            print(f"\n[{i+1}/{len(keep_ratios)}] Graph={graph_ratio:.2f}, Model={model_ratio:.2f}")
            
            # Create masks
            graph_mask = self.create_graph_mask(graph_scores, graph_ratio)
            channel_masks = self.create_channel_masks(
                model_scores, model_ratio, strategy=channel_strategy
            )
            
            # Evaluate
            metrics = self.evaluate_joint_pruning(graph_mask, channel_masks)
            
            # Store result
            result = {
                'graph_unit': self.graph_unit,
                'graph_scoring_method': graph_scoring_method,
                'model_scoring_method': model_scoring_method,
                'channel_strategy': channel_strategy,
                'graph_keep_ratio': graph_ratio,
                'model_keep_ratio': model_ratio,
                'graph_sparsity': metrics['graph_sparsity'],
                'model_sparsity': metrics['model_sparsity'],
                'val_loss': metrics['val_loss'],
                'val_acc': metrics['val_acc'],
                'test_loss': metrics['test_loss'],
                'test_acc': metrics['test_acc'],
                'seed': self.seed,
                # Model hyperparameters
                'model_hidden_dim': self.model.hidden_dim,
                'model_num_gcn_layers': self.model.num_gcn_layers,
                'model_num_mlp_layers': self.model.num_mlp_layers,
                'model_dropout': self.model.dropout,
            }
            
            results.append(result)
            
            print(f"  Graph sparsity: {metrics['graph_sparsity']:.2%}")
            print(f"  Model sparsity: {metrics['model_sparsity']:.2%}")
            print(f"  Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_acc']:.4f}")
            print(f"  Test Loss: {metrics['test_loss']:.4f}, Test Acc: {metrics['test_acc']:.4f}")
        
        # Save results
        self._save_results(results, output_dir)
        
        print("\n" + "=" * 70)
        print(f"✓ Experiment complete! Results saved to {output_dir}")
        print("=" * 70)
        
        return results
    
    def _save_results(self, results: List[Dict], output_dir: Path):
        """Save results to JSON and CSV."""
        # JSON
        json_path = output_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[Runner] Saved results to {json_path}")
        
        # CSV
        csv_path = output_dir / 'results.csv'
        if results:
            keys = results[0].keys()
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            print(f"[Runner] Saved results to {csv_path}")


def create_keep_ratio_grid(
    graph_ratios: List[float],
    model_ratios: List[float]
) -> List[Tuple[float, float]]:
    """
    Create 2D grid of keep ratios.
    
    Args:
        graph_ratios: List of graph keep ratios
        model_ratios: List of model keep ratios
    
    Returns:
        List of (graph_ratio, model_ratio) tuples
    """
    grid = []
    for gr in graph_ratios:
        for mr in model_ratios:
            grid.append((gr, mr))
    return grid


def create_keep_ratio_sequence(
    ratios: List[float],
    independent: bool = True
) -> List[Tuple[float, float]]:
    """
    Create sequence of keep ratios.
    
    Args:
        ratios: List of keep ratios
        independent: If True, vary independently. If False, use same ratio.
    
    Returns:
        List of (graph_ratio, model_ratio) tuples
    """
    if independent:
        # Full combinations
        return create_keep_ratio_grid(ratios, ratios)
    else:
        # Same ratio for both
        return [(r, r) for r in ratios]
