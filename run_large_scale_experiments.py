"""
Large-Scale Joint Pruning Experiments

Comprehensive experiments comparing:
1. Different scoring methods (baseline vs Shapley)
2. Graph units (node vs edge pruning)
3. Channel strategies (global vs layerwise)
4. Sparsity levels (0% to 90%)
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import GCNNet
from data import load_dataset
from experiments import JointPruningRunner, create_keep_ratio_grid
from train import train_gcn
from utils import set_seed, setup_logger
from config import Config

from coalition_builder import build_edge_coalitions, build_channel_coalitions


class LargeScaleExperiment:
    """
    Manager for large-scale pruning experiments.
    
    Runs systematic comparisons across multiple dimensions:
    - Scoring methods
    - Graph units
    - Channel strategies
    - Sparsity levels
    - Multiple seeds for robustness
    """
    
    def __init__(
        self,
        base_output_dir: str = 'outputs/large_scale',
        device: str = 'cpu',
        seeds: list = None
    ):
        """
        Initialize experiment manager.
        
        Args:
            base_output_dir: Base directory for all experiments
            device: Device to use
            seeds: List of random seeds for robustness
        """
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(device)
        self.seeds = seeds if seeds is not None else [42, 123, 456]
        # self.seeds = seeds if seeds is not None else [42]
        
        # Setup logger
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.base_output_dir / f'experiment_{timestamp}.log'
        self.logger = setup_logger('large_scale', str(log_file))
        
        self.logger.info("=" * 70)
        self.logger.info("Large-Scale Joint Pruning Experiments")
        self.logger.info("=" * 70)
        self.logger.info(f"Output directory: {self.base_output_dir}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Seeds: {self.seeds}")
        self.logger.info("=" * 70)
    
    def train_base_model(self, seed: int = 42) -> GCNNet:
        """
        Train base model for pruning experiments.
        
        Args:
            seed: Random seed
        
        Returns:
            Trained model
        """
        self.logger.info(f"\n[Training] Training base model (seed={seed})...")
        
        set_seed(seed, deterministic=True)
        
        # Create config
        config = Config()
        config.exp_name = f'base_model_seed{seed}'
        config.seed = seed
        config.epochs = 200
        config.hidden_dim = 64
        config.num_gcn_layers = 3
        config.num_mlp_layers = 1
        config.dropout = 0.5
        config.lr = 0.01
        config.weight_decay = 5e-4
        config.early_stop_patience = 50
        config.device = 'cpu'
        config.output_dir = str(self.base_output_dir / 'base_models')
        
        # Train
        model, results = train_gcn(config)
        
        self.logger.info(f"  Base model trained:")
        self.logger.info(f"    Val Acc: {results['best_val_metric']:.4f}")
        self.logger.info(f"    Test Acc: {results['test_acc']:.4f}")
        
        return model
    
    # def experiment_1_baseline_methods(self):
    #     """
    #     Experiment 1: Compare baseline scoring methods.

    #     Adds two Shapley-based scoring configs (edge+channel).
    #     """
    #     self.logger.info("\n" + "=" * 70)
    #     self.logger.info("EXPERIMENT 1: Baseline Scoring Methods")
    #     self.logger.info("=" * 70)

    #     exp_dir = self.base_output_dir / 'exp1_baseline_methods'
    #     exp_dir.mkdir(exist_ok=True)

    #     # Load data
    #     data = load_dataset('Cora')

    #     # Sparsity levels
    #     keep_ratios = [
    #         (1.0, 1.0),
    #         (0.9, 0.9),
    #         (0.8, 0.8),
    #         (0.7, 0.7),
    #         (0.6, 0.6),
    #         (0.5, 0.5),
    #     ]

    #     # Methods to compare (✅新增两组 Shapley)
    #     methods = [
    #         ('feature_l2', 'feature_l2'),
    #         ('grad', 'grad'),
    #         ('taylor', 'taylor'),
    #         ('sensitivity', 'sensitivity'),
    #         ('random', 'random'),
    #         ('approximation', 'approximation'),  # permutation MC Shapley
    #         ('owen', 'owen'),          # edge Owen + channel permutation MC (最稳)
    #     ]

    #     # Shapley kwargs（只在 Shapley 时使用，避免非 Shapley 走 kwargs）
    #     shapley_graph_kwargs = {
    #         "num_players": 200,   # sampled edges as players
    #         "num_samples": 30,    # MC permutations for 'approximation'
    #         # Owen 相关参数可选：runner 会转发给 compute_shapley_scores
    #         "K": 5,
    #         "C_R": 1.0,
    #         "C_S": 1.0,
    #     }
    #     shapley_model_kwargs = {
    #         "num_players": 16,    # channels per layer
    #         "num_samples": 20,
    #     }

    #     shapley_methods = {"approximation", "owen"}  # runner graph侧识别的 Shapley method 名

    #     all_results = []

    #     for seed in self.seeds:
    #         self.logger.info(f"\n--- Seed {seed} ---")

    #         # Train base model
    #         model = self.train_base_model(seed)

    #         for graph_method, model_method in methods:
    #             self.logger.info(f"\n  Method: {graph_method} + {model_method}")

    #             runner = JointPruningRunner(
    #                 model=model,
    #                 data=data,
    #                 graph_unit='edge',
    #                 device=self.device,
    #                 seed=seed
    #             )

    #             # ✅最小改动：仅当 graph_method 是 shapley 时才传 kwargs
    #             extra_kwargs = {}
    #             if graph_method in shapley_methods:
    #                 extra_kwargs["graph_scoring_kwargs"] = shapley_graph_kwargs
    #                 extra_kwargs["model_scoring_kwargs"] = shapley_model_kwargs

    #             results = runner.run_experiment(
    #                 graph_scoring_method=graph_method,
    #                 model_scoring_method=model_method,
    #                 keep_ratios=keep_ratios,
    #                 channel_strategy='global',
    #                 output_dir=str(exp_dir / f'seed{seed}_{graph_method}_{model_method}'),
    #                 **extra_kwargs
    #             )

    #             for r in results:
    #                 r['experiment'] = 'exp1_baseline'
    #                 r['seed'] = seed

    #             all_results.extend(results)

    #     self._save_aggregated_results(all_results, exp_dir / 'all_results.json')
    #     self._analyze_and_plot(all_results, exp_dir)

    #     self.logger.info(f"\n✓ Experiment 1 complete! Results in {exp_dir}")

    def experiment_1_baseline_methods(self):
        """
        Experiment 1: Compare baseline scoring methods.

        保留原有 baseline 配置，并额外增加增强版 Owen 组合：
        - 多种 edge coalition strategy
        - channel Owen 32 / 64 两种 player 截取规模
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXPERIMENT 1: Baseline Scoring Methods")
        self.logger.info("=" * 70)

        exp_dir = self.base_output_dir / 'exp1_baseline_methods'
        exp_dir.mkdir(exist_ok=True)

        # Load data
        data = load_dataset('Cora')

        # Sparsity levels
        keep_ratios = [
            (1.0, 1.0),
            (0.9, 0.9),
            (0.8, 0.8),
            (0.7, 0.7),
            (0.6, 0.6),
            (0.5, 0.5),
        ]

        # =========================
        # 1) 原有 baseline methods：保持不变
        # =========================
        baseline_methods = [
            ('feature_l2', 'activation_l2'),
            ('grad', 'grad'),
            ('taylor', 'taylor'),
            ('sensitivity', 'sensitivity'),
            ('random', 'random'),
            ('approximation', 'approximation'),
        ]

        # approximation 的 kwargs：保持原有风格
        shapley_graph_kwargs = {
            "num_players": 200,
            "num_samples": 30,
            "K": 5,
            "C_R": 1.0,
            "C_S": 1.0,
        }
        shapley_model_kwargs = {
            "num_players": 16,
            "num_samples": 20,
        }

        # =========================
        # 2) 增强版 Owen 配置
        # =========================

        # edge coalition strategies
        edge_owen_variants = [
            {
                "name": "owen_source_node_ch32",
                "edge_strategy": "source_node",
                "edge_kwargs": {},
                "channel_player_cap": 32,
            },
            {
                "name": "owen_target_node_ch32",
                "edge_strategy": "target_node",
                "edge_kwargs": {},
                "channel_player_cap": 32,
            },
            {
                "name": "owen_community_ch32",
                "edge_strategy": "community",
                "edge_kwargs": {"method": "louvain"},
                "channel_player_cap": 32,
            },
            # {
            #     "name": "owen_motif_bridge_ch32",
            #     "edge_strategy": "motif_bridge",
            #     "edge_kwargs": {},
            #     "channel_player_cap": 32,
            # },
            # {
            #     "name": "owen_motif_triangle_ch32",
            #     "edge_strategy": "motif_triangle",
            #     "edge_kwargs": {},
            #     "channel_player_cap": 32,
            # },
            # {
            #     "name": "owen_motif_clustering_ch32",
            #     "edge_strategy": "motif_clustering",
            #     "edge_kwargs": {},
            #     "channel_player_cap": 32,
            # },
            {
                "name": "owen_source_node_ch64",
                "edge_strategy": "source_node",
                "edge_kwargs": {},
                "channel_player_cap": 64,
            },
            {
                "name": "owen_target_node_ch64",
                "edge_strategy": "target_node",
                "edge_kwargs": {},
                "channel_player_cap": 64,
            },
            {
                "name": "owen_community_ch64",
                "edge_strategy": "community",
                "edge_kwargs": {"method": "louvain"},
                "channel_player_cap": 64,
            },
            # {
            #     "name": "owen_motif_bridge_ch64",
            #     "edge_strategy": "motif_bridge",
            #     "edge_kwargs": {},
            #     "channel_player_cap": 64,
            # },
            # {
            #     "name": "owen_motif_triangle_ch64",
            #     "edge_strategy": "motif_local_clustering",
            #     "edge_kwargs": {},
            #     "channel_player_cap": 64,
            # },
            # {
            #     "name": "owen_motif_clustering_ch64",
            #     "edge_strategy": "motif_triangle",
            #     "edge_kwargs": {},
            #     "channel_player_cap": 64,
            # }
        ]

        # 你示例里的 layerwise channel coalition 参数
        # 如与你模型实际结构不一致，只改这里即可
        channel_layer_channel_counts = [64, 64, 64, 64]
        channel_layer_names = ['gcn_layers.0', 'gcn_layers.1', 'gcn_layers.2', 'mlp_layers.0']

        # Owen 额外参数
        max_edge_players = 100
        edge_num_samples_q = 5
        edge_num_samples_p = 5
        edge_K = 10

        channel_num_samples_q = 5
        channel_num_samples_p = 5

        all_results = []

        # -------------------------
        # 小工具函数：只在本实验函数内部用
        # -------------------------
        def _restrict_coalitions_to_players(coalitions, players):
            """
            Restrict full coalitions to the given players only.
            Ensures the returned coalitions form a partition of `players`.
            """
            player_set = set(players)
            restricted = []

            for group in coalitions:
                sub_group = [p for p in group if p in player_set]
                if len(sub_group) > 0:
                    restricted.append(sub_group)

            flat_players = set()
            for group in restricted:
                flat_players.update(group)

            if flat_players != player_set:
                missing = player_set - flat_players
                extra = flat_players - player_set
                raise ValueError(
                    f"Restricted coalitions do not match players. "
                    f"missing={missing}, extra={extra}"
                )

            return restricted

        def _build_edge_owen_kwargs(data, edge_strategy, edge_kwargs):
            coalition_result = build_edge_coalitions(
                strategy=edge_strategy,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                **edge_kwargs
            )

            # 采样 players
            players = list(coalition_result.player_to_group.keys())[:max_edge_players]

            # 关键：coalitions 也要同步裁剪到 players
            restricted_coalitions = _restrict_coalitions_to_players(
                coalition_result.coalitions,
                players
            )

            return {
                "players": players,
                "coalitions": restricted_coalitions,
                "num_samples_q": edge_num_samples_q,
                "num_samples_p": edge_num_samples_p,
                "K": edge_K,
            }

        def _build_channel_owen_kwargs(player_cap):
            coalition_result = build_channel_coalitions(
                'layerwise',
                layer_channel_counts=channel_layer_channel_counts,
                layer_names=channel_layer_names
            )

            layer_0_players = coalition_result.coalitions[0][:player_cap]

            restricted_coalitions = _restrict_coalitions_to_players(
                coalition_result.coalitions,
                layer_0_players
            )

            return {
                "players": layer_0_players,
                "coalitions": restricted_coalitions,
                "target_layer": "gcn_layers.0",
                "num_samples_q": channel_num_samples_q,
                "num_samples_p": channel_num_samples_p,
            }

        for seed in self.seeds:
            self.logger.info(f"\n--- Seed {seed} ---")

            # Train base model
            model = self.train_base_model(seed)

            # =========================
            # A. 原有 baseline 部分
            # =========================
            for graph_method, model_method in baseline_methods:
                self.logger.info(f"\n  Method: {graph_method} + {model_method}")

                runner = JointPruningRunner(
                    model=model,
                    data=data,
                    graph_unit='edge',
                    device=self.device,
                    seed=seed
                )

                extra_kwargs = {}
                if graph_method == "approximation":
                    extra_kwargs["graph_scoring_kwargs"] = shapley_graph_kwargs
                    extra_kwargs["model_scoring_kwargs"] = shapley_model_kwargs

                results = runner.run_experiment(
                    graph_scoring_method=graph_method,
                    model_scoring_method=model_method,
                    keep_ratios=keep_ratios,
                    channel_strategy='layerwise',
                    output_dir=str(exp_dir / f'seed{seed}_{graph_method}_{model_method}'),
                    **extra_kwargs
                )

                for r in results:
                    r['experiment'] = 'exp1_baseline'
                    r['seed'] = seed
                    r['graph_variant'] = graph_method
                    r['model_variant'] = model_method
                    r['display_method'] = f'{graph_method}+{model_method}'

                all_results.extend(results)

            # =========================
            # B. 增强版 Owen 追加实验
            # =========================
            for variant in edge_owen_variants:
                variant_name = variant["name"]
                edge_strategy = variant["edge_strategy"]
                edge_kwargs = variant["edge_kwargs"]
                channel_player_cap = variant["channel_player_cap"]

                self.logger.info(f"\n  Owen Variant: {variant_name}")

                runner = JointPruningRunner(
                    model=model,
                    data=data,
                    graph_unit='edge',
                    device=self.device,
                    seed=seed
                )

                graph_owen_kwargs = _build_edge_owen_kwargs(
                    data=data,
                    edge_strategy=edge_strategy,
                    edge_kwargs=edge_kwargs
                )
                model_owen_kwargs = _build_channel_owen_kwargs(
                    player_cap=channel_player_cap
                )

                results = runner.run_experiment(
                    graph_scoring_method='owen',
                    model_scoring_method='owen',
                    keep_ratios=keep_ratios,
                    channel_strategy='layerwise',
                    output_dir=str(exp_dir / f'seed{seed}_{variant_name}'),
                    graph_scoring_kwargs=graph_owen_kwargs,
                    model_scoring_kwargs=model_owen_kwargs
                )

                for r in results:
                    r['experiment'] = 'exp1_baseline'
                    r['seed'] = seed
                    r['graph_variant'] = edge_strategy
                    r['model_variant'] = f'channel_owen_{channel_player_cap}'
                    r['display_method'] = variant_name

                all_results.extend(results)

        self._save_aggregated_results(all_results, exp_dir / 'all_results.json')
        self._analyze_and_plot(all_results, exp_dir)

        self.logger.info(f"\n✓ Experiment 1 complete! Results in {exp_dir}")


    
    def experiment_3_channel_strategies(self):
        """
        Experiment 3: Compare global vs layerwise channel pruning.
        
        Tests:
        - Strategy: global vs layerwise
        - Model sparsity: 0% to 70%
        - No graph pruning (focus on model)
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXPERIMENT 3: Channel Pruning Strategies")
        self.logger.info("=" * 70)
        
        exp_dir = self.base_output_dir / 'exp3_channel_strategies'
        exp_dir.mkdir(exist_ok=True)
        
        data = load_dataset('Cora')
        
        # Model-only pruning
        keep_ratios = [(1.0, r) for r in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]]
        
        all_results = []
        
        for seed in self.seeds:
            self.logger.info(f"\n--- Seed {seed} ---")
            model = self.train_base_model(seed)
            
            for strategy in ['global', 'layerwise']:
                self.logger.info(f"\n  Strategy: {strategy}")
                
                runner = JointPruningRunner(
                    model=model,
                    data=data,
                    graph_unit='node',
                    device=self.device,
                    seed=seed
                )
                
                results = runner.run_experiment(
                    graph_scoring_method='degree',
                    model_scoring_method='magnitude',
                    keep_ratios=keep_ratios,
                    channel_strategy=strategy,
                    output_dir=str(exp_dir / f'seed{seed}_{strategy}')
                )
                
                for r in results:
                    r['experiment'] = 'exp3_strategies'
                    r['seed'] = seed
                
                all_results.extend(results)
        
        self._save_aggregated_results(all_results, exp_dir / 'all_results.json')
        self._analyze_and_plot(all_results, exp_dir)
        
        self.logger.info(f"\n✓ Experiment 3 complete! Results in {exp_dir}")
    
    def experiment_4_joint_pruning_grid(self):
        """
        Experiment 4: 2D grid search over joint pruning (Owen scores only).

        Tests:
        - Graph sparsity (keep ratio): 1.0, 0.8, 0.6, 0.4
        - Model sparsity (keep ratio): 1.0, 0.8, 0.6, 0.4
        - Full 4×4 grid

        Scoring:
        - Graph unit (edge): Owen value with a fixed coalition strategy
        - Model unit (channel): Owen value on target layer channels

        Notes:
        - This version is adapted from the enhanced Owen setup in Experiment 1.
        - Only one Owen variant is used here to build the accuracy matrix.
        """
        self.logger.info("\n" + "=" * 70)
        self.logger.info("EXPERIMENT 4: Joint Pruning Grid Search (Owen Scores)")
        self.logger.info("=" * 70)

        exp_dir = self.base_output_dir / "exp4_joint_grid_owen"
        exp_dir.mkdir(exist_ok=True)

        data = load_dataset("Cora")

        # 4×4 grid
        graph_ratios = [1.0, 0.8, 0.6, 0.4]
        model_ratios = [1.0, 0.8, 0.6, 0.4]
        keep_ratios = create_keep_ratio_grid(graph_ratios, model_ratios)

        self.logger.info(f"  Grid: {len(graph_ratios)}×{len(model_ratios)} = {len(keep_ratios)} configs")

        # =========================================================
        # Fixed Owen variant for Experiment 4
        # You can change these two lines only if needed
        # =========================================================
        owen_variant_name = "owen_target_node_ch32"
        edge_strategy = "target_node"
        edge_kwargs = {}
        channel_player_cap = 32

        # Channel coalition config (same style as Experiment 1)
        channel_layer_channel_counts = [64, 64, 64, 64]
        channel_layer_names = ['gcn_layers.0', 'gcn_layers.1', 'gcn_layers.2', 'mlp_layers.0']

        # Owen hyperparams (same style as Experiment 1)
        max_edge_players = 100
        edge_num_samples_q = 5
        edge_num_samples_p = 5
        edge_K = 10

        channel_num_samples_q = 5
        channel_num_samples_p = 5

        all_results = []

        # -------------------------
        # Local helper functions
        # -------------------------
        def _restrict_coalitions_to_players(coalitions, players):
            """
            Restrict full coalitions to the given players only.
            Ensures the returned coalitions form a partition of `players`.
            """
            player_set = set(players)
            restricted = []

            for group in coalitions:
                sub_group = [p for p in group if p in player_set]
                if len(sub_group) > 0:
                    restricted.append(sub_group)

            flat_players = set()
            for group in restricted:
                flat_players.update(group)

            if flat_players != player_set:
                missing = player_set - flat_players
                extra = flat_players - player_set
                raise ValueError(
                    f"Restricted coalitions do not match players. "
                    f"missing={missing}, extra={extra}"
                )

            return restricted

        def _build_edge_owen_kwargs(data, edge_strategy, edge_kwargs):
            coalition_result = build_edge_coalitions(
                strategy=edge_strategy,
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                **edge_kwargs
            )

            players = list(coalition_result.player_to_group.keys())[:max_edge_players]

            restricted_coalitions = _restrict_coalitions_to_players(
                coalition_result.coalitions,
                players
            )

            return {
                "players": players,
                "coalitions": restricted_coalitions,
                "num_samples_q": edge_num_samples_q,
                "num_samples_p": edge_num_samples_p,
                "K": edge_K,
            }

        def _build_channel_owen_kwargs(player_cap):
            coalition_result = build_channel_coalitions(
                'layerwise',
                layer_channel_counts=channel_layer_channel_counts,
                layer_names=channel_layer_names
            )

            layer_0_players = coalition_result.coalitions[0][:player_cap]

            restricted_coalitions = _restrict_coalitions_to_players(
                coalition_result.coalitions,
                layer_0_players
            )

            return {
                "players": layer_0_players,
                "coalitions": restricted_coalitions,
                "target_layer": "gcn_layers.0",
                "num_samples_q": channel_num_samples_q,
                "num_samples_p": channel_num_samples_p,
            }

        # Build fixed Owen kwargs once per experiment
        graph_owen_kwargs = _build_edge_owen_kwargs(
            data=data,
            edge_strategy=edge_strategy,
            edge_kwargs=edge_kwargs
        )
        model_owen_kwargs = _build_channel_owen_kwargs(
            player_cap=channel_player_cap
        )

        for seed in self.seeds:
            self.logger.info(f"\n--- Seed {seed} ---")
            model = self.train_base_model(seed)

            runner = JointPruningRunner(
                model=model,
                data=data,
                graph_unit="edge",
                device=self.device,
                seed=seed,
            )

            results = runner.run_experiment(
                graph_scoring_method="owen",
                model_scoring_method="owen",
                keep_ratios=keep_ratios,
                channel_strategy="layerwise",
                output_dir=str(exp_dir / f"seed{seed}_{owen_variant_name}"),
                graph_scoring_kwargs=graph_owen_kwargs,
                model_scoring_kwargs=model_owen_kwargs,
            )

            for r in results:
                r["experiment"] = "exp4_joint_grid_owen"
                r["seed"] = seed
                r["graph_variant"] = edge_strategy
                r["model_variant"] = f"channel_owen_{channel_player_cap}"
                r["display_method"] = owen_variant_name

                # 记录 Owen 配置，方便后续画矩阵和写论文
                r["graph_owen_cfg"] = {
                    "edge_strategy": edge_strategy,
                    "edge_kwargs": edge_kwargs,
                    "max_edge_players": max_edge_players,
                    "num_samples_q": edge_num_samples_q,
                    "num_samples_p": edge_num_samples_p,
                    "K": edge_K,
                }
                r["model_owen_cfg"] = {
                    "target_layer": "gcn_layers.0",
                    "channel_player_cap": channel_player_cap,
                    "num_samples_q": channel_num_samples_q,
                    "num_samples_p": channel_num_samples_p,
                }

            all_results.extend(results)

        self._save_aggregated_results(all_results, exp_dir / "all_results.json")
        self._analyze_and_plot(all_results, exp_dir)

        self.logger.info(f"\n✓ Experiment 4 complete! Results in {exp_dir}")
    
    
        
    
    def _save_aggregated_results(self, results: list, output_path: Path):
        """Save aggregated results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"\n[Results] Saved aggregated results to {output_path}")
    
    def _analyze_and_plot(self, results: list, exp_dir: Path):
        """Analyze results and create summary."""
        if not results:
            return
        
        # Group by configuration
        configs = {}
        for r in results:
            key = (r['graph_keep_ratio'], r['model_keep_ratio'], 
                   r.get('channel_strategy', 'global'))
            if key not in configs:
                configs[key] = []
            configs[key].append(r)
        
        # Compute statistics
        summary = []
        for key, group in configs.items():
            graph_ratio, model_ratio, strategy = key
            
            test_accs = [r['test_acc'] for r in group]
            val_accs = [r['val_acc'] for r in group]
            
            summary.append({
                'graph_keep_ratio': graph_ratio,
                'model_keep_ratio': model_ratio,
                'channel_strategy': strategy,
                'num_runs': len(group),
                'test_acc_mean': np.mean(test_accs),
                'test_acc_std': np.std(test_accs),
                'test_acc_min': np.min(test_accs),
                'test_acc_max': np.max(test_accs),
                'val_acc_mean': np.mean(val_accs),
                'val_acc_std': np.std(val_accs),
            })
        
        # Save summary
        summary_path = exp_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create readable report
        report_path = exp_dir / 'report.txt'
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("Experiment Summary\n")
            f.write("=" * 70 + "\n\n")
            
            for s in sorted(summary, key=lambda x: (-x['graph_keep_ratio'], -x['model_keep_ratio'])):
                f.write(f"Keep Ratios: Graph={s['graph_keep_ratio']:.2f}, Model={s['model_keep_ratio']:.2f}\n")
                f.write(f"  Strategy: {s['channel_strategy']}\n")
                f.write(f"  Runs: {s['num_runs']}\n")
                f.write(f"  Test Acc: {s['test_acc_mean']:.4f} ± {s['test_acc_std']:.4f}\n")
                f.write(f"  Val Acc:  {s['val_acc_mean']:.4f} ± {s['val_acc_std']:.4f}\n")
                f.write("\n")
        
        self.logger.info(f"[Analysis] Summary saved to {summary_path}")
        self.logger.info(f"[Analysis] Report saved to {report_path}")



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Large-Scale Pruning Experiments')
    parser.add_argument('--mode', type=str, default='exp4',
                       choices=['exp1', 'exp2', 'exp3', 'exp4', 'exp5'],
                       help='Experiment mode')
    
    args = parser.parse_args()
    
    
    manager = LargeScaleExperiment()
    
    exp_map = {
        'exp1': manager.experiment_1_baseline_methods,
        'exp3': manager.experiment_3_channel_strategies,
        'exp4': manager.experiment_4_joint_pruning_grid,
    }
    
    exp_map[args.mode]()


if __name__ == '__main__':
    main()
