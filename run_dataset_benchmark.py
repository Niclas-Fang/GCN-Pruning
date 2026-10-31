"""
Benchmark GCN on Planetoid datasets (Cora, CiteSeer, PubMed).

The script:
  1. trains a node-classification model with the project's training loop;
  2. optionally runs the project's joint graph+channel pruning pipeline;
  3. writes a JSON summary.

The Graph Transformer architecture lives on the `feature/graph-transformer`
branch; this baseline branch supports GCN only.

Examples:
    python run_dataset_benchmark.py --dataset CiteSeer --model GCN \
        --exp_name citeseer_gcn --prune cheap
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from data import load_dataset, get_dataset_info
from train import train_gcn
from utils import set_seed, save_results

SUPPORTED_MODELS = ('GCN',)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Dataset benchmark for GCN')

    # Dataset / model
    parser.add_argument('--dataset', type=str, default='Cora',
                        choices=['Cora', 'CiteSeer', 'PubMed'])
    parser.add_argument('--model', type=str, default='GCN', choices=SUPPORTED_MODELS,
                        help='Architecture to train')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Directory containing the PyG dataset cache')

    # Architecture
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_gcn_layers', type=int, default=3)
    parser.add_argument('--num_mlp_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--early_stop_patience', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    # Output
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs/benchmark')
    parser.add_argument('--save_checkpoint', action='store_true', default=True)

    # Pruning
    parser.add_argument('--prune', type=str, default='none', choices=['none', 'cheap', 'shapley'],
                        help='none: train only; cheap: degree/magnitude; shapley: graph Shapley')
    parser.add_argument('--graph_unit', type=str, default='node', choices=['node', 'edge'])
    parser.add_argument('--graph_method', type=str, default=None,
                        help='degree/random for node, betweenness/feature_l2/random for edge, '
                             'or owen/approximation for Shapley')
    parser.add_argument('--model_method', type=str, default='magnitude',
                        choices=['magnitude', 'random', 'activation_l2', 'grad', 'taylor'])
    parser.add_argument('--channel_strategy', type=str, default='global',
                        choices=['global', 'layerwise'])
    parser.add_argument('--keep_ratios', type=str, default='1.0,0.9,0.8,0.7',
                        help='Comma-separated keep ratios (same for graph and model)')
    parser.add_argument('--num_players', type=int, default=16,
                        help='Number of players used by Shapley scoring')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of coalition samples per Shapley player')
    parser.add_argument('--num_samples_q', type=int, default=10)
    parser.add_argument('--num_samples_p', type=int, default=10)
    parser.add_argument('--shapley_K', type=int, default=5)
    parser.add_argument('--target_layer', type=str, default=None,
                        help='Optional single layer for channel Shapley')
    parser.add_argument('--seed_scores', type=int, default=None,
                        help='Optional separate seed for score computation')

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    config = Config()

    exp_name = args.exp_name
    if exp_name is None:
        exp_name = f"{args.dataset.lower()}_{args.model.lower()}_seed{args.seed}"
    config.exp_name = exp_name

    config.dataset = args.dataset
    config.data_dir = args.data_dir
    config.model = args.model
    config.hidden_dim = args.hidden_dim
    config.num_gcn_layers = args.num_gcn_layers
    config.num_mlp_layers = args.num_mlp_layers
    config.dropout = args.dropout

    config.epochs = args.epochs
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    config.early_stop_patience = args.early_stop_patience
    config.seed = args.seed
    config.device = args.device

    config.output_dir = str(Path(args.output_dir))
    config.save_checkpoint = args.save_checkpoint
    config.deterministic = True

    return config


def parse_keep_ratios(raw: str) -> List[Tuple[float, float]]:
    ratios = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        ratio = float(part)
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"Keep ratio must be in (0, 1], got {ratio}")
        ratios.append((ratio, ratio))
    if not ratios:
        raise ValueError('At least one keep ratio is required')
    return ratios


def default_graph_method(graph_unit: str) -> str:
    return 'degree' if graph_unit == 'node' else 'betweenness'


def run_pruning(
    args: argparse.Namespace,
    config: Config,
    model: torch.nn.Module,
    device: torch.device,
) -> List[Dict]:
    from run_joint_pruning import JointPruningRunner

    data = load_dataset(config.dataset, config.data_dir)
    keep_ratios = parse_keep_ratios(args.keep_ratios)

    graph_method = args.graph_method or default_graph_method(args.graph_unit)
    if args.prune == 'cheap' and graph_method in ('owen', 'approximation'):
        raise ValueError('Use --prune shapley for owen/approximation graph scoring')
    if args.prune == 'shapley' and graph_method not in ('owen', 'approximation'):
        graph_method = 'owen'

    runner = JointPruningRunner(
        model=model,
        data=data,
        graph_unit=args.graph_unit,
        device=device,
        seed=args.seed,
    )

    graph_kwargs = {}
    if args.prune == 'shapley':
        graph_kwargs = {
            'num_players': args.num_players,
            'num_samples': args.num_samples,
            'num_samples_q': args.num_samples_q,
            'num_samples_p': args.num_samples_p,
            'K': args.shapley_K,
        }

    model_kwargs = {}
    if args.target_layer is not None:
        model_kwargs['target_layer'] = args.target_layer

    out_dir = Path(config.output_dir) / config.exp_name / 'pruning'
    results = runner.run_experiment(
        graph_scoring_method=graph_method,
        model_scoring_method=args.model_method,
        keep_ratios=keep_ratios,
        channel_strategy=args.channel_strategy,
        output_dir=str(out_dir),
        graph_scoring_kwargs=graph_kwargs,
        model_scoring_kwargs=model_kwargs,
    )
    return results


def main():
    args = parse_args()
    config = build_config(args)

    print("=" * 70)
    print("Dataset Benchmark")
    print("=" * 70)
    print(f"Dataset:     {args.dataset}")
    print(f"Model:       {args.model}")
    print(f"Device:      {args.device} (cuda available: {torch.cuda.is_available()})")
    print(f"Exp name:    {config.exp_name}")
    print(f"Output dir:  {config.output_dir}")
    print(f"Prune mode:  {args.prune}")
    print("=" * 70)

    info = get_dataset_info(args.dataset)
    print(f"Dataset info: {info}")

    set_seed(config.seed, config.deterministic)

    start = time.time()
    model, train_results = train_gcn(config)
    train_seconds = time.time() - start

    device = torch.device(
        config.device if torch.cuda.is_available() and config.device == 'cuda' else 'cpu'
    )

    pruning_results = []
    if args.prune != 'none':
        start = time.time()
        pruning_results = run_pruning(args, config, model, device)
        prune_seconds = time.time() - start
    else:
        prune_seconds = 0.0

    summary = {
        'dataset': args.dataset,
        'model': args.model,
        'seed': args.seed,
        'hidden_dim': args.hidden_dim,
        'num_gcn_layers': args.num_gcn_layers,
        'num_mlp_layers': args.num_mlp_layers,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'train_seconds': train_seconds,
        'prune_seconds': prune_seconds,
        'training': train_results,
        'pruning_mode': args.prune,
        'graph_unit': args.graph_unit if args.prune != 'none' else None,
        'graph_method': (
            (args.graph_method or default_graph_method(args.graph_unit))
            if args.prune != 'none' else None
        ),
        'model_method': args.model_method if args.prune != 'none' else None,
        'pruning': pruning_results,
    }

    out_dir = Path(config.output_dir) / config.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'benchmark_summary.json'
    save_results(str(summary_path), summary)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Best epoch:  {train_results['best_epoch']}")
    print(f"Val metric:  {train_results['best_val_metric']:.4f}")
    print(f"Test acc:    {train_results['test_acc']:.4f}")
    if pruning_results:
        print("-" * 70)
        print(f"{'graph_keep':>10s} {'model_keep':>10s} {'graph_sparsity':>14s} "
              f"{'model_sparsity':>14s} {'val_acc':>8s} {'test_acc':>9s}")
        for r in pruning_results:
            print(f"{r['graph_keep_ratio']:>10.2f} {r['model_keep_ratio']:>10.2f} "
                  f"{r['graph_sparsity']:>14.2%} {r['model_sparsity']:>14.2%} "
                  f"{r['val_acc']:>8.4f} {r['test_acc']:>9.4f}")
    print("=" * 70)
    print(f"Summary saved to: {summary_path}")

    return summary


if __name__ == '__main__':
    main()
