# GCN Training and Pruning Project

A reproducible framework for Graph Convolutional Network (GCN) training and pruning experiments.

## Project Structure

```
gcn_pruning_project/
├── models/          # Model architectures (GCN definitions)
├── data/            # Data loading and preprocessing
├── train/           # Training loops and optimization
├── pruning/         # Pruning algorithms and methods
├── scorers/         # Scoring functions for pruning
├── experiments/     # Experiment runners and workflows
├── utils/           # Utility functions
│   ├── seed.py      # Random seed management
│   ├── io.py        # Checkpoint I/O
│   └── logger.py    # Logging utilities
├── outputs/         # Experiment outputs
│   └── {exp_name}/
│       ├── checkpoints/  # Model checkpoints
│       ├── scores/       # Pruning scores
│       ├── results/      # Result files
│       └── logs/         # Log files
├── config.py        # Configuration management
├── main.py          # Main entry point
└── requirements.txt # Python dependencies
```

## Features

### Reproducibility
- **Seed Control**: Fixed random seeds for numpy, torch, and CUDA
- **Deterministic Mode**: Configurable cudnn deterministic behavior
- **Checkpoint System**: Save/load model states with metadata
- **Git Integration**: Optional git hash tracking in checkpoints

### Configuration
- Centralized configuration management
- Command-line argument parsing
- Organized output directory structure
- JSON config export for reproducibility

### Logging
- Dual logging to console and file
- Environment information tracking
- Metrics logging with CSV export
- Structured experiment tracking

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run a basic experiment:

```bash
python main.py --exp_name test_exp --seed 42
```

With custom configuration:

```bash
python main.py \
    --exp_name my_experiment \
    --dataset Cora \
    --hidden_dim 128 \
    --epochs 300 \
    --pruning_ratio 0.7 \
    --seed 123
```

## Configuration Options

### Experiment Settings
- `--exp_name`: Experiment name (default: 'default_exp')
- `--seed`: Random seed (default: 42)
- `--deterministic`: Enable deterministic mode

### Device Settings
- `--device`: Device to use ('cuda' or 'cpu')
- `--gpu_id`: GPU ID for CUDA

### Data Settings
- `--dataset`: Dataset name ('Cora', 'CiteSeer', 'PubMed')

### Model Settings
- `--model`: Model architecture (default: 'GCN')
- `--hidden_dim`: Hidden dimension (default: 64)
- `--num_layers`: Number of GCN layers (default: 2)
- `--dropout`: Dropout rate (default: 0.5)

### Training Settings
- `--epochs`: Number of training epochs (default: 200)
- `--lr`: Learning rate (default: 0.01)
- `--weight_decay`: Weight decay (default: 5e-4)

### Pruning Settings
- `--pruning_method`: Pruning method ('magnitude', 'random', 'gradient')
- `--pruning_ratio`: Pruning ratio 0-1 (default: 0.5)

## Output Structure

Each experiment creates a structured output directory:

```
outputs/{exp_name}/
├── checkpoints/
│   ├── model_best.pth
│   ├── model_best_config.json
│   └── model_epoch_*.pth
├── scores/
│   └── pruning_scores.npy
├── results/
│   ├── config.json
│   ├── summary.json
│   └── final_results.json
└── logs/
    ├── experiment.log
    └── metrics.csv
```

## Checkpoint Format

Checkpoints include:
- `model_state_dict`: Model parameters
- `config`: Configuration dictionary
- `metrics`: Performance metrics
- `epoch`: Training epoch
- `optimizer_state_dict`: Optimizer state (optional)
- `git_hash`: Git commit hash (optional)

## Utilities

### Seed Management
```python
from utils import set_seed, get_seed_state, set_seed_state

# Set seed for reproducibility
set_seed(42, deterministic=True)

# Save/restore RNG state
state = get_seed_state()
set_seed_state(state)
```

### Checkpoint I/O
```python
from utils import save_checkpoint, load_checkpoint

# Save checkpoint
save_checkpoint(
    'path/to/checkpoint.pth',
    model_state=model.state_dict(),
    config=config.to_dict(),
    metrics={'val_acc': 0.85, 'test_acc': 0.83},
    epoch=100
)

# Load checkpoint
checkpoint = load_checkpoint('path/to/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Logging
```python
from utils import setup_logger, MetricsLogger

# Setup logger
logger = setup_logger('my_logger', log_file='experiment.log')

# Track metrics
metrics_logger = MetricsLogger('metrics.csv')
metrics_logger.log(epoch=1, train_loss=0.5, val_acc=0.8)
best_acc, best_epoch = metrics_logger.get_best('val_acc')
```

## Development Status

- [x] Project structure
- [x] Configuration system
- [x] Utility functions (seed, I/O, logging)
- [ ] Data loading
- [ ] Model implementations
- [ ] Training loops
- [ ] Pruning algorithms
- [ ] Scoring functions
- [ ] Experiment workflows

## TODO

1. Implement data loading for graph datasets
2. Define GCN model architectures
3. Create training loops with early stopping
4. Implement pruning methods (magnitude, gradient-based)
5. Add scoring functions for pruning importance
6. Build experiment runners for training + pruning workflows

## License

MIT License
