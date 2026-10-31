# Quick Start Guide

## Installation

1. **Clone/Download the project**
```bash
cd gcn_pruning_project/
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

Required packages:
- torch >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0

## Verify Setup

Run the test script to verify everything is working:

```bash
python main_test.py
```

Expected output:
```
============================================================
GCN Pruning Project - Setup Verification
============================================================
...
✓ All tests passed! Project setup is complete.
============================================================
```

## Run First Experiment

```bash
python main.py --exp_name first_test --seed 42
```

This will:
1. Create output directory: `outputs/first_test/`
2. Initialize logger
3. Set random seed to 42
4. Print configuration and environment info
5. Save config to `outputs/first_test/results/config.json`

## Output Files

After running, check:
```
outputs/first_test/
├── checkpoints/    # (empty for now)
├── scores/         # (empty for now)
├── results/
│   ├── config.json    # Full configuration
│   └── summary.json   # Experiment summary
└── logs/
    └── experiment.log # Detailed log file
```

## Configuration Examples

### Change Dataset
```bash
python main.py --dataset CiteSeer --exp_name citeseer_test
```

### Adjust Model Architecture
```bash
python main.py --hidden_dim 128 --num_layers 3 --dropout 0.3
```

### Configure Pruning
```bash
python main.py \
    --pruning_method magnitude \
    --pruning_ratio 0.7 \
    --exp_name high_prune
```

### Full Custom Run
```bash
python main.py \
    --exp_name custom_gcn \
    --dataset Cora \
    --seed 123 \
    --hidden_dim 256 \
    --num_layers 2 \
    --dropout 0.5 \
    --epochs 300 \
    --lr 0.005 \
    --weight_decay 1e-4 \
    --pruning_method magnitude \
    --pruning_ratio 0.6
```

## Understanding the Configuration

The config file (`config.py`) organizes settings into categories:

**Experiment Settings**
- `exp_name`: Name of your experiment
- `seed`: Random seed for reproducibility
- `deterministic`: Enable deterministic mode

**Model Settings**
- `model`: 'GCN' (more models coming)
- `hidden_dim`: Hidden layer dimension
- `num_layers`: Number of GCN layers
- `dropout`: Dropout rate

**Training Settings**
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `weight_decay`: L2 regularization

**Pruning Settings**
- `pruning_method`: 'magnitude', 'random', 'gradient'
- `pruning_ratio`: Fraction of weights to prune (0.0-1.0)
- `pruning_schedule`: 'oneshot' or 'iterative'

## Next Steps

After verifying the setup:

1. **Implement Data Loading** - Add dataset loaders in `data/`
2. **Define GCN Model** - Implement GCN architecture in `models/`
3. **Create Training Loop** - Add training logic in `train/`
4. **Add Pruning Methods** - Implement pruning in `pruning/`

See `PROJECT_OVERVIEW.md` for detailed implementation plan.

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError`:
```bash
# Make sure you're in the project directory
cd gcn_pruning_project/

# Try running from project root
python main.py
```

### CUDA Issues
If CUDA is not available but you want to use GPU:
```bash
# Force CPU mode
python main.py --device cpu
```

### Permission Errors
If you can't create output directories:
```bash
# Check permissions
ls -la outputs/

# Or run with appropriate permissions
sudo python main.py  # (not recommended)
```

## Tips

1. **Always use a unique exp_name** to avoid overwriting results
2. **Set seed** for reproducibility of experiments
3. **Use deterministic mode** for exact reproducibility (slower)
4. **Check logs** in `outputs/{exp_name}/logs/` for debugging
5. **Review config.json** to verify experiment settings

## Help

View all available options:
```bash
python main.py --help
```

For detailed documentation, see:
- `README.md` - Full documentation
- `PROJECT_OVERVIEW.md` - System overview
- Code docstrings in each module
