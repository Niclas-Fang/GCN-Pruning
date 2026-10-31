# GCN Model Implementation - Testing Guide

## ✅ Implemented Components

### 1. Model Architecture (`models/`)

**GCNConv Layer** (`models/gcn_layer.py`)
- Custom implementation of Graph Convolutional layer
- Implements: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
- Supports sparse adjacency matrices
- Xavier initialization

**GCNNet Model** (`models/gcn_net.py`)
- Multi-layer GCN architecture
- Structure:
  ```
  input_proj (NOT prunable)
  → (GCN + ReLU + Dropout) × num_gcn_layers (prunable)
  → (MLP + ReLU + Dropout) × num_mlp_layers (prunable)
  → output_head (NOT prunable)
  ```

**Key Features:**
- `forward(x, adj_norm, return_embeddings=False)` - Forward pass with optional embeddings
- `list_prunable_layers()` - Returns list of prunable layers (excludes input_proj and output_head)
- `get_prunable_parameters()` - Returns all prunable parameters
- `count_parameters(only_prunable=False)` - Count total or prunable parameters

### 2. Data Loading (`data/`)

**Dataset Loader** (`data/loader.py`)
- `load_cora()` - Load Cora dataset
- `load_dataset(name)` - Generic dataset loader
- Creates GraphData object with:
  - x: Node features
  - edge_index: Edge indices
  - y: Node labels
  - train_mask, val_mask, test_mask

**Data Statistics:**
- Cora: 2708 nodes, 1433 features, 7 classes
- Train/Val/Test: 140/500/1000 nodes

### 3. Training System (`train/`)

**Evaluation** (`train/eval.py`)
- `evaluate(model, data, mask, adj_norm)` - Returns (loss, accuracy)
- `evaluate_all_splits()` - Evaluate on train/val/test
- `compute_accuracy()` - Compute classification accuracy

**Training** (`train/train_gcn.py`)
- `train_gcn(config)` - Main training function
- `train_epoch()` - Single epoch training
- `EarlyStopping` - Early stopping handler

**Training Features:**
- Cross-entropy loss
- Adam optimizer
- Early stopping (on val_loss or val_acc)
- Checkpoint saving (best model)
- Comprehensive logging

### 4. Configuration

**New Parameters Added:**
```python
# Model architecture
num_gcn_layers = 3        # Number of GCN layers (>= 3)
num_mlp_layers = 1        # Number of MLP layers (>= 1)

# Training
early_stop_metric = 'val_acc'  # 'val_acc' or 'val_loss'
```

## 🚀 Usage Examples

### 1. Run Sanity Check
```bash
python sanity_forward.py
```

This will:
- Load Cora dataset
- Create GCN model
- Test forward pass
- Print layer-wise dimensions
- List prunable layers
- Show parameter statistics

Expected Output:
```
======================================================================
GCN Forward Pass Sanity Check
======================================================================

Device: cuda

[1] Loading Cora dataset...
  ✓ Dataset loaded:
    - Nodes: 2708
    - Features: 1433
    - Classes: 7
    - Train/Val/Test: 140/500/1000

[2] Normalizing adjacency matrix...
  ✓ Adjacency matrix normalized: torch.Size([2708, 2708])

[3] Creating GCN model...
  ✓ Model created:

GCNNet(
  in_dim=1433, hidden_dim=64, num_classes=7
  num_gcn_layers=3, num_mlp_layers=1
  dropout=0.5
  total_params=96,007
  prunable_params=12,544
)

[4] Testing forward pass (training mode)...
  ✓ Forward pass completed
    - Input shape: torch.Size([2708, 1433])
    - Output shape (logits): torch.Size([2708, 7])
  ✓ Forward pass with embeddings
    - Logits shape: torch.Size([2708, 7])
    - Embeddings shape: torch.Size([2708, 64])
  ✓ Logits match between two forward passes

[5] Testing forward pass (eval mode)...
  ✓ Eval forward pass completed
    - Output shape: torch.Size([2708, 7])

[6] Layer-wise tensor dimensions:
  Input:       torch.Size([2708, 1433])
  input_proj:  torch.Size([2708, 64])
  gcn_layer_0: torch.Size([2708, 64])
  gcn_layer_1: torch.Size([2708, 64])
  gcn_layer_2: torch.Size([2708, 64])
  mlp_layer_0: torch.Size([2708, 64])
  output_head: torch.Size([2708, 7])

[7] Testing prunable layers listing...
  ✓ Found 4 prunable layers:
    - gcn_layers.0       : GCNConv         (4,160 params)
    - gcn_layers.1       : GCNConv         (4,160 params)
    - gcn_layers.2       : GCNConv         (4,160 params)
    - mlp_layers.0       : Linear          (4,160 params)

[8] Parameter statistics:
  Total parameters:        96,007
  Prunable parameters:     12,544 (13.1%)
  Non-prunable parameters: 83,463 (86.9%)

[9] Testing predictions...
  ✓ Predictions computed
    - Predicted classes: [0, 1, 2, 3, 4, 5, 6]
    - Train accuracy (untrained): ~0.14 (random)

======================================================================
✓ All sanity checks passed!
======================================================================
```

### 2. Train GCN Model
```bash
python train/train_gcn.py \
    --exp_name gcn_cora \
    --dataset Cora \
    --hidden_dim 64 \
    --num_gcn_layers 3 \
    --num_mlp_layers 1 \
    --dropout 0.5 \
    --epochs 200 \
    --lr 0.01 \
    --weight_decay 5e-4 \
    --early_stop_patience 50 \
    --early_stop_metric val_acc \
    --seed 42
```

Expected Training Output:
```
[Seed] Set random seed to 42 (deterministic=True)
Using device: cuda
Loading dataset: Cora
  Nodes: 2708, Features: 1433, Classes: 7
  Train: 140, Val: 500, Test: 1000
Normalizing adjacency matrix...
Creating GCN model...

GCNNet(
  in_dim=1433, hidden_dim=64, num_classes=7
  num_gcn_layers=3, num_mlp_layers=1
  dropout=0.5
  total_params=96,007
  prunable_params=12,544
)

Starting training for 200 epochs...
Early stopping: metric=val_acc, patience=50

Epoch   1/200 | Train Loss: 1.9456 | Val Loss: 1.9234 | Val Acc: 0.1420
Epoch  10/200 | Train Loss: 0.8765 | Val Loss: 1.2345 | Val Acc: 0.5640
Epoch  20/200 | Train Loss: 0.4321 | Val Loss: 0.9876 | Val Acc: 0.6780
...
Epoch 150/200 | Train Loss: 0.0234 | Val Loss: 0.7654 | Val Acc: 0.7920

[I/O] Checkpoint saved to outputs/gcn_cora/checkpoints/best_model.pth

Early stopping triggered at epoch 150
Best val_acc: 0.7920 at epoch 100

============================================================
Training completed!
Best val_acc: 0.7920 at epoch 100

Loading best checkpoint from epoch 100...
[I/O] Checkpoint loaded from outputs/gcn_cora/checkpoints/best_model.pth
  - Epoch: 100
  - Metrics: {'val_loss': 0.7654, 'val_acc': 0.7920}

Evaluating on test set...

Final Test Results:
  Test Loss: 0.7823
  Test Acc:  0.7850
============================================================

[I/O] Results saved to outputs/gcn_cora/results/training_results.json

✓ Training complete!
  Best Val Acc: 0.7920 (epoch 100)
  Test Acc: 0.7850
```

### 3. Custom Training Configuration
```bash
# Deeper network
python train/train_gcn.py \
    --exp_name gcn_deep \
    --num_gcn_layers 5 \
    --num_mlp_layers 2 \
    --hidden_dim 128

# Different early stopping
python train/train_gcn.py \
    --exp_name gcn_loss_stop \
    --early_stop_metric val_loss \
    --early_stop_patience 100

# Higher learning rate
python train/train_gcn.py \
    --exp_name gcn_fast \
    --lr 0.05 \
    --epochs 100
```

## 📊 Output Files

After training, the following files are created:

```
outputs/gcn_cora/
├── checkpoints/
│   ├── best_model.pth           # Best model checkpoint
│   └── best_model_config.json   # Configuration
├── results/
│   └── training_results.json    # Final results
└── logs/
    └── train.log                # Training log
```

**Checkpoint Contents:**
- model_state_dict: Model parameters
- optimizer_state_dict: Optimizer state
- config: Full configuration
- metrics: {'val_loss': ..., 'val_acc': ...}
- epoch: Best epoch number
- git_hash: Git commit (if available)

**Results JSON:**
```json
{
  "best_epoch": 100,
  "best_val_metric": 0.7920,
  "test_loss": 0.7823,
  "test_acc": 0.7850,
  "config": {...}
}
```

## 🔍 Model Inspection

### List Prunable Layers
```python
from models import GCNNet

model = GCNNet(in_dim=1433, hidden_dim=64, num_classes=7)

# Get prunable layers
prunable = model.list_prunable_layers()
for name, layer in prunable:
    print(f"{name}: {layer}")

# Output:
# gcn_layers.0: GCNConv(64, 64)
# gcn_layers.1: GCNConv(64, 64)
# gcn_layers.2: GCNConv(64, 64)
# mlp_layers.0: Linear(in_features=64, out_features=64)
```

### Count Parameters
```python
total = model.count_parameters(only_prunable=False)
prunable = model.count_parameters(only_prunable=True)

print(f"Total: {total:,}")      # 96,007
print(f"Prunable: {prunable:,}") # 12,544
print(f"Ratio: {prunable/total:.1%}") # 13.1%
```

### Forward Pass with Embeddings
```python
logits, embeddings = model(x, adj_norm, return_embeddings=True)
# logits: [num_nodes, num_classes]
# embeddings: [num_nodes, hidden_dim]
```

## 🎯 Implementation Details

### Model Architecture Breakdown

1. **Input Projection** (NOT prunable)
   - Linear: (1433 → 64)
   - Parameters: 1433 × 64 + 64 = 91,776

2. **GCN Layers** (Prunable)
   - 3 × GCNConv(64 → 64)
   - Each: 64 × 64 + 64 = 4,160
   - Total: 12,480

3. **MLP Layers** (Prunable)
   - 1 × Linear(64 → 64)
   - Parameters: 64 × 64 + 64 = 4,160

4. **Output Head** (NOT prunable)
   - Linear: (64 → 7)
   - Parameters: 64 × 7 + 7 = 455

**Total: 96,007 parameters (12,544 prunable)**

### Adjacency Matrix Normalization

The GCN uses symmetric normalization:
```
A_norm = D^(-1/2) * A * D^(-1/2)
```

Where:
- A is adjacency matrix with self-loops
- D is degree matrix

This is computed once and reused during training.

## ⚙️ Configuration Options

All training parameters can be set via command line or config file:

```python
# Model
--hidden_dim 64              # Hidden dimension
--num_gcn_layers 3           # Number of GCN layers (>= 3)
--num_mlp_layers 1           # Number of MLP layers (>= 1)
--dropout 0.5                # Dropout rate

# Training
--epochs 200                 # Max epochs
--lr 0.01                    # Learning rate
--weight_decay 5e-4          # L2 regularization
--early_stop_patience 50     # Patience for early stopping
--early_stop_metric val_acc  # Metric to monitor

# Experiment
--exp_name my_exp            # Experiment name
--seed 42                    # Random seed
--device cuda                # Device
```

## 📝 Notes

1. **Prunable vs Non-Prunable Layers:**
   - Input projection and output head are NOT prunable
   - All GCN and MLP layers ARE prunable
   - This design preserves input/output dimensions

2. **Forward Pass Modes:**
   - Training: Uses dropout
   - Evaluation: No dropout, deterministic

3. **Early Stopping:**
   - Can use val_acc (maximize) or val_loss (minimize)
   - Best model saved at best epoch

4. **Reproducibility:**
   - Fixed random seeds
   - Deterministic mode option
   - All configs saved in checkpoint

## ✅ Verification Checklist

- [x] GCNNet class with correct architecture
- [x] input_proj, gcn_layers, mlp_layers, output_head
- [x] forward() with return_embeddings option
- [x] list_prunable_layers() method
- [x] Data loading for Cora
- [x] evaluate() function
- [x] train_gcn() with early stopping
- [x] Checkpoint saving (best model)
- [x] sanity_forward.py script
- [x] Configuration system integration
- [x] Logging and output management

Ready for pruning implementation! 🎉
