# Joint Pruning Runner Documentation

## Overview

The Joint Pruning Runner enables systematic experiments combining graph pruning (nodes/edges) and model pruning (channels) with various scoring methods and keep ratios.

## 🎯 Core Concepts

### Pruning Units

**Graph Units:**
- `node`: Prune nodes using `apply_node_mask`
  - Semantics: Pruned nodes keep self-loops but don't aggregate neighbors
- `edge`: Prune edges using `apply_edge_mask`
  - Semantics: Remove edges from graph structure

**Model Units:**
- `channel`: Prune channels from all prunable layers
  - Applies to GCN layers and MLP layers
  - Excludes input projection and output head

### Scoring Methods

**Graph Scoring:**
- `degree`: Node degree (number of neighbors)
- `random`: Random baseline
- `betweenness`: Edge betweenness (proxy)
- `owen`: Owen-Shapley values
- `approximation`: Monte Carlo Shapley approximation

**Model Scoring:**
- `magnitude`: L2 norm of weights per channel
- `random`: Random baseline
- `owen`: Owen-Shapley values (per layer)
- `approximation`: Monte Carlo Shapley approximation (per layer)

### Channel Pruning Strategies

**Global Strategy:**
- Concatenate scores from all layers
- Rank all channels globally
- Keep top-k% across all layers
- May remove entire layers if all channels pruned

**Layerwise Strategy:**
- Keep top-k% within each layer independently
- Preserves layer structure
- More balanced pruning across layers

## 📦 Usage

### Basic Setup

```python
from models import GCNNet
from data import load_dataset
from experiments import JointPruningRunner

# Load data and model
data = load_dataset('Cora')
model = GCNNet(...)

# Create runner
runner = JointPruningRunner(
    model=model,
    data=data,
    graph_unit='node',      # 'node' or 'edge'
    device=device,
    seed=42
)
```

### Simple Experiment

```python
# Define keep ratios: (graph_ratio, model_ratio)
keep_ratios = [
    (1.0, 1.0),  # No pruning (baseline)
    (0.9, 0.9),  # 10% pruning
    (0.8, 0.8),  # 20% pruning
    (0.7, 0.7),  # 30% pruning
]

# Run experiment
results = runner.run_experiment(
    graph_scoring_method='degree',
    model_scoring_method='magnitude',
    keep_ratios=keep_ratios,
    channel_strategy='global',
    output_dir='outputs/my_experiment'
)
```

### Grid Search

```python
from experiments import create_keep_ratio_grid

# Create 2D grid
graph_ratios = [1.0, 0.9, 0.8, 0.7, 0.6]
model_ratios = [1.0, 0.9, 0.8, 0.7, 0.6]
keep_ratios = create_keep_ratio_grid(graph_ratios, model_ratios)
# Result: 5 × 5 = 25 configurations

results = runner.run_experiment(
    graph_scoring_method='degree',
    model_scoring_method='magnitude',
    keep_ratios=keep_ratios,
    output_dir='outputs/grid_search'
)
```

### Custom Configurations

```python
# Asymmetric pruning
keep_ratios = [
    (0.9, 1.0),  # Prune only graph
    (1.0, 0.9),  # Prune only model
    (0.8, 0.95), # More graph pruning
    (0.95, 0.8), # More model pruning
]

results = runner.run_experiment(...)
```

### Shapley-based Pruning

```python
# Use Shapley values for scoring
results = runner.run_experiment(
    graph_scoring_method='approximation',  # Shapley approximation
    model_scoring_method='approximation',
    keep_ratios=[(0.9, 0.9), (0.8, 0.8)],
    graph_scoring_kwargs={'num_samples': 50, 'num_players': 100},
    model_scoring_kwargs={'num_samples': 20, 'num_players': 16}
)
```

## 📊 Output Format

### JSON Output

File: `outputs/{exp_name}/results.json`

```json
[
  {
    "graph_unit": "node",
    "graph_scoring_method": "degree",
    "model_scoring_method": "magnitude",
    "channel_strategy": "global",
    "graph_keep_ratio": 0.9,
    "model_keep_ratio": 0.9,
    "graph_sparsity": 0.1,
    "model_sparsity": 0.102,
    "val_loss": 0.6543,
    "val_acc": 0.7820,
    "test_loss": 0.6789,
    "test_acc": 0.7750,
    "seed": 42,
    "model_hidden_dim": 64,
    "model_num_gcn_layers": 3,
    "model_num_mlp_layers": 1,
    "model_dropout": 0.5
  },
  ...
]
```

### CSV Output

File: `outputs/{exp_name}/results.csv`

```csv
graph_unit,graph_scoring_method,model_scoring_method,channel_strategy,graph_keep_ratio,model_keep_ratio,graph_sparsity,model_sparsity,val_loss,val_acc,test_loss,test_acc,seed,model_hidden_dim,model_num_gcn_layers,model_num_mlp_layers,model_dropout
node,degree,magnitude,global,1.0,1.0,0.0,0.0,0.5234,0.8120,0.5432,0.8050,42,64,3,1,0.5
node,degree,magnitude,global,0.9,0.9,0.1,0.102,0.6543,0.7820,0.6789,0.7750,42,64,3,1,0.5
...
```

### Scores Cache

Scores are cached to avoid recomputation:
- `outputs/{exp_name}/scores_node_degree.pkl`
- `outputs/{exp_name}/scores_channel_magnitude.pkl`

## 🔧 JointPruningRunner API

### Initialization

```python
runner = JointPruningRunner(
    model: nn.Module,
    data: GraphData,
    graph_unit: str = 'node',    # 'node' or 'edge'
    model_unit: str = 'channel',
    device: torch.device = None,
    seed: int = 42
)
```

### Methods

**compute_graph_scores**
```python
scores = runner.compute_graph_scores(
    method: str,                    # Scoring method
    save_path: str = None,          # Path to save/load
    **kwargs                        # Method-specific args
) -> torch.Tensor
```

**compute_model_scores**
```python
scores = runner.compute_model_scores(
    method: str,
    save_path: str = None,
    **kwargs
) -> Dict[str, torch.Tensor]       # {layer_name: scores}
```

**create_graph_mask**
```python
mask = runner.create_graph_mask(
    scores: torch.Tensor,
    keep_ratio: float               # 0.0 to 1.0
) -> torch.Tensor                   # Boolean mask
```

**create_channel_masks**
```python
masks = runner.create_channel_masks(
    scores: Dict[str, torch.Tensor],
    keep_ratio: float,
    strategy: str = 'global'        # 'global' or 'layerwise'
) -> Dict[str, torch.Tensor]        # {layer_name: mask}
```

**evaluate_joint_pruning**
```python
metrics = runner.evaluate_joint_pruning(
    graph_mask: torch.Tensor,
    channel_masks: Dict[str, torch.Tensor]
) -> Dict[str, float]
# Returns: {val_loss, val_acc, test_loss, test_acc, 
#           graph_sparsity, model_sparsity}
```

**run_experiment**
```python
results = runner.run_experiment(
    graph_scoring_method: str,
    model_scoring_method: str,
    keep_ratios: List[Tuple[float, float]],
    channel_strategy: str = 'global',
    output_dir: str = 'outputs/joint_pruning',
    **kwargs
) -> List[Dict[str, Any]]
```

## 📈 Example Workflows

### Workflow 1: Simple Pruning

```python
# 1. Create runner
runner = JointPruningRunner(model, data, graph_unit='node')

# 2. Run with simple methods
results = runner.run_experiment(
    graph_scoring_method='degree',
    model_scoring_method='magnitude',
    keep_ratios=[(1.0, 1.0), (0.8, 0.8), (0.6, 0.6)],
    output_dir='outputs/simple'
)

# 3. Analyze
for r in results:
    print(f"Sparsity: {r['graph_sparsity']:.1%}/{r['model_sparsity']:.1%}")
    print(f"Test Acc: {r['test_acc']:.4f}")
```

### Workflow 2: Method Comparison

```python
methods = [
    ('random', 'random'),
    ('degree', 'magnitude'),
]

for graph_method, model_method in methods:
    runner = JointPruningRunner(model, data)
    
    results = runner.run_experiment(
        graph_scoring_method=graph_method,
        model_scoring_method=model_method,
        keep_ratios=[(0.8, 0.8)],
        output_dir=f'outputs/{graph_method}_{model_method}'
    )
    
    print(f"{graph_method}/{model_method}: {results[0]['test_acc']:.4f}")
```

### Workflow 3: Strategy Comparison

```python
for strategy in ['global', 'layerwise']:
    runner = JointPruningRunner(model, data)
    
    results = runner.run_experiment(
        graph_scoring_method='degree',
        model_scoring_method='magnitude',
        keep_ratios=[(1.0, 0.7)],
        channel_strategy=strategy,
        output_dir=f'outputs/strategy_{strategy}'
    )
    
    print(f"{strategy}: {results[0]['test_acc']:.4f}")
```

## 🎯 Design Decisions

### Why Separate Graph and Model Units?

Allows independent control:
- Prune only graph: `(0.8, 1.0)`
- Prune only model: `(1.0, 0.8)`
- Prune both: `(0.8, 0.8)`

### Why Two Channel Strategies?

**Global**: Best for finding truly unimportant channels
**Layerwise**: More robust, preserves architecture

### Why Cache Scores?

- Expensive to compute (especially Shapley)
- Reuse across keep ratio experiments
- Save time on repeated runs

### Why JSON and CSV?

- JSON: Complete structured data
- CSV: Easy analysis in Excel/pandas

## 📊 Expected Results

### Baseline Performance

```
No pruning (1.0, 1.0):
  Val Acc: ~0.81, Test Acc: ~0.80

Light pruning (0.9, 0.9):
  Val Acc: ~0.79, Test Acc: ~0.78
  Minimal accuracy drop

Moderate pruning (0.7, 0.7):
  Val Acc: ~0.74, Test Acc: ~0.73
  Noticeable drop

Heavy pruning (0.5, 0.5):
  Val Acc: ~0.65, Test Acc: ~0.64
  Significant drop
```

### Method Comparison

```
Random < Degree/Magnitude < Shapley
(for same sparsity level)
```

### Strategy Comparison

```
Global: Better at high sparsity (finds weak channels)
Layerwise: Better at low sparsity (preserves balance)
```

## ⚠️ Notes

### Computational Cost

**Simple methods** (degree, magnitude):
- Fast: seconds per experiment
- Suitable for large grids

**Shapley methods** (owen, approximation):
- Slow: minutes to hours
- Use small player sets (50-100)
- Cache results

### Keep Ratio Guidelines

**Conservative**: 0.9-0.7 (10-30% pruning)
**Moderate**: 0.7-0.5 (30-50% pruning)
**Aggressive**: 0.5-0.3 (50-70% pruning)

Start conservative, increase gradually.

### Graph vs Model Pruning

**Graph pruning**: Often more impactful
- Directly reduces information flow
- Harder to recover from

**Model pruning**: More robust
- Redundancy in parameters
- Fine-tuning can recover accuracy

### Best Practices

1. Always include baseline: `(1.0, 1.0)`
2. Use multiple seeds for robustness
3. Start with simple methods
4. Compare strategies on same data
5. Visualize results (accuracy vs sparsity)

## 🚀 Next Steps

After joint pruning experiments:
1. **Fine-tuning**: Retrain pruned models
2. **Iterative pruning**: Gradual sparsity increase
3. **Structured pruning**: Remove entire neurons/filters
4. **Hardware acceleration**: Measure actual speedup

Ready for complete pruning pipelines! 🎉
