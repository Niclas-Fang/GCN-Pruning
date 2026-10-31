# Scoring and Shapley Value System Documentation

## Overview

This document describes the unified scoring system and Shapley value computation for joint pruning condition evaluation.

## 📦 System Architecture

### 1. Unified Scoring Interface (`scorers/base.py`)

**Core Function:**
```python
score(
    unit_type: str,          # 'node', 'edge', 'channel'
    method: str,             # Scoring method name
    model,                   # GCN model
    data,                    # Graph data
    base_masks: Dict = None, # Conditional pruning state
    save_path: str = None,   # Optional save path
    seed: int = None,        # Random seed
    **kwargs                 # Method-specific args
) -> Dict[str, Any]
```

**Base Masks Format:**
```python
base_masks = {
    'node_mask': torch.Tensor,        # [num_nodes] boolean
    'edge_mask': torch.Tensor,        # [num_edges] boolean
    'channel_masks': {                # dict of channel masks
        'gcn_layers.0': torch.Tensor, # [hidden_dim] float (0 or 1)
        'gcn_layers.1': torch.Tensor,
        ...
    }
}
```

**Output Format:**
```python
result = {
    'unit_type': str,              # 'node', 'edge', or 'channel'
    'method': str,                 # Method name
    'scores': Dict[str, Tensor],   # Computed scores
    'base_masks': Dict,            # Base masks used
    'elapsed_time': float,         # Computation time
    'seed': int,                   # Random seed
    'config': Dict                 # Additional config
}
```

**Saving Format:**
Scores are saved to `scores/{unit_type}_{method}.pkl` containing:
- All fields from result dictionary
- Timestamp
- Configuration metadata

### 2. Baseline Scoring Methods

#### Node Scoring (`scorers/node_scorers.py`)

**Degree-based:**
```python
result = score(unit_type='node', method='degree', model=model, data=data)
# Higher degree = more important
```

**Random:**
```python
result = score(unit_type='node', method='random', model=model, data=data, seed=42)
# Baseline for comparison
```

#### Edge Scoring (`scorers/edge_scorers.py`)

**Random:**
```python
result = score(unit_type='edge', method='random', model=model, data=data, seed=42)
```

**Betweenness (Proxy):**
```python
result = score(unit_type='edge', method='betweenness', model=model, data=data)
# Uses endpoint degree product as proxy
```

#### Channel Scoring (`scorers/channel_scorers.py`)

**Magnitude (L2 norm):**
```python
result = score(unit_type='channel', method='magnitude', model=model, data=data)
# Returns dict: {layer_name: scores}
```

**Random:**
```python
result = score(unit_type='channel', method='random', model=model, data=data, seed=42)
```

### 3. Shapley Value Computation (`scorers/shapley.py`)

#### Value Function

**Definition:**
```python
v(S) = -val_loss
```
Where `val_loss` is computed on validation set under coalition S.

**Conditional Evaluation:**
- **Node coalition**: Changes `node_mask`, fixes `channel_masks` and graph structure
- **Edge coalition**: Changes `edge_mask`, fixes `channel_masks` and nodes
- **Channel coalition**: Changes `channel_mask` for one layer, fixes graph structure

**Cache Key:**
```python
cache_key = (base_masks_hash, unit_type, coalition_tuple)
```
Base masks hash ensures different pruning states have separate caches.

#### Algorithms

**1. Monte Carlo Shapley Approximation**
```python
from scorers.shapley import compute_shapley_scores

result = compute_shapley_scores(
    model=model,
    data=data,
    adj_norm=adj_norm,
    unit_type='node',           # or 'edge', 'channel'
    base_masks=base_masks,
    players=[0, 1, 2, ...],     # Units to evaluate
    method='approximation',
    num_samples=100,            # Number of permutations
    seed=42,
    device=device
)
```

**Algorithm:**
1. Sample random permutation of players
2. For each player in permutation:
   - Compute marginal contribution: v(S ∪ {i}) - v(S)
   - Add to player's Shapley value
3. Average over all permutations

**2. Two-stage Owen-Shapley**
```python
result = compute_shapley_scores(
    model=model,
    data=data,
    adj_norm=adj_norm,
    unit_type='node',
    base_masks=base_masks,
    players=players,
    method='owen',
    num_samples_q=10,           # Samples for q (outer)
    num_samples_p=10,           # Samples for p (middle)
    K=5,                        # Mini-batches
    seed=42
)
```

**Algorithm** (from provided image):
```
For q in {0, 1/C_R·C_S·K, 2/C_R·C_S·K, ..., 1}:
    For p in {0, 1/C_S·K, 2/C_S·K, ..., 1}:
        Initialize e_j = 0 for all j
        For k in 1, 2, ..., K:
            Sample I^(q) ~ Bern(q)
            Sample I^(p) ~ Bern(p)
            
            For each j:
                h(q,p)_k,j = v(I^(q) ⊙ B + I^(p) ⊙ B_(j) + X^(j))
                           - v(I^(q) ⊙ B + I^(p) ⊙ B_(j))
                
                e_j += h(q,p)_k,j
        
        Sh_j(v) += e_j
        
Sh_j(v) /= (C_R · C_S · K)
```

#### Coalition Value Function

```python
from scorers.shapley import CoalitionValueFunction

value_fn = CoalitionValueFunction(
    model=model,
    data=data,
    adj_norm=adj_norm,
    unit_type='node',           # 'node', 'edge', 'channel'
    base_masks=base_masks,
    criterion=nn.CrossEntropyLoss(),
    device=device,
    cache={}                    # Shared cache
)

# Evaluate coalition
value = value_fn([0, 1, 2, 3])  # Returns -val_loss
```

**Caching:**
- Automatic memoization of coalition evaluations
- Cache key includes base_masks hash
- Significant speedup for repeated evaluations

## 🚀 Usage Examples

### Example 1: Basic Node Scoring

```python
from scorers import score
from data import load_dataset
from models import GCNNet

# Load data and model
data = load_dataset('Cora')
model = GCNNet(...)

# Compute degree-based scores
result = score(
    unit_type='node',
    method='degree',
    model=model,
    data=data
)

node_scores = result['scores']['node_scores']  # [num_nodes]
```

### Example 2: Conditional Scoring

```python
# Create base masks with 50% channel pruning
base_masks = {
    'node_mask': torch.ones(data.num_nodes, dtype=torch.bool),
    'edge_mask': torch.ones(data.edge_index.size(1), dtype=torch.bool),
    'channel_masks': {
        'gcn_layers.0': torch.cat([torch.ones(32), torch.zeros(32)]),
        'gcn_layers.1': torch.cat([torch.ones(32), torch.zeros(32)]),
        ...
    }
}

# Score nodes under this channel pruning
result = score(
    unit_type='node',
    method='degree',
    model=model,
    data=data,
    base_masks=base_masks  # Conditional evaluation
)
```

### Example 3: Shapley Values (Small Scale)

```python
from scorers.shapley import compute_shapley_scores
from models import normalize_adjacency

# Prepare
adj_norm = normalize_adjacency(data.edge_index, data.num_nodes)
base_masks = create_default_masks(model, data)

# Select 50 nodes to evaluate
val_nodes = data.val_mask.nonzero(as_tuple=True)[0][:50]
players = val_nodes.tolist()

# Compute Shapley values
result = compute_shapley_scores(
    model=model,
    data=data,
    adj_norm=adj_norm,
    unit_type='node',
    base_masks=base_masks,
    players=players,
    method='approximation',
    num_samples=100,
    seed=42
)

shapley_values = result['shapley_values']  # [num_players]
```

### Example 4: Channel Importance

```python
# Magnitude-based channel scoring
result = score(
    unit_type='channel',
    method='magnitude',
    model=model,
    data=data
)

# Result is dict mapping layers to scores
for layer_name, scores in result['scores'].items():
    print(f"{layer_name}: {scores.shape}")
    # Find top-10 channels
    top10 = torch.topk(scores, 10).indices
    print(f"  Top-10: {top10.tolist()}")
```

### Example 5: Save and Load Scores

```python
# Compute and save
result = score(
    unit_type='node',
    method='degree',
    model=model,
    data=data,
    save_path='scores/node_degree.pkl',
    seed=42
)

# Load later
from scorers.base import PruningScorer
loaded = PruningScorer.load_scores('scores/node_degree.pkl')

print(loaded['unit_type'])      # 'node'
print(loaded['method'])          # 'degree'
print(loaded['elapsed_time'])    # computation time
```

## 🔧 Implementation Details

### Conditional Evaluation Mechanism

**For Node Coalitions:**
1. Create node_mask with only coalition nodes
2. Apply node mask → new edge_index
3. Normalize adjacency
4. Evaluate with fixed channel_masks
5. Return -val_loss

**For Edge Coalitions:**
1. Create edge_mask with only coalition edges
2. Apply edge mask → new edge_index
3. Normalize adjacency
4. Evaluate with fixed channel_masks
5. Return -val_loss

**For Channel Coalitions:**
1. Create channel_mask with only coalition channels
2. Update MaskedModelWrapper
3. Evaluate with fixed graph structure
4. Return -val_loss

### Cache Hash Computation

```python
def hash_masks(base_masks):
    # Node mask bytes
    + edge_mask bytes
    + for each layer (sorted):
        + layer_name bytes
        + channel_mask bytes
    → MD5 hash
```

This ensures:
- Different pruning states have different hashes
- Same pruning state → same hash (cache hit)
- Cache is valid across runs with same masks

### Performance Considerations

**Baseline Methods:**
- Degree: O(E) where E = num_edges
- Random: O(N) where N = num_units
- Magnitude: O(P) where P = num_parameters

**Shapley Methods:**
- Approximation: O(S × N) where S = num_samples, N = num_players
- Owen: O(Q × P × K × N) where Q, P = samples per stage
- Cache reduces redundant evaluations significantly

**Small Scale Recommendations:**
- Nodes: 50-100 players, 20-50 samples
- Edges: 50-100 players, 20-50 samples
- Channels: 16-32 players, 20-50 samples

## 📊 Expected Results

### Node Scoring

**Degree-based:**
```
High-degree nodes (hubs): High scores
Low-degree nodes: Low scores
Correlation with centrality: Strong
```

**Shapley-based:**
```
Nodes critical for validation accuracy: High values
Isolated nodes: Near-zero values
More nuanced than degree: Yes
```

### Edge Scoring

**Random:**
```
Uniform distribution
Good baseline for comparison
```

**Shapley-based:**
```
Edges connecting important nodes: High values
Redundant edges: Lower values
Bridge edges: High values
```

### Channel Scoring

**Magnitude:**
```
Channels with large weights: High scores
Dead channels (near-zero weights): Low scores
Fast to compute: Yes
```

**Shapley-based:**
```
Channels critical for accuracy: High values
Redundant channels: Lower values
Computationally expensive: Yes
```

## 🧪 Testing

### Small Scale Test (`test_shapley_small.py`)

```bash
python test_shapley_small.py
```

**Tests:**
1. Node Shapley (50 nodes, 20 samples)
2. Edge Shapley (50 edges, 20 samples)
3. Channel Shapley (16 channels, 20 samples)
4. Conditional evaluation with base pruning

**Expected Output:**
```
TEST 1: Node Shapley Values (50 nodes)
  Shapley values computed: 50
  Value range: [-0.123456, 0.234567]
  Num evaluations: ~500
  Cache size: ~400
  Elapsed time: ~15.2s
  ✓ Node Shapley test passed!
```

### Usage Examples (`example_scoring.py`)

```bash
python example_scoring.py
```

Demonstrates:
1. Basic scoring (degree, random, magnitude)
2. Conditional scoring with base masks
3. Saving and loading scores
4. Comparing multiple methods

## 📝 Notes

### Shapley Value Interpretation

- **Positive value**: Unit contributes positively (improves accuracy)
- **Negative value**: Unit hurts performance
- **Zero value**: No impact on accuracy
- **Magnitude**: Importance level

### Computational Cost

**Baseline methods**: Very fast (seconds)
**Shapley approximation**: Moderate (minutes for 50-100 units)
**Owen-Shapley**: Slower but more accurate (tens of minutes)

### Cache Benefits

Example cache hit rates:
- Without cache: 1000 evaluations
- With cache: 200-300 evaluations
- Speedup: 3-5×

### Limitations

1. **Small scale only**: Full-scale Shapley is prohibitive
2. **Validation loss**: Proxy for true importance
3. **Approximation error**: MC methods have variance
4. **Conditional evaluation**: Assumes independence

## 🎯 Next Steps

After scoring implementation:
1. **Pruning methods** - Use scores to create masks
2. **Pruning schedulers** - Oneshot vs iterative
3. **Complete workflows** - Train → Score → Prune → Evaluate
4. **Experiments** - Compare methods and baselines

Ready for pruning algorithm implementation! 🚀
