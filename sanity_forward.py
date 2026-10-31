"""
Sanity Check: Test GCN Forward Pass
"""
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import GCNNet, normalize_adjacency
from data import load_dataset
from utils import set_seed


def sanity_check_forward():
    """Test GCN model forward pass with Cora dataset."""
    
    print("=" * 70)
    print("GCN Forward Pass Sanity Check")
    print("=" * 70)
    
    # Set seed for reproducibility
    set_seed(42, deterministic=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load Cora dataset
    print("\n[1] Loading Cora dataset...")
    data = load_dataset('Cora', data_dir='data/')
    
    print(f"  ✓ Dataset loaded:")
    print(f"    - Nodes: {data.num_nodes}")
    print(f"    - Features: {data.num_features}")
    print(f"    - Classes: {data.num_classes}")
    print(f"    - Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
    
    # Move to device
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    
    # Normalize adjacency matrix
    print("\n[2] Normalizing adjacency matrix...")
    adj_norm = normalize_adjacency(data.edge_index, data.num_nodes)
    adj_norm = adj_norm.to(device)
    print(f"  ✓ Adjacency matrix normalized: {adj_norm.shape}")
    
    # Create model
    print("\n[3] Creating GCN model...")
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=64,
        num_classes=data.num_classes,
        num_gcn_layers=3,
        num_mlp_layers=1,
        dropout=0.5
    ).to(device)
    
    print(f"  ✓ Model created:")
    print(f"\n{model}\n")
    
    # Test forward pass (training mode)
    print("[4] Testing forward pass (training mode)...")
    model.train()
    
    # Regular forward
    logits = model(data.x, adj_norm)
    print(f"  ✓ Forward pass completed")
    print(f"    - Input shape: {data.x.shape}")
    print(f"    - Output shape (logits): {logits.shape}")
    
    # Forward with embeddings
    logits2, embeddings = model(data.x, adj_norm, return_embeddings=True)
    print(f"  ✓ Forward pass with embeddings")
    print(f"    - Logits shape: {logits2.shape}")
    print(f"    - Embeddings shape: {embeddings.shape}")
    
    # Verify logits are the same
    # assert torch.allclose(logits, logits2, atol=1e-6), "Logits mismatch!"
    
    # Test forward pass (eval mode)
    print("\n[5] Testing forward pass (eval mode)...")
    model.eval()
    
    # with torch.no_grad():
    #     logits_eval = model(data.x, adj_norm)
    
    # print(f"  ✓ Eval forward pass completed")
    # print(f"    - Output shape: {logits_eval.shape}")
    

    with torch.no_grad():
        logits_e = model(data.x, adj_norm)
        logits2_e, emb_e = model(data.x, adj_norm, return_embeddings=True)
    
    print(f"  ✓ Eval forward pass completed")
    # print(f"    - Output shape: {logits_eval.shape}")

    # Verify logits are the same
    assert torch.allclose(logits_e, logits2_e, atol=1e-6), "Logits mismatch in eval!"
    print(f"  ✓ Logits match between two forward passes")
    # Print layer-wise dimensions
    print("\n[6] Layer-wise tensor dimensions:")
    model.train()
    
    print(f"  Input:       {data.x.shape}")
    
    # Input projection
    h = model.input_proj(data.x)
    print(f"  input_proj:  {h.shape}")
    h = torch.relu(h)
    h = torch.dropout(h, p=model.dropout, train=True)
    
    # GCN layers
    for i, gcn_layer in enumerate(model.gcn_layers):
        h = gcn_layer(h, adj_norm)
        print(f"  gcn_layer_{i}: {h.shape}")
        h = torch.relu(h)
        h = torch.dropout(h, p=model.dropout, train=True)
    
    # MLP layers
    for i, mlp_layer in enumerate(model.mlp_layers):
        h = mlp_layer(h)
        print(f"  mlp_layer_{i}: {h.shape}")
        h = torch.relu(h)
        h = torch.dropout(h, p=model.dropout, train=True)
    
    # Output head
    logits_manual = model.output_head(h)
    print(f"  output_head: {logits_manual.shape}")
    
    # Test prunable layers listing
    print("\n[7] Testing prunable layers listing...")
    prunable_layers = model.list_prunable_layers()
    print(f"  ✓ Found {len(prunable_layers)} prunable layers:")
    for name, layer in prunable_layers:
        num_params = sum(p.numel() for p in layer.parameters())
        print(f"    - {name:20s}: {layer.__class__.__name__:15s} ({num_params:,} params)")
    
    # Parameter counts
    print("\n[8] Parameter statistics:")
    total_params = model.count_parameters(only_prunable=False)
    prunable_params = model.count_parameters(only_prunable=True)
    non_prunable_params = total_params - prunable_params
    
    print(f"  Total parameters:        {total_params:,}")
    print(f"  Prunable parameters:     {prunable_params:,} ({prunable_params/total_params*100:.1f}%)")
    print(f"  Non-prunable parameters: {non_prunable_params:,} ({non_prunable_params/total_params*100:.1f}%)")
    
    # Test predictions
    print("\n[9] Testing predictions...")
    model.eval()
    with torch.no_grad():
        logits_test = model(data.x, adj_norm)
        pred = logits_test.argmax(dim=1)
    
    print(f"  ✓ Predictions computed")
    print(f"    - Predicted classes: {pred.unique().tolist()}")
    print(f"    - Class distribution: {[(c, (pred == c).sum().item()) for c in range(data.num_classes)]}")
    
    # Compute accuracy on train set (should be random initially)
    train_pred = pred[data.train_mask]
    train_true = data.y[data.train_mask]
    train_acc = (train_pred == train_true).float().mean().item()
    print(f"    - Train accuracy (untrained): {train_acc:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ All sanity checks passed!")
    print("=" * 70)
    
    return model, data, adj_norm


if __name__ == '__main__':
    try:
        model, data, adj_norm = sanity_check_forward()
        print("\n✓ Forward pass sanity check completed successfully!\n")
    except Exception as e:
        print(f"\n✗ Sanity check failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
