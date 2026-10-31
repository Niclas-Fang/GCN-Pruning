"""
Training Script for GCN Model
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import GCNNet, normalize_adjacency
from data import load_dataset
from train.eval import evaluate
from utils import set_seed, setup_logger, save_checkpoint
from config import Config


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(self, patience: int = 50, mode: str = 'min', delta: float = 0.0):
        """
        Args:
            patience: Number of epochs to wait for improvement
            mode: 'min' for loss, 'max' for accuracy
            delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if should stop early.
        
        Args:
            score: Current metric value
            epoch: Current epoch
        
        Returns:
            True if should stop early
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        # Check if improved
        if self.mode == 'min':
            improved = score < self.best_score - self.delta
        else:  # mode == 'max'
            improved = score > self.best_score + self.delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def train_epoch(
    model: nn.Module,
    data,
    adj_norm: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch:int=None
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: GCN model
        data: Graph data
        adj_norm: Normalized adjacency matrix
        optimizer: Optimizer
        criterion: Loss function
        device: Device
    
    Returns:
        Training loss
    """
    model.train()
    optimizer.zero_grad()

    # 记录一步更新前的分类头权重
    if epoch == 1:
        w0 = model.output_head.weight.detach().clone()

    logits = model(data.x, adj_norm)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    loss.backward()

    if epoch == 1:
        g = model.output_head.weight.grad
        print("[Debug] grad output_head mean|g| =", g.abs().mean().item(), "max|g| =", g.abs().max().item())

    optimizer.step()

    if epoch == 1:
        dw = (model.output_head.weight.detach() - w0).abs().mean().item()
        print("[Debug] mean |ΔW| output_head =", dw)
    if epoch == 1:
        with torch.no_grad():
            logits = model(data.x, adj_norm)
            print("[Debug] logits mean/std:", logits.mean().item(), logits.std().item())
            pred = logits.argmax(dim=1)
            print("[Debug] pred unique:", pred.unique().numel())


    return loss.item()


def train_gcn(config: Config):
    """
    Main training function.
    
    Args:
        config: Configuration object
    """
    # Setup directories
    config.create_output_dirs()
    output_dirs = config.get_output_dirs()
    
    # Setup logger
    logger = setup_logger(
        name='gcn_training',
        log_file=str(output_dirs['logs'] / 'train.log')
    )
    
    # Set random seed
    set_seed(config.seed, config.deterministic)
    
    # Device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset}")
    data = load_dataset(config.dataset, config.data_dir)
    logger.info(f"  Nodes: {data.num_nodes}, Features: {data.num_features}, Classes: {data.num_classes}")
    logger.info(f"  Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
    
    print("[Debug] x mean/std:", data.x.mean().item(), data.x.std().item())
    print("[Debug] x nonzero ratio:", (data.x != 0).float().mean().item())
    print("[Debug] y dtype/min/max:", data.y.dtype, int(data.y.min()), int(data.y.max()))
    print("[Debug] class counts:", torch.bincount(data.y.cpu()).tolist())
    print("[Debug] train_mask dtype:", data.train_mask.dtype)


    # Move data to device
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)
    
    # Normalize adjacency matrix
    logger.info("Normalizing adjacency matrix...")
    adj_norm = normalize_adjacency(data.edge_index, data.num_nodes)
    adj_norm = adj_norm.to(device)
    
    print("[Debug] adj_norm is_sparse:", adj_norm.is_sparse)
    print("[Debug] adj_norm nnz:", adj_norm._nnz())
    print("[Debug] adj_norm values min/max:", adj_norm.values().min().item(), adj_norm.values().max().item())


    # Create model
    logger.info("Creating GCN model...")
    model = GCNNet(
        in_dim=data.num_features,
        hidden_dim=config.hidden_dim,
        num_classes=data.num_classes,
        num_gcn_layers=config.num_gcn_layers,
        num_mlp_layers=config.num_mlp_layers,
        dropout=config.dropout
    ).to(device)

    logger.info(f"\n{model}")
    
    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stop_patience,
        mode='max' if config.early_stop_metric == 'val_acc' else 'min'
    )
    
    # Training loop
    logger.info(f"\nStarting training for {config.epochs} epochs...")
    logger.info(f"Early stopping: metric={config.early_stop_metric}, patience={config.early_stop_patience}")
    
    best_val_metric = -float('inf') if config.early_stop_metric == 'val_acc' else float('inf')
    best_epoch = 0
    
    for epoch in range(1, config.epochs + 1):
        # Train
        train_loss = train_epoch(model, data, adj_norm, optimizer, criterion, device, epoch)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, data, data.val_mask, adj_norm, criterion)
        
        # Log progress
        if epoch % config.log_interval == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:3d}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )
        
        # Check for improvement
        current_metric = val_acc if config.early_stop_metric == 'val_acc' else val_loss
        is_best = False
        
        if config.early_stop_metric == 'val_acc':
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_epoch = epoch
                is_best = True
        else:
            if current_metric < best_val_metric:
                best_val_metric = current_metric
                best_epoch = epoch
                is_best = True
        
        # Save best checkpoint
        if is_best and config.save_checkpoint:
            checkpoint_path = output_dirs['checkpoints'] / 'best_model.pth'
            save_checkpoint(
                save_path=str(checkpoint_path),
                model_state=model.state_dict(),
                config=config.to_dict(),
                metrics={'val_loss': val_loss, 'val_acc': val_acc},
                epoch=epoch,
                optimizer_state=optimizer.state_dict()
            )
        
        # Early stopping check
        if early_stopping(current_metric, epoch):
            logger.info(f"\nEarly stopping triggered at epoch {epoch}")
            logger.info(f"Best {config.early_stop_metric}: {best_val_metric:.4f} at epoch {best_epoch}")
            break
    
    # Load best model and evaluate on test set
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best {config.early_stop_metric}: {best_val_metric:.4f} at epoch {best_epoch}")
    
    # Load best checkpoint
    best_checkpoint_path = output_dirs['checkpoints'] / 'best_model.pth'
    if best_checkpoint_path.exists():
        logger.info(f"\nLoading best checkpoint from epoch {best_epoch}...")
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(model, data, data.test_mask, adj_norm, criterion)
    
    logger.info(f"\nFinal Test Results:")
    logger.info(f"  Test Loss: {test_loss:.4f}")
    logger.info(f"  Test Acc:  {test_acc:.4f}")
    logger.info("=" * 60)
    
    # Save final results
    results = {
        'best_epoch': best_epoch,
        'best_val_metric': best_val_metric,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'config': config.to_dict()
    }
    
    from utils import save_results
    results_path = output_dirs['results'] / 'training_results.json'
    save_results(str(results_path), results)
    
    logger.info(f"\nResults saved to {results_path}")
    
    return model, results


def main():
    """Main entry point for training script."""
    from config import parse_args, get_config
    
    # Parse arguments
    args = parse_args()
    config = get_config(args)
    
    # Add training-specific defaults if not set
    if not hasattr(config, 'num_gcn_layers'):
        config.num_gcn_layers = 3
    if not hasattr(config, 'num_mlp_layers'):
        config.num_mlp_layers = 1
    if not hasattr(config, 'early_stop_metric'):
        config.early_stop_metric = 'val_acc'  # or 'val_loss'
    
    # Train model
    model, results = train_gcn(config)
    
    print(f"\n✓ Training complete!")
    print(f"  Best Val Acc: {results['best_val_metric']:.4f} (epoch {results['best_epoch']})")
    print(f"  Test Acc: {results['test_acc']:.4f}")


if __name__ == '__main__':
    main()
