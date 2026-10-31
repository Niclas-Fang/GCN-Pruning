"""
Configuration Management
"""
import argparse
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for GCN training and pruning."""
    
    def __init__(self):
        # Experiment settings
        self.exp_name = 'default_exp'
        self.seed = 42
        self.deterministic = True
        
        # Device settings
        self.device = 'cuda'
        self.gpu_id = 0
        
        # Data settings
        self.dataset = 'Cora'
        self.data_dir = 'data/'
        
        # Model settings
        self.model = 'GCN'
        self.hidden_dim = 64
        self.num_layers = 2
        self.dropout = 0.5
        
        # Training settings
        self.epochs = 200
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.early_stop_patience = 50
        self.early_stop_metric = 'val_acc'  # or 'val_loss'
        self.num_gcn_layers = 3
        self.num_mlp_layers = 1

        # Pruning settings
        self.pruning_method = 'magnitude'
        self.pruning_ratio = 0.5
        self.pruning_schedule = 'oneshot'  # or 'iterative'
        
        # Output settings
        self.output_dir = 'outputs/'
        self.save_checkpoint = True
        self.save_interval = 50
        
        # Logging settings
        self.log_interval = 10
        self.verbose = True
    
    def update(self, config_dict: Dict[str, Any]):
        """Update config from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"[Warning] Unknown config key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def get_output_dirs(self) -> Dict[str, Path]:
        """Get all output directories for this experiment."""
        base_dir = Path(self.output_dir) / self.exp_name
        return {
            'base': base_dir,
            'checkpoints': base_dir / 'checkpoints',
            'scores': base_dir / 'scores',
            'results': base_dir / 'results',
            'logs': base_dir / 'logs',
        }
    
    def create_output_dirs(self):
        """Create all output directories."""
        dirs = self.get_output_dirs()
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        print(f"[Config] Created output directories at {dirs['base']}")
    
    def __repr__(self) -> str:
        """String representation of config."""
        lines = ["Configuration:"]
        lines.append("=" * 60)
        
        categories = {
            'Experiment': ['exp_name', 'seed', 'deterministic'],
            'Device': ['device', 'gpu_id'],
            'Data': ['dataset', 'data_dir'],
            'Model': ['model', 'hidden_dim', 'num_layers', 'dropout'],
            'Training': ['epochs', 'lr', 'weight_decay', 'early_stop_patience', 
                         'early_stop_metric', 'num_gcn_layers', 'num_mlp_layers'],
            'Pruning': ['pruning_method', 'pruning_ratio', 'pruning_schedule'],
            'Output': ['output_dir', 'save_checkpoint', 'save_interval'],
        }
        
        for category, keys in categories.items():
            lines.append(f"\n{category}:")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    lines.append(f"  {key:25s}: {value}")
        
        lines.append("=" * 60)
        return '\n'.join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GCN Training and Pruning')
    
    # Experiment
    parser.add_argument('--exp_name', type=str, default='default_exp',
                       help='Experiment name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Enable deterministic mode')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    # Data
    parser.add_argument('--dataset', type=str, default='Cora',
                       choices=['Cora', 'CiteSeer', 'PubMed'],
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data/',
                       help='Directory containing the PyG dataset cache')
    
    # Model
    parser.add_argument('--model', type=str, default='GCN',
                       help='Model architecture')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--early_stop_patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--early_stop_metric', type=str, default='val_acc',
                       choices=['val_acc', 'val_loss'],
                       help='Metric for early stopping')
    parser.add_argument('--num_gcn_layers', type=int, default=3,
                       help='Number of GCN layers (>= 3)')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                       help='Number of MLP layers (>= 1)')
    
    # Pruning
    parser.add_argument('--pruning_method', type=str, default='magnitude',
                       choices=['magnitude', 'random', 'gradient'],
                       help='Pruning method')
    parser.add_argument('--pruning_ratio', type=float, default=0.5,
                       help='Pruning ratio (0-1)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/',
                       help='Output directory')
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Verbose output')
    
    return parser.parse_args()


def get_config(args: argparse.Namespace = None) -> Config:
    """Get configuration from arguments."""
    config = Config()
    
    if args is not None:
        config.update(vars(args))
    
    return config
