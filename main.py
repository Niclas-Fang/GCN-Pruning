"""
Main Entry Point for GCN Training and Pruning
"""
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config, parse_args, get_config
from utils import (
    set_seed,
    setup_logger,
    log_environment_info,
    MetricsLogger,
    save_results
)


def main():
    """Main function to run experiments."""
    
    # Parse arguments and get config
    args = parse_args()
    config = get_config(args)
    
    # Create output directories
    config.create_output_dirs()
    output_dirs = config.get_output_dirs()
    
    # Setup logger
    log_file = output_dirs['logs'] / 'experiment.log'
    logger = setup_logger(
        name='gcn_pruning',
        log_file=str(log_file),
        console=True
    )
    
    # Log environment information
    log_environment_info(logger)
    
    # Set random seed
    set_seed(config.seed, config.deterministic)
    
    # Print configuration
    logger.info("\n" + str(config))
    
    # Determine device
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{config.gpu_id}')
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(config.gpu_id)})")
    else:
        device = torch.device('cpu')
        logger.info(f"Using device: {device}")
        if config.device == 'cuda':
            logger.warning("CUDA requested but not available, using CPU")
    
    # Log project structure
    logger.info("\nProject Structure:")
    logger.info(f"  Project root: {project_root}")
    logger.info(f"  Output base:  {output_dirs['base']}")
    logger.info(f"  Checkpoints:  {output_dirs['checkpoints']}")
    logger.info(f"  Scores:       {output_dirs['scores']}")
    logger.info(f"  Results:      {output_dirs['results']}")
    logger.info(f"  Logs:         {output_dirs['logs']}")
    
    # Save configuration to JSON
    config_save_path = output_dirs['results'] / 'config.json'
    save_results(str(config_save_path), config.to_dict())
    
    logger.info("\n" + "=" * 60)
    logger.info("Setup complete! Ready for training and pruning.")
    logger.info("=" * 60)
    
    # Placeholder for future implementation
    logger.info("\n[TODO] Implement the following components:")
    logger.info("  1. Data loading (data/)")
    logger.info("  2. Model definition (models/)")
    logger.info("  3. Training loop (train/)")
    logger.info("  4. Pruning methods (pruning/)")
    logger.info("  5. Scoring functions (scorers/)")
    logger.info("  6. Experiment runners (experiments/)")
    
    # Create a simple summary
    summary = {
        'experiment_name': config.exp_name,
        'seed': config.seed,
        'device': str(device),
        'dataset': config.dataset,
        'model': config.model,
        'status': 'setup_complete'
    }
    
    summary_path = output_dirs['results'] / 'summary.json'
    save_results(str(summary_path), summary)
    
    logger.info(f"\nExperiment setup saved to: {output_dirs['results']}")


if __name__ == '__main__':
    main()
