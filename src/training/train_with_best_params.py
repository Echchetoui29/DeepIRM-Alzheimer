import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import logging
from dotenv import load_dotenv

from models import create_alexnet, get_model_summary
from dataset import AlzheimerMRIDataset
from train import Trainer, create_data_transforms

# Load environment
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_best_params() -> dict:
    """Load best hyperparameters from search results."""
    best_params_path = Path(__file__).parent.parent / "checkpoints" / "best_hyperparameters.json"
    
    if not best_params_path.exists():
        logger.error("Best hyperparameters not found!")
        logger.error("Run hyperparameter_search.py first to find optimal parameters.")
        raise FileNotFoundError(f"File not found: {best_params_path}")
    
    with open(best_params_path, 'r') as f:
        results = json.load(f)
    
    return results['params']


def create_optimizer(model: nn.Module, params: dict) -> torch.optim.Optimizer:
    """Create optimizer based on parameters."""
    optimizer_type = params.get('optimizer', 'Adam')
    lr = params.get('learning_rate', 0.001)
    wd = params.get('weight_decay', 0.0001)
    
    if optimizer_type == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        logger.warning(f"Unknown optimizer: {optimizer_type}. Using Adam.")
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def create_scheduler(optimizer: torch.optim.Optimizer, params: dict, num_epochs: int):
    """Create learning rate scheduler based on parameters."""
    scheduler_type = params.get('scheduler', 'cosine')
    
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
    elif scheduler_type == 'exponential':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.1,
            patience=5,
            verbose=True
        )
    else:
        logger.warning(f"Unknown scheduler: {scheduler_type}. Using cosine.")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )


def main():
    """Train with best hyperparameters from search."""
    
    print("\n" + "="*80)
    print("TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("="*80)
    
    # Load best parameters
    try:
        best_params = load_best_params()
        print("\nLoaded best hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key:<20} : {value}")
        print("="*80 + "\n")
    except FileNotFoundError:
        logger.error("Please run hyperparameter_search.py first!")
        return
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load configuration
    DATASET_BASE = os.getenv("DATASET_PATH", "/mnt/c/Users/ghout/Desktop/augmented-alzheimer-mri-dataset")
    stats_path = Path(__file__).parent.parent / "config" / "dataset_statistics.json"
    
    # Find dataset
    datasets_to_check = [
        ("BalancedAlzheimerDataset", "Balanced"),
        ("BalancedBalancedFromAugmented", "BalancedFromCombined"),
        ("BalancedFromAugmented", "BalancedFromAugmented"),
        ("CombinedAlzheimerDataset", "Combined"),
    ]
    
    dataset_path = None
    for dataset_dir, name in datasets_to_check:
        full_path = os.path.join(DATASET_BASE, dataset_dir)
        if os.path.exists(full_path):
            dataset_path = full_path
            logger.info(f"✓ Found dataset: {name}")
            break
    
    if not dataset_path:
        logger.error("No dataset found!")
        return
    
    # Create transforms
    train_transform, val_transform = create_data_transforms(str(stats_path))
    
    # Create dataset
    logger.info("Loading dataset...")
    full_dataset = AlzheimerMRIDataset(
        root_dir=dataset_path,
        transform=train_transform
    )
    
    # Split into train and validation (80-20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Update validation dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders with optimized batch size
    batch_size = best_params.get('batch_size', 32)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Batch size: {batch_size}")
    
    # Create model with optimized dropout
    dropout_rate = best_params.get('dropout_rate', 0.5)
    model = create_alexnet(
        num_classes=4,
        input_channels=1,
        dropout_rate=dropout_rate
    )
    model.to(device)
    get_model_summary(model)
    
    # Create trainer with optimized parameters
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=best_params.get('learning_rate', 0.001),
        weight_decay=best_params.get('weight_decay', 0.0001),
        model_name="alexnet_optimized"
    )
    
    # Override optimizer and scheduler with best ones
    trainer.optimizer = create_optimizer(model, best_params)
    trainer.scheduler = create_scheduler(trainer.optimizer, best_params, num_epochs=50)
    
    # Train with extended epochs for best performance
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        patience=30  # Increased patience for better convergence
    )
    
    # Save training history
    history_path = Path(__file__).parent.parent / "checkpoints" / "training_history_optimized.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"✓ Training history saved: {history_path}")
    
    # Print final results
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best Validation Accuracy: {history['best_val_accuracy']:.2f}%")
    print(f"Final Training Accuracy: {history['train_accuracies'][-1]:.2f}%")
    print(f"Epochs Trained: {len(history['train_losses'])}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
