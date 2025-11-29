import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from itertools import product
import random
from tqdm import tqdm
from dotenv import load_dotenv
from datetime import datetime

from models import create_alexnet
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


class GridSearchCV:
    """Grid Search for hyperparameter tuning with simple train/val split."""
    
    def __init__(
        self,
        device: torch.device,
        dataset: AlzheimerMRIDataset,
        val_split: float = 0.2,
        stats_path: str = None
    ):
        """
        Initialize GridSearchCV.
        
        Args:
            device: Device to train on
            dataset: Full dataset
            val_split: Validation split ratio
            stats_path: Path to dataset statistics for normalization
        """
        self.device = device
        self.dataset = dataset
        self.val_split = val_split
        self.stats_path = stats_path
        self.results = []
        
        # Create results directory (for stats only)
        self.results_dir = Path(__file__).parent.parent / "checkpoints" / "hyperparameter_search"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def search(
        self,
        param_grid: Dict[str, List],
        num_epochs: int = 30,
        patience: int = 10
    ) -> Dict:
        """
        Perform grid search over parameter combinations.
        
        Args:
            param_grid: Dictionary with parameter names as keys and lists of values
                Example:
                {
                    'learning_rate': [0.0001, 0.001, 0.01],
                    'weight_decay': [0, 0.0001, 0.001],
                    'batch_size': [16, 32, 64],
                    'optimizer': ['Adam', 'SGD', 'AdamW'],
                    'scheduler': ['step', 'cosine', 'exponential'],
                    'dropout_rate': [0.3, 0.5, 0.7]
                }
            num_epochs: Epochs to train
            patience: Early stopping patience
        
        Returns:
            Best configuration dictionary
        """
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(product(*param_values))
        
        print("\n" + "="*80)
        print("GRID SEARCH FOR HYPERPARAMETER TUNING")
        print("="*80)
        print(f"Total combinations to test: {len(param_combinations)}")
        print(f"Validation split: {self.val_split:.1%}")
        print(f"Epochs per combination: {num_epochs}")
        print("="*80 + "\n")
        
        # Split dataset once (80-20 train-val split)
        train_size = int(len(self.dataset) * (1 - self.val_split))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        for combo_idx, params in enumerate(param_combinations):
            param_dict = dict(zip(param_names, params))
            
            print(f"\n{'='*80}")
            print(f"COMBINATION {combo_idx+1}/{len(param_combinations)}")
            print(f"{'='*80}")
            print("Parameters:")
            for key, value in param_dict.items():
                print(f"  {key:<20} : {value}")
            print(f"{'='*80}\n")
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=param_dict.get('batch_size', 32),
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=param_dict.get('batch_size', 32),
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Create model
            model = create_alexnet(
                num_classes=4,
                input_channels=1,
                dropout_rate=param_dict.get('dropout_rate', 0.5)
            )
            model.to(self.device)
            
            # Create trainer
            trainer = self._create_trainer(model, param_dict)
            
            # Train
            print(f"  Training...", end=" ", flush=True)
            try:
                history = trainer.train(
                    train_loader,
                    val_loader,
                    num_epochs=num_epochs,
                    patience=patience
                )
                
                best_val_acc = history['best_val_accuracy']
                best_val_loss = min(history['val_losses'])
                
                print(f"✓ Best Val Acc: {best_val_acc:.2f}%")
                
                result = {
                    'params': param_dict,
                    'best_val_accuracy': float(best_val_acc),
                    'best_val_loss': float(best_val_loss),
                    'train_losses': [float(x) for x in history['train_losses']],
                    'val_losses': [float(x) for x in history['val_losses']],
                    'train_accuracies': [float(x) for x in history['train_accuracies']],
                    'val_accuracies': [float(x) for x in history['val_accuracies']]
                }
                self.results.append(result)
                
            except Exception as e:
                logger.error(f"Error training: {e}")
                print(f"✗ FAILED: {e}")
                continue
        
        # Sort by best val accuracy (descending)
        self.results.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
        
        # Display top results
        print("\n" + "="*80)
        print("TOP 10 CONFIGURATIONS")
        print("="*80)
        for i, result in enumerate(self.results[:10]):
            print(f"\n{i+1}. Val Accuracy: {result['best_val_accuracy']:.2f}% | Loss: {result['best_val_loss']:.4f}")
            print(f"   Parameters:")
            for key, value in result['params'].items():
                print(f"     {key:<20} : {value}")
        
        # Save results
        self._save_results()
        
        return self.results[0] if self.results else None
    
    def _create_trainer(self, model: nn.Module, param_dict: Dict) -> Trainer:
        """Create trainer with specified hyperparameters."""
        
        # Create base trainer
        trainer = Trainer(
            model=model,
            device=self.device,
            learning_rate=param_dict.get('learning_rate', 0.001),
            weight_decay=param_dict.get('weight_decay', 0.0001),
            model_name="hyperopt"
        )
        
        # Override optimizer based on param_dict
        optimizer_type = param_dict.get('optimizer', 'SGD')
        lr = param_dict.get('learning_rate', 0.001)
        wd = param_dict.get('weight_decay', 0.0001)
        
        if optimizer_type == 'Adam':
            trainer.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=wd
            )
        elif optimizer_type == 'AdamW':
            trainer.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=wd
            )
        elif optimizer_type == 'SGD':
            trainer.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=wd
            )
        
        # Override scheduler based on param_dict
        scheduler_type = param_dict.get('scheduler', 'step')
        
        if scheduler_type == 'cosine':
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer,
                T_max=30,
                eta_min=1e-6
            )
        elif scheduler_type == 'exponential':
            trainer.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                trainer.optimizer,
                gamma=0.95
            )
        elif scheduler_type == 'step':
            trainer.scheduler = torch.optim.lr_scheduler.StepLR(
                trainer.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif scheduler_type == 'reduce_on_plateau':
            trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                trainer.optimizer,
                mode='max',
                factor=0.1,
                patience=5,
                verbose=True
            )
        
        return trainer
    
    def _save_results(self) -> None:
        """Save search results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"grid_search_results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"✓ Results saved to: {results_path}")


class RandomSearchCV:
    """Random Search for hyperparameter tuning with simple train/val split."""
    
    def __init__(
        self,
        device: torch.device,
        dataset: AlzheimerMRIDataset,
        val_split: float = 0.2,
        stats_path: str = None
    ):
        """
        Initialize RandomSearchCV.
        
        Args:
            device: Device to train on
            dataset: Full dataset
            val_split: Validation split ratio
            stats_path: Path to dataset statistics
        """
        self.device = device
        self.dataset = dataset
        self.val_split = val_split
        self.stats_path = stats_path
        self.results = []
        
        # Create results directory (for stats only)
        self.results_dir = Path(__file__).parent.parent / "checkpoints" / "hyperparameter_search"
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def search(
        self,
        param_distributions: Dict[str, List],
        n_iter: int = 20,
        num_epochs: int = 30,
        patience: int = 10
    ) -> Dict:
        """
        Perform random search over parameter space.
        
        Args:
            param_distributions: Dict of parameter distributions
                Example:
                {
                    'learning_rate': [0.00001, 0.01],  # Will sample log-uniformly
                    'weight_decay': [0, 0.001],
                    'batch_size': [16, 32, 64],  # Will sample uniformly
                    'optimizer': ['Adam', 'SGD', 'AdamW'],
                    'scheduler': ['step', 'cosine', 'exponential'],
                    'dropout_rate': [0.3, 0.7]
                }
            n_iter: Number of random combinations to try
            num_epochs: Epochs to train
            patience: Early stopping patience
        
        Returns:
            Best configuration dictionary
        """
        
        print("\n" + "="*80)
        print("RANDOM SEARCH FOR HYPERPARAMETER TUNING")
        print("="*80)
        print(f"Random combinations to test: {n_iter}")
        print(f"Validation split: {self.val_split:.1%}")
        print(f"Epochs per combination: {num_epochs}")
        print("="*80 + "\n")
        
        # Set random seed
        random.seed(42)
        np.random.seed(42)
        
        # Split dataset once (80-20 train-val split)
        train_size = int(len(self.dataset) * (1 - self.val_split))
        val_size = len(self.dataset) - train_size
        train_dataset, val_dataset = random_split(
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        for iter_idx in range(n_iter):
            # Sample random parameters
            param_dict = self._sample_parameters(param_distributions)
            
            print(f"\n{'='*80}")
            print(f"ITERATION {iter_idx+1}/{n_iter}")
            print(f"{'='*80}")
            print("Parameters:")
            for key, value in param_dict.items():
                print(f"  {key:<20} : {value}")
            print(f"{'='*80}\n")
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=param_dict['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=param_dict['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Create model
            model = create_alexnet(
                num_classes=4,
                input_channels=1,
                dropout_rate=param_dict['dropout_rate']
            )
            model.to(self.device)
            
            # Create trainer
            trainer = self._create_trainer(model, param_dict)
            
            # Train
            print(f"  Training...", end=" ", flush=True)
            try:
                history = trainer.train(
                    train_loader,
                    val_loader,
                    num_epochs=num_epochs,
                    patience=patience
                )
                
                best_val_acc = history['best_val_accuracy']
                best_val_loss = min(history['val_losses'])
                
                print(f"✓ Best Val Acc: {best_val_acc:.2f}%")
                
                result = {
                    'params': param_dict,
                    'best_val_accuracy': float(best_val_acc),
                    'best_val_loss': float(best_val_loss),
                    'train_losses': [float(x) for x in history['train_losses']],
                    'val_losses': [float(x) for x in history['val_losses']],
                    'train_accuracies': [float(x) for x in history['train_accuracies']],
                    'val_accuracies': [float(x) for x in history['val_accuracies']]
                }
                self.results.append(result)
                
            except Exception as e:
                logger.error(f"Error training: {e}")
                print(f"✗ FAILED: {e}")
                continue
        
        # Sort by best val accuracy
        self.results.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
        
        # Display top results
        print("\n" + "="*80)
        print("TOP 10 CONFIGURATIONS")
        print("="*80)
        for i, result in enumerate(self.results[:10]):
            print(f"\n{i+1}. Val Accuracy: {result['best_val_accuracy']:.2f}% | Loss: {result['best_val_loss']:.4f}")
            print(f"   Parameters:")
            for key, value in result['params'].items():
                print(f"     {key:<20} : {value}")
        
        # Save results
        self._save_results()
        
        return self.results[0] if self.results else None
    
    def _sample_parameters(self, param_distributions: Dict) -> Dict:
        """Sample random parameters from distributions."""
        param_dict = {}
        
        for param_name, param_range in param_distributions.items():
            if isinstance(param_range, list):
                if len(param_range) == 2 and all(isinstance(x, (int, float)) for x in param_range):
                    # Numeric range - sample based on parameter type
                    if 'learning_rate' in param_name.lower():
                        # Log-uniform sampling for learning rate
                        log_min = np.log10(param_range[0])
                        log_max = np.log10(param_range[1])
                        param_dict[param_name] = 10 ** np.random.uniform(log_min, log_max)
                    elif 'dropout' in param_name.lower():
                        # Uniform sampling for dropout
                        param_dict[param_name] = np.random.uniform(param_range[0], param_range[1])
                    elif 'batch_size' in param_name.lower():
                        # Sample from powers of 2 within range
                        min_pow = int(np.log2(param_range[0]))
                        max_pow = int(np.log2(param_range[1]))
                        param_dict[param_name] = 2 ** np.random.randint(min_pow, max_pow + 1)
                    else:
                        # Uniform sampling for other numeric params
                        param_dict[param_name] = np.random.uniform(param_range[0], param_range[1])
                else:
                    # Discrete choices
                    param_dict[param_name] = random.choice(param_range)
            else:
                param_dict[param_name] = param_range
        
        # Ensure batch_size is int
        if 'batch_size' in param_dict:
            param_dict['batch_size'] = int(param_dict['batch_size'])
        
        return param_dict
    
    def _create_trainer(self, model: nn.Module, param_dict: Dict) -> Trainer:
        """Create trainer with specified hyperparameters."""
        
        trainer = Trainer(
            model=model,
            device=self.device,
            learning_rate=param_dict['learning_rate'],
            weight_decay=param_dict['weight_decay'],
            model_name="hyperopt"
        )
        
        # Override optimizer
        optimizer_type = param_dict['optimizer']
        lr = param_dict['learning_rate']
        wd = param_dict['weight_decay']
        
        if optimizer_type == 'Adam':
            trainer.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=wd
            )
        elif optimizer_type == 'AdamW':
            trainer.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=wd
            )
        elif optimizer_type == 'SGD':
            trainer.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=wd
            )
        
        # Override scheduler
        scheduler_type = param_dict['scheduler']
        
        if scheduler_type == 'cosine':
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer,
                T_max=30,
                eta_min=1e-6
            )
        elif scheduler_type == 'exponential':
            trainer.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                trainer.optimizer,
                gamma=0.95
            )
        elif scheduler_type == 'step':
            trainer.scheduler = torch.optim.lr_scheduler.StepLR(
                trainer.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif scheduler_type == 'reduce_on_plateau':
            trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                trainer.optimizer,
                mode='max',
                factor=0.1,
                patience=5
            )
        
        return trainer
    
    def _save_results(self) -> None:
        """Save search results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"random_search_results_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"✓ Results saved to: {results_path}")


def main():
    """Main function for hyperparameter search."""
    
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
    train_transform, _ = create_data_transforms(str(stats_path))
    
    # Load full dataset
    logger.info("Loading dataset...")
    full_dataset = AlzheimerMRIDataset(
        root_dir=dataset_path,
        transform=train_transform
    )
    logger.info(f"Dataset size: {len(full_dataset)}")
    
    # Choose search method
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    print("Choose search method:")
    print("  1. Grid Search (exhaustive, slower)")
    print("  2. Random Search (faster, recommended)")
    print("="*80)
    
    choice = input("Enter choice (1 or 2, default=2): ").strip() or "2"
    
    if choice == "1":
        # GRID SEARCH
        param_grid = {
            'learning_rate': [0.0001, 0.001],
            'weight_decay': [0.0001, 0.001],
            'batch_size': [32, 64],
            'optimizer': ['Adam', 'AdamW'],
            'scheduler': ['cosine', 'step'],
            'dropout_rate': [0.5, 0.6]
        }
        
        searcher = GridSearchCV(
            device=device,
            dataset=full_dataset,
            val_split=0.2,
            stats_path=str(stats_path)
        )
        
        best_result = searcher.search(
            param_grid=param_grid,
            num_epochs=30,
            patience=10
        )
        
    else:
        # RANDOM SEARCH (RECOMMENDED)
        param_distributions = {
            'learning_rate': [0.00001, 0.01],  # Log scale
            'weight_decay': [0, 0.001],
            'batch_size': [16, 32, 64],
            'optimizer': ['Adam', 'AdamW', 'SGD'],
            'scheduler': ['cosine', 'step', 'exponential'],
            'dropout_rate': [0.3, 0.7]
        }
        
        searcher = RandomSearchCV(
            device=device,
            dataset=full_dataset,
            val_split=0.2,
            stats_path=str(stats_path)
        )
        
        best_result = searcher.search(
            param_distributions=param_distributions,
            n_iter=20,  # Try 20 random combinations
            num_epochs=30,
            patience=10
        )
    
    if best_result is None:
        logger.error("No successful training runs!")
        return
    
    # Display best result
    print("\n" + "="*80)
    print("BEST CONFIGURATION FOUND")
    print("="*80)
    print(f"Best Val Accuracy: {best_result['best_val_accuracy']:.2f}%")
    print(f"Best Val Loss: {best_result['best_val_loss']:.4f}")
    print("\nBest Parameters:")
    for key, value in best_result['params'].items():
        print(f"  {key:<20} : {value}")
    print("="*80)
    
    # Save best params for easy reuse
    best_params_path = Path(__file__).parent.parent / "checkpoints" / "best_hyperparameters.json"
    with open(best_params_path, 'w') as f:
        json.dump(best_result, f, indent=2)
    
    logger.info(f"✓ Best parameters saved to: {best_params_path}")
    
    print("\nTo use these parameters, update your train.py with the best configuration!")


if __name__ == "__main__":
    main()
