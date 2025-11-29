"""
Script de fine-tuning du modèle AlexNet sur un nouveau jeu d’images réalistes.

Ce script permet d'affiner un modèle AlexNet préentraîné en utilisant un dataset
réaliste externe. Il applique des techniques de transfert d’apprentissage,
offre l’option de geler les couches de caractéristiques et utilise un taux
d’apprentissage réduit pour un ajustement plus stable.

Utilisation :
    python finetune_alexnet_realistic.py \
        --checkpoint_path ../checkpoints/alexnet_optimized_epoch_35_acc_98.05.pt \
        --realistic_data_path /chemin/vers/les/images \
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from models import AlexNet, get_model_summary
from dataset import AlzheimerMRIDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('finetune.log')
    ]
)
logger = logging.getLogger(__name__)


class FineTuner:
    """Classe de fine-tuning pour un modèle AlexNet préentraîné."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0001,
        freeze_features: bool = False,
        model_name: str = "alexnet_finetuned"
    ):
        """
        Initialise le fine-tuner.
        
        Args:
            model: Modèle PyTorch préentraîné.
            device: Appareil de calcul (CPU ou GPU).
            learning_rate: Taux d’apprentissage réduit pour le fine-tuning.
            weight_decay: Poids de régularisation L2.
            freeze_features: Geler les couches convolutionnelles si True.
            model_name: Nom des checkpoints sauvegardés.
        """
        self.model = model
        self.device = device
        self.model_name = model_name
        self.freeze_features = freeze_features
        
        if freeze_features:
            logger.info("Gel des couches de caractéristiques…")
            for param in self.model.features.parameters():
                param.requires_grad = False
            logger.info("Couches de caractéristiques gelées.")
        
        self.criterion = nn.CrossEntropyLoss()
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        
        self.optimizer = torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        self.checkpoint_dir = Path(__file__).parent.parent / "checkpoints" / "finetuned"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Paramètres totaux : {total_params:,}")
        logger.info(f"Paramètres entraînables : {trainable_params_count:,} ({100*trainable_params_count/total_params:.1f}%)")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Entraîne une époque complète.
        
        Args:
            train_loader: DataLoader d'entraînement.
        
        Returns:
            Perte moyenne et précision moyenne.
        """
        self.model.train()
        
        if self.freeze_features:
            self.model.features.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Entraînement", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """
        Valide le modèle sur le jeu de validation.
        
        Args:
            val_loader: DataLoader de validation.
        
        Returns:
            Perte moyenne, précision et statistiques par classe.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                for label, pred in zip(labels, predicted):
                    label_idx = label.item()
                    ok = (pred == label).item()
                    
                    class_total[label_idx] = class_total.get(label_idx, 0) + 1
                    class_correct[label_idx] = class_correct.get(label_idx, 0) + ok
                
                pbar.set_postfix({'acc': f'{100*correct/total:.2f}%'})
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        per_class_metrics = {
            idx: {
                'correct': class_correct[idx],
                'total': class_total[idx],
                'accuracy': 100 * class_correct[idx] / class_total[idx]
            }
            for idx in class_total
        }
        
        return avg_loss, accuracy, per_class_metrics
    
    def finetune(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 20, patience: int = 10) -> Dict:
        """
        Lance le fine-tuning du modèle.
        
        Args:
            train_loader: Données d'entraînement.
            val_loader: Données de validation.
            num_epochs: Nombre maximal d’époques.
            patience: Patience du early stopping.
        
        Returns:
            Historique d'entraînement.
        """
        print("\n" + "=" * 70)
        print("DÉBUT DU FINE-TUNING")
        print("=" * 70)
        
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            val_loss, val_acc, per_class = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            self.scheduler.step(val_acc)
            
            print(f"\n--- Époque {epoch+1}/{num_epochs} ---")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                self.save_checkpoint(epoch, val_acc, train_loss, val_loss)
                print(f"✓ Nouveau meilleur modèle sauvegardé ({val_acc:.2f}%)")
            else:
                patience_counter += 1
                print(f"Aucune amélioration ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print("Arrêt anticipé activé.")
                    break
        
        print("\nFINE-TUNING TERMINÉ")
        print(f"Meilleure précision validation : {best_val_accuracy:.2f}%\n")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': best_val_accuracy
        }
    
    def save_checkpoint(self, epoch: int, val_accuracy: float, train_loss: float, val_loss: float) -> None:
        """Sauvegarde un checkpoint du modèle."""
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch+1}_acc_{val_accuracy:.2f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'freeze_features': self.freeze_features,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        logger.info(f"Checkpoint sauvegardé : {checkpoint_path}")


def load_pretrained_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Charge un modèle préentraîné depuis un checkpoint.

    Args:
        checkpoint_path: Chemin du fichier checkpoint.
        device: Appareil cible.
    
    Returns:
        Modèle chargé.
    """
    logger.info(f"Chargement du modèle depuis : {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = AlexNet(
        num_classes=4,
        input_channels=1,
        dropout_rate=0.498070764044508
    )
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    logger.info("Modèle préentraîné chargé avec succès.")
    
    return model


def create_data_transforms(mean: float, std: float, augment: bool = True) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Crée les transformations pour l’entraînement et la validation.

    Args:
        mean: Moyenne pour normalisation.
        std: Écart-type pour normalisation.
        augment: Activer ou non l’augmentation.
    
    Returns:
        Transformations d'entraînement et de validation.
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((200, 190)),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((200, 190)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((200, 190)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])
    
    return train_transform, val_transform


def main():
    """Point d'entrée principal du script de fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tuning d’AlexNet sur un dataset réaliste")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Chemin du modèle préentraîné')
    parser.add_argument('--realistic_data_path', type=str, required=True, help='Chemin du dataset réaliste')
    parser.add_argument('--freeze_features', action='store_true', help='Geler les couches convolutionnelles')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Taux d’apprentissage')
    parser.add_ARGUwENT('--batch_size', type=int, default=16, help='Taille des batchs')
    parser.add_argument('--num_epochs', type=int, default=20, help='Nombre maximal d’époques')
    parser.add_argument('--patience', type=int, default=10, help='Patience du early stopping')
    parser.add_argument('--train_split', type=float, default=0.8, help='Ratio train/val')
    parser.add_argument('--no_augment', action='store_true', help='Désactiver l’augmentation')

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device utilisé : {device}")
    
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).parent / checkpoint_path
    
    model = load_pretrained_model(str(checkpoint_path), device)
    get_model_summary(model)
    
    stats_path = Path(__file__).parent.parent / "config" / "dataset_statistics.json"
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        mean = stats['normalization']['mean']
        std = stats['normalization']['std']
    else:
        mean = 0.2945542335510254
        std = 0.3180045485496521
    
    train_transform, val_transform = create_data_transforms(mean, std, augment=not args.no_augment)
    
    full_dataset = AlzheimerMRIDataset(
        root_dir=args.realistic_data_path,
        transform=train_transform
    )
    
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    finetuner = FineTuner(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=0.0001,
        freeze_features=args.freeze_features,
        model_name=f"alexnet_finetuned_{'frozen' if args.freeze_features else 'full'}"
    )
    
    history = finetuner.finetune(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        patience=args.patience
    )
    
    history_path = finetuner.checkpoint_dir / f"finetune_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'history': history,
            'final_metrics': {
                'best_val_accuracy': history['best_val_accuracy'],
                'final_train_accuracy': history['train_accuracies'][-1],
                'final_val_accuracy': history['val_accuracies'][-1]
            }
        }, f, indent=2)
    
    logger.info(f"Historique sauvegardé : {history_path}")


if __name__ == "__main__":
    main()
