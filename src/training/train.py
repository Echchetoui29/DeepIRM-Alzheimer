"""
Script principal pour l'entraînement du modèle AlexNet sur le jeu de données d'IRM pour la maladie d'Alzheimer.

Ce script orchestre l'ensemble du processus :
1.  Chargement de la configuration (chemins, etc.) depuis un fichier .env.
2.  Configuration du logging pour le suivi.
3.  Définition de la classe `Trainer` qui encapsule toute la logique d'entraînement,
    de validation, de sauvegarde/chargement de checkpoints et d'arrêt anticipé (early stopping).
4.  Création des transformations de données (data augmentation pour l'entraînement).
5.  Recherche, chargement et division du jeu de données en ensembles d'entraînement et de validation.
6.  Création des DataLoaders PyTorch.
7.  Instanciation du modèle AlexNet.
8.  Lancement de l'entraînement via la classe `Trainer`.
9.  Sauvegarde de l'historique d'entraînement dans un fichier JSON.
"""

import os
import json
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from models import create_alexnet, get_model_summary
from dataset import AlzheimerMRIDataset

# Charger les variables d'environnement depuis le fichier .env
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Classe encapsulant la logique d'entraînement pour le modèle AlexNet."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        model_name: str = "alexnet_alzheimer"
    ):
        """
        Initialise l'entraîneur.

        Args:
            model (nn.Module): Le modèle PyTorch à entraîner.
            device (torch.device): L'appareil sur lequel effectuer l'entraînement (CPU ou GPU).
            learning_rate (float): Le taux d'apprentissage pour l'optimiseur.
            weight_decay (float): La pondération de la régularisation L2 (weight decay).
            model_name (str): Le nom de base pour les fichiers de checkpoint sauvegardés.
        """
        self.model = model
        self.device = device
        self.model_name = model_name
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
        
        # Planificateur pour ajuster le taux d'apprentissage
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10,
            gamma=0.1
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Créer le répertoire pour les checkpoints
        self.checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Exécute une seule époque d'entraînement.

        Args:
            train_loader (DataLoader): Le DataLoader pour les données d'entraînement.

        Returns:
            Tuple[float, float]: La perte moyenne et la précision sur l'époque.
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Entraînement", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Passe avant (Forward)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Passe arrière (Backward) et optimisation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Calcul des statistiques
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.update(1)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float, Dict]:
        """
        Évalue le modèle sur le jeu de données de validation.

        Args:
            val_loader (DataLoader): Le DataLoader pour les données de validation.

        Returns:
            Tuple[float, float, Dict]: La perte moyenne, la précision globale,
                                      et un dictionnaire de métriques par classe.
        """
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        class_correct, class_total = {}, {}
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Suivi des métriques par classe
                for label, pred in zip(labels, predicted):
                    label_idx = label.item()
                    is_correct = (pred == label).item()
                    
                    class_total.setdefault(label_idx, 0)
                    class_correct.setdefault(label_idx, 0)
                    
                    class_total[label_idx] += 1
                    if is_correct:
                        class_correct[label_idx] += 1
                
                pbar.update(1)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        # Calcul de la précision par classe
        per_class_metrics = {
            class_idx: {
                'correct': class_correct[class_idx],
                'total': class_total[class_idx],
                'accuracy': 100 * class_correct[class_idx] / class_total[class_idx]
            } for class_idx in class_total
        }
        
        return avg_loss, accuracy, per_class_metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        patience: int = 15
    ) -> Dict:
        """
        Lance le processus d'entraînement complet avec arrêt anticipé (early stopping).

        Args:
            train_loader (DataLoader): Le DataLoader pour l'entraînement.
            val_loader (DataLoader): Le DataLoader pour la validation.
            num_epochs (int): Le nombre maximum d'époques.
            patience (int): Le nombre d'époques à attendre sans amélioration
                          de la précision de validation avant d'arrêter l'entraînement.

        Returns:
            Dict: Un dictionnaire contenant l'historique de l'entraînement.
        """
        print("\n" + "=" * 70)
        print("DÉBUT DE L'ENTRAÎNEMENT")
        print("=" * 70)
        print(f"Appareil : {self.device}")
        print(f"Modèle : {self.model.__class__.__name__}")
        print(f"Époques : {num_epochs}")
        print(f"Patience (Early Stopping) : {patience}")
        print("=" * 70 + "\n")
        
        best_val_accuracy = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            val_loss, val_acc, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            self.scheduler.step()
            
            print(f"\nÉpoque [{epoch+1}/{num_epochs}]")
            print(f"  Perte Entraînement : {train_loss:.4f} | Précision Entraînement : {train_acc:.2f}%")
            print(f"  Perte Validation :   {val_loss:.4f} | Précision Validation :   {val_acc:.2f}%")
            
            # Logique d'arrêt anticipé
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                patience_counter = 0
                self.save_checkpoint(epoch, val_acc)
                print(f"  ✓ Meilleur modèle sauvegardé (Précision Val : {val_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️  Arrêt anticipé déclenché (patience : {patience})")
                    break
        
        print("\n" + "=" * 70)
        print("ENTRAÎNEMENT TERMINÉ")
        print("=" * 70 + "\n")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_accuracy': best_val_accuracy
        }
    
    def save_checkpoint(self, epoch: int, val_accuracy: float) -> None:
        """Sauvegarde un checkpoint du modèle."""
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch+1}_acc_{val_accuracy:.2f}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_accuracy': val_accuracy
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Charge un modèle depuis un fichier de checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✓ Checkpoint chargé : {checkpoint_path}")


def create_data_transforms(stats_path: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Crée les pipelines de transformation pour l'entraînement et la validation.

    Args:
        stats_path (str): Le chemin vers le fichier JSON contenant les statistiques
                          (moyenne, écart-type) du jeu de données.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: Un tuple contenant les transformations
                                                       pour l'entraînement et la validation.
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    mean = stats['normalization']['mean']
    std = stats['normalization']['std']
    
    # Transformations pour l'entraînement (avec data augmentation)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    # Transformations pour la validation (juste normalisation)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std]),
    ])
    
    return train_transform, val_transform


def main():
    """Fonction principale qui orchestre le processus d'entraînement."""
    # Configuration de l'appareil
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilisation de l'appareil : {device}")
    
    # Chargement des chemins depuis les variables d'environnement
    DATASET_BASE = os.getenv("DATASET_PATH", "c:/Users/ghout/Desktop/augmented-alzheimer-mri-dataset")
    stats_path = Path(__file__).parent.parent / "config" / "dataset_statistics.json"
    
    # Recherche automatique du répertoire du jeu de données
    datasets_to_check = [
        ("BalancedBalancedFromAugmented", "BalancedBalancedFromAugmented"),
        ("BalancedAlzheimerDataset", "Balanced"),
        ("BalancedFromCombined", "BalancedFromCombined"),
        ("BalancedFromAugmented", "BalancedFromAugmented"),
        ("CombinedAlzheimerDataset", "Combined"),
    ]
    
    dataset_path = None
    for dataset_dir, name in datasets_to_check:
        full_path = os.path.join(DATASET_BASE, dataset_dir)
        if os.path.exists(full_path):
            dataset_path = full_path
            logger.info(f"✓ Jeu de données trouvé : {name}")
            break
    
    if not dataset_path:
        logger.error("Aucun jeu de données trouvé dans les chemins spécifiés !")
        return
    
    # Création des transformations
    train_transform, val_transform = create_data_transforms(str(stats_path))
    
    # Création du jeu de données complet
    logger.info("Chargement du jeu de données...")
    full_dataset = AlzheimerMRIDataset(root_dir=dataset_path, transform=train_transform)
    
    # Division en ensembles d'entraînement et de validation (80-20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Appliquer les transformations de validation uniquement à l'ensemble de validation
    val_dataset.dataset.transform = val_transform
    
    # Création des DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"Échantillons d'entraînement : {len(train_dataset)}")
    logger.info(f"Échantillons de validation : {len(val_dataset)}")
    
    # Création du modèle
    model = create_alexnet(num_classes=4, input_channels=1, lite=False)
    model.to(device)
    get_model_summary(model)
    
    # Instanciation et lancement de l'entraîneur
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=0.001,
        weight_decay=0.0001,
        model_name="alexnet_alzheimer"
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        patience=15
    )
    
    # Sauvegarde de l'historique d'entraînement
    history_path = Path(__file__).parent.parent / "checkpoints" / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f"✓ Historique d'entraînement sauvegardé : {history_path}")


if __name__ == "__main__":
    main()