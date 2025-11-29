"""
Evaluation script for AlexNet models on Alzheimer MRI datasets.

This script evaluates a trained AlexNet checkpoint on a dataset and computes
various classification metrics including accuracy, precision, recall, F1-score,
AUC, confusion matrices, and optionally ROC curves. Results are saved in a
structured directory with images, text reports, logs, and predictions CSV.

Features:
    - Computes per-class and overall metrics (micro/macro/weighted)
    - Confusion matrices (raw and normalized)
    - ROC curves (one-vs-rest)
    - Saves structured outputs in JSON, TXT, CSV, and PNG formats
    - Supports dataset sampling limits
    - Automatic dataset path resolution from command-line or environment

Directory structure for outputs:

    output_dir/
        images/
            confusion_matrix.png
            confusion_matrix_normalized.png
            roc_curves.png
        text/
            evaluation_report.json
            classification_report.txt
            summary.txt
            predictions.csv
        logs/
            evaluate.log

Usage:
    python evaluate.py --checkpoint_path ../checkpoints/alexnet_epoch35.pt
    python evaluate.py --checkpoint_path ../checkpoints/model.pt --data_path /custom/dataset --limit_samples 2000
    python evaluate.py --checkpoint_path ../checkpoints/model.pt --no_roc --no_csv
"""

import os
import argparse
import json
from pathlib import Path
import logging
from typing import Optional

from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import AlzheimerMRIDataset
from models import AlexNet

env_path = Path(__file__).parent.parent / "config" / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_transforms(mean: float, std: float):
    return transforms.Compose([
        transforms.Resize((200, 190)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])


def load_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    logger.info(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = AlexNet(num_classes=4, input_channels=1, dropout_rate=0.498070764044508)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.to(device)
    model.eval()
    logger.info("Model loaded and set to eval mode")
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    class_names,
    generate_roc: bool = True,
    save_csv: bool = True
):
    images_dir = output_dir / 'images'
    text_dir = output_dir / 'text'
    logs_dir = output_dir / 'logs'
    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(logs_dir / 'evaluate.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        logger.addHandler(file_handler)

    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    if len(all_probs) == 0:
        logger.error("No samples were processed. Dataset appears empty or unreadable.")
        empty_report = {'error': 'No samples processed', 'num_samples': 0}
        with open(text_dir / 'evaluation_report.json', 'w') as f:
            json.dump(empty_report, f, indent=2)
        logger.info(f"Saved empty evaluation report to: {text_dir / 'evaluation_report.json'}")
        return empty_report

    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    accuracy = float(accuracy_score(all_labels, all_preds))

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names)), zero_division=0
    )
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='micro', zero_division=0
    )
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    p_weighted, r_weighted, f_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    with np.errstate(all='ignore'):
        cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    try:
        y_bin = label_binarize(all_labels, classes=list(range(len(class_names))))
        auc_macro_ovr = float(roc_auc_score(y_bin, all_probs, average='macro', multi_class='ovr'))
        auc_micro_ovr = float(roc_auc_score(y_bin, all_probs, average='micro', multi_class='ovr'))
        auc_weighted_ovr = float(roc_auc_score(y_bin, all_probs, average='weighted', multi_class='ovr'))
        per_class_auc = {}
        for i, name in enumerate(class_names):
            try:
                per_class_auc[name] = float(roc_auc_score(y_bin[:, i], all_probs[:, i]))
            except Exception:
                per_class_auc[name] = None
    except Exception:
        auc_macro_ovr = None
        auc_micro_ovr = None
        auc_weighted_ovr = None
        per_class_auc = {name: None for name in class_names}

    per_class = {
        name: {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i]),
            'auc': per_class_auc.get(name)
        }
        for i, name in enumerate(class_names)
    }

    report = {
        'accuracy': accuracy,
        'precision': {'micro': float(p_micro), 'macro': float(p_macro), 'weighted': float(p_weighted)},
        'recall': {'micro': float(r_micro), 'macro': float(r_macro), 'weighted': float(r_weighted)},
        'f1': {'micro': float(f_micro), 'macro': float(f_macro), 'weighted': float(f_weighted)},
        'auc': {
            'macro_ovr': auc_macro_ovr,
            'micro_ovr': auc_micro_ovr,
            'weighted_ovr': auc_weighted_ovr,
            'per_class': per_class_auc
        },
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_normalized': cm_norm.tolist()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(text_dir / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    summary_path = text_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (micro/macro/weighted): {p_micro:.4f} / {p_macro:.4f} / {p_weighted:.4f}\n")
        f.write(f"Recall   (micro/macro/weighted): {r_micro:.4f} / {r_macro:.4f} / {r_weighted:.4f}\n")
        f.write(f"F1       (micro/macro/weighted): {f_micro:.4f} / {f_macro:.4f} / {f_weighted:.4f}\n")
        f.write(f"AUC (macro/micro/weighted ovr): {auc_macro_ovr if auc_macro_ovr else 'N/A'} / {auc_micro_ovr if auc_micro_ovr else 'N/A'} / {auc_weighted_ovr if auc_weighted_ovr else 'N/A'}\n\n")
        f.write("Per-class metrics:\n")
        for name in class_names:
            m = per_class[name]
            f.write(f"- {name}: precision={m['precision']:.4f}, recall={m['recall']:.4f}, f1={m['f1']:.4f}, auc={m['auc'] if m['auc'] is not None else 'N/A'}, support={m['support']}\n")

    cls_rep_path = text_dir / 'classification_report.txt'
    try:
        rep_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        with open(cls_rep_path, 'w') as f:
            f.write(rep_str)
    except Exception:
        pass

    if save_csv:
        try:
            import csv
            preds_csv_path = text_dir / 'predictions.csv'
            with open(preds_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['true_label', 'pred_label'] + [f'prob_{name}' for name in class_names])
                for tl, pl, probs in zip(all_labels, all_preds, all_probs):
                    writer.writerow([int(tl), int(pl)] + [float(p) for p in probs])
        except Exception:
            pass

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(images_dir / 'confusion_matrix.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Purples', xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.savefig(images_dir / 'confusion_matrix_normalized.png')
    plt.close()

    if generate_roc:
        try:
            y_bin = label_binarize(all_labels, classes=list(range(len(class_names))))
            plt.figure(figsize=(8, 6))
            for i in range(len(class_names)):
                fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
                plt.plot(fpr, tpr, label=f"{class_names[i]}")
            plt.plot([0, 1], [0, 1], 'k--', lw=0.5)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves (one-vs-rest)')
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(images_dir / 'roc_curves.png')
            plt.close()
        except Exception:
            pass

    return report


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint on dataset')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=False)
    parser.add_argument('--output_dir', type=str, default='../evaluation_results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--stats_path', type=str, default='../config/dataset_statistics.json')
    parser.add_argument('--limit_samples', type=int, default=None)
    parser.add_argument('--no_roc', action='store_true')
    parser.add_argument('--no_csv', action='store_true')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))

    stats_path = Path(args.stats_path)
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        mean = stats['normalization']['mean']
        std = stats['normalization']['std']
    else:
        mean = 0.2945542335510254
        std = 0.3180045485496521

    transform = create_transforms(mean, std)

    dataset_path: Optional[Path] = None
    if args.data_path and Path(args.data_path).exists():
        dataset_path = Path(args.data_path)
    if dataset_path is None:
        for env_var in ['EVAL_DATASET_PATH', 'DATASET_PATH']:
            p = os.getenv(env_var)
            if p and Path(p).exists():
                dataset_path = Path(p)
                break
    if dataset_path is None:
        raise FileNotFoundError("Failed to resolve dataset path. Provide --data_path or set DATASET_PATH/EVAL_DATASET_PATH.")

    dataset = AlzheimerMRIDataset(root_dir=str(dataset_path), transform=transform)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Aborting evaluation.")

    if args.limit_samples and args.limit_samples < len(dataset):
        rng = np.random.default_rng(seed=42)
        subset_indices = rng.choice(len(dataset), size=args.limit_samples, replace=False)
        samples = [dataset[i] for i in subset_indices]
        images = torch.stack([s[0] for s in samples])
        labels = torch.tensor([s[1] for s in samples], dtype=torch.long)
        from torch.utils.data import TensorDataset
        loader = DataLoader(TensorDataset(images, labels), batch_size=args.batch_size, shuffle=False)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=(device.type=='cuda'))

    model = load_model(args.checkpoint_path, device)
    report = evaluate(model, loader, device, Path(args.output_dir), dataset.CLASS_NAMES, generate_roc=not args.no_roc, save_csv=not args.no_csv)
    logger.info(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
