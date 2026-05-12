import os
import json

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.models.MultiTaskBiLSTM import MultiTaskBiLSTM
from src.data.bilstm_dataset import BiLSTMDataset
from src.utils.metrics import CoreSetEvaluator


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_splits(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation entry-point
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_bilstm(config_path: str) -> None:
    config = load_config(config_path)
    splits = load_splits("configs/data_splits.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
    # Sequences are already pre-standardised; no max_frames argument needed.
    test_dataset = BiLSTMDataset(
        config['data_dir'],
        splits['test'],
        augment=False,
        cache=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MultiTaskBiLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=0.5,
    ).to(device)

    checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_bilstm.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support checkpoints saved with or without the 'model_state_dict' wrapper.
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    evaluator = CoreSetEvaluator()

    # ── Inference ─────────────────────────────────────────────────────────────
    logits_all  : list[torch.Tensor] = []
    labels_all  : list[torch.Tensor] = []
    pred_counts : list[float]        = []
    true_counts : list[float]        = []

    with torch.no_grad():
        for x, y, reps in test_loader:
            x = x.to(device, non_blocking=True)

            logits, rep_pred = model(x)

            logits_all.append(logits.cpu())
            labels_all.append(y.cpu())
            pred_counts.extend(rep_pred.cpu().numpy().flatten().tolist())
            true_counts.extend(reps.cpu().numpy().flatten().tolist())

    logits_all = torch.cat(logits_all)   # (N, num_classes)
    labels_all = torch.cat(labels_all)   # (N,)

    # ── Classification metrics ────────────────────────────────────────────────
    acc = evaluator.calculate_classification_accuracy(logits_all, labels_all)

    probs     = torch.softmax(logits_all, dim=1).numpy()
    preds     = probs.argmax(axis=1)
    labels_np = labels_all.numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        labels_np, preds, average=None, zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels_np, preds, average='macro', zero_division=0
    )

    # ── AUC ───────────────────────────────────────────────────────────────────
    labels_bin = label_binarize(labels_np, classes=np.arange(config['num_classes']))
    macro_auc  = roc_auc_score(labels_bin, probs, average='macro', multi_class='ovr')

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(labels_np, preds)

    # ── Repetition-counting metrics ───────────────────────────────────────────
    mae  = evaluator.calculate_normalized_mae(pred_counts, true_counts)
    rmse = evaluator.calculate_rmse(pred_counts, true_counts)
    obo  = evaluator.calculate_obo_accuracy(pred_counts, true_counts)

    # ── Report ────────────────────────────────────────────────────────────────
    class_names = ['bench_press', 'pull_up', 'push_up', 'squat']

    print("\n" + "=" * 65)
    print("  BiLSTM Baseline — Evaluation Report")
    print(f"  Checkpoint : {checkpoint_path}")
    print("=" * 65)

    print("\n── Classification Metrics ──────────────────────────────────")
    print(f"  Top-1 Accuracy   : {acc * 100:6.2f}%")
    print(f"  Macro AUC        : {macro_auc:.4f}")
    print(f"  Macro Precision  : {macro_precision:.4f}")
    print(f"  Macro Recall     : {macro_recall:.4f}")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")

    print("\n── Per-Class Breakdown ─────────────────────────────────────")
    print(f"{'Class':15s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'N':>6s}")
    print("-" * 48)
    for idx, name in enumerate(class_names):
        print(
            f"{name:15s} "
            f"{precision[idx]:7.4f} "
            f"{recall[idx]:7.4f} "
            f"{f1[idx]:7.4f} "
            f"{support[idx]:6d}"
        )

    print("\n── Confusion Matrix ────────────────────────────────────────")
    print("  Rows = Ground Truth   Cols = Predicted")
    print(f"  Classes: {class_names}")
    for row in cm:
        print(" ", " ".join(f"{v:6d}" for v in row))

    print("\n── Repetition Counting Metrics ─────────────────────────────")
    print(f"  Normalized MAE   : {mae:.4f}")
    print(f"  RMSE             : {rmse:.4f}")
    print(f"  OBO Accuracy     : {obo * 100:.2f}%")
    print("\n" + "=" * 65)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    evaluate_bilstm("configs/bilstm_config.yaml")