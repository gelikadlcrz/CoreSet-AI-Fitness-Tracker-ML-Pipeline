import torch
from torch.utils.data import DataLoader

import yaml
import json
import numpy as np
import os

from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score
)

from sklearn.preprocessing import label_binarize

from src.models.MultiTaskBiLSTM import MultiTaskBiLSTM
from src.data.bilstm_dataset import BiLSTMDataset
from src.utils.metrics import CoreSetEvaluator


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_splits(path):
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_bilstm(config_path):

    config = load_config(config_path)

    splits = load_splits("configs/data_splits.json")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    test_dataset = BiLSTMDataset(
        config['data_dir'],
        splits['test'],
        config['max_frames']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = MultiTaskBiLSTM(
        config['input_size'],
        config['hidden_size'],
        config['num_layers'],
        config['num_classes']
    ).to(device)

    checkpoint_path = os.path.join(
        config['checkpoint_dir'],
        'best_bilstm.pth'
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model.eval()

    evaluator = CoreSetEvaluator()

    # ---------------------------------------------------------
    # Storage
    # ---------------------------------------------------------

    logits_all = []
    labels_all = []

    pred_counts = []
    true_counts = []

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    with torch.no_grad():

        for x, y, reps in test_loader:

            x = x.to(device)


            logits, rep_pred = model(x)

            logits_all.append(logits.cpu())

            labels_all.append(y.cpu())

            pred_counts.extend(
                rep_pred.cpu().numpy().flatten()
            )

            true_counts.extend(
                reps.cpu().numpy().flatten()
            )

    logits_all = torch.cat(logits_all)

    labels_all = torch.cat(labels_all)

    # ---------------------------------------------------------
    # Classification Metrics
    # ---------------------------------------------------------

    acc = evaluator.calculate_classification_accuracy(
        logits_all,
        labels_all
    )

    probs = torch.softmax(logits_all, dim=1).numpy()

    preds = probs.argmax(axis=1)

    labels_np = labels_all.numpy()

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels_np,
            preds,
            average=None,
            zero_division=0
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels_np,
            preds,
            average='macro',
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # AUC
    # ---------------------------------------------------------

    labels_bin = label_binarize(
        labels_np,
        classes=np.arange(config['num_classes'])
    )

    macro_auc = roc_auc_score(
        labels_bin,
        probs,
        average='macro',
        multi_class='ovr'
    )

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    cm = confusion_matrix(labels_np, preds)

    # ---------------------------------------------------------
    # Counting Metrics
    # ---------------------------------------------------------

    mae = evaluator.calculate_normalized_mae(
        pred_counts,
        true_counts
    )

    rmse = evaluator.calculate_rmse(
        pred_counts,
        true_counts
    )

    obo = evaluator.calculate_obo_accuracy(
        pred_counts,
        true_counts
    )

    # ---------------------------------------------------------
    # Class Names
    # ---------------------------------------------------------

    class_names = [
        'bench_press',
        'pull_up',
        'push_up',
        'squat'
    ]

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print("\n" + "=" * 65)

    print("  CoreSet BiLSTM — Evaluation Report")

    print(f"  Checkpoint : {checkpoint_path}")

    print("=" * 65)

    # ---------------------------------------------------------
    # Classification Metrics
    # ---------------------------------------------------------

    print("\n── Classification Metrics ──────────────────────────────────")

    print(f"  Top-1 Accuracy   : {acc*100:6.2f}%")

    print(f"  Macro AUC        : {macro_auc:.4f}")

    print(f"  Macro Precision  : {macro_precision:.4f}")

    print(f"  Macro Recall     : {macro_recall:.4f}")

    print(f"  Macro F1-Score   : {macro_f1:.4f}")

    # ---------------------------------------------------------
    # Per-Class
    # ---------------------------------------------------------

    print("\n── Per-Class Breakdown ─────────────────────────────────────")

    print(f"{'Class':15s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'N':>6s}")

    print("-" * 48)

    for idx, class_name in enumerate(class_names):

        print(
            f"{class_name:15s} "
            f"{precision[idx]:7.4f} "
            f"{recall[idx]:7.4f} "
            f"{f1[idx]:7.4f} "
            f"{support[idx]:6d}"
        )

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    print("\n── Confusion Matrix ────────────────────────────────────────")

    print("  Rows = Ground Truth   Cols = Predicted")

    print(f"  Classes: {class_names}")

    for row in cm:
        print(" ", " ".join(f"{v:6d}" for v in row))

    # ---------------------------------------------------------
    # Counting Metrics
    # ---------------------------------------------------------

    print("\n── Repetition Counting Metrics ─────────────────────────────")

    print(f"  Normalized MAE   : {mae:.4f}")

    print(f"  RMSE             : {rmse:.4f}")

    print(f"  OBO Accuracy     : {obo*100:.2f}%")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    evaluate_bilstm("configs/bilstm_config.yaml")