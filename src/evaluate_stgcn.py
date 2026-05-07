"""
evaluate_stgcn.py — Full held-out test-set evaluation for the CoreSet ST-GCN.

Metrics computed (per the Validation, Testing & Deployment methodology):

  Counting metrics  (density-map → rep-count via peak detection):
    • Normalized MAE   — L1 fidelity, robust to outliers (Izadi et al., 2022)
    • RMSE             — L2 penalty, flags catastrophic failures
    • OBO Accuracy     — off-by-one tolerance (Hu et al., 2022)

  Classification metrics:
    • Top-1 Accuracy   — baseline scalar
    • AUC              — periodicity detection (Dwibedi et al., 2020 / RepNet)
    • Precision        — per-class & macro-averaged (Grandini et al., 2020)
    • Recall           — per-class & macro-averaged
    • Macro F1-Score   — primary unified classification metric

  Results are printed to stdout and saved to:
    • results/stgcn_evaluation_results.json   (machine-readable)
    • results/stgcn_evaluation_report.txt     (human-readable)

Usage:
    python evaluate_stgcn.py
    python evaluate_stgcn.py --config configs/stgcn_config.yaml
                             --checkpoint checkpoint/best_stgcn_model.pth
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from src.data.coreset_dataset import CoreSetGCN_Dataset, ANGLE_FEATURE_DIM
from src.models.stgcn_multitask import CoreSetSTGCN_MultiTask
from src.utils.metrics import CoreSetEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Inference pass
# ---------------------------------------------------------------------------

def run_inference(model, loader, feat_mean, feat_std, device):
    """
    Run a single forward pass over the DataLoader.

    Returns
    -------
    all_logits       : np.ndarray  (N, num_classes)
    all_labels       : np.ndarray  (N,)
    all_density_pred : np.ndarray  (N, max_frames)  — sigmoid outputs
    all_density_gt   : np.ndarray  (N, max_frames)  — ground-truth maps
    """
    model.eval()

    all_logits, all_labels = [], []
    all_density_pred, all_density_gt = [], []

    with torch.no_grad():
        for inputs, labels, density_gts in loader:
            inputs      = inputs.to(device)
            inputs      = (inputs - feat_mean.to(device)) / feat_std.to(device)
            labels      = labels.to(device)
            density_gts = density_gts.to(device)

            logits, density_maps = model(inputs)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_density_pred.append(density_maps.cpu().numpy())
            all_density_gt.append(density_gts.cpu().numpy())

    return (
        np.concatenate(all_logits,       axis=0),
        np.concatenate(all_labels,       axis=0),
        np.concatenate(all_density_pred, axis=0),
        np.concatenate(all_density_gt,   axis=0),
    )


# ---------------------------------------------------------------------------
# Counting metrics
# ---------------------------------------------------------------------------

def compute_counting_metrics(density_pred: np.ndarray,
                              density_gt: np.ndarray,
                              evaluator: CoreSetEvaluator):
    """
    Convert per-sample density maps to integer rep-counts and compute
    MAE, RMSE, and OBO accuracy.

    Ground-truth counts are derived from the density-map integrals
    (sum of the GT map ≈ rep_count by construction in coreset_dataset.py).
    """
    pred_counts = []
    gt_counts   = []

    for pred_map, gt_map in zip(density_pred, density_gt):
        # Predicted count: peak-detection on sigmoid output
        pred_count = CoreSetEvaluator.density_map_to_count(pred_map)
        pred_counts.append(pred_count)

        # Ground-truth count: integral of the normalised Gaussian map
        # (sum ≈ rep_count by construction in _get_ground_truth_density_map)
        gt_counts.append(round(float(gt_map.sum())))

    mae  = evaluator.calculate_normalized_mae(pred_counts, gt_counts)
    rmse = evaluator.calculate_rmse(pred_counts, gt_counts)
    obo  = evaluator.calculate_obo_accuracy(pred_counts, gt_counts)

    return mae, rmse, obo, pred_counts, gt_counts


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(logits: np.ndarray,
                                    labels: np.ndarray,
                                    evaluator: CoreSetEvaluator,
                                    class_names):
    """
    Compute Top-1 accuracy, macro F1, per-class precision/recall/F1,
    and macro-averaged AUC (one-vs-rest).

    AUC follows the periodicity-detection framing of RepNet
    (Dwibedi et al., 2020): the density-map sigmoid values are the
    classifier scores and the binary GT (any rep in this frame?) is the
    label.  Here we compute AUC on the *classification* softmax probabilities
    using one-vs-rest multi-class ROC AUC, which is the natural extension
    for a 4-class exercise taxonomy.
    """
    # Softmax probabilities
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)          # (N, C)

    preds = probs.argmax(axis=1)

    # Top-1 accuracy
    accuracy = float((preds == labels).mean())

    # Macro precision / recall / F1  (Grandini et al., 2020)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(len(class_names))),
        zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro",
        labels=list(range(len(class_names))),
        zero_division=0
    )

    # Multi-class AUC (one-vs-rest)
    # If only one class is present in labels sklearn will raise; guard it.
    try:
        auc = float(roc_auc_score(
            labels, probs, multi_class="ovr", average="macro"
        ))
    except ValueError:
        auc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))

    per_class = {}
    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i]),
        }

    return {
        "accuracy":        accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall":    float(macro_recall),
        "macro_f1":        float(macro_f1),
        "macro_auc":       auc,
        "per_class":       per_class,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 65

def format_report(cls_metrics: dict,
                  mae: float, rmse: float, obo: float,
                  class_names,
                  checkpoint_path: str) -> str:
    lines = [
        SEPARATOR,
        "  CoreSet ST-GCN — Evaluation Report",
        f"  Checkpoint : {checkpoint_path}",
        SEPARATOR,
        "",
        "── Classification Metrics ──────────────────────────────────",
        f"  Top-1 Accuracy   : {cls_metrics['accuracy']*100:6.2f}%",
        f"  Macro AUC        : {cls_metrics['macro_auc']:.4f}"
          "   (one-vs-rest, following RepNet / Dwibedi et al., 2020)",
        f"  Macro Precision  : {cls_metrics['macro_precision']:.4f}",
        f"  Macro Recall     : {cls_metrics['macro_recall']:.4f}",
        f"  Macro F1-Score   : {cls_metrics['macro_f1']:.4f}"
          "   ← primary classification metric (Grandini et al., 2020)",
        "",
        "── Per-Class Breakdown ─────────────────────────────────────",
        f"  {'Class':<14} {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'N':>5}",
        "  " + "-" * 42,
    ]

    for cls in class_names:
        m = cls_metrics["per_class"][cls]
        lines.append(
            f"  {cls:<14} {m['precision']:6.4f}  {m['recall']:6.4f}"
            f"  {m['f1']:6.4f}  {m['support']:5d}"
        )

    lines += [
        "",
        "── Confusion Matrix ────────────────────────────────────────",
        f"  Rows = Ground Truth   Cols = Predicted",
        f"  Classes: {class_names}",
    ]
    cm = cls_metrics["confusion_matrix"]
    for row in cm:
        lines.append("  " + "  ".join(f"{v:5d}" for v in row))

    lines += [
        "",
        "── Repetition Counting Metrics ─────────────────────────────",
        f"  Normalized MAE   : {mae:.4f}"
          "   (L1, per RepCount benchmark — Hu et al., 2022)",
        f"  RMSE             : {rmse:.4f}"
          "   (L2, flags catastrophic failures — Izadi et al., 2022)",
        f"  OBO Accuracy     : {obo*100:.2f}%"
          "   (off-by-one tolerance — Hu et al., 2022)",
        "",
        SEPARATOR,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(config_path: str = "configs/stgcn_config.yaml",
             checkpoint_path: str = "checkpoint/best_stgcn_model.pth"):

    config     = load_config(config_path)
    device     = select_device()
    evaluator  = CoreSetEvaluator()
    class_names = evaluator.classes

    print(SEPARATOR)
    print("  CoreSet ST-GCN — Evaluation")
    print(f"  Device     : {device.type.upper()}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(SEPARATOR)

    # ------------------------------------------------------------------ #
    #  Load checkpoint                                                     #
    # ------------------------------------------------------------------ #
    ckpt      = load_checkpoint(checkpoint_path, device)
    feat_mean = ckpt["feat_mean"].to(device)   # (1, C, 1, 1, 1)
    feat_std  = ckpt["feat_std"].to(device)    # (1, C, 1, 1, 1)

    # ------------------------------------------------------------------ #
    #  Build model and load weights                                        #
    # ------------------------------------------------------------------ #
    model = CoreSetSTGCN_MultiTask(
        in_channels=ANGLE_FEATURE_DIM,
        num_classes=config["num_classes"],
        max_frames=config["max_frames"],
        node_count=config["node_count"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint  "
          f"(val loss {ckpt.get('val_loss', float('nan')):.4f}, "
          f"val acc {ckpt.get('val_accuracy', float('nan'))*100:.2f}%)\n")

    # ------------------------------------------------------------------ #
    #  Test DataLoader                                                     #
    # ------------------------------------------------------------------ #
    split_file = os.path.join("configs", "data_splits.json")

    test_dataset = CoreSetGCN_Dataset(
        data_dir=config["data_dir"],
        split_file=split_file,
        split_type="test",
        max_frames=config["max_frames"],
        augment=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    print(f"  Test samples : {len(test_dataset)}")
    print("-" * 65)

    # ------------------------------------------------------------------ #
    #  Inference                                                           #
    # ------------------------------------------------------------------ #
    logits, labels, density_pred, density_gt = run_inference(
        model, test_loader, feat_mean, feat_std, device
    )

    # ------------------------------------------------------------------ #
    #  Compute metrics                                                     #
    # ------------------------------------------------------------------ #
    cls_metrics = compute_classification_metrics(
        logits, labels, evaluator, class_names
    )

    mae, rmse, obo, pred_counts, gt_counts = compute_counting_metrics(
        density_pred, density_gt, evaluator
    )

    # ------------------------------------------------------------------ #
    #  Print & save results                                                #
    # ------------------------------------------------------------------ #
    report = format_report(
        cls_metrics, mae, rmse, obo, class_names, checkpoint_path
    )
    print(report)

    os.makedirs("results", exist_ok=True)

    # JSON
    results_dict = {
        "checkpoint":  checkpoint_path,
        "test_samples": len(test_dataset),
        "classification": cls_metrics,
        "counting": {
            "normalized_mae": mae,
            "rmse":           rmse,
            "obo_accuracy":   obo,
        },
    }
    json_path = "results/stgcn_evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    # Text report
    txt_path = "results/stgcn_evaluation_report.txt"
    with open(txt_path, "w") as f:
        f.write(report)

    print(f"\n  Results saved to:")
    print(f"    {json_path}")
    print(f"    {txt_path}")
    print(SEPARATOR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CoreSet ST-GCN on the held-out test set."
    )
    parser.add_argument(
        "--config",
        default="configs/stgcn_config.yaml",
        help="Path to YAML config (default: configs/stgcn_config.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint/best_stgcn_model.pth",
        help="Path to best model checkpoint (default: checkpoint/best_stgcn_model.pth)",
    )
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)