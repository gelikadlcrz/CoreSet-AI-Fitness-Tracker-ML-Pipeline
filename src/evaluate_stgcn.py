"""
evaluate_stgcn.py — Full held-out test-set evaluation for the CoreSet ST-GCN.

This version follows the clean versioned checkpoint workflow:

    checkpoint/
      latest_stgcn_checkpoint.txt

      stgcn_v001/
        final_stgcn_model.pth

      stgcn_v002/
        final_stgcn_model.pth

Default usage:

    python -m src.evaluate_stgcn

This resolves:

    checkpoint/latest_stgcn_checkpoint.txt

which should point to:

    checkpoint/stgcn_v###/final_stgcn_model.pth

Results are saved to:

    results/stgcn_v###/stgcn_evaluation_results.json
    results/stgcn_v###/stgcn_evaluation_report.txt
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.data.coreset_dataset import (
    ANGLE_FEATURE_DIM,
    CoreSetGCN_Dataset,
)
from src.models.stgcn_multitask import CoreSetSTGCN_MultiTask
from src.utils.metrics import CoreSetEvaluator
from src.utils.versioning import resolve_checkpoint_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(path: str | Path, device: torch.device):
    return torch.load(
        path,
        map_location=device,
        weights_only=False,
    )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def get_results_dir(checkpoint_path: str | Path) -> Path:
    """
    Save evaluation outputs under the matching ST-GCN version folder.

    Example:
        checkpoint/stgcn_v003/final_stgcn_model.pth
        results/stgcn_v003/
    """
    checkpoint_path = Path(checkpoint_path)
    version_name = checkpoint_path.parent.name

    if not version_name.startswith("stgcn_v"):
        version_name = "stgcn_eval"

    results_dir = Path("results") / version_name
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model,
    loader,
    feat_mean,
    feat_std,
    device,
):
    """
    Run a single inference pass over the test loader.
    """
    model.eval()

    all_logits = []
    all_labels = []
    all_density_pred = []
    all_density_gt = []

    feat_mean = feat_mean.to(device)
    feat_std = feat_std.to(device)

    with torch.no_grad():
        for inputs, labels, density_gts in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            density_gts = density_gts.to(device)

            inputs = (inputs - feat_mean) / feat_std

            logits, density_maps = model(inputs)

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_density_pred.append(density_maps.cpu().numpy())
            all_density_gt.append(density_gts.cpu().numpy())

    return (
        np.concatenate(all_logits, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(all_density_pred, axis=0),
        np.concatenate(all_density_gt, axis=0),
    )


# ---------------------------------------------------------------------------
# Counting Metrics
# ---------------------------------------------------------------------------

def compute_counting_metrics(
    density_pred: np.ndarray,
    density_gt: np.ndarray,
    evaluator: CoreSetEvaluator,
    postprocess: dict | None = None,
):
    """
    Convert density maps to counts with adaptive/tuned peak detection.
    """
    postprocess = postprocess or {
        "adaptive": True,
        "threshold_ratio": 0.30,
        "threshold_floor": 0.15,
        "min_distance": 10,
        "prominence_ratio": 0.05,
    }

    pred_counts = []
    gt_counts = []

    for pred_map, gt_map in zip(density_pred, density_gt):
        pred_count = CoreSetEvaluator.density_map_to_count(
            pred_map,
            threshold=postprocess.get("threshold"),
            min_distance=int(postprocess.get("min_distance", 10)),
            adaptive=bool(postprocess.get("adaptive", True)),
            threshold_ratio=float(postprocess.get("threshold_ratio", 0.30)),
            threshold_floor=float(postprocess.get("threshold_floor", 0.15)),
            prominence_ratio=float(postprocess.get("prominence_ratio", 0.05)),
        )

        pred_counts.append(pred_count)
        gt_counts.append(round(float(gt_map.sum())))

    mae = evaluator.calculate_normalized_mae(pred_counts, gt_counts)
    rmse = evaluator.calculate_rmse(pred_counts, gt_counts)
    obo = evaluator.calculate_obo_accuracy(pred_counts, gt_counts)

    return mae, rmse, obo, pred_counts, gt_counts


# ---------------------------------------------------------------------------
# Classification Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    evaluator: CoreSetEvaluator,
    class_names,
):
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)

    accuracy = float((preds == labels).mean())

    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="macro",
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    try:
        auc = float(
            roc_auc_score(
                labels,
                probs,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(
        labels,
        preds,
        labels=list(range(len(class_names))),
    )

    per_class = {}

    for i, cls in enumerate(class_names):
        per_class[cls] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    return {
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "macro_auc": auc,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Report Formatting
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 65


def format_report(
    cls_metrics: dict,
    mae: float,
    rmse: float,
    obo: float,
    class_names,
    checkpoint_path: str,
    postprocess: dict | None = None,
):
    macro_auc = cls_metrics["macro_auc"]

    if np.isnan(macro_auc):
        macro_auc_text = "N/A"
    else:
        macro_auc_text = f"{macro_auc:.4f}"

    lines = [
        SEPARATOR,
        "  CoreSet ST-GCN — Evaluation Report",
        f"  Checkpoint : {checkpoint_path}",
        SEPARATOR,
        "",
        "── Classification Metrics ──────────────────────────────────",
        f"  Top-1 Accuracy   : {cls_metrics['accuracy'] * 100:6.2f}%",
        f"  Macro AUC        : {macro_auc_text}"
        "   (one-vs-rest, following RepNet / Dwibedi et al., 2020)",
        f"  Macro Precision  : {cls_metrics['macro_precision']:.4f}",
        f"  Macro Recall     : {cls_metrics['macro_recall']:.4f}",
        f"  Macro F1-Score   : {cls_metrics['macro_f1']:.4f}"
        "   ← primary classification metric (Grandini et al., 2020)",
        "",
        "── Per-Class Breakdown ─────────────────────────────────────",
        f"  {'Class':<14} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}",
        "  " + "-" * 42,
    ]

    for cls in class_names:
        m = cls_metrics["per_class"][cls]

        lines.append(
            f"  {cls:<14} "
            f"{m['precision']:6.4f}  "
            f"{m['recall']:6.4f}  "
            f"{m['f1']:6.4f}  "
            f"{m['support']:5d}"
        )

    lines += [
        "",
        "── Confusion Matrix ────────────────────────────────────────",
        "  Rows = Ground Truth   Cols = Predicted",
        f"  Classes: {class_names}",
    ]

    cm = cls_metrics["confusion_matrix"]

    for row in cm:
        lines.append(
            "  " + " ".join(f"{v:5d}" for v in row)
        )

    postprocess = postprocess or {}

    lines += [
        "",
        "── Repetition Counting Metrics ─────────────────────────────",
        "  Peak Detection   : "
        f"adaptive={bool(postprocess.get('adaptive', True))}, "
        f"ratio={postprocess.get('threshold_ratio', 0.30)}, "
        f"floor={postprocess.get('threshold_floor', 0.15)}, "
        f"distance={postprocess.get('min_distance', 10)}",
        f"  Normalized MAE   : {mae:.4f}"
        "   (L1, per RepCount benchmark — Hu et al., 2022)",
        f"  RMSE             : {rmse:.4f}"
        "   (L2, flags catastrophic failures — Izadi et al., 2022)",
        f"  OBO Accuracy     : {obo * 100:.2f}%"
        "   (off-by-one tolerance — Hu et al., 2022)",
        "",
        SEPARATOR,
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    config_path="configs/stgcn_config.yaml",
    checkpoint_path="latest",
    count_threshold: float | None = None,
    count_threshold_ratio: float | None = None,
    count_threshold_floor: float | None = None,
    count_min_distance: int | None = None,
    count_prominence_ratio: float | None = None,
    adaptive_counting: bool = True,
):
    config = load_config(config_path)

    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path,
        checkpoint_root=config.get("checkpoint_dir", "checkpoint"),
        prefix="stgcn",
    )

    checkpoint_path = Path(checkpoint_path)

    device = select_device()
    evaluator = CoreSetEvaluator()
    class_names = evaluator.classes

    print(SEPARATOR)
    print("  CoreSet ST-GCN — Evaluation")
    print(f"  Device     : {device.type.upper()}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(SEPARATOR)

    # ----------------------------------------------------------------------
    # Load checkpoint
    # ----------------------------------------------------------------------

    ckpt = load_checkpoint(
        checkpoint_path,
        device,
    )

    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)

    # Prefer checkpoint-embedded config if available.
    checkpoint_config = ckpt.get("config")

    if checkpoint_config:
        config.update(checkpoint_config)

    # Counting post-processing:
    # Prefer tuned validation parameters stored in the checkpoint,
    # then allow CLI overrides.
    postprocess = ckpt.get(
        "counting_postprocess",
        {
            "adaptive": True,
            "threshold_ratio": 0.30,
            "threshold_floor": 0.15,
            "min_distance": 10,
            "prominence_ratio": 0.05,
        },
    ).copy()

    postprocess["adaptive"] = bool(adaptive_counting)

    if count_threshold is not None:
        postprocess["threshold"] = float(count_threshold)
        postprocess["adaptive"] = False

    if count_threshold_ratio is not None:
        postprocess["threshold_ratio"] = float(count_threshold_ratio)

    if count_threshold_floor is not None:
        postprocess["threshold_floor"] = float(count_threshold_floor)

    if count_min_distance is not None:
        postprocess["min_distance"] = int(count_min_distance)

    if count_prominence_ratio is not None:
        postprocess["prominence_ratio"] = float(count_prominence_ratio)

    # ----------------------------------------------------------------------
    # Build model
    # ----------------------------------------------------------------------

    model = CoreSetSTGCN_MultiTask(
        in_channels=ANGLE_FEATURE_DIM,
        num_classes=config["num_classes"],
        max_frames=config["max_frames"],
        node_count=config["node_count"],
    ).to(device)

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    print(
        f"  Loaded epoch {ckpt.get('epoch', '?')} checkpoint "
        f"(val loss {ckpt.get('val_loss', float('nan')):.4f}, "
        f"val acc {ckpt.get('val_accuracy', float('nan')) * 100:.2f}%)\n"
    )

    # ----------------------------------------------------------------------
    # Test Dataset
    # ----------------------------------------------------------------------

    split_file = os.path.join(
        "configs",
        "data_splits.json",
    )

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

    # ----------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------

    logits, labels, density_pred, density_gt = run_inference(
        model,
        test_loader,
        feat_mean,
        feat_std,
        device,
    )

    # ----------------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------------

    cls_metrics = compute_classification_metrics(
        logits,
        labels,
        evaluator,
        class_names,
    )

    mae, rmse, obo, pred_counts, gt_counts = compute_counting_metrics(
        density_pred,
        density_gt,
        evaluator,
        postprocess,
    )

    # ----------------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------------

    report = format_report(
        cls_metrics,
        mae,
        rmse,
        obo,
        class_names,
        str(checkpoint_path),
        postprocess,
    )

    print(report)

    # ----------------------------------------------------------------------
    # Save Results
    # ----------------------------------------------------------------------

    results_dir = get_results_dir(checkpoint_path)

    results_dict = {
        "checkpoint": checkpoint_path.as_posix(),
        "test_samples": len(test_dataset),
        "classification": cls_metrics,
        "counting": {
            "normalized_mae": mae,
            "rmse": rmse,
            "obo_accuracy": obo,
            "postprocess": postprocess,
            "pred_counts": pred_counts,
            "gt_counts": gt_counts,
        },
    }

    json_path = results_dir / "stgcn_evaluation_results.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    txt_path = results_dir / "stgcn_evaluation_report.txt"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n  Results saved to:")
    print(f"    {json_path}")
    print(f"    {txt_path}")
    print(SEPARATOR)

    return results_dict


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate CoreSet ST-GCN on the held-out test set."
    )

    parser.add_argument(
        "--config",
        default="configs/stgcn_config.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="Path to checkpoint, or 'latest' to use the latest final ST-GCN model",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional fixed peak threshold. Overrides adaptive counting.",
    )

    parser.add_argument(
        "--threshold-ratio",
        type=float,
        default=None,
        help="Adaptive threshold ratio multiplied by each sample's max peak.",
    )

    parser.add_argument(
        "--threshold-floor",
        type=float,
        default=None,
        help="Adaptive threshold floor.",
    )

    parser.add_argument(
        "--min-distance",
        type=int,
        default=None,
        help="Minimum frames between detected peaks.",
    )

    parser.add_argument(
        "--prominence-ratio",
        type=float,
        default=None,
        help="Minimum prominence as a fraction of the sample's max peak.",
    )

    parser.add_argument(
        "--no-adaptive-counting",
        action="store_true",
        help="Use fixed-threshold counting instead of adaptive counting.",
    )

    args = parser.parse_args()

    evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        count_threshold=args.threshold,
        count_threshold_ratio=args.threshold_ratio,
        count_threshold_floor=args.threshold_floor,
        count_min_distance=args.min_distance,
        count_prominence_ratio=args.prominence_ratio,
        adaptive_counting=not args.no_adaptive_counting,
    )