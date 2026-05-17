"""
evaluate_bilstm.py — Full held-out test-set evaluation for the CoreSet BiLSTM.

Follows the clean versioned checkpoint workflow:

    saved_models/
      latest_bilstm_checkpoint.txt

      bilstm_v001/
        final_bilstm_model.pth

      bilstm_v002/
        final_bilstm_model.pth

Default usage:

    python -m src.evaluate_bilstm

This resolves:

    saved_models/latest_bilstm_checkpoint.txt

which should point to:

    saved_models/bilstm_v###/final_bilstm_model.pth

Results are saved to:

    results/bilstm_v###/bilstm_evaluation_results.json
    results/bilstm_v###/bilstm_evaluation_report.txt
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
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from src.data.bilstm_dataset import BiLSTMDataset
from src.models.MultiTaskBiLSTM import MultiTaskBiLSTM
from src.utils.metrics import CoreSetEvaluator
from src.utils.bilstm_versioning import resolve_checkpoint_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Evaluation entry-point
# ---------------------------------------------------------------------------

def evaluate_bilstm(
    config_path: str = "configs/bilstm_config.yaml",
    checkpoint: str = "latest",
) -> None:
    config = load_config(config_path)
    splits = load_splits("configs/data_splits.json")

    device = select_device()

    checkpoint_root = Path(config["checkpoint_dir"])
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=checkpoint,
        checkpoint_root=checkpoint_root,
    )

    print("CoreSet BiLSTM — Evaluation")
    print("=" * 70)
    print(f"  Hardware     : {device.type.upper()}")
    print(f"  Checkpoint   : {checkpoint_path.as_posix()}")
    print("-" * 70)

    # ── Dataset ───────────────────────────────────────────────────────────
    test_dataset = BiLSTMDataset(
        config["data_dir"], splits["test"], augment=False, cache=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=int(config.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )

    print(f"  Test samples : {len(test_dataset)}")
    print("-" * 70)

    # ── Model ─────────────────────────────────────────────────────────────
    ckpt        = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config", config)

    model = MultiTaskBiLSTM(
        input_size=ckpt_config["input_size"],
        hidden_size=ckpt_config["hidden_size"],
        num_layers=ckpt_config["num_layers"],
        num_classes=ckpt_config["num_classes"],
        dropout=float(ckpt_config.get("dropout", 0.5)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    evaluator   = CoreSetEvaluator()
    num_classes = ckpt_config["num_classes"]

    # ── Inference ─────────────────────────────────────────────────────────
    all_logits:  list[torch.Tensor] = []
    all_labels:  list[torch.Tensor] = []
    pred_counts: list[float]        = []
    true_counts: list[float]        = []

    with torch.no_grad():
        for x, y, reps in test_loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            logits, rep_pred = model(x)
            all_logits.append(logits.float().cpu())
            all_labels.append(y.cpu())
            pred_counts.extend(rep_pred.cpu().numpy().flatten().tolist())
            true_counts.extend(reps.cpu().numpy().flatten().tolist())

    logits_all = torch.cat(all_logits)
    labels_all = torch.cat(all_labels)

    # ── Classification metrics ─────────────────────────────────────────────
    acc       = evaluator.calculate_classification_accuracy(logits_all, labels_all)
    probs     = torch.softmax(logits_all, dim=1).numpy()
    preds     = probs.argmax(axis=1)
    labels_np = labels_all.numpy()

    precision, recall, f1, support = precision_recall_fscore_support(
        labels_np, preds, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels_np, preds, average="macro", zero_division=0
    )
    labels_bin = label_binarize(labels_np, classes=np.arange(num_classes))
    macro_auc  = roc_auc_score(labels_bin, probs, average="macro", multi_class="ovr")
    cm         = confusion_matrix(labels_np, preds)

    # ── Counting metrics ───────────────────────────────────────────────────
    nmae = evaluator.calculate_normalized_mae(pred_counts, true_counts)
    rmse = evaluator.calculate_rmse(pred_counts, true_counts)
    obo  = evaluator.calculate_obo_accuracy(pred_counts, true_counts)

    class_names = ["bench_press", "pull_up", "push_up", "squat"]

    # ── Report ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BiLSTM Baseline — Evaluation Report")
    print(f"  Checkpoint : {checkpoint_path.as_posix()}")
    print("=" * 70)

    print("\n── Classification Metrics ──────────────────────────────────────────")
    print(f"  Top-1 Accuracy   : {acc * 100:6.2f}%")
    print(f"  Macro AUC        : {macro_auc:.4f}")
    print(f"  Macro Precision  : {macro_precision:.4f}")
    print(f"  Macro Recall     : {macro_recall:.4f}")
    print(f"  Macro F1-Score   : {macro_f1:.4f}")

    print("\n── Per-Class Breakdown ─────────────────────────────────────────────")
    print(f"{'Class':15s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'N':>6s}")
    print("-" * 50)
    for i, name in enumerate(class_names):
        print(f"{name:15s} {precision[i]:7.4f} {recall[i]:7.4f} {f1[i]:7.4f} {support[i]:6d}")

    print("\n── Confusion Matrix ────────────────────────────────────────────────")
    print("  Rows = Ground Truth   Cols = Predicted")
    print(f"  Classes: {class_names}")
    for row in cm:
        print(" ", " ".join(f"{v:6d}" for v in row))

    print("\n── Repetition Counting Metrics ─────────────────────────────────────")
    print(f"  Normalized MAE   : {nmae:.4f}")
    print(f"  RMSE             : {rmse:.4f}")
    print(f"  OBO Accuracy     : {obo * 100:.2f}%")
    print("\n" + "=" * 70)

    # ── Save outputs ──────────────────────────────────────────────────────
    results_dir = Path("results") / checkpoint_path.parent.name
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint":      checkpoint_path.as_posix(),
        "test_accuracy":   round(acc,             6),
        "macro_auc":       round(macro_auc,        6),
        "macro_precision": round(macro_precision,  6),
        "macro_recall":    round(macro_recall,     6),
        "macro_f1":        round(macro_f1,         6),
        "per_class": {
            name: {
                "precision": round(float(precision[i]), 6),
                "recall":    round(float(recall[i]),    6),
                "f1":        round(float(f1[i]),        6),
                "support":   int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
        "test_obo":        round(obo,  6),
        "test_nmae":       round(nmae, 6),
        "test_rmse":       round(rmse, 6),
    }

    results_json = results_dir / "bilstm_evaluation_results.json"
    report_txt   = results_dir / "bilstm_evaluation_report.txt"

    with results_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report_lines = [
        "CoreSet BiLSTM — Evaluation Report",
        f"Checkpoint : {checkpoint_path.as_posix()}",
        "",
        f"Top-1 Accuracy : {acc * 100:.2f}%",
        f"Macro AUC      : {macro_auc:.4f}",
        f"Macro F1       : {macro_f1:.4f}",
        f"OBO Accuracy   : {obo * 100:.2f}%",
        f"nMAE           : {nmae:.4f}",
        f"RMSE           : {rmse:.4f}",
    ]
    report_txt.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"  Results JSON : {results_json.as_posix()}")
    print(f"  Report TXT   : {report_txt.as_posix()}")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM checkpoint.")
    parser.add_argument("--config",     default="configs/bilstm_config.yaml")
    parser.add_argument("--checkpoint", default="latest",
                        help="Path to .pth or 'latest' (default).")
    args = parser.parse_args()
    evaluate_bilstm(config_path=args.config, checkpoint=args.checkpoint)