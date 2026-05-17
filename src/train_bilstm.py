"""
train_bilstm.py — CoreSet BiLSTM Baseline training.

Clean checkpoint workflow:
- Each training run creates a new version folder:

      saved_models/bilstm_v001/
      saved_models/bilstm_v002/
      saved_models/bilstm_v003/

- Each version folder contains only one .pth model file:

      final_bilstm_model.pth

This one final checkpoint is used for:
    1. evaluation
    2. ONNX / TFLite export
    3. mobile deployment

The script does NOT create:
    best_bilstm.pth
    best_bilstm_baseline.pth
    epoch_*.pth
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from src.models.MultiTaskBiLSTM import MultiTaskBiLSTM
from src.data.bilstm_dataset import BiLSTMDataset
from src.utils.metrics import CoreSetEvaluator
from src.utils.bilstm_versioning import (
    create_versioned_run_dir,
    write_latest_checkpoint_pointer,
    write_latest_run_pointer,
)


# ---------------------------------------------------------------------------
# Config / Reproducibility
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_splits(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

def make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    config: dict,
) -> DataLoader:
    """Windows-safe DataLoader. num_workers defaults to 0."""
    requested_workers = int(config.get("num_workers", 0))
    if requested_workers <= 0:
        num_workers = 0
    else:
        num_workers = min(requested_workers, max(os.cpu_count() or 1, 1))

    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": shuffle,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))

    return DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def append_history_row(history_path: Path, row: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Evaluation Helper
# ---------------------------------------------------------------------------

def run_split_evaluation(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion_cls: nn.Module,
    criterion_reg: nn.Module,
    lambda_cls: float,
    lambda_reg: float,
    device: torch.device,
    use_amp: bool,
) -> dict:
    model.eval()

    total_loss = cls_loss_sum = reg_loss_sum = 0.0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    pred_counts: list[float] = []
    true_counts: list[float] = []

    def autocast_ctx():
        return torch.cuda.amp.autocast(enabled=True) if use_amp else nullcontext()

    with torch.no_grad():
        for x, y, reps in loader:
            x    = x.to(device, non_blocking=(device.type == "cuda"))
            y    = y.long().to(device, non_blocking=(device.type == "cuda"))
            reps = reps.float().unsqueeze(1).to(device, non_blocking=(device.type == "cuda"))

            with autocast_ctx():
                logits, rep_pred = model(x)
                loss_cls = criterion_cls(logits, y)
                loss_reg = criterion_reg(rep_pred, reps)
                loss     = lambda_cls * loss_cls + lambda_reg * loss_reg

            total_loss   += float(loss.item())
            cls_loss_sum += float(loss_cls.item())
            reg_loss_sum += float(loss_reg.item())

            all_logits.append(logits.float().cpu())
            all_labels.append(y.cpu())
            pred_counts.extend(rep_pred.cpu().numpy().flatten().tolist())
            true_counts.extend(reps.cpu().numpy().flatten().tolist())

    n = max(len(loader), 1)
    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)

    return {
        "loss":        total_loss   / n,
        "loss_cls":    cls_loss_sum / n,
        "loss_reg":    reg_loss_sum / n,
        "accuracy":    classification_accuracy(logits_cat, labels_cat),
        "logits":      logits_cat,
        "labels":      labels_cat,
        "pred_counts": pred_counts,
        "true_counts": true_counts,
    }


# ---------------------------------------------------------------------------
# Checkpoint Saving
# ---------------------------------------------------------------------------

def save_final_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    val_accuracy: float,
    score: float,
    config: dict,
    config_path: str,
    run_dir: Path,
) -> None:
    """
    Save exactly one model file inside the current version folder.

    Overwrites only:
        saved_models/bilstm_v###/final_bilstm_model.pth
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss":             val_loss,
            "val_accuracy":         val_accuracy,
            "score":                score,
            "config":               config,
            "config_path":          config_path,
            "run_dir":              run_dir.as_posix(),
        },
        checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------

def train_bilstm(config_path: str = "configs/bilstm_config.yaml") -> Path:
    config = load_config(config_path)
    splits = load_splits("configs/data_splits.json")

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device  = select_device()
    use_amp = device.type == "cuda" and bool(config.get("use_amp", True))

    # ── Versioned output folders ───────────────────────────────────────────
    checkpoint_root      = Path(config["checkpoint_dir"])
    run_dir              = create_versioned_run_dir(checkpoint_root)
    checkpoint_path      = run_dir / "final_bilstm_model.pth"
    history_path         = run_dir / "training_history.csv"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    metadata_path        = run_dir / "run_metadata.json"

    with config_snapshot_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # ── Banner ────────────────────────────────────────────────────────────
    print("CoreSet BiLSTM — Training")
    print("=" * 70)
    print(f"  Hardware         : {device.type.upper()}")
    print(f"  AMP              : {use_amp}")
    print(f"  Seed             : {seed}")
    print(f"  Run folder       : {run_dir.as_posix()}")
    print(f"  Final checkpoint : {checkpoint_path.as_posix()}")
    print("-" * 70)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_dataset = BiLSTMDataset(config["data_dir"], splits["train"], augment=True,  cache=True)
    val_dataset   = BiLSTMDataset(config["data_dir"], splits["val"],   augment=False, cache=True)
    test_dataset  = BiLSTMDataset(config["data_dir"], splits["test"],  augment=False, cache=True)

    print(f"  Data             : {len(train_dataset)} train | {len(val_dataset)} val | {len(test_dataset)} test")
    print("-" * 70)

    # ── DataLoaders ───────────────────────────────────────────────────────
    train_loader = make_loader(train_dataset, config["batch_size"], True,  device, config)
    val_loader   = make_loader(val_dataset,   config["batch_size"], False, device, config)
    test_loader  = make_loader(test_dataset,  config["batch_size"], False, device, config)

    # ── Model ─────────────────────────────────────────────────────────────
    model = MultiTaskBiLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        dropout=float(config.get("dropout", 0.5)),
    ).to(device)

    print(f"  Parameters       : {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer (selective weight decay) ─────────────────────────────────
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in ("bn", "norm", "bias")):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = optim.AdamW(
        [
            {"params": decay,    "weight_decay": float(config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
    )

    # ── LR Scheduler ──────────────────────────────────────────────────────
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(config["patience_lr"])
    )

    # ── Loss functions ────────────────────────────────────────────────────
    criterion_cls = nn.CrossEntropyLoss(
        label_smoothing=float(config.get("label_smoothing", 0.0))
    )
    criterion_reg = nn.MSELoss()

    lambda_cls = float(config["lambda_cls"])
    lambda_reg = float(config["lambda_reg"])

    scaler              = GradScaler(enabled=use_amp)
    evaluator           = CoreSetEvaluator()
    patience_early_stop = int(config["patience_early_stop"])
    grad_clip           = float(config.get("grad_clip", 1.0))

    best_score        = float("-inf")
    best_epoch        = 0
    best_val_accuracy = 0.0
    no_improve        = 0

    print(f"  lambda_cls       : {lambda_cls}  lambda_reg : {lambda_reg}")
    print(f"  dropout          : {config.get('dropout', 0.5)}")
    print(f"  LR               : {config['learning_rate']}  weight_decay : {config['weight_decay']}")
    print(f"  Patience LR / ES : {config['patience_lr']} / {patience_early_stop}")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════════════════
    # Training Loop
    # ══════════════════════════════════════════════════════════════════════

    def autocast_ctx():
        return torch.cuda.amp.autocast(enabled=True) if use_amp else nullcontext()

    for epoch in range(int(config["epochs"])):
        start_time = time.time()
        model.train()

        total_loss = cls_loss_sum = reg_loss_sum = 0.0

        for x, y, reps in train_loader:
            x    = x.to(device, non_blocking=(device.type == "cuda"))
            y    = y.long().to(device, non_blocking=(device.type == "cuda"))
            reps = reps.float().unsqueeze(1).to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                logits, rep_pred = model(x)
                loss_cls = criterion_cls(logits, y)
                loss_reg = criterion_reg(rep_pred, reps)
                loss     = lambda_cls * loss_cls + lambda_reg * loss_reg

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss   += float(loss.item())
            cls_loss_sum += float(loss_cls.item())
            reg_loss_sum += float(loss_reg.item())

        n         = max(len(train_loader), 1)
        avg_train = total_loss   / n
        avg_cls   = cls_loss_sum / n
        avg_reg   = reg_loss_sum / n

        # ── Validation ────────────────────────────────────────────────────
        val_result   = run_split_evaluation(
            model=model, loader=val_loader,
            criterion_cls=criterion_cls, criterion_reg=criterion_reg,
            lambda_cls=lambda_cls, lambda_reg=lambda_reg,
            device=device, use_amp=use_amp,
        )

        val_loss     = val_result["loss"]
        val_accuracy = val_result["accuracy"]
        val_obo      = evaluator.calculate_obo_accuracy(val_result["pred_counts"], val_result["true_counts"])
        val_mae      = evaluator.calculate_normalized_mae(val_result["pred_counts"], val_result["true_counts"])

        score = val_accuracy   # BiLSTM has no density head; score = accuracy

        scheduler.step(val_loss)
        lr_now     = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_time

        # ── Epoch log (ST-GCN style) ──────────────────────────────────────
        print(
            f"Ep [{epoch + 1:3d}/{config['epochs']}] "
            f"loss {avg_train:.3f} "
            f"(cls {avg_cls:.3f} reg {avg_reg:.3f}) "
            f"| val {val_loss:.3f} "
            f"| acc {val_accuracy * 100:.1f}% "
            f"obo {val_obo * 100:.1f}% "
            f"nmae {val_mae:.3f} "
            f"| score {score:.3f} "
            f"| lr {lr_now:.1e} "
            f"| {epoch_time:.1f}s"
        )

        append_history_row(history_path, {
            "epoch":          epoch + 1,
            "train_loss":     round(avg_train,    6),
            "train_cls_loss": round(avg_cls,      6),
            "train_reg_loss": round(avg_reg,      6),
            "val_loss":       round(val_loss,     6),
            "val_accuracy":   round(val_accuracy, 6),
            "val_obo":        round(val_obo,      6),
            "val_nmae":       round(val_mae,      6),
            "score":          round(score,        6),
            "lr":             lr_now,
        })

        # ── Checkpoint ────────────────────────────────────────────────────
        if score > best_score:
            best_score        = score
            best_epoch        = epoch + 1
            best_val_accuracy = val_accuracy
            no_improve        = 0

            save_final_checkpoint(
                checkpoint_path,
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch + 1, val_loss=val_loss, val_accuracy=val_accuracy,
                score=score, config=config, config_path=config_path, run_dir=run_dir,
            )
            write_latest_checkpoint_pointer(checkpoint_root, checkpoint_path)
            write_latest_run_pointer(checkpoint_root, run_dir)

            print(f"  ✓ updated final model: {checkpoint_path.as_posix()}")

        else:
            no_improve += 1

        if no_improve >= patience_early_stop:
            print(f"\n  Early stop at epoch {epoch + 1}.")
            break

    if not checkpoint_path.exists():
        raise RuntimeError("Training finished without producing final_bilstm_model.pth.")

    # ══════════════════════════════════════════════════════════════════════
    # Final Test Evaluation Using Saved Final Checkpoint
    # ══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("  Loading final checkpoint for test evaluation")
    print("=" * 70)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_result   = run_split_evaluation(
        model=model, loader=test_loader,
        criterion_cls=criterion_cls, criterion_reg=criterion_reg,
        lambda_cls=lambda_cls, lambda_reg=lambda_reg,
        device=device, use_amp=use_amp,
    )

    test_accuracy = test_result["accuracy"]
    test_obo      = evaluator.calculate_obo_accuracy(test_result["pred_counts"], test_result["true_counts"])
    test_mae      = evaluator.calculate_normalized_mae(test_result["pred_counts"], test_result["true_counts"])
    test_rmse     = evaluator.calculate_rmse(test_result["pred_counts"], test_result["true_counts"])

    metadata = {
        "run_dir":           run_dir.as_posix(),
        "final_checkpoint":  checkpoint_path.as_posix(),
        "best_epoch":        best_epoch,
        "best_score":        best_score,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy":     test_accuracy,
        "test_obo":          test_obo,
        "test_nmae":         test_mae,
        "test_rmse":         test_rmse,
        "history_csv":       history_path.as_posix(),
        "config_snapshot":   config_snapshot_path.as_posix(),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Test Accuracy    : {test_accuracy * 100:.2f}%")
    print(f"  Test OBO         : {test_obo * 100:.2f}%")
    print(f"  Test nMAE        : {test_mae:.4f}")
    print(f"  Test RMSE        : {test_rmse:.4f}")
    print("=" * 70)
    print(f"  Final model      : {checkpoint_path.as_posix()}")
    print(f"  Latest checkpoint: {(checkpoint_root / 'latest_bilstm_checkpoint.txt').as_posix()}")
    print(f"  Latest run       : {(checkpoint_root / 'latest_bilstm_run.txt').as_posix()}")
    print(f"  History CSV      : {history_path.as_posix()}")
    print(f"  Metadata JSON    : {metadata_path.as_posix()}")
    print("=" * 70)

    return checkpoint_path


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_bilstm("configs/bilstm_config.yaml")