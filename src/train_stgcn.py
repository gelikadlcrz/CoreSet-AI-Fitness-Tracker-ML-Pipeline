"""
train_stgcn.py — CoreSet ST-GCN training.

Clean checkpoint workflow:
- Each training run creates a new version folder:

      checkpoint/stgcn_v001/
      checkpoint/stgcn_v002/
      checkpoint/stgcn_v003/

- Each version folder contains only one .pth model file:

      final_stgcn_model.pth

This one final checkpoint is used for:
    1. evaluation
    2. ONNX / TFLite export
    3. mobile deployment

The script does NOT create:
    score_epoch_*.pth
    accuracy_epoch_*.pth
    counting_epoch_*.pth
    best_stgcn_by_accuracy.pth
    best_stgcn_by_counting.pth
    best_stgcn_model.pth
"""

from __future__ import annotations

import csv
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from src.data.coreset_dataset import ANGLE_FEATURE_DIM, CoreSetGCN_Dataset
from src.models.stgcn_multitask import CoreSetSTGCN_MultiTask
from src.utils.versioning import (
    create_versioned_run_dir,
    write_latest_checkpoint_pointer,
    write_latest_run_pointer,
)


# ---------------------------------------------------------------------------
# Config / Reproducibility
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


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
    """
    Windows-safe DataLoader.

    num_workers defaults to 0 because Windows multiprocessing can cause issues
    during local training.
    """
    requested_workers = int(config.get("num_workers", 0))

    if requested_workers <= 0:
        num_workers = 0
    else:
        max_workers = max(os.cpu_count() or 1, 1)
        num_workers = min(requested_workers, max_workers)

    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))

    return DataLoader(dataset, **kwargs)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_norm_stats_online(
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute channel-wise mean/std from the non-augmented training split only.
    """
    n = 0
    sum_ = None
    sumsq_ = None

    with torch.no_grad():
        for inputs, _, _ in loader:
            inputs = inputs.to(
                device,
                non_blocking=(device.type == "cuda"),
            ).double()

            _, channels, _, _, _ = inputs.shape

            flat = (
                inputs
                .permute(0, 2, 3, 4, 1)
                .contiguous()
                .view(-1, channels)
            )

            if sum_ is None:
                sum_ = torch.zeros(channels, dtype=torch.float64, device=device)
                sumsq_ = torch.zeros(channels, dtype=torch.float64, device=device)

            sum_ += flat.sum(dim=0)
            sumsq_ += (flat * flat).sum(dim=0)
            n += flat.size(0)

    if n == 0:
        raise RuntimeError("Cannot compute normalization statistics from an empty loader.")

    mean = sum_ / n
    var = (sumsq_ / n - mean * mean).clamp(min=1e-12)
    std = var.sqrt().clamp(min=1e-6)

    return (
        mean.float().view(1, -1, 1, 1, 1),
        std.float().view(1, -1, 1, 1, 1),
    )


def normalize_density_gt(density_gt: torch.Tensor) -> torch.Tensor:
    """
    Normalize each sample density map by its own maximum.

    Empty density maps remain zero.
    """
    max_vals = density_gt.amax(dim=1, keepdim=True).clamp(min=1e-6)
    return density_gt / max_vals


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class PeakWeightedBCELoss(nn.Module):
    """
    BCE density loss with larger weight near repetition peaks.
    """

    def __init__(self, alpha: float = 3.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, pred: torch.Tensor, gt_norm: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, gt_norm, reduction="none")
        weights = 1.0 + self.alpha * gt_norm
        return (bce * weights).mean()


# ---------------------------------------------------------------------------
# Training Helpers
# ---------------------------------------------------------------------------

def mixup_cls(
    x: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.0,
    device: torch.device | str = "cpu",
):
    """
    Classification-only mixup.

    Density labels are not mixed because density maps belong to the original
    temporal sequence.
    """
    if alpha <= 0:
        return x, labels, labels, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=device)

    mixed_x = lam * x + (1.0 - lam) * x[idx]

    return mixed_x, labels, labels[idx], lam


def compute_class_weights(
    labels: list[int],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Balanced class weights normalized to mean 1.0.
    """
    counts = np.bincount(
        np.asarray(labels, dtype=int),
        minlength=num_classes,
    ).astype(np.float32)

    counts = np.maximum(counts, 1.0)

    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32, device=device)


def classification_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())


def append_history_row(history_path: Path, row: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = history_path.exists()

    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# ---------------------------------------------------------------------------
# Counting Metrics
# ---------------------------------------------------------------------------

def count_density_map(
    density_map: np.ndarray,
    threshold_ratio: float = 0.30,
    threshold_floor: float = 0.15,
    min_distance: int = 10,
    prominence_ratio: float = 0.05,
) -> int:
    """
    Convert a predicted density curve into a repetition count using peak detection.
    """
    from scipy.signal import find_peaks

    density_map = np.asarray(density_map, dtype=np.float32).reshape(-1)

    if density_map.size == 0:
        return 0

    max_val = float(np.nanmax(density_map))

    if not np.isfinite(max_val) or max_val <= 0:
        return 0

    threshold = max(
        max_val * float(threshold_ratio),
        float(threshold_floor),
    )

    threshold = min(threshold, max_val * 0.95)

    prominence = max(
        max_val * float(prominence_ratio),
        1e-6,
    )

    peaks, _ = find_peaks(
        density_map,
        height=threshold,
        distance=int(min_distance),
        prominence=prominence,
    )

    return int(len(peaks))


def counting_metrics_from_maps(
    den_pred: np.ndarray,
    den_gt: np.ndarray,
    postprocess: Optional[dict] = None,
):
    """
    Return nMAE, RMSE, OBO, predicted counts, and ground-truth counts.
    """
    postprocess = postprocess or {}

    pred_counts = []
    gt_counts = []

    for pred_map, gt_map in zip(den_pred, den_gt):
        pred_count = count_density_map(
            pred_map,
            threshold_ratio=float(postprocess.get("threshold_ratio", 0.30)),
            threshold_floor=float(postprocess.get("threshold_floor", 0.15)),
            min_distance=int(postprocess.get("min_distance", 10)),
            prominence_ratio=float(postprocess.get("prominence_ratio", 0.05)),
        )

        gt_count = round(float(np.asarray(gt_map).sum()))

        pred_counts.append(pred_count)
        gt_counts.append(gt_count)

    pred_arr = np.asarray(pred_counts, dtype=float)
    gt_arr = np.asarray(gt_counts, dtype=float)

    valid = gt_arr > 0

    if valid.any():
        nmae = float(np.mean(np.abs(pred_arr[valid] - gt_arr[valid]) / gt_arr[valid]))
    else:
        nmae = 0.0

    if len(pred_arr):
        rmse = float(np.sqrt(np.mean((pred_arr - gt_arr) ** 2)))
        obo = float((np.abs(pred_arr - gt_arr) <= 1).mean())
    else:
        rmse = 0.0
        obo = 0.0

    return nmae, rmse, obo, pred_counts, gt_counts


def tune_counting_postprocess(
    den_pred: np.ndarray,
    den_gt: np.ndarray,
    config: dict,
) -> tuple[dict, float, float, float]:
    """
    Tune peak detection parameters on validation predictions only.
    """
    if not bool(config.get("tune_counting_postprocess", True)):
        params = {
            "adaptive": True,
            "threshold_ratio": float(config.get("count_threshold_ratio", 0.30)),
            "threshold_floor": float(config.get("count_threshold_floor", 0.15)),
            "min_distance": int(config.get("count_min_distance", 10)),
            "prominence_ratio": float(config.get("count_prominence_ratio", 0.05)),
            "selection_metric": "default",
        }

        nmae, rmse, obo, _, _ = counting_metrics_from_maps(
            den_pred,
            den_gt,
            params,
        )

        return params, nmae, rmse, obo

    ratio_grid = config.get(
        "count_threshold_ratio_grid",
        [0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
    )

    floor_grid = config.get(
        "count_threshold_floor_grid",
        [0.05, 0.10, 0.15, 0.20],
    )

    distance_grid = config.get(
        "count_min_distance_grid",
        [6, 8, 10, 12, 15, 18],
    )

    prominence_grid = config.get(
        "count_prominence_ratio_grid",
        [0.03, 0.05, 0.08],
    )

    best = None

    for ratio in ratio_grid:
        for floor in floor_grid:
            for dist in distance_grid:
                for prom in prominence_grid:
                    params = {
                        "adaptive": True,
                        "threshold_ratio": float(ratio),
                        "threshold_floor": float(floor),
                        "min_distance": int(dist),
                        "prominence_ratio": float(prom),
                        "selection_metric": "max_obo_then_min_nmae",
                    }

                    nmae, rmse, obo, _, _ = counting_metrics_from_maps(
                        den_pred,
                        den_gt,
                        params,
                    )

                    key = (obo, -nmae, -rmse)

                    if best is None or key > best[0]:
                        best = (key, params, nmae, rmse, obo)

    _, params, nmae, rmse, obo = best

    return params, nmae, rmse, obo


# ---------------------------------------------------------------------------
# Evaluation Helper
# ---------------------------------------------------------------------------

def run_split_evaluation(
    *,
    model: nn.Module,
    loader: DataLoader,
    loss_cls: nn.Module,
    loss_density: nn.Module,
    lambda_cls: float,
    lambda_density: float,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    device: torch.device,
    use_amp: bool,
):
    model.eval()

    total_loss = 0.0
    all_logits = []
    all_labels = []
    all_density_pred = []
    all_density_gt = []

    def autocast_context():
        if use_amp:
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()

    with torch.no_grad():
        for inputs, labels, density_gts in loader:
            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            density_gts = density_gts.to(device, non_blocking=(device.type == "cuda"))

            inputs = (inputs - feat_mean) / feat_std
            density_norm = normalize_density_gt(density_gts)

            with autocast_context():
                logits, density_maps = model(inputs)
                l_cls = loss_cls(logits, labels)
                l_den = loss_density(density_maps, density_norm)
                loss = lambda_cls * l_cls + lambda_density * l_den

            total_loss += float(loss.item())

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_density_pred.append(density_maps.detach().cpu().numpy())
            all_density_gt.append(density_gts.detach().cpu().numpy())

    logits_cat = torch.cat(all_logits)
    labels_cat = torch.cat(all_labels)

    avg_loss = total_loss / max(len(loader), 1)
    acc = classification_accuracy(logits_cat, labels_cat)

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "logits": logits_cat,
        "labels": labels_cat,
        "density_pred": np.concatenate(all_density_pred),
        "density_gt": np.concatenate(all_density_gt),
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
    val_obo: float,
    val_nmae: float,
    val_rmse: float,
    score: float,
    score_formula: dict,
    counting_postprocess: dict,
    config: dict,
    config_path: str,
    run_dir: Path,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
) -> None:
    """
    Save exactly one model file inside the current version folder.

    This overwrites only:
        checkpoint/stgcn_v###/final_stgcn_model.pth

    It does not create extra .pth files.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_obo": val_obo,
            "val_nmae": val_nmae,
            "val_rmse": val_rmse,
            "score": score,
            "score_formula": score_formula,
            "counting_postprocess": counting_postprocess,
            "config": config,
            "config_path": config_path,
            "run_dir": run_dir.as_posix(),
            "feat_mean": feat_mean.detach().cpu(),
            "feat_std": feat_std.detach().cpu(),
        },
        checkpoint_path,
    )


# ---------------------------------------------------------------------------
# Main Training
# ---------------------------------------------------------------------------

def train_stgcn_model(config_path: str = "configs/stgcn_config.yaml"):
    config = load_config(config_path)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = select_device()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    checkpoint_root = Path(config["checkpoint_dir"])
    run_dir = create_versioned_run_dir(checkpoint_root, prefix="stgcn")
    checkpoint_path = run_dir / "final_stgcn_model.pth"

    history_path = run_dir / "training_history.csv"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    metadata_path = run_dir / "run_metadata.json"

    with config_snapshot_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("CoreSet ST-GCN — Training")
    print("=" * 70)
    print(f"  Hardware         : {device.type.upper()}")
    print(f"  Seed             : {seed}")
    print(f"  Run folder       : {run_dir.as_posix()}")
    print(f"  Final checkpoint : {checkpoint_path.as_posix()}")
    print("-" * 70)

    split_file = os.path.join("configs", "data_splits.json")

    max_frames = int(config["max_frames"])
    batch_size = int(config["batch_size"])
    num_classes = int(config["num_classes"])
    node_count = int(config["node_count"])

    train_ds = CoreSetGCN_Dataset(
        config["data_dir"],
        split_file,
        "train",
        max_frames,
        augment=True,
    )

    stats_ds = CoreSetGCN_Dataset(
        config["data_dir"],
        split_file,
        "train",
        max_frames,
        augment=False,
    )

    val_ds = CoreSetGCN_Dataset(
        config["data_dir"],
        split_file,
        "val",
        max_frames,
        augment=False,
    )

    test_ds = CoreSetGCN_Dataset(
        config["data_dir"],
        split_file,
        "test",
        max_frames,
        augment=False,
    )

    print(f"  Data             : {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")
    print("-" * 70)

    train_loader = make_loader(train_ds, batch_size, True, device, config)
    stats_loader = make_loader(stats_ds, batch_size, False, device, config)
    val_loader = make_loader(val_ds, batch_size, False, device, config)
    test_loader = make_loader(test_ds, batch_size, False, device, config)

    print("  Computing normalization statistics from non-augmented train split...")

    feat_mean, feat_std = compute_norm_stats_online(stats_loader, device)

    print(
        f"  mean [{feat_mean.min():.4f}, {feat_mean.max():.4f}] "
        f"std [{feat_std.min():.4f}, {feat_std.max():.4f}]"
    )

    print("-" * 70)

    model = CoreSetSTGCN_MultiTask(
        in_channels=ANGLE_FEATURE_DIM,
        num_classes=num_classes,
        max_frames=max_frames,
        node_count=node_count,
    ).to(device)

    print(f"  Parameters       : {sum(p.numel() for p in model.parameters()):,}")

    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lname = name.lower()

        if any(nd in lname for nd in ("bn", "norm", "bias", "adj")):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": float(config["weight_decay"]),
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
            },
        ],
        lr=float(config["learning_rate"]),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config.get("scheduler_t0", 20)),
        T_mult=int(config.get("scheduler_t_mult", 2)),
        eta_min=float(config.get("eta_min", 5e-6)),
    )

    class_weights = None

    if bool(config.get("use_class_weights", True)):
        class_weights = compute_class_weights(
            train_ds.labels,
            num_classes,
            device,
        )

        print(
            "  Class weights    : "
            f"{[round(float(w), 3) for w in class_weights.detach().cpu()]}"
        )

    loss_cls = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(config.get("label_smoothing", 0.03)),
    )

    loss_density = PeakWeightedBCELoss(
        alpha=float(config.get("density_alpha", 3.0)),
    )

    lambda_cls = float(config.get("lambda_cls", 1.0))
    lambda_density_max = float(config.get("lambda_density_max", 0.20))
    warmup_epochs = int(config.get("warmup_epochs", 12))
    mixup_alpha = float(config.get("mixup_alpha", 0.05))
    patience = int(config.get("patience", 30))
    noise_std = float(config.get("noise_std", 0.01))
    grad_clip = float(config.get("grad_clip", 1.0))
    accumulation_steps = max(1, int(config.get("accumulation_steps", 1)))

    score_acc_weight = float(config.get("score_acc_weight", 0.85))
    score_obo_weight = float(config.get("score_obo_weight", 0.15))
    score_nmae_weight = float(config.get("score_nmae_weight", 0.05))

    score_formula = {
        "accuracy_weight": score_acc_weight,
        "obo_weight": score_obo_weight,
        "nmae_penalty_weight": score_nmae_weight,
    }

    use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"  AMP enabled      : {use_amp}")
    print(f"  Patience         : {patience}")
    print(
        "  Score formula    : "
        f"{score_acc_weight:.2f}*accuracy "
        f"+ {score_obo_weight:.2f}*OBO "
        f"- {score_nmae_weight:.2f}*nMAE"
    )
    print("=" * 70)

    def autocast_context():
        if use_amp:
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()

    best_score = float("-inf")
    best_epoch = 0
    best_val_accuracy = 0.0
    best_val_obo = 0.0
    best_val_nmae = float("inf")
    best_val_rmse = float("inf")
    best_counting_postprocess = None

    no_improve = 0

    # -----------------------------------------------------------------------
    # Training Loop
    # -----------------------------------------------------------------------

    for epoch in range(int(config["epochs"])):
        if epoch < warmup_epochs:
            lambda_density = 0.0
        else:
            ramp = min(
                1.0,
                (epoch - warmup_epochs) / max(warmup_epochs, 1),
            )
            lambda_density = lambda_density_max * ramp

        model.train()

        total_loss = 0.0
        cls_loss_sum = 0.0
        den_loss_sum = 0.0

        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (inputs, labels, density_gts) in enumerate(train_loader):
            is_update_step = (
                ((batch_idx + 1) % accumulation_steps == 0)
                or ((batch_idx + 1) == len(train_loader))
            )

            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            density_gts = density_gts.to(device, non_blocking=(device.type == "cuda"))

            inputs = (inputs - feat_mean) / feat_std

            if noise_std > 0:
                inputs = inputs + torch.randn_like(inputs) * noise_std

            density_norm = normalize_density_gt(density_gts)

            inputs, labels_a, labels_b, lam = mixup_cls(
                inputs,
                labels,
                alpha=mixup_alpha,
                device=device,
            )

            with autocast_context():
                logits, density_maps = model(inputs)

                l_cls = (
                    lam * loss_cls(logits, labels_a)
                    + (1.0 - lam) * loss_cls(logits, labels_b)
                )

                l_den = loss_density(density_maps, density_norm)

                loss = lambda_cls * l_cls + lambda_density * l_den
                loss_for_backward = loss / accumulation_steps

            if use_amp:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if is_update_step:
                if use_amp:
                    scaler.unscale_(optimizer)

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=grad_clip,
                    )

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            cls_loss_sum += float(l_cls.item())
            den_loss_sum += float(l_den.item())

        n_batches = max(len(train_loader), 1)

        avg_train = total_loss / n_batches
        avg_cls = cls_loss_sum / n_batches
        avg_den = den_loss_sum / n_batches

        scheduler.step(epoch + 1)

        # -------------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------------

        val_result = run_split_evaluation(
            model=model,
            loader=val_loader,
            loss_cls=loss_cls,
            loss_density=loss_density,
            lambda_cls=lambda_cls,
            lambda_density=lambda_density,
            feat_mean=feat_mean,
            feat_std=feat_std,
            device=device,
            use_amp=use_amp,
        )

        val_postprocess, val_nmae, val_rmse, val_obo = tune_counting_postprocess(
            val_result["density_pred"],
            val_result["density_gt"],
            config,
        )

        val_accuracy = val_result["accuracy"]
        val_loss = val_result["loss"]

        score = (
            score_acc_weight * val_accuracy
            + score_obo_weight * val_obo
            - score_nmae_weight * val_nmae
        )

        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"Ep [{epoch + 1:3d}/{config['epochs']}] "
            f"loss {avg_train:.3f} "
            f"(cls {avg_cls:.3f} den {avg_den:.3f}) "
            f"| val {val_loss:.3f} "
            f"| acc {val_accuracy * 100:.1f}% "
            f"obo {val_obo * 100:.1f}% "
            f"nmae {val_nmae:.3f} "
            f"| score {score:.3f} "
            f"| lr {lr_now:.1e}"
        )

        append_history_row(
            history_path,
            {
                "epoch": epoch + 1,
                "train_loss": round(avg_train, 6),
                "train_cls_loss": round(avg_cls, 6),
                "train_density_loss": round(avg_den, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_accuracy, 6),
                "val_obo": round(val_obo, 6),
                "val_nmae": round(val_nmae, 6),
                "val_rmse": round(val_rmse, 6),
                "score": round(score, 6),
                "count_threshold_ratio": val_postprocess.get("threshold_ratio"),
                "count_threshold_floor": val_postprocess.get("threshold_floor"),
                "count_min_distance": val_postprocess.get("min_distance"),
                "count_prominence_ratio": val_postprocess.get("prominence_ratio"),
                "lr": lr_now,
                "lambda_density": round(lambda_density, 6),
            },
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_val_accuracy = val_accuracy
            best_val_obo = val_obo
            best_val_nmae = val_nmae
            best_val_rmse = val_rmse
            best_counting_postprocess = val_postprocess
            no_improve = 0

            save_final_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                val_obo=val_obo,
                val_nmae=val_nmae,
                val_rmse=val_rmse,
                score=score,
                score_formula=score_formula,
                counting_postprocess=val_postprocess,
                config=config,
                config_path=config_path,
                run_dir=run_dir,
                feat_mean=feat_mean,
                feat_std=feat_std,
            )

            write_latest_checkpoint_pointer(
                checkpoint_root,
                checkpoint_path,
                prefix="stgcn",
            )

            write_latest_run_pointer(
                checkpoint_root,
                run_dir,
                prefix="stgcn",
            )

            print(f"  ✓ updated final model: {checkpoint_path.as_posix()}")

        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n  Early stop at epoch {epoch + 1}.")
            break

    if not checkpoint_path.exists():
        raise RuntimeError("Training finished without producing final_stgcn_model.pth.")

    # -----------------------------------------------------------------------
    # Final Test Evaluation Using Saved Final Checkpoint
    # -----------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("  Loading final checkpoint for test evaluation")
    print("=" * 70)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    feat_mean = checkpoint["feat_mean"].to(device)
    feat_std = checkpoint["feat_std"].to(device)

    test_result = run_split_evaluation(
        model=model,
        loader=test_loader,
        loss_cls=loss_cls,
        loss_density=loss_density,
        lambda_cls=lambda_cls,
        lambda_density=lambda_density_max,
        feat_mean=feat_mean,
        feat_std=feat_std,
        device=device,
        use_amp=use_amp,
    )

    test_postprocess = checkpoint.get(
        "counting_postprocess",
        best_counting_postprocess
        or {
            "adaptive": True,
            "threshold_ratio": 0.30,
            "threshold_floor": 0.15,
            "min_distance": 10,
            "prominence_ratio": 0.05,
            "selection_metric": "fallback",
        },
    )

    test_nmae, test_rmse, test_obo, _, _ = counting_metrics_from_maps(
        test_result["density_pred"],
        test_result["density_gt"],
        test_postprocess,
    )

    test_accuracy = test_result["accuracy"]

    metadata = {
        "run_dir": run_dir.as_posix(),
        "final_checkpoint": checkpoint_path.as_posix(),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_val_accuracy": best_val_accuracy,
        "best_val_obo": best_val_obo,
        "best_val_nmae": best_val_nmae,
        "best_val_rmse": best_val_rmse,
        "counting_postprocess": test_postprocess,
        "test_accuracy": test_accuracy,
        "test_obo": test_obo,
        "test_nmae": test_nmae,
        "test_rmse": test_rmse,
        "history_csv": history_path.as_posix(),
        "config_snapshot": config_snapshot_path.as_posix(),
    }

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Test Accuracy    : {test_accuracy * 100:.2f}%")
    print(f"  Test OBO         : {test_obo * 100:.2f}%")
    print(f"  Test nMAE        : {test_nmae:.4f}")
    print(f"  Test RMSE        : {test_rmse:.4f}")
    print("=" * 70)
    print(f"  Final model      : {checkpoint_path.as_posix()}")
    print(f"  Latest checkpoint: {(checkpoint_root / 'latest_stgcn_checkpoint.txt').as_posix()}")
    print(f"  Latest run       : {(checkpoint_root / 'latest_stgcn_run.txt').as_posix()}")
    print(f"  History CSV      : {history_path.as_posix()}")
    print(f"  Metadata JSON    : {metadata_path.as_posix()}")
    print("=" * 70)

    return checkpoint_path


if __name__ == "__main__":
    train_stgcn_model("configs/stgcn_config.yaml")