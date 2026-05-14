"""
train_stgcn.py — CoreSet ST-GCN training with safer optimization + versioned outputs.

What changed in this version
----------------------------
1. Every training run is saved in a new folder:
       checkpoint/stgcn_v001/
       checkpoint/stgcn_v002/
       checkpoint/stgcn_v003/
   This prevents a newly trained model from overwriting an older trained model.

2. Best checkpoints inside each run use unique filenames with the epoch/score,
   then the final best checkpoint is copied once to that run folder as:
       best_stgcn_model.pth

3. Training is more configurable and faster on supported GPUs:
   - optional CUDA AMP mixed precision
   - configurable patience, warmup, mixup, density weight, noise, grad clipping
   - cleaner optimizer groups and scheduler
   - history CSV + config snapshot saved per versioned run

4. Normalization statistics are computed from the training split only using a
   deterministic non-augmented loader, then stored inside every checkpoint.
"""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from src.data.coreset_dataset import ANGLE_FEATURE_DIM, CoreSetGCN_Dataset
from src.models.stgcn_multitask import CoreSetSTGCN_MultiTask
from src.utils.metrics import CoreSetEvaluator
from src.utils.versioning import (
    create_versioned_run_dir,
    write_latest_checkpoint_pointer,
    write_latest_run_pointer,
)


# ---------------------------------------------------------------------------
# Config / reproducibility
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
# Density GT normalization
# ---------------------------------------------------------------------------

def normalize_density_gt(density_gt: torch.Tensor) -> torch.Tensor:
    """
    Normalize each sample's raw density map by its per-sample maximum.
    Peaks become 1.0 and background remains close to 0. Zero maps stay zero.
    """
    max_vals = density_gt.amax(dim=1, keepdim=True).clamp(min=1e-6)
    return density_gt / max_vals


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class PeakWeightedBCELoss(nn.Module):
    """
    BCE with higher weight on peak frames.
    L = mean((1 + alpha * gt_norm) * BCE(pred, gt_norm))
    """

    def __init__(self, alpha: float = 4.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, gt_norm: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, gt_norm, reduction="none")
        weights = 1.0 + self.alpha * gt_norm
        return (bce * weights).mean()


# ---------------------------------------------------------------------------
# Mixup — classification only
# ---------------------------------------------------------------------------

def mixup_cls(x: torch.Tensor, labels: torch.Tensor, alpha: float = 0.1, device="cpu"):
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], labels, labels[idx], lam


# ---------------------------------------------------------------------------
# Normalization stats
# ---------------------------------------------------------------------------

def compute_norm_stats_online(loader: DataLoader, device: torch.device):
    """
    Compute channel-wise mean/std over the training split only.
    This version uses sum/squared-sum accumulation, which is fast and stable
    enough for standardized angle features.
    """
    n = 0
    sum_ = None
    sumsq_ = None

    with torch.no_grad():
        for inputs, _, _ in loader:
            inputs = inputs.to(device, non_blocking=(device.type == "cuda")).double()
            _, c, _, _, _ = inputs.shape
            flat = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, c)

            if sum_ is None:
                sum_ = torch.zeros(c, dtype=torch.float64, device=device)
                sumsq_ = torch.zeros(c, dtype=torch.float64, device=device)

            sum_ += flat.sum(dim=0)
            sumsq_ += (flat * flat).sum(dim=0)
            n += flat.size(0)

    if n == 0:
        raise RuntimeError("Cannot compute normalization statistics from an empty loader.")

    mean = sum_ / n
    var = (sumsq_ / n - mean * mean).clamp(min=1e-12)
    std = var.sqrt().clamp(min=1e-6)
    return mean.float().view(1, -1, 1, 1, 1), std.float().view(1, -1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def append_history_row(history_path: Path, row: dict) -> None:
    """Append one epoch of metrics to the run history CSV.

    Creates the file and header on the first write. If later rows contain
    extra keys, the function preserves the original header and writes only
    matching fields so training does not crash.
    """
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        fieldnames = list(row.keys())
        with history_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return

    with history_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            fieldnames = next(reader)
        except StopIteration:
            fieldnames = list(row.keys())

    safe_row = {key: row.get(key, "") for key in fieldnames}
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(safe_row)


def grad_norm(params) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


def count_density_map(
    density_map: np.ndarray,
    threshold_ratio: float = 0.30,
    threshold_floor: float = 0.15,
    min_distance: int = 10,
    prominence_ratio: float = 0.05,
) -> int:
    """Convert a normalized density curve into a repetition count."""
    from scipy.signal import find_peaks

    density_map = np.asarray(density_map, dtype=np.float32).reshape(-1)
    if density_map.size == 0:
        return 0

    max_val = float(np.nanmax(density_map))
    if not np.isfinite(max_val) or max_val <= 0:
        return 0

    threshold = max(max_val * float(threshold_ratio), float(threshold_floor))
    threshold = min(threshold, max_val * 0.95)
    prominence = max(max_val * float(prominence_ratio), 1e-6)

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
    postprocess: dict | None = None,
):
    """Return nMAE, RMSE, OBO, predicted counts and ground-truth counts."""
    postprocess = postprocess or {}
    pred_c, gt_c = [], []
    for pm, gm in zip(den_pred, den_gt):
        pred_c.append(
            count_density_map(
                pm,
                threshold_ratio=float(postprocess.get("threshold_ratio", 0.30)),
                threshold_floor=float(postprocess.get("threshold_floor", 0.15)),
                min_distance=int(postprocess.get("min_distance", 10)),
                prominence_ratio=float(postprocess.get("prominence_ratio", 0.05)),
            )
        )
        gt_c.append(round(float(gm.sum())))

    pa = np.asarray(pred_c, dtype=float)
    ga = np.asarray(gt_c, dtype=float)
    valid = ga > 0
    if valid.any():
        nmae = float(np.mean(np.abs(pa[valid] - ga[valid]) / ga[valid]))
    else:
        nmae = 0.0
    rmse = float(np.sqrt(np.mean((pa - ga) ** 2))) if len(pa) else 0.0
    obo = float((np.abs(pa - ga) <= 1).mean()) if len(pa) else 0.0
    return nmae, rmse, obo, pred_c, gt_c


def tune_counting_postprocess(den_pred: np.ndarray, den_gt: np.ndarray, config: dict) -> tuple[dict, float, float, float]:
    """Tune peak-detection parameters on validation predictions."""
    if not bool(config.get("tune_counting_postprocess", True)):
        params = {
            "adaptive": True,
            "threshold_ratio": float(config.get("count_threshold_ratio", 0.30)),
            "threshold_floor": float(config.get("count_threshold_floor", 0.15)),
            "min_distance": int(config.get("count_min_distance", 10)),
            "prominence_ratio": float(config.get("count_prominence_ratio", 0.05)),
            "selection_metric": "default",
        }
        nmae, rmse, obo, _, _ = counting_metrics_from_maps(den_pred, den_gt, params)
        return params, nmae, rmse, obo

    ratio_grid = config.get("count_threshold_ratio_grid", [0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    floor_grid = config.get("count_threshold_floor_grid", [0.05, 0.10, 0.15, 0.20])
    distance_grid = config.get("count_min_distance_grid", [6, 8, 10, 12, 15, 18])
    prominence_grid = config.get("count_prominence_ratio_grid", [0.03, 0.05, 0.08])

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
                    nmae, rmse, obo, _, _ = counting_metrics_from_maps(den_pred, den_gt, params)
                    key = (obo, -nmae, -rmse)
                    if best is None or key > best[0]:
                        best = (key, params, nmae, rmse, obo)

    _, params, nmae, rmse, obo = best
    return params, nmae, rmse, obo


def compute_val_obo(den_pred: np.ndarray, den_gt: np.ndarray) -> float:
    """Backward-compatible default OBO helper."""
    _, _, obo, _, _ = counting_metrics_from_maps(den_pred, den_gt)
    return obo



def make_loader(dataset, batch_size: int, shuffle: bool, device: torch.device, config: dict) -> DataLoader:
    """
    Build a DataLoader safely across Windows/macOS/Linux.

    The previous v2 file called make_loader() but the helper was missing,
    which caused training to stop before epoch 1.
    """
    requested_workers = int(config.get("num_workers", 0))

    # Safer default for Windows + CPU training. Users can still set
    # num_workers in configs/stgcn_config.yaml if they want multiprocessing.
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

def compute_class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    """Balanced class weights normalized to mean 1.0."""
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_stgcn_model(config_path: str = "configs/stgcn_config.yaml"):
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    checkpoint_root = Path(config["checkpoint_dir"])
    run_dir = create_versioned_run_dir(checkpoint_root, prefix="stgcn")
    write_latest_run_pointer(checkpoint_root, run_dir, prefix="stgcn")

    history_path = run_dir / "training_history.csv"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    with config_snapshot_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("CoreSet ST-GCN — Training")
    print("=" * 70)

    device = select_device()
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    print(f"  Hardware       : {device.type.upper()}")
    print(f"  Seed           : {seed}")
    print(f"  Run folder     : {run_dir.as_posix()}")

    split_file = os.path.join("configs", "data_splits.json")
    max_frames = int(config["max_frames"])

    train_ds = CoreSetGCN_Dataset(config["data_dir"], split_file, "train", max_frames, augment=True)
    stats_ds = CoreSetGCN_Dataset(config["data_dir"], split_file, "train", max_frames, augment=False)
    val_ds = CoreSetGCN_Dataset(config["data_dir"], split_file, "val", max_frames, augment=False)
    test_ds = CoreSetGCN_Dataset(config["data_dir"], split_file, "test", max_frames, augment=False)

    print(f"  Data           : {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")
    print("-" * 70)

    batch_size = int(config["batch_size"])
    train_loader = make_loader(train_ds, batch_size, True, device, config)
    stats_loader = make_loader(stats_ds, batch_size, False, device, config)
    val_loader = make_loader(val_ds, batch_size, False, device, config)
    test_loader = make_loader(test_ds, batch_size, False, device, config)

    print("  Computing normalization statistics from non-augmented train split...")
    feat_mean, feat_std = compute_norm_stats_online(stats_loader, device)
    print(
        f"  mean [{feat_mean.min():.4f}, {feat_mean.max():.4f}]  "
        f"std [{feat_std.min():.4f}, {feat_std.max():.4f}]"
    )
    print("-" * 70)

    model = CoreSetSTGCN_MultiTask(
        in_channels=ANGLE_FEATURE_DIM,
        num_classes=int(config["num_classes"]),
        max_frames=max_frames,
        node_count=int(config["node_count"]),
    ).to(device)

    print(f"  Parameters     : {sum(p.numel() for p in model.parameters()):,}")

    # ---- Optimizer -------------------------------------------------------
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in name.lower() for nd in ("bn", "norm", "bias", "adj")):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(config.get("scheduler_t0", 20)),
        T_mult=int(config.get("scheduler_t_mult", 2)),
        eta_min=float(config.get("eta_min", 5e-6)),
    )

    # ---- Loss and hyperparameters ---------------------------------------
    class_weights = None
    if bool(config.get("use_class_weights", True)):
        class_weights = compute_class_weights(train_ds.labels, int(config["num_classes"]), device)
        print(f"  Class weights  : {[round(float(w), 3) for w in class_weights.detach().cpu()]}")

    loss_cls = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(config.get("label_smoothing", 0.03)),
    )
    loss_density = PeakWeightedBCELoss(alpha=float(config.get("density_alpha", 3.0)))

    lambda_cls = float(config.get("lambda_cls", 1.0))
    # Keep density auxiliary. A lower default protects classification accuracy.
    lambda_density_max = float(config.get("lambda_density_max", 0.20))
    warmup_epochs = int(config.get("warmup_epochs", 12))
    mixup_alpha = float(config.get("mixup_alpha", 0.05))
    patience = int(config.get("patience", 30))
    noise_std = float(config.get("noise_std", 0.01))
    grad_clip = float(config.get("grad_clip", 1.0))
    accumulation_steps = max(1, int(config.get("accumulation_steps", 1)))

    # Checkpoint selection is now classification-first, with counting as a
    # secondary objective. This avoids choosing a low-accuracy model just
    # because its OBO is slightly better.
    score_acc_weight = float(config.get("score_acc_weight", 0.85))
    score_obo_weight = float(config.get("score_obo_weight", 0.15))
    score_nmae_weight = float(config.get("score_nmae_weight", 0.05))

    use_amp = bool(config.get("use_amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"  AMP enabled    : {use_amp}")
    print(f"  Patience       : {patience}")
    print(
        f"  Score formula  : {score_acc_weight:.2f}*accuracy "
        f"+ {score_obo_weight:.2f}*OBO - {score_nmae_weight:.2f}*nMAE"
    )
    print("=" * 70)

    evaluator = CoreSetEvaluator()
    best_score = float("-inf")
    best_val_acc = 0.0
    best_obo = 0.0
    best_nmae = float("inf")
    best_checkpoint_path: Path | None = None
    best_acc = float("-inf")
    best_acc_checkpoint_path: Path | None = None
    best_count_score = float("-inf")
    best_count_checkpoint_path: Path | None = None
    no_improve = 0

    def autocast_context():
        if use_amp:
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()

    def save_checkpoint(
        epoch: int,
        val_loss: float,
        val_acc: float,
        obo: float,
        nmae: float,
        rmse: float,
        score: float,
        postprocess: dict,
        tag: str = "score",
    ) -> Path:
        ckpt_path = run_dir / (
            f"{tag}_epoch_{epoch:03d}_score_{score:.4f}_"
            f"acc_{val_acc:.4f}_obo_{obo:.4f}_nmae_{nmae:.4f}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_nmae": nmae,
                "val_rmse": rmse,
                "obo": obo,
                "score": score,
                "score_formula": {
                    "accuracy_weight": score_acc_weight,
                    "obo_weight": score_obo_weight,
                    "nmae_penalty_weight": score_nmae_weight,
                },
                "counting_postprocess": postprocess,
                "config": config,
                "config_path": config_path,
                "run_dir": run_dir.as_posix(),
                "feat_mean": feat_mean.detach().cpu(),
                "feat_std": feat_std.detach().cpu(),
            },
            ckpt_path,
        )
        if tag == "score":
            (run_dir / "best_checkpoint.txt").write_text(ckpt_path.as_posix(), encoding="utf-8")
            write_latest_checkpoint_pointer(checkpoint_root, ckpt_path, prefix="stgcn")
        return ckpt_path


    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    for epoch in range(int(config["epochs"])):
        if epoch < warmup_epochs:
            lambda_den = 0.0
        else:
            ramp = min(1.0, (epoch - warmup_epochs) / max(warmup_epochs, 1))
            lambda_den = lambda_density_max * ramp

        model.train()
        total_loss = 0.0
        cls_loss_sum = 0.0
        den_loss_sum = 0.0
        first_logged_step = False

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (inputs, labels, density_gts) in enumerate(train_loader):
            is_update_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))

            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))
            density_gts = density_gts.to(device, non_blocking=(device.type == "cuda"))

            inputs = (inputs - feat_mean) / feat_std
            if noise_std > 0:
                inputs = inputs + torch.randn_like(inputs) * noise_std
            density_norm = normalize_density_gt(density_gts)

            inputs, la, lb, lam = mixup_cls(inputs, labels, alpha=mixup_alpha, device=device)

            with autocast_context():
                logits, density_maps = model(inputs)
                l_cls = lam * loss_cls(logits, la) + (1 - lam) * loss_cls(logits, lb)
                l_den = loss_density(density_maps, density_norm)
                loss = lambda_cls * l_cls + lambda_den * l_den
                loss_for_backward = loss / accumulation_steps

            if use_amp:
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            if is_update_step:
                if use_amp:
                    scaler.unscale_(optimizer)

                if (not first_logged_step) and epoch % 10 == 0:
                    cls_params = list(model.head_classification.parameters())
                    den_params = list(model.head_density.parameters())
                    gn_cls = grad_norm(cls_params)
                    gn_den = grad_norm(den_params)
                    ratio = gn_den / (gn_cls + 1e-8)
                    print(
                        f"  [Grad balance epoch {epoch + 1}] "
                        f"||grad cls||={gn_cls:.3f}  ||grad den||={gn_den:.3f}  "
                        f"ratio={ratio:.2f}  lambda_den={lambda_den:.3f}"
                    )
                    first_logged_step = True

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            cls_loss_sum += l_cls.item()
            den_loss_sum += l_den.item()

        n_batches = max(len(train_loader), 1)
        avg_train = total_loss / n_batches
        avg_cls = cls_loss_sum / n_batches
        avg_den = den_loss_sum / n_batches
        scheduler.step(epoch + 1)

        # ---- Validate ----------------------------------------------------
        model.eval()
        total_val = 0.0
        all_logits, all_labels_v = [], []
        all_dp, all_dg = [], []

        with torch.no_grad():
            for inputs, labels, density_gts in val_loader:
                inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
                labels = labels.to(device, non_blocking=(device.type == "cuda"))
                density_gts = density_gts.to(device, non_blocking=(device.type == "cuda"))

                inputs = (inputs - feat_mean) / feat_std
                density_norm = normalize_density_gt(density_gts)

                with autocast_context():
                    logits, density_maps = model(inputs)
                    l_cls = loss_cls(logits, labels)
                    l_den = loss_density(density_maps, density_norm)
                    total_val += (lambda_cls * l_cls + lambda_den * l_den).item()

                all_logits.append(logits.detach().cpu())
                all_labels_v.append(labels.detach().cpu())
                all_dp.append(density_maps.detach().cpu().numpy())
                all_dg.append(density_gts.detach().cpu().numpy())

        avg_val = total_val / max(len(val_loader), 1)
        val_acc = evaluator.calculate_classification_accuracy(torch.cat(all_logits), torch.cat(all_labels_v))
        val_postprocess, val_nmae, val_rmse, val_obo = tune_counting_postprocess(
            np.concatenate(all_dp),
            np.concatenate(all_dg),
            config,
        )
        score = (
            score_acc_weight * val_acc
            + score_obo_weight * val_obo
            - score_nmae_weight * val_nmae
        )
        count_score = val_obo - val_nmae

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Ep [{epoch + 1:3d}/{config['epochs']}] "
            f"loss {avg_train:.3f} (cls {avg_cls:.3f} den {avg_den:.3f}) "
            f"| val {avg_val:.3f} "
            f"| acc {val_acc * 100:.1f}% obo {val_obo * 100:.1f}% "
            f"nmae {val_nmae:.3f} "
            f"| score {score:.3f} | lr {lr_now:.1e}"
        )

        append_history_row(
            history_path,
            {
                "epoch": epoch + 1,
                "train_loss": round(avg_train, 6),
                "train_cls_loss": round(avg_cls, 6),
                "train_density_loss": round(avg_den, 6),
                "val_loss": round(avg_val, 6),
                "val_accuracy": round(val_acc, 6),
                "val_obo": round(val_obo, 6),
                "val_nmae": round(val_nmae, 6),
                "val_rmse": round(val_rmse, 6),
                "score": round(score, 6),
                "count_threshold_ratio": val_postprocess.get("threshold_ratio"),
                "count_threshold_floor": val_postprocess.get("threshold_floor"),
                "count_min_distance": val_postprocess.get("min_distance"),
                "count_prominence_ratio": val_postprocess.get("prominence_ratio"),
                "lr": lr_now,
                "lambda_density": round(lambda_den, 6),
            },
        )

        # ---- Versioned checkpoints --------------------------------------
        improved_overall = score > best_score
        improved_acc = val_acc > best_acc
        improved_count = count_score > best_count_score

        if improved_overall:
            best_score = score
            best_val_acc = val_acc
            best_obo = val_obo
            best_nmae = val_nmae
            no_improve = 0
            best_checkpoint_path = save_checkpoint(
                epoch + 1, avg_val, val_acc, val_obo, val_nmae, val_rmse,
                score, val_postprocess, tag="score"
            )
            print(f"    ✓ saved new overall best: {best_checkpoint_path.as_posix()}")
        else:
            no_improve += 1

        if improved_acc:
            best_acc = val_acc
            best_acc_checkpoint_path = save_checkpoint(
                epoch + 1, avg_val, val_acc, val_obo, val_nmae, val_rmse,
                score, val_postprocess, tag="accuracy"
            )
            print(f"    ✓ saved new accuracy best: {best_acc_checkpoint_path.as_posix()}")

        if improved_count:
            best_count_score = count_score
            best_count_checkpoint_path = save_checkpoint(
                epoch + 1, avg_val, val_acc, val_obo, val_nmae, val_rmse,
                score, val_postprocess, tag="counting"
            )

        if no_improve >= patience:
            print(f"\n  Early stop at epoch {epoch + 1}.")
            break


    if best_checkpoint_path is None:
        raise RuntimeError("Training finished without producing a checkpoint.")

    # Make stable names inside this new run folder. This does not overwrite old
    # runs because each training session has its own stgcn_vXXX folder.
    stable_best_path = run_dir / "best_stgcn_model.pth"
    shutil.copy2(best_checkpoint_path, stable_best_path)
    write_latest_checkpoint_pointer(checkpoint_root, stable_best_path, prefix="stgcn")

    stable_acc_path = None
    if best_acc_checkpoint_path is not None:
        stable_acc_path = run_dir / "best_stgcn_by_accuracy.pth"
        shutil.copy2(best_acc_checkpoint_path, stable_acc_path)

    stable_count_path = None
    if best_count_checkpoint_path is not None:
        stable_count_path = run_dir / "best_stgcn_by_counting.pth"
        shutil.copy2(best_count_checkpoint_path, stable_count_path)

    # ---- Test evaluation -------------------------------------------------
    print("\n" + "=" * 70)
    ckpt = torch.load(stable_best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    feat_mean = ckpt["feat_mean"].to(device)
    feat_std = ckpt["feat_std"].to(device)
    model.eval()

    all_logits, all_labels_t = [], []
    all_dp, all_dg = [], []
    with torch.no_grad():
        for inputs, labels, density_gts in test_loader:
            inputs = (inputs.to(device, non_blocking=(device.type == "cuda")) - feat_mean) / feat_std
            with autocast_context():
                logits, density_maps = model(inputs)
            all_logits.append(logits.detach().cpu())
            all_labels_t.append(labels.detach().cpu())
            all_dp.append(density_maps.detach().cpu().numpy())
            all_dg.append(density_gts.numpy())

    test_acc = evaluator.calculate_classification_accuracy(torch.cat(all_logits), torch.cat(all_labels_t))
    test_postprocess = ckpt.get("counting_postprocess", {
        "adaptive": True,
        "threshold_ratio": 0.30,
        "threshold_floor": 0.15,
        "min_distance": 10,
        "prominence_ratio": 0.05,
    })
    test_nmae, test_rmse, test_obo, _, _ = counting_metrics_from_maps(
        np.concatenate(all_dp),
        np.concatenate(all_dg),
        test_postprocess,
    )

    metadata = {
        "run_dir": run_dir.as_posix(),
        "best_checkpoint": stable_best_path.as_posix(),
        "best_accuracy_checkpoint": stable_acc_path.as_posix() if stable_acc_path else None,
        "best_counting_checkpoint": stable_count_path.as_posix() if stable_count_path else None,
        "best_epoch_checkpoint": best_checkpoint_path.as_posix(),
        "best_score": best_score,
        "best_val_accuracy": best_val_acc,
        "best_val_obo": best_obo,
        "best_val_nmae": best_nmae,
        "counting_postprocess": test_postprocess,
        "test_accuracy": test_acc,
        "test_obo": test_obo,
        "test_nmae": test_nmae,
        "test_rmse": test_rmse,
        "history_csv": history_path.as_posix(),
        "config_snapshot": config_snapshot_path.as_posix(),
    }
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Test Accuracy   : {test_acc * 100:.2f}%")
    print(f"  Test OBO        : {test_obo * 100:.2f}%")
    print(f"  Test nMAE       : {test_nmae:.4f}")
    print(f"  Test RMSE       : {test_rmse:.4f}")
    print("=" * 70)
    print(f"  Best checkpoint : {stable_best_path.as_posix()}")
    if stable_acc_path is not None:
        print(f"  Accuracy best   : {stable_acc_path.as_posix()}")
    if stable_count_path is not None:
        print(f"  Counting best   : {stable_count_path.as_posix()}")
    print(f"  History CSV     : {history_path.as_posix()}")
    print(
        f"  Best score      : {best_score:.3f}  "
        f"(acc {best_val_acc * 100:.1f}%  obo {best_obo * 100:.1f}%  "
        f"nMAE {best_nmae:.3f})"
    )


    return stable_best_path


if __name__ == "__main__":
    train_stgcn_model("configs/stgcn_config.yaml")
