"""
train_stgcn.py — Optimized training for CoreSet ST-GCN v3.

Key changes in this version
---------------------------
1.  DENSITY GT NORMALIZATION (most important fix):
    The raw density GT from the dataset sums to the rep count
    (e.g. 8.0 spread across 150 frames → peak values ≈ 0.05–0.2).
    Standard MSE on a sigmoid output ∈ [0,1] makes predicting zero
    everywhere near-optimal for background frames, which are ~95% of the
    signal. Peak detection at threshold=0.5 then finds nothing → OBO=9%.

    Fix: before computing the density loss, normalize each sample's GT
    by its per-sample maximum so that peaks = 1.0, background ≈ 0.
    The model learns peak shape and location. Rep count is recovered by
    counting peaks in the [0,1] predicted map — no integral needed.

2.  PEAK-WEIGHTED BCE LOSS replaces focal MSE:
    Binary cross-entropy with per-frame weights:
        w(t) = 1 + alpha * gt_norm(t)
    BCE is more appropriate than MSE for binary peak/no-peak prediction
    (which is what the normalized target encodes). alpha=9 means peak
    frames get 10× the gradient signal of background frames.

3.  DUAL CHECKPOINT: saves best-by-val-loss AND best-by-OBO separately.
    Previously the checkpoint was saved only by val-loss, so the best
    counting model could be discarded in favor of a better classifier.
    At the end of training, the OBO checkpoint is selected if its OBO
    score beats the val-loss checkpoint's OBO by a margin.

4.  VALIDATION OBO MONITORING every epoch (fast approximation):
    Run peak-detection on val density maps each epoch so early stopping
    can factor in counting quality, not just cross-entropy.

5.  DISABLE MIXUP FOR DENSITY (keep for classification only):
    Mixing two density maps produces an ambiguous target for peak detection
    (two overlapping peak patterns become indistinguishable). Mixup is
    retained for classification labels but density uses the primary sample.

6.  Lower noise σ=0.02 (was 0.05) — less perturbation after normalization.

7.  Patience increased to 25 to give the density head time to stabilize.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import yaml
from torch.utils.data import DataLoader

from src.data.coreset_dataset import CoreSetGCN_Dataset, ANGLE_FEATURE_DIM
from src.models.stgcn_multitask import CoreSetSTGCN_MultiTask
from src.utils.metrics import CoreSetEvaluator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Density GT normalization
# ---------------------------------------------------------------------------

def normalize_density_gt(density_gt: torch.Tensor) -> torch.Tensor:
    """
    Normalize each sample's density map by its per-sample maximum.

    Before: density_gt sums to rep_count; peaks ≈ 0.05–0.2 (tiny)
    After:  density_gt_norm in [0, 1]; peaks = 1.0; background ≈ 0

    Args:
        density_gt: (B, T) raw density maps from dataset

    Returns:
        density_norm: (B, T) normalized, same device
    """
    max_vals = density_gt.amax(dim=1, keepdim=True).clamp(min=1e-6)
    return density_gt / max_vals


# ---------------------------------------------------------------------------
# Peak-weighted BCE density loss
# ---------------------------------------------------------------------------

class PeakWeightedBCELoss(nn.Module):
    """
    Binary cross-entropy with higher weight on peak frames.

    L = mean( w(t) * BCE(pred(t), gt_norm(t)) )
    w(t) = 1 + alpha * gt_norm(t)

    With alpha=9: peak frames (gt_norm=1) get 10× weight vs background (gt_norm≈0).
    BCE is more principled than MSE for a [0,1] target because it is the
    proper log-loss for Bernoulli-distributed binary predictions.

    Args:
        alpha (float): Peak up-weighting. Default 9 (→ 10× for peaks).
    """
    def __init__(self, alpha: float = 9.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, gt_norm: torch.Tensor) -> torch.Tensor:
        # gt_norm should already be normalized to [0,1] by normalize_density_gt()
        bce     = F.binary_cross_entropy(pred, gt_norm, reduction='none')
        weights = 1.0 + self.alpha * gt_norm
        return (bce * weights).mean()

import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mixup (classification only)
# ---------------------------------------------------------------------------

def mixup_cls(x, labels, alpha: float = 0.2, device='cpu'):
    """
    Mixup for input and classification labels only.
    Density GT is NOT mixed (peak patterns from two samples are ambiguous).

    Returns: mixed_x, labels_a, labels_b, lam
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], labels, labels[idx], lam


# ---------------------------------------------------------------------------
# Online normalization (Welford)
# ---------------------------------------------------------------------------

def compute_norm_stats_online(loader, device):
    """Compute per-channel mean/std over training set in O(1) memory."""
    n = 0
    mean_ = None
    M2_   = None

    with torch.no_grad():
        for inputs, _, _ in loader:
            inputs = inputs.to(device)
            b, c, t, v, m = inputs.shape
            flat = inputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, c)
            batch_n    = flat.size(0)
            batch_mean = flat.mean(dim=0)

            if mean_ is None:
                mean_ = batch_mean
                M2_   = torch.zeros(c, device=device)
                n     = 0

            n_new  = n + batch_n
            delta  = batch_mean - mean_
            mean_  = mean_ + delta * batch_n / n_new
            M2_   += ((flat - mean_) ** 2).sum(dim=0)
            n      = n_new

    std_ = (M2_ / max(n - 1, 1)).sqrt().clamp(min=1e-6)
    return mean_.view(1, -1, 1, 1, 1), std_.view(1, -1, 1, 1, 1)


# ---------------------------------------------------------------------------
# Fast OBO approximation for validation monitoring
# ---------------------------------------------------------------------------

def compute_val_obo(density_pred_np: np.ndarray,
                    density_gt_np:   np.ndarray) -> float:
    """
    Compute OBO on a batch of predicted (normalized) and GT (raw) density maps.

    pred: (N, T) in [0,1] — model output
    gt:   (N, T) raw (sums to rep count)
    """
    from scipy.signal import find_peaks

    pred_counts, gt_counts = [], []
    for pred_map, gt_map in zip(density_pred_np, density_gt_np):
        # Adaptive threshold: half the predicted map's max
        thresh = max(float(pred_map.max()) * 0.3, 0.15)
        peaks, _ = find_peaks(pred_map, height=thresh, distance=10)
        pred_counts.append(len(peaks))
        gt_counts.append(round(float(gt_map.sum())))

    pred_counts = np.array(pred_counts, dtype=float)
    gt_counts   = np.array(gt_counts,   dtype=float)
    return float((np.abs(pred_counts - gt_counts) <= 1).mean())


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_stgcn_model(config_path: str = 'configs/stgcn_config.yaml'):
    config = load_config(config_path)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    print("CoreSet ST-GCN v3 (Optimized) — Training")
    print("=" * 62)

    # Hardware
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"  Hardware : {device.type.upper()}")

    # Datasets
    split_file = os.path.join('configs', 'data_splits.json')
    train_ds = CoreSetGCN_Dataset(config['data_dir'], split_file, 'train',
                                   config['max_frames'], augment=True)
    val_ds   = CoreSetGCN_Dataset(config['data_dir'], split_file, 'val',
                                   config['max_frames'], augment=False)
    test_ds  = CoreSetGCN_Dataset(config['data_dir'], split_file, 'test',
                                   config['max_frames'], augment=False)

    print(f"  Data     : {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test")
    print("-" * 62)

    nw = min(4, os.cpu_count() or 0)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                               shuffle=True,  num_workers=nw,
                               pin_memory=(device.type == 'cuda'),
                               persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                               shuffle=False, num_workers=nw,
                               persistent_workers=(nw > 0))
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'],
                               shuffle=False, num_workers=nw,
                               persistent_workers=(nw > 0))

    # Normalization
    print("  Computing normalization statistics...")
    feat_mean, feat_std = compute_norm_stats_online(train_loader, device)
    print(f"  mean [{feat_mean.min():.4f}, {feat_mean.max():.4f}]  "
          f"std [{feat_std.min():.4f}, {feat_std.max():.4f}]")
    print("-" * 62)

    # Model
    model = CoreSetSTGCN_MultiTask(
        in_channels=ANGLE_FEATURE_DIM,
        num_classes=config['num_classes'],
        max_frames=config['max_frames'],
        node_count=config['node_count']
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")
    print("=" * 62)

    # Optimizer — AdamW with param group weight decay
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in name for nd in ('bn', 'norm', 'bias', 'adj')):
            no_decay.append(p)
        else:
            decay.append(p)

    optimizer = optim.AdamW(
        [{'params': decay,    'weight_decay': config['weight_decay']},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=config['learning_rate']
    )

    # LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=5e-6
    )

    # Loss functions
    loss_cls     = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_density = PeakWeightedBCELoss(alpha=9.0)

    LAMBDA_CLS     = 1.0
    LAMBDA_DENSITY = 2.0    # up-weight density — it needs more gradient
    MIXUP_ALPHA    = 0.2

    # Training state
    evaluator          = CoreSetEvaluator()
    best_val_loss      = float('inf')
    best_obo           = 0.0
    best_val_accuracy  = 0.0
    early_stop_patience = 25
    epochs_no_improve   = 0

    ckpt_loss = os.path.join(config['checkpoint_dir'], 'best_stgcn_model.pth')
    ckpt_obo  = os.path.join(config['checkpoint_dir'], 'best_stgcn_obo.pth')

    def _save(path, epoch, val_loss, val_acc, obo):
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss, 'val_accuracy': val_acc,
            'obo': obo,
            'config': config, 'feat_mean': feat_mean, 'feat_std': feat_std,
        }, path)

    for epoch in range(config['epochs']):

        # ---- Train -------------------------------------------------------
        model.train()
        total_train_loss = 0.0

        for inputs, labels, density_gts in train_loader:
            inputs      = inputs.to(device)
            labels      = labels.to(device)
            density_gts = density_gts.to(device)

            inputs = (inputs - feat_mean) / feat_std
            inputs = inputs + torch.randn_like(inputs) * 0.02

            # Normalize density GT to [0,1] per sample
            density_norm = normalize_density_gt(density_gts)

            # Mixup on inputs + cls labels only (density stays primary)
            inputs, labels_a, labels_b, lam = mixup_cls(
                inputs, labels, alpha=MIXUP_ALPHA, device=device
            )

            optimizer.zero_grad()
            logits, density_maps = model(inputs)

            l_cls = (lam * loss_cls(logits, labels_a)
                     + (1 - lam) * loss_cls(logits, labels_b))
            l_den = loss_density(density_maps, density_norm)
            loss  = LAMBDA_CLS * l_cls + LAMBDA_DENSITY * l_den

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        scheduler.step(epoch + 1)

        # ---- Validate ----------------------------------------------------
        model.eval()
        total_val_loss = 0.0
        all_logits, all_labels_v = [], []
        all_den_pred, all_den_gt = [], []

        with torch.no_grad():
            for inputs, labels, density_gts in val_loader:
                inputs      = inputs.to(device)
                labels      = labels.to(device)
                density_gts = density_gts.to(device)

                inputs       = (inputs - feat_mean) / feat_std
                density_norm = normalize_density_gt(density_gts)

                logits, density_maps = model(inputs)
                l_cls = loss_cls(logits, labels)
                l_den = loss_density(density_maps, density_norm)
                loss  = LAMBDA_CLS * l_cls + LAMBDA_DENSITY * l_den

                total_val_loss += loss.item()
                all_logits.append(logits.cpu())
                all_labels_v.append(labels.cpu())
                all_den_pred.append(density_maps.cpu().numpy())
                all_den_gt.append(density_gts.cpu().numpy())

        avg_val   = total_val_loss / len(val_loader)
        val_acc   = evaluator.calculate_classification_accuracy(
            torch.cat(all_logits), torch.cat(all_labels_v)
        )
        val_obo   = compute_val_obo(
            np.concatenate(all_den_pred),
            np.concatenate(all_den_gt)
        )

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch+1:3d}/{config['epochs']}] "
            f"| Train {avg_train:.4f} | Val {avg_val:.4f} "
            f"| Acc {val_acc*100:.1f}% | OBO {val_obo*100:.1f}% "
            f"| LR {lr_now:.2e}"
        )

        # ---- Checkpoint --------------------------------------------------
        improved = False

        if avg_val < best_val_loss:
            best_val_loss     = avg_val
            best_val_accuracy = val_acc
            _save(ckpt_loss, epoch+1, avg_val, val_acc, val_obo)
            print(f"    ✓ best-loss saved  (OBO {val_obo*100:.1f}%)")
            improved = True

        if val_obo > best_obo:
            best_obo = val_obo
            _save(ckpt_obo, epoch+1, avg_val, val_acc, val_obo)
            print(f"    ✓ best-OBO  saved  (OBO {val_obo*100:.1f}%)")
            improved = True

        if not improved:
            epochs_no_improve += 1
            if epochs_no_improve % 5 == 0:
                print(f"    No improvement ({epochs_no_improve}/{early_stop_patience})")
        else:
            epochs_no_improve = 0

        if epochs_no_improve >= early_stop_patience:
            print(f"\n  Early stopping at epoch {epoch+1}.")
            break

    # ---- Select best checkpoint ----------------------------------------
    # Use the OBO checkpoint if it exists and has a better OBO
    selected_ckpt = ckpt_loss
    if os.path.exists(ckpt_obo):
        obo_ckpt_data  = torch.load(ckpt_obo,  map_location='cpu')
        loss_ckpt_data = torch.load(ckpt_loss, map_location='cpu')
        if obo_ckpt_data.get('obo', 0) > loss_ckpt_data.get('obo', 0) + 0.02:
            selected_ckpt = ckpt_obo
            print(f"\n  Selecting OBO checkpoint (OBO "
                  f"{obo_ckpt_data['obo']*100:.1f}% vs "
                  f"{loss_ckpt_data.get('obo',0)*100:.1f}%)")

    # Copy selected to canonical path if not already there
    if selected_ckpt != ckpt_loss:
        import shutil
        shutil.copy(selected_ckpt, ckpt_loss)

    # ---- Quick test eval ------------------------------------------------
    print("\n" + "=" * 62)
    ckpt = torch.load(ckpt_loss, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    feat_mean = ckpt['feat_mean'].to(device)
    feat_std  = ckpt['feat_std'].to(device)

    model.eval()
    all_logits, all_labels_t = [], []
    all_den_pred, all_den_gt = [], []

    with torch.no_grad():
        for inputs, labels, density_gts in test_loader:
            inputs = (inputs.to(device) - feat_mean) / feat_std
            logits, density_maps = model(inputs)
            all_logits.append(logits.cpu())
            all_labels_t.append(labels)
            all_den_pred.append(density_maps.cpu().numpy())
            all_den_gt.append(density_gts.numpy())

    test_acc = evaluator.calculate_classification_accuracy(
        torch.cat(all_logits), torch.cat(all_labels_t)
    )
    test_obo = compute_val_obo(
        np.concatenate(all_den_pred),
        np.concatenate(all_den_gt)
    )
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test OBO      : {test_obo*100:.2f}%")
    print("=" * 62)
    print(f"  Checkpoint: {ckpt_loss}")


if __name__ == '__main__':
    train_stgcn_model('configs/stgcn_config.yaml')
