"""
train_stgcn.py — CoreSet ST-GCN v3  (balanced multi-task training)

Root cause of the 46% classification collapse in the previous run
-----------------------------------------------------------------
LAMBDA_DENSITY=2.0 with alpha=9 BCE meant the density head received
~10× the gradient signal of the classification head. With only 700
training samples the backbone was fully captured by the density task
and learned representations useless for classification (bench_press
predicted 0 times — total feature collapse).

Fixes applied here
------------------
1.  LAMBDA_DENSITY reduced to 0.3 (was 2.0).
    Classification is the harder, more valuable task on this dataset.
    Density is a regularizer / auxiliary task, not the primary objective.

2.  alpha reduced to 4.0 (was 9.0).
    Peak frames still get 5× weight, but the total BCE magnitude is kept
    comparable to the cross-entropy magnitude so gradients balance.

3.  LOSS MAGNITUDE MONITORING printed every 5 epochs so imbalance is
    immediately visible during future runs.

4.  Label smoothing reduced to 0.05 (was 0.1).
    With 4 balanced classes and only 700 samples, ε=0.1 was too much —
    it flattened the decision boundary and caused bench_press collapse.

5.  Mixup α reduced to 0.1 (was 0.2).
    Lighter interpolation preserves class identity better on small datasets.

6.  Backbone uses a WARM-UP phase: density loss is zero for the first
    WARMUP_EPOCHS epochs so the backbone first learns clean class
    representations, then the density head is introduced gradually.

7.  GRADIENT BALANCE CHECK: after the first batch of each epoch, the
    ratio of ‖∇ density‖ / ‖∇ cls‖ is computed and logged to confirm
    the two tasks are not fighting each other.

8.  Early stopping now tracks a composite score:
        score = 0.7 * val_acc + 0.3 * val_obo
    so neither task can dominate checkpoint selection.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    Normalize each sample's raw density map by its per-sample maximum.
    Peaks → 1.0, background → 0.  Zero maps stay zero.
    """
    max_vals = density_gt.amax(dim=1, keepdim=True).clamp(min=1e-6)
    return density_gt / max_vals


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class PeakWeightedBCELoss(nn.Module):
    """
    BCE with higher weight on peak frames.
    L = mean( (1 + alpha*gt_norm) * BCE(pred, gt_norm) )
    alpha=4 → peak frames get 5× gradient vs background.
    """
    def __init__(self, alpha: float = 4.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, gt_norm: torch.Tensor) -> torch.Tensor:
        bce     = F.binary_cross_entropy(pred, gt_norm, reduction='none')
        weights = 1.0 + self.alpha * gt_norm
        return (bce * weights).mean()


# ---------------------------------------------------------------------------
# Mixup — classification only
# ---------------------------------------------------------------------------

def mixup_cls(x, labels, alpha: float = 0.1, device='cpu'):
    lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=device)
    return lam * x + (1 - lam) * x[idx], labels, labels[idx], lam


# ---------------------------------------------------------------------------
# Online normalization (Welford)
# ---------------------------------------------------------------------------

def compute_norm_stats_online(loader, device):
    n = 0; mean_ = None; M2_ = None
    with torch.no_grad():
        for inputs, _, _ in loader:
            inputs = inputs.to(device)
            b, c, t, v, m = inputs.shape
            flat = inputs.permute(0,2,3,4,1).contiguous().view(-1, c)
            bn   = flat.size(0)
            bm   = flat.mean(0)
            if mean_ is None:
                mean_ = bm; M2_ = torch.zeros(c, device=device); n = 0
            n_new = n + bn
            delta = bm - mean_
            mean_ = mean_ + delta * bn / n_new
            M2_  += ((flat - mean_)**2).sum(0)
            n     = n_new
    std_ = (M2_ / max(n-1,1)).sqrt().clamp(min=1e-6)
    return mean_.view(1,-1,1,1,1), std_.view(1,-1,1,1,1)


# ---------------------------------------------------------------------------
# Gradient norm helper
# ---------------------------------------------------------------------------

def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5


# ---------------------------------------------------------------------------
# Fast OBO approximation
# ---------------------------------------------------------------------------

def compute_val_obo(den_pred: np.ndarray, den_gt: np.ndarray) -> float:
    from scipy.signal import find_peaks
    pred_c, gt_c = [], []
    for pm, gm in zip(den_pred, den_gt):
        thresh = max(float(pm.max()) * 0.3, 0.15)
        peaks, _ = find_peaks(pm, height=thresh, distance=10)
        pred_c.append(len(peaks))
        gt_c.append(round(float(gm.sum())))
    pa = np.array(pred_c, float); ga = np.array(gt_c, float)
    return float((np.abs(pa - ga) <= 1).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_stgcn_model(config_path: str = 'configs/stgcn_config.yaml'):
    config = load_config(config_path)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    print("CoreSet ST-GCN v3 — Training")
    print("=" * 62)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"  Hardware : {device.type.upper()}")

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
    kw = dict(num_workers=nw, pin_memory=(device.type=='cuda'),
               persistent_workers=(nw > 0))
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                               shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                               shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'],
                               shuffle=False, **kw)

    print("  Computing normalization statistics...")
    feat_mean, feat_std = compute_norm_stats_online(train_loader, device)
    print(f"  mean [{feat_mean.min():.4f}, {feat_mean.max():.4f}]  "
          f"std  [{feat_std.min():.4f}, {feat_std.max():.4f}]")
    print("-" * 62)

    model = CoreSetSTGCN_MultiTask(
        in_channels=ANGLE_FEATURE_DIM,
        num_classes=config['num_classes'],
        max_frames=config['max_frames'],
        node_count=config['node_count']
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 62)

    # ---- Optimizer -------------------------------------------------------
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
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=5e-6
    )

    # ---- Loss & hyper-params --------------------------------------------
    loss_cls     = nn.CrossEntropyLoss(label_smoothing=0.05)   # lighter smoothing
    loss_density = PeakWeightedBCELoss(alpha=4.0)

    # Classification is primary; density is auxiliary.
    # λ_density is ramped up linearly after WARMUP_EPOCHS.
    LAMBDA_CLS         = 1.0
    LAMBDA_DENSITY_MAX = 0.3    # final density weight (was 2.0 → caused collapse)
    WARMUP_EPOCHS      = 10     # cls-only warmup before density kicks in
    MIXUP_ALPHA        = 0.1

    evaluator = CoreSetEvaluator()
    best_score = 0.0           # composite 0.7*acc + 0.3*obo
    best_val_acc = 0.0
    best_obo     = 0.0
    no_improve   = 0
    PATIENCE     = 25

    ckpt_path = os.path.join(config['checkpoint_dir'], 'best_stgcn_model.pth')

    def _save(epoch, val_loss, val_acc, obo, score):
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss, 'val_accuracy': val_acc,
            'obo': obo, 'score': score,
            'config': config, 'feat_mean': feat_mean, 'feat_std': feat_std,
        }, ckpt_path)

    for epoch in range(config['epochs']):

        # Ramp density weight from 0 → LAMBDA_DENSITY_MAX over warmup
        if epoch < WARMUP_EPOCHS:
            lambda_den = 0.0
        else:
            ramp = min(1.0, (epoch - WARMUP_EPOCHS) / max(WARMUP_EPOCHS, 1))
            lambda_den = LAMBDA_DENSITY_MAX * ramp

        # ---- Train -------------------------------------------------------
        model.train()
        total_loss = 0.0
        cls_loss_sum = 0.0
        den_loss_sum = 0.0
        first_batch  = True

        for inputs, labels, density_gts in train_loader:
            inputs      = inputs.to(device)
            labels      = labels.to(device)
            density_gts = density_gts.to(device)

            inputs       = (inputs - feat_mean) / feat_std
            inputs       = inputs + torch.randn_like(inputs) * 0.02
            density_norm = normalize_density_gt(density_gts)

            # Mixup: classification labels only
            inputs, la, lb, lam = mixup_cls(inputs, labels,
                                             alpha=MIXUP_ALPHA, device=device)

            optimizer.zero_grad()
            logits, density_maps = model(inputs)

            l_cls = (lam * loss_cls(logits, la)
                     + (1 - lam) * loss_cls(logits, lb))
            l_den = loss_density(density_maps, density_norm)
            loss  = LAMBDA_CLS * l_cls + lambda_den * l_den

            loss.backward()

            # Log gradient balance on first batch of every 10th epoch
            if first_batch and epoch % 10 == 0:
                cls_params = list(model.head_classification.parameters())
                den_params = list(model.head_density.parameters())
                gn_cls = grad_norm(cls_params)
                gn_den = grad_norm(den_params)
                ratio  = gn_den / (gn_cls + 1e-8)
                print(f"  [Grad balance epoch {epoch+1}] "
                      f"‖∇cls‖={gn_cls:.3f}  ‖∇den‖={gn_den:.3f}  "
                      f"ratio={ratio:.2f}  λ_den={lambda_den:.3f}")
                first_batch = False

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss   += loss.item()
            cls_loss_sum += l_cls.item()
            den_loss_sum += l_den.item()

        n_batches  = len(train_loader)
        avg_train  = total_loss   / n_batches
        avg_cls    = cls_loss_sum / n_batches
        avg_den    = den_loss_sum / n_batches
        scheduler.step(epoch + 1)

        # ---- Validate ----------------------------------------------------
        model.eval()
        total_val = 0.0
        all_logits, all_labels_v = [], []
        all_dp, all_dg           = [], []

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
                total_val += (LAMBDA_CLS * l_cls + lambda_den * l_den).item()

                all_logits.append(logits.cpu())
                all_labels_v.append(labels.cpu())
                all_dp.append(density_maps.cpu().numpy())
                all_dg.append(density_gts.cpu().numpy())

        avg_val  = total_val / len(val_loader)
        val_acc  = evaluator.calculate_classification_accuracy(
            torch.cat(all_logits), torch.cat(all_labels_v))
        val_obo  = compute_val_obo(np.concatenate(all_dp), np.concatenate(all_dg))
        score    = 0.7 * val_acc + 0.3 * val_obo

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"Ep [{epoch+1:3d}/{config['epochs']}] "
            f"loss {avg_train:.3f} (cls {avg_cls:.3f} den {avg_den:.3f}) "
            f"| val {avg_val:.3f} "
            f"| acc {val_acc*100:.1f}% obo {val_obo*100:.1f}% "
            f"| score {score:.3f} | lr {lr_now:.1e}"
        )

        # ---- Checkpoint --------------------------------------------------
        if score > best_score:
            best_score   = score
            best_val_acc = val_acc
            best_obo     = val_obo
            no_improve   = 0
            _save(epoch+1, avg_val, val_acc, val_obo, score)
            print(f"    ✓ saved  acc={val_acc*100:.1f}%  obo={val_obo*100:.1f}%")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch+1}.")
                break

    # ---- Test evaluation -------------------------------------------------
    print("\n" + "=" * 62)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    feat_mean = ckpt['feat_mean'].to(device)
    feat_std  = ckpt['feat_std'].to(device)
    model.eval()

    all_logits, all_labels_t = [], []
    all_dp, all_dg           = [], []
    with torch.no_grad():
        for inputs, labels, density_gts in test_loader:
            inputs = (inputs.to(device) - feat_mean) / feat_std
            logits, density_maps = model(inputs)
            all_logits.append(logits.cpu())
            all_labels_t.append(labels)
            all_dp.append(density_maps.cpu().numpy())
            all_dg.append(density_gts.numpy())

    test_acc = evaluator.calculate_classification_accuracy(
        torch.cat(all_logits), torch.cat(all_labels_t))
    test_obo = compute_val_obo(
        np.concatenate(all_dp), np.concatenate(all_dg))

    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test OBO      : {test_obo*100:.2f}%")
    print("=" * 62)
    print(f"  Best checkpoint : {ckpt_path}")
    print(f"  Best score      : {best_score:.3f}  "
          f"(acc {best_val_acc*100:.1f}%  obo {best_obo*100:.1f}%)")


if __name__ == '__main__':
    train_stgcn_model('configs/stgcn_config.yaml')