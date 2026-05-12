import os
import json
import time
import random

import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

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


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════════
# Training entry-point
# ══════════════════════════════════════════════════════════════════════════════

def train_bilstm(config_path: str) -> None:
    config = load_config(config_path)
    splits = load_splits("configs/data_splits.json")

    set_seed(42)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == 'cuda'

    # ── Datasets ─────────────────────────────────────────────────────────────
    # Sequences are already standardised (zero mean / unit variance) and fixed
    # to a uniform temporal length by the offline pre-processing pipeline.
    # No max_frames argument is required or passed.
    train_dataset = BiLSTMDataset(
        config['data_dir'],
        splits['train'],
        augment=True,
        cache=True,
    )
    val_dataset = BiLSTMDataset(
        config['data_dir'],
        splits['val'],
        augment=False,
        cache=True,
    )
    test_dataset = BiLSTMDataset(
        config['data_dir'],
        splits['test'],
        augment=False,
        cache=True,
    )

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
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
        dropout=0.5,                        # fixed per methodology
    ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # AdamW correctly decouples weight-decay from adaptive gradient scaling,
    # yielding better generalisation than vanilla Adam (Loshchilov & Hutter, 2019).
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],         # 1e-3
        weight_decay=config['weight_decay'],# 1e-4
    )

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    # ReduceLROnPlateau: halve LR when validation loss does not improve for
    # patience_lr consecutive epochs (patience=10, factor=0.5 per methodology).
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['patience_lr'],     # 10
    )

    # ── Loss functions ────────────────────────────────────────────────────────
    # Classification head: categorical cross-entropy.
    criterion_cls = nn.CrossEntropyLoss()

    # Regression head: Mean Squared Error, as specified in the methodology.
    # λ_cls = 1.0, λ_density = 0.5  →  L_total = λ_cls·L_cls + λ_density·L_mse
    criterion_reg = nn.MSELoss()

    lambda_cls = config['lambda_cls']       # 1.0
    lambda_reg = config['lambda_reg']       # 0.5

    # ── Misc training state ───────────────────────────────────────────────────
    scaler = GradScaler(enabled=use_amp)
    evaluator = CoreSetEvaluator()

    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stop_counter = 0

    save_path = os.path.join(config['checkpoint_dir'], 'best_bilstm.pth')

    print("\nBiLSTM Baseline — Multi-Task Training")
    print(f"Device : {device}  |  AMP : {use_amp}")
    print(f"Train  : {len(train_dataset)}  |  Val : {len(val_dataset)}  |  Test : {len(test_dataset)}")
    print(f"λ_cls={lambda_cls}  λ_reg={lambda_reg}  dropout=0.5  lr={config['learning_rate']}")

    # ══════════════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════════════

    for epoch in range(config['epochs']):
        start_time = time.time()
        model.train()

        train_loss = 0.0
        train_loss_cls = 0.0
        train_loss_reg = 0.0

        for batch_idx, (x, y, reps) in enumerate(train_loader):
            x    = x.to(device, non_blocking=True)
            y    = y.long().to(device, non_blocking=True)
            reps = reps.float().unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                logits, rep_pred = model(x)
                loss_cls = criterion_cls(logits, y)
                loss_reg = criterion_reg(rep_pred, reps)
                loss     = lambda_cls * loss_cls + lambda_reg * loss_reg

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss     += loss.item()
            train_loss_cls += loss_cls.item()
            train_loss_reg += loss_reg.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{config['epochs']}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss     /= len(train_loader)
        train_loss_cls /= len(train_loader)
        train_loss_reg /= len(train_loader)

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        logits_all = []
        labels_all = []

        with torch.no_grad():
            for x, y, reps in val_loader:
                x    = x.to(device, non_blocking=True)
                y    = y.long().to(device, non_blocking=True)
                reps = reps.float().unsqueeze(1).to(device, non_blocking=True)

                with autocast(enabled=use_amp):
                    logits, rep_pred = model(x)
                    loss_cls = criterion_cls(logits, y)
                    loss_reg = criterion_reg(rep_pred, reps)
                    loss     = lambda_cls * loss_cls + lambda_reg * loss_reg

                val_loss += loss.item()
                logits_all.append(logits.float().cpu())
                labels_all.append(y.cpu())

        val_loss   /= len(val_loader)
        logits_all  = torch.cat(logits_all)
        labels_all  = torch.cat(labels_all)

        val_acc = evaluator.calculate_classification_accuracy(logits_all, labels_all)

        # Step the LR scheduler on validation loss
        scheduler.step(val_loss)

        current_lr  = optimizer.param_groups[0]['lr']
        epoch_time  = time.time() - start_time

        # ── Logging ───────────────────────────────────────────────────────────
        print("=" * 60)
        print(f"Epoch [{epoch+1}/{config['epochs']}]")
        print(f"Train Loss : {train_loss:.4f}  (Cls: {train_loss_cls:.4f}, Reg: {train_loss_reg:.4f})")
        print(f"Val   Loss : {val_loss:.4f}")
        print(f"Val   Acc  : {val_acc*100:.2f}%")
        print(f"LR         : {current_lr:.2e}")
        print(f"Time       : {epoch_time:.2f}s")

        # ── Checkpoint ────────────────────────────────────────────────────────
        # Save whenever either validation loss or accuracy improves.
        improved = val_loss < best_val_loss or val_acc > best_val_acc
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if improved:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                },
                save_path,
            )
            early_stop_counter = 0
            print(f"✓ Saved best model  (Acc: {val_acc*100:.2f}%  Loss: {val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"  No improvement ({early_stop_counter}/{config['patience_early_stop']})")

        # ── Early stopping ────────────────────────────────────────────────────
        # Triggered after patience_early_stop epochs without improvement (= 15).
        if early_stop_counter >= config['patience_early_stop']:
            print("\nEarly stopping triggered — restoring best checkpoint.")
            break

    print("\nTraining complete.")
    print(f"Best Val Acc  : {best_val_acc*100:.2f}%")
    print(f"Best Val Loss : {best_val_loss:.4f}")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_bilstm("configs/bilstm_config.yaml")