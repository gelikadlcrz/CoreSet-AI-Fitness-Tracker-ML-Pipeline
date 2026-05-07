import os
import json
import time
import random

import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from src.models.MultiTaskBiLSTM import MultiTaskBiLSTM
from src.data.bilstm_dataset import BiLSTMDataset
from src.utils.metrics import CoreSetEvaluator


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_splits(path):
    with open(path, 'r') as f:
        return json.load(f)


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_bilstm(config_path):

    config = load_config(config_path)
    splits = load_splits("configs/data_splits.json")

    set_seed(42)

    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------

    train_dataset = BiLSTMDataset(
        config['data_dir'],
        splits['train'],
        config['max_frames']
    )

    val_dataset = BiLSTMDataset(
        config['data_dir'],
        splits['val'],
        config['max_frames']
    )

    test_dataset = BiLSTMDataset(
        config['data_dir'],
        splits['test'],
        config['max_frames']
    )

    # ---------------------------------------------------------
    # Loaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
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

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config['patience_lr']
    )

    # ---------------------------------------------------------
    # Losses
    # ---------------------------------------------------------

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    lambda_cls = config.get('lambda_cls', 1.0)
    lambda_reg = config.get('lambda_reg', 0.5)

    evaluator = CoreSetEvaluator()

    best_val_loss = float('inf')
    early_stop_counter = 0

    save_path = os.path.join(
        config['checkpoint_dir'],
        'best_bilstm.pth'
    )

    print("\nBiLSTM Multi-Task Training")
    print(
        f"Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)}"
    )

    # =========================================================
    # TRAINING LOOP
    # =========================================================

    for epoch in range(config['epochs']):

        start_time = time.time()

        model.train()

        train_loss = 0
        train_loss_cls = 0
        train_loss_reg = 0

        # -----------------------------------------------------
        # TRAIN
        # -----------------------------------------------------

        for batch_idx, (x, y, reps) in enumerate(train_loader):

            x = x.to(device)
            y = y.long().to(device)

            reps = reps.float().unsqueeze(1).to(device)

           

            optimizer.zero_grad()

            logits, rep_pred = model(x)

            loss_cls = criterion_cls(logits, y)

            loss_reg = criterion_reg(rep_pred, reps)

            loss = (
                lambda_cls * loss_cls
                + lambda_reg * loss_reg
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_loss += loss.item()
            train_loss_cls += loss_cls.item()
            train_loss_reg += loss_reg.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{config['epochs']}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss /= len(train_loader)
        train_loss_cls /= len(train_loader)
        train_loss_reg /= len(train_loader)

        # -----------------------------------------------------
        # VALIDATION
        # -----------------------------------------------------

        model.eval()

        val_loss = 0

        logits_all = []
        labels_all = []

        with torch.no_grad():

            for x, y, reps in val_loader:

                x = x.to(device)

                y = y.long().to(device)

                reps = reps.float().unsqueeze(1).to(device)

                

                logits, rep_pred = model(x)

                loss_cls = criterion_cls(logits, y)

                loss_reg = criterion_reg(rep_pred, reps)

                loss = (
                    lambda_cls * loss_cls
                    + lambda_reg * loss_reg
                )

                val_loss += loss.item()

                logits_all.append(logits.cpu())
                labels_all.append(y.cpu())

        val_loss /= len(val_loader)

        logits_all = torch.cat(logits_all)
        labels_all = torch.cat(labels_all)

        val_acc = evaluator.calculate_classification_accuracy(
            logits_all,
            labels_all
        )

        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - start_time

        # -----------------------------------------------------
        # LOGGING
        # -----------------------------------------------------

        print("=" * 60)

        print(f"Epoch [{epoch+1}/{config['epochs']}]")

        print(
            f"Train Loss: {train_loss:.4f} "
            f"(Cls: {train_loss_cls:.4f}, "
            f"Reg: {train_loss_reg:.4f})"
        )

        print(f"Val Loss  : {val_loss:.4f}")

        print(f"Val Acc   : {val_acc*100:.2f}%")

        print(f"LR        : {current_lr:.2e}")

        print(f"Time      : {epoch_time:.2f}s")

        scheduler.step(val_loss)

        # -----------------------------------------------------
        # SAVE BEST
        # -----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                save_path
            )

            early_stop_counter = 0

            print("✓ Saved new best model")

        else:
            early_stop_counter += 1

        # -----------------------------------------------------
        # EARLY STOPPING
        # -----------------------------------------------------

        if early_stop_counter >= config['patience_early_stop']:

            print("Early stopping triggered")

            break

    print("\nTraining complete")


if __name__ == "__main__":
    train_bilstm("configs/bilstm_config.yaml")