import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import json

from src.models.bilstm import MultiTaskBiLSTM
from src.data.bilstm_dataset import BiLSTMDataset
from src.utils.metrics import CoreSetEvaluator


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_splits(path):
    with open(path, 'r') as f:
        return json.load(f)


def train_bilstm(config_path):
    config = load_config(config_path)
    splits = load_splits("config/data_splits.json")

    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = BiLSTMDataset(config['data_dir'], splits['train'], config['max_frames'])
    val_dataset   = BiLSTMDataset(config['data_dir'], splits['val'], config['max_frames'])
    test_dataset  = BiLSTMDataset(config['data_dir'], splits['test'], config['max_frames'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = MultiTaskBiLSTM(
        config['input_size'],
        config['hidden_size'],
        config['num_layers'],
        config['num_classes']
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['patience_lr']
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    lambda_cls = config.get('lambda_cls', 1.0)
    lambda_reg = config.get('lambda_reg', 0.5)

    evaluator = CoreSetEvaluator()

    best_val_loss = float('inf')
    early_stop_counter = 0

    save_path = os.path.join(config['checkpoint_dir'], 'best_bilstm.pth')

    print("BiLSTM Multi-Task Training")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0

        for x, y, reps in train_loader:
            x, y, reps = x.to(device), y.to(device), reps.to(device)

            optimizer.zero_grad()

            logits, rep_pred = model(x)

            loss_cls = criterion_cls(logits, y)
            loss_reg = criterion_reg(rep_pred, reps)

            loss = lambda_cls * loss_cls + lambda_reg * loss_reg

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        logits_all = []
        labels_all = []

        with torch.no_grad():
            for x, y, reps in val_loader:
                x, y, reps = x.to(device), y.to(device), reps.to(device)

                logits, rep_pred = model(x)

                loss_cls = criterion_cls(logits, y)
                loss_reg = criterion_reg(rep_pred, reps)
                loss = lambda_cls * loss_cls + lambda_reg * loss_reg

                val_loss += loss.item()

                logits_all.append(logits.cpu())
                labels_all.append(y.cpu())

        val_loss /= len(val_loader)

        val_acc = evaluator.calculate_classification_accuracy(
            torch.cat(logits_all),
            torch.cat(labels_all)
        )

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
            print("Saved new best model")
        else:
            early_stop_counter += 1

        if early_stop_counter >= config['patience_early_stop']:
            print("Early stopping triggered")
            break

    print("Training complete")