import torch
from torch.utils.data import DataLoader
import yaml
import json
import numpy as np

from src.models.bilstm import MultiTaskBiLSTM
from src.data.bilstm_dataset import BiLSTMDataset
from src.utils.metrics import CoreSetEvaluator


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_splits(path):
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_bilstm(config_path):
    config = load_config(config_path)
    splits = load_splits("config/data_splits.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = BiLSTMDataset(
        config['data_dir'],
        splits['test'],
        config['max_frames']
    )

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = MultiTaskBiLSTM(
        config['input_size'],
        config['hidden_size'],
        config['num_layers'],
        config['num_classes']
    ).to(device)

    model.load_state_dict(torch.load("saved_models/best_bilstm.pth", weights_only=True))
    model.eval()

    evaluator = CoreSetEvaluator()

    logits_all = []
    labels_all = []
    pred_counts = []
    true_counts = []

    with torch.no_grad():
        for x, y, reps in test_loader:
            x = x.to(device)

            logits, rep_pred = model(x)

            logits_all.append(logits.cpu())
            labels_all.append(y)
            pred_counts.extend(rep_pred.cpu().numpy())
            true_counts.extend(reps.numpy())

    logits_all = torch.cat(logits_all)
    labels_all = torch.cat(labels_all)

    acc = evaluator.calculate_classification_accuracy(logits_all, labels_all)
    mae = evaluator.calculate_normalized_mae(pred_counts, true_counts)
    rmse = evaluator.calculate_rmse(pred_counts, true_counts)
    obo = evaluator.calculate_obo_accuracy(pred_counts, true_counts)

    print("\nEvaluation Results")
    print("----------------------------------------")
    print(f"Classification Accuracy : {acc:.4f}")
    print(f"Normalized MAE          : {mae:.4f}")
    print(f"RMSE                    : {rmse:.4f}")
    print(f"OBO Accuracy            : {obo:.4f}")
    print("----------------------------------------")


if __name__ == "__main__":
    evaluate_bilstm("configs/bilstm_config.yaml")