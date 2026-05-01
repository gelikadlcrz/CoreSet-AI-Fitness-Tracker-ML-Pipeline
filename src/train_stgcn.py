"""
train_stgcn.py — Training entry point for the CoreSet ST-GCN framework.

Implements the full training protocol described in the methodology:

  Optimiser       : AdamW (Loshchilov & Hutter, 2019).
                    The methodology cites "Adam optimizer (Kingma & Ba, 2015)"
                    but AdamW is used here because it correctly decouples the
                    weight-decay regularisation from the adaptive gradient
                    scaling, which produces better generalisation.  The paper
                    should cite both references and note AdamW was used.

  LR schedule     : ReduceLROnPlateau — reduce by factor 0.5 when validation
                    loss does not improve for 10 consecutive epochs, per the
                    methodology.

  Early stopping  : Training halts if validation loss does not improve for
                    15 consecutive epochs; best checkpoint is restored.

  Loss function   : Multi-task composite loss
                        L_total = λ₁ · L_cls + λ₂ · L_density
                    where λ₁ = 1.0 (CrossEntropyLoss) and
                          λ₂ = 0.5 (MSELoss), per the methodology.

  Partitioning    : Subject-wise GroupShuffleSplit (70 / 15 / 15).

  Dropout         : 0.5 before classification head, 0.3 before density head.

  Weight decay    : 1×10⁻⁴ (L2 regularisation via AdamW).

  Batch size      : 32.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from src.data.coreset_dataset import CoreSetGCN_Dataset, ANGLE_FEATURE_DIM
from src.models.stgcn_multitask import CoreSetSTGCN_MultiTask
from src.utils.metrics import CoreSetEvaluator


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_stgcn_model(config_path: str = 'configs/stgcn_config.yaml'):
    config = load_config(config_path)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    print("CoreSet ST-GCN — Training (Subject-Wise Split)")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Hardware detection                                                  #
    # ------------------------------------------------------------------ #
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"  Hardware: {device.type.upper()}")

    # ------------------------------------------------------------------ #
    #  Dataset — augmented for training, clean for val/test               #
    # ------------------------------------------------------------------ #
    full_train_dataset = CoreSetGCN_Dataset(
        data_dir=config['data_dir'],
        max_frames=config['max_frames'],
        augment=True
    )
    full_eval_dataset = CoreSetGCN_Dataset(
        data_dir=config['data_dir'],
        max_frames=config['max_frames'],
        augment=False
    )

    total_files = len(full_train_dataset)
    print(f"  Total files in {config['data_dir']}: {total_files}")
    for exercise, count in full_train_dataset.class_counts.items():
        print(f"    {exercise.replace('_', ' ').title()}: {count}")
    print("-" * 60)

    # ------------------------------------------------------------------ #
    #  Subject-wise partitioning (70 / 15 / 15)                           #
    #  GroupShuffleSplit ensures no subject appears in multiple splits.   #
    # ------------------------------------------------------------------ #
    subject_ids = full_train_dataset.get_subject_ids()
    indices = np.arange(total_files)

    gss_main = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=42)
    train_idx, val_test_idx = next(
        gss_main.split(indices, groups=subject_ids)
    )

    val_test_subjects = np.array(subject_ids)[val_test_idx]
    gss_val_test = GroupShuffleSplit(
        n_splits=1, train_size=0.50, random_state=42
    )
    val_rel_idx, test_rel_idx = next(
        gss_val_test.split(val_test_idx, groups=val_test_subjects)
    )
    val_idx  = val_test_idx[val_rel_idx]
    test_idx = val_test_idx[test_rel_idx]

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset   = Subset(full_eval_dataset,  val_idx)
    test_dataset  = Subset(full_eval_dataset,  test_idx)

    print(f"  Partition (subject-isolated): "
          f"{len(train_dataset)} train | "
          f"{len(val_dataset)} val | "
          f"{len(test_dataset)} test")

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=0, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=0
    )

    # ------------------------------------------------------------------ #
    #  Model                                                               #
    #  in_channels = ANGLE_FEATURE_DIM (14 joint angles per node)         #
    # ------------------------------------------------------------------ #
    model = CoreSetSTGCN_MultiTask(
        in_channels=ANGLE_FEATURE_DIM,
        num_classes=config['num_classes'],
        max_frames=config['max_frames'],
        node_count=config['node_count']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Optimiser — AdamW with L2 weight decay                             #
    #  (Loshchilov & Hutter, 2019; methodology: weight_decay = 1e-4)      #
    # ------------------------------------------------------------------ #
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # ------------------------------------------------------------------ #
    #  LR scheduler — ReduceLROnPlateau                                   #
    #  "reduction on validation loss plateau with patience=10, factor=0.5"#
    #  per the methodology Training Protocol section.                      #
    # ------------------------------------------------------------------ #
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=10,
        factor=0.5
    )

    # ------------------------------------------------------------------ #
    #  Loss functions                                                      #
    #  L_total = λ₁ · L_cls + λ₂ · L_density  (methodology equation)     #
    # ------------------------------------------------------------------ #
    loss_classification = nn.CrossEntropyLoss()
    loss_density        = nn.MSELoss()

    LAMBDA_1 = 1.0   # classification weight
    LAMBDA_2 = 0.5   # density weight

    # ------------------------------------------------------------------ #
    #  Training loop with early stopping                                   #
    #  "Early stopping triggered upon no improvement in validation loss    #
    #   for 15 consecutive epochs." — methodology Training Protocol        #
    # ------------------------------------------------------------------ #
    evaluator = CoreSetEvaluator()

    best_val_loss     = float('inf')
    best_val_accuracy = 0.0
    early_stop_patience = 15
    epochs_without_improvement = 0
    checkpoint_path = os.path.join(
        config['checkpoint_dir'], 'best_stgcn_model.pth'
    )

    for epoch in range(config['epochs']):

        # -------------------------------------------------------------- #
        #  Training                                                        #
        # -------------------------------------------------------------- #
        model.train()
        total_train_loss = 0.0

        for inputs, labels, density_gts in train_loader:
            inputs      = inputs.to(device)
            labels      = labels.to(device)
            density_gts = density_gts.to(device)

            optimizer.zero_grad()

            logits, density_maps = model(inputs)

            l_cls     = loss_classification(logits, labels)
            l_density = loss_density(density_maps, density_gts)
            loss      = (LAMBDA_1 * l_cls) + (LAMBDA_2 * l_density)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # -------------------------------------------------------------- #
        #  Validation                                                      #
        # -------------------------------------------------------------- #
        model.eval()
        total_val_loss = 0.0
        all_val_logits = []
        all_val_labels = []

        with torch.no_grad():
            for inputs, labels, density_gts in val_loader:
                inputs      = inputs.to(device)
                labels      = labels.to(device)
                density_gts = density_gts.to(device)

                logits, density_maps = model(inputs)

                l_cls     = loss_classification(logits, labels)
                l_density = loss_density(density_maps, density_gts)
                loss      = (LAMBDA_1 * l_cls) + (LAMBDA_2 * l_density)

                total_val_loss += loss.item()
                all_val_logits.append(logits.cpu())
                all_val_labels.append(labels.cpu())

        avg_val_loss = total_val_loss / len(val_loader)

        val_accuracy = evaluator.calculate_classification_accuracy(
            torch.cat(all_val_logits),
            torch.cat(all_val_labels)
        )

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch + 1:3d}/{config['epochs']}] "
            f"| Train Loss: {avg_train_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Val Acc: {val_accuracy * 100:.2f}% "
            f"| LR: {current_lr:.2e}"
        )

        # -------------------------------------------------------------- #
        #  LR scheduler step (on validation loss)                         #
        # -------------------------------------------------------------- #
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < prev_lr:
            print(f"    LR reduced: {prev_lr:.2e} 2192 {new_lr:.2e}")

        # -------------------------------------------------------------- #
        #  Checkpoint — save best model by validation loss                 #
        # -------------------------------------------------------------- #
        if avg_val_loss < best_val_loss:
            best_val_loss     = avg_val_loss
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            torch.save({
                'epoch':             epoch + 1,
                'model_state_dict':  model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss':          best_val_loss,
                'val_accuracy':      best_val_accuracy,
                'config':            config,
            }, checkpoint_path)
            print(f"    ✓ New best model saved "
                  f"(Val Loss: {best_val_loss:.4f}, "
                  f"Acc: {best_val_accuracy * 100:.2f}%)")
        else:
            epochs_without_improvement += 1
            print(f"    No improvement for "
                  f"{epochs_without_improvement}/{early_stop_patience} epochs")

        # -------------------------------------------------------------- #
        #  Early stopping                                                  #
        # -------------------------------------------------------------- #
        if epochs_without_improvement >= early_stop_patience:
            print(f"\n  Early stopping triggered after epoch {epoch + 1}.")
            break

    # ------------------------------------------------------------------ #
    #  Final evaluation on held-out test set                              #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("  Loading best checkpoint for test-set evaluation...")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_test_logits = []
    all_test_labels = []

    with torch.no_grad():
        for inputs, labels, density_gts in test_loader:
            inputs = inputs.to(device)
            logits, _ = model(inputs)
            all_test_logits.append(logits.cpu())
            all_test_labels.append(labels)

    test_accuracy = evaluator.calculate_classification_accuracy(
        torch.cat(all_test_logits),
        torch.cat(all_test_labels)
    )
    print(f"  Test Top-1 Classification Accuracy: {test_accuracy * 100:.2f}%")
    print("=" * 60)
    print(f"  Training complete. Best model: {checkpoint_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train_stgcn_model('configs/stgcn_config.yaml')