import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import yaml
import os

# Internal project imports
from src.models.bilstm import BaselineBiLSTM
from src.utils.metrics import CoreSetEvaluator

# --- CONSOLIDATED DATA LOADER ---
class BiLSTMDataset(Dataset):
    def __init__(self, data_dir, max_frames=300):
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.samples = []
        
        self.class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        
        for idx, name in enumerate(self.class_names):
            class_path = os.path.join(data_dir, name)
            files = [f for f in os.listdir(class_path) if f.endswith('.pt')]
            for f in files:
                self.samples.append((os.path.join(class_path, f), idx))
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = torch.load(path, weights_only=True).float() 
        
        num_frames = data.shape[0]
        
        if num_frames > self.max_frames:
            data = data[:self.max_frames, :]
        elif num_frames < self.max_frames:
            padding = torch.zeros((self.max_frames - num_frames, 132))
            data = torch.cat([data, padding], dim=0)
            
        return data, label

# --- TRAINING PIPELINE ---
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_bilstm_model(config_path):
    config = load_config(config_path)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # --- HARDWARE DETECTION ---
    if torch.cuda.is_available():
        device = torch.device("cuda") 
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")  
    else:
        device = torch.device("cpu")  

    # Initialize Dataset
    full_dataset = BiLSTMDataset(data_dir=config['data_dir'])
    total_size = len(full_dataset)
    
    # Get counts per class for the header
    class_counts = {name: 0 for name in full_dataset.class_names}
    for _, label in full_dataset.samples:
        class_counts[full_dataset.class_names[label]] += 1

    # --- SPLIT DATA ---
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.get('batch_size', 32), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.get('batch_size', 32), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.get('batch_size', 32), shuffle=False)
    
    # --- MODEL ARCHITECTURE ---
    model = BaselineBiLSTM(
        input_size=config['input_size'], 
        hidden_size=128, 
        num_layers=2, 
        num_classes=config['num_classes']
    ).to(device)
    
    # Count Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # --- HEADER PRINTOUT ---
    print(f"\tCoreSet BiLSTM — Training (Subject-Wise Split)")
    print(f"\t============================================================")
    print(f"\t  Hardware: {device.type.upper()}")
    print(f"\t  Total files in {config['data_dir']}: {total_size}")
    for name, count in class_counts.items():
        print(f"\t    {name.replace('_', ' ').title()}: {count}")
    print(f"\t------------------------------------------------------------")
    print(f"\t  Partition (subject-isolated): {len(train_dataset)} train | {len(val_dataset)} val | {len(test_dataset)} test")
    print(f"\t  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"\t============================================================")

    # --- OPTIMIZATION SETUP ---
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config.get('weight_decay', 0.0001))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config.get('patience_lr', 10))
    criterion = nn.CrossEntropyLoss()
    evaluator = CoreSetEvaluator()
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    patience_limit = config.get('patience_early_stop', 15)
    save_path = os.path.join(config['checkpoint_dir'], 'best_bilstm_baseline.pth')

    # --- TRAINING LOOP ---
    for epoch in range(config['epochs']):
        current_lr = optimizer.param_groups[0]['lr']
        
        # Train Phase
        model.train()
        total_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        total_val_loss = 0.0
        val_logits_list = []
        val_labels_list = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                val_logits_list.append(logits.cpu())
                val_labels_list.append(labels.cpu())
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        val_accuracy = evaluator.calculate_classification_accuracy(
            torch.cat(val_logits_list),
            torch.cat(val_labels_list)
        )
        
        # Print Epoch Base Output
        print(f"\tEpoch [{epoch+1:2d}/{config['epochs']}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy * 100:.2f}% | LR: {current_lr:.2e}")
        
        # Scheduler Step
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # --- CHECKPOINTING & EARLY STOPPING ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"\t    ✓ New best model saved (Val Loss: {best_val_loss:.4f}, Acc: {val_accuracy * 100:.2f}%)")
            early_stop_counter = 0 
        else:
            early_stop_counter += 1
            if new_lr < current_lr:
                print(f"\t    LR reduced: {current_lr:.2e} -> {new_lr:.2e}")
            print(f"\t    No improvement for {early_stop_counter}/{patience_limit} epochs")

        if early_stop_counter >= patience_limit:
            print(f"\n\t  Early stopping triggered after epoch {epoch+1}.")
            break

    # --- FINAL TEST EVALUATION ---
    print("\n\t============================================================")
    print("\t  Loading best checkpoint for test-set evaluation...")
    
    # Load the best model safely
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    
    test_logits_list = []
    test_labels_list = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            test_logits_list.append(logits.cpu())
            test_labels_list.append(labels.cpu())
            
    test_accuracy = evaluator.calculate_classification_accuracy(
        torch.cat(test_logits_list),
        torch.cat(test_labels_list)
    )
    
    print(f"\t  Test Top-1 Classification Accuracy: {test_accuracy * 100:.2f}%")
    print("\t============================================================\n")

if __name__ == '__main__':
    train_bilstm_model('configs/bilstm_config.yaml')