import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data.coreset_dataset import CoreSetGCN_Dataset
from src.models.stgcn_multitask import CoreSetSTGCN_MultiTask
from src.utils.metrics import CoreSetEvaluator
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_stgcn_model(config_path):
    config = load_config(config_path)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    print("🚀 Initiating CoreSet ST-GCN Training (Randomized Split)...")
    
    full_dataset = CoreSetGCN_Dataset(data_dir=config['data_dir'], max_frames=config['max_frames'])
    total_files = len(full_dataset)
    print(f"   Found {total_files} total files in {config['data_dir']}.")

    train_size = int(0.70 * total_files)
    val_size = int(0.15 * total_files)
    test_size = total_files - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42) 
    )
    
    print(f"   Partitioned: {train_size} Train | {val_size} Val | {test_size} Test")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    model = CoreSetSTGCN_MultiTask(
        num_classes=config['num_classes'], 
        max_frames=config['max_frames'], 
        node_count=config['node_count']
    ).cuda()
    
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    loss_classification = nn.CrossEntropyLoss()
    
    evaluator = CoreSetEvaluator()
    best_val_accuracy = 0.0

    for epoch in range(config['epochs']):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            logits, density_maps = model(inputs)
            
            loss = loss_classification(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        val_logits_list = []
        val_labels_list = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                logits, density_maps = model(inputs)
                
                loss = loss_classification(logits, labels)
                total_val_loss += loss.item()
                
                val_logits_list.append(logits.cpu())
                val_labels_list.append(labels.cpu())
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        val_accuracy = evaluator.calculate_classification_accuracy(
            torch.cat(val_logits_list),
            torch.cat(val_labels_list)
        )
        
        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} "
              f"| Val Top-1 Accuracy: {val_accuracy * 100:.2f}%")
              
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_path = os.path.join(config['checkpoint_dir'], 'best_stgcn_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"   🌟 New best model saved! (Accuracy: {best_val_accuracy * 100:.2f}%)")

    print(f"\n✅ Training Complete. Best model saved to {config['checkpoint_dir']}/best_stgcn_model.pth")

if __name__ == '__main__':
    train_stgcn_model('../configs/stgcn_config.yaml')