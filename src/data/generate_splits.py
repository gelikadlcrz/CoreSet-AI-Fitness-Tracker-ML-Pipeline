import os
import re
import json
import random
from collections import defaultdict

# --- CONFIGURATION ---
# Safely anchor paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Point to your actual data folder shown in the screenshot
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/final_data"))
OUTPUT_SPLIT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../../configs/data_splits.json"))

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
# ---------------------

def extract_subject_id(filename):
    """
    Extracts the subject ID from the filename.
    Matches 'person1', 'person2', etc.
    """
    # Ensure we only check the filename, not the folder path
    basename = os.path.basename(filename)
    
    match = re.search(r'(person\d+)', basename)
    if match:
        return match.group(1)
        
    # If no person ID is found, treat the base filename as a unique subject
    base_name = re.sub(r'_aug_.*', '', basename) 
    return base_name.replace('.npz', '')

def generate_splits():
    random.seed(SEED)
    
    # Dictionary to group files by subject
    subject_groups = defaultdict(list)
    
    # 1. Traverse directories to find all NPZ files
    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            if not filename.endswith('.npz'):
                continue
                
            # Get the path relative to 'final_data'
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, DATA_DIR)
            
            subject_id = extract_subject_id(filename)
            subject_groups[subject_id].append(rel_path)

    subjects = list(subject_groups.keys())
    random.shuffle(subjects)
    
    # 2. Partition subjects while tracking file counts
    total_files = sum(len(files) for files in subject_groups.values())
    
    train_target = total_files * TRAIN_RATIO
    val_target = total_files * VAL_RATIO
    
    train_files, val_files, test_files = [], [], []
    train_subjects, val_subjects, test_subjects = [], [], []
    
    current_train_count = 0
    current_val_count = 0
    
    for subject in subjects:
        files = subject_groups[subject]
        num_files = len(files)
        
        # Distribute subjects to the splits
        if current_train_count + num_files <= train_target or len(train_files) == 0:
            train_files.extend(files)
            train_subjects.append(subject)
            current_train_count += num_files
        elif current_val_count + num_files <= val_target or len(val_files) == 0:
            val_files.extend(files)
            val_subjects.append(subject)
            current_val_count += num_files
        else:
            test_files.extend(files)
            test_subjects.append(subject)
            
    # 3. Save the splits
    splits = {
        "metadata": {
            "total_files": total_files,
            "train_files_count": len(train_files),
            "val_files_count": len(val_files),
            "test_files_count": len(test_files)
        },
        "train": train_files,
        "val": val_files,
        "test": test_files,
        "subjects": {
            "train": train_subjects,
            "val": val_subjects,
            "test": test_subjects
        }
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_SPLIT_FILE), exist_ok=True)
    
    with open(OUTPUT_SPLIT_FILE, 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Splits successfully generated and saved to {OUTPUT_SPLIT_FILE}")
    print(f"Train: {len(train_files)} files | Val: {len(val_files)} files | Test: {len(test_files)} files")

if __name__ == "__main__":
    generate_splits()