import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import RAW_DATA_DIR, SPLITS_DIR, RANDOM_SEED

def create_splits(lang_code):
    raw_file = os.path.join(RAW_DATA_DIR, f"{lang_code}_corpus.txt")
    if not os.path.exists(raw_file):
        print(f"Raw data for {lang_code} not found at {raw_file}")
        return None, None, None
        
    with open(raw_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    # Clean up empty lines and normalize
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    random.seed(RANDOM_SEED)
    random.shuffle(cleaned_lines)
    
    n = len(cleaned_lines)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_data = cleaned_lines[:train_end]
    val_data = cleaned_lines[train_end:val_end]
    test_data = cleaned_lines[val_end:]
    
    os.makedirs(SPLITS_DIR, exist_ok=True)
    
    def save_split(data, split_name):
        path = os.path.join(SPLITS_DIR, f"{lang_code}_{split_name}.txt")
        with open(path, "w", encoding="utf-8") as f:
            for line in data:
                f.write(line + "\n")
        return path
        
    train_path = save_split(train_data, "train")
    val_path = save_split(val_data, "val")
    test_path = save_split(test_data, "test")
    
    print(f"Created splits for {lang_code}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test.")
    return train_path, val_path, test_path

def load_split(lang_code, split_name):
    path = os.path.join(SPLITS_DIR, f"{lang_code}_{split_name}.txt")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]
