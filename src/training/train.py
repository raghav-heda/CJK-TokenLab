import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import SMOKE_TEST_PARAMS, LOGS_DIR, MODELS_DIR

class TokenizedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        
        # Determine if it's proposed or baseline
        is_proposed = hasattr(tokenizer, 'encode') and not hasattr(tokenizer, 'spm_processor')
        # Encode all texts
        print("Tokenizing dataset...")
        for text in texts:
            if hasattr(tokenizer, 'spm_processor') and tokenizer.spm_processor:
                ids = tokenizer.spm_processor.encode_as_ids(text)
            elif hasattr(tokenizer, 'encode'):
                ids = tokenizer.encode(text)
                if hasattr(ids, 'ids'):
                    ids = ids.ids
            else:
                raise ValueError("Unknown tokenizer type")
            
            # Simple chunking for LM
            for i in range(0, len(ids), max_seq_len):
                chunk = ids[i:i+max_seq_len]
                if len(chunk) > 1:
                    self.data.append(chunk)
                    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        chunk = self.data[idx]
        # Pad if necessary
        if len(chunk) < self.max_seq_len:
            chunk = chunk + [0] * (self.max_seq_len - len(chunk))
        
        chunk = torch.tensor(chunk, dtype=torch.long)
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def train_model(model, train_loader, val_loader, epochs, lr, device, save_path):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    
    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        epoch_time = time.time() - start_time
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epoch_times"].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.2f}s")
        
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    with open(os.path.join(save_path, "history.json"), "w") as f:
        json.dump(history, f)
        
    return history
