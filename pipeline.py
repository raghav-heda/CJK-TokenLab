import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from config.config import MODELS_DIR, TOKENIZERS_DIR, RESULTS_DIR, SMOKE_TEST_PARAMS, LANGUAGES
from src.data.prepare_execute_dataset import prepare_execute_dataset
from src.preprocessing.prep import create_splits, load_split
from src.tokenization.baseline_tokenizers import BaselineTokenizer
from src.tokenization.cjk_tokenizer import CJKHybridTokenizer
from src.training.model import SimpleTransformerLM
from src.training.train import TokenizedDataset, train_model
from src.evaluation.metrics import calculate_tokenization_metrics

def main():
    print("Starting CJK-TokenLab Smoke Test Pipeline...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    metrics_results = {}
    
    # 1. Prepare Data
    # For smoke test, limit to 100 sentences. This handles all languages internally.
    print("\n--- Preparing EXECUTE Dataset ---")
    success = prepare_execute_dataset(max_sentences=1000)
    if not success:
        print("Failed to prepare dataset.")
        return

    for lang_code in LANGUAGES.keys():
        print(f"\n{'='*40}")
        print(f"Processing Language: {LANGUAGES[lang_code]} ({lang_code})")
        print(f"{'='*40}")
        
        train_path, val_path, test_path = create_splits(lang_code)
        
        if not train_path:
            continue
            
        train_texts = load_split(lang_code, "train")
        val_texts = load_split(lang_code, "val")
        
        # 2. Train Tokenizers
        print("\n--- Training Tokenizers ---")
        
        baseline_tok = BaselineTokenizer(lang_code, vocab_size=SMOKE_TEST_PARAMS["d_model"] * 2, method="bpe")
        baseline_save_path = os.path.join(TOKENIZERS_DIR, f"{lang_code}_baseline_bpe")
        baseline_tok.fit_on_texts(train_texts, baseline_save_path)
        
        # Byte-Level BPE Baseline (modern GPT-2-style)
        byte_bpe_tok = BaselineTokenizer(lang_code, vocab_size=SMOKE_TEST_PARAMS["d_model"] * 2, method="byte_bpe")
        byte_bpe_save_path = os.path.join(TOKENIZERS_DIR, f"{lang_code}_byte_bpe")
        byte_bpe_tok.fit_on_texts(train_texts, byte_bpe_save_path)
        
        proposed_tok = CJKHybridTokenizer(lang_code, vocab_size=SMOKE_TEST_PARAMS["d_model"] * 2, min_frequency_for_normal_char=2)
        proposed_save_path = os.path.join(TOKENIZERS_DIR, f"{lang_code}_proposed")
        proposed_tok.fit_on_texts(train_texts)
        proposed_tok.save(proposed_save_path)
        
        # 3. Tokenization Metrics
        print("\n--- Tokenization Metrics ---")
        base_metrics = calculate_tokenization_metrics(baseline_tok, val_texts)
        byte_bpe_metrics = calculate_tokenization_metrics(byte_bpe_tok, val_texts)
        prop_metrics = calculate_tokenization_metrics(proposed_tok, val_texts)
        
        metrics_results[lang_code] = {
            "Baseline": base_metrics,
            "Byte-Level BPE": byte_bpe_metrics,
            "Proposed": prop_metrics
        }
        
        # 4. Train Models
        print("\n--- Training Models ---")
        
        # Baseline Model
        print("Training Baseline Model...")
        train_dataset_base = TokenizedDataset(train_texts, baseline_tok, SMOKE_TEST_PARAMS["max_seq_len"])
        val_dataset_base = TokenizedDataset(val_texts, baseline_tok, SMOKE_TEST_PARAMS["max_seq_len"])
        
        train_loader_base = DataLoader(train_dataset_base, batch_size=SMOKE_TEST_PARAMS["batch_size"], shuffle=True)
        val_loader_base = DataLoader(val_dataset_base, batch_size=SMOKE_TEST_PARAMS["batch_size"])
        
        model_base = SimpleTransformerLM(
            vocab_size=baseline_tok.tokenizer.get_vocab_size() if hasattr(baseline_tok, 'tokenizer') else baseline_tok.vocab_size,
            d_model=SMOKE_TEST_PARAMS["d_model"],
            nhead=SMOKE_TEST_PARAMS["nhead"],
            num_layers=SMOKE_TEST_PARAMS["num_layers"],
            dim_feedforward=SMOKE_TEST_PARAMS["dim_feedforward"],
            max_seq_len=SMOKE_TEST_PARAMS["max_seq_len"]
        )
        train_model(model_base, train_loader_base, val_loader_base, SMOKE_TEST_PARAMS["epochs"], SMOKE_TEST_PARAMS["lr"], device, os.path.join(MODELS_DIR, f"{lang_code}_baseline"))
        
        # Proposed Model
        print("Training Proposed Model...")
        train_dataset_prop = TokenizedDataset(train_texts, proposed_tok, SMOKE_TEST_PARAMS["max_seq_len"])
        val_dataset_prop = TokenizedDataset(val_texts, proposed_tok, SMOKE_TEST_PARAMS["max_seq_len"])
        
        train_loader_prop = DataLoader(train_dataset_prop, batch_size=SMOKE_TEST_PARAMS["batch_size"], shuffle=True)
        val_loader_prop = DataLoader(val_dataset_prop, batch_size=SMOKE_TEST_PARAMS["batch_size"])
        
        model_prop = SimpleTransformerLM(
            vocab_size=proposed_tok.tokenizer.get_vocab_size(),
            d_model=SMOKE_TEST_PARAMS["d_model"],
            nhead=SMOKE_TEST_PARAMS["nhead"],
            num_layers=SMOKE_TEST_PARAMS["num_layers"],
            dim_feedforward=SMOKE_TEST_PARAMS["dim_feedforward"],
            max_seq_len=SMOKE_TEST_PARAMS["max_seq_len"]
        )
        train_model(model_prop, train_loader_prop, val_loader_prop, SMOKE_TEST_PARAMS["epochs"], SMOKE_TEST_PARAMS["lr"], device, os.path.join(MODELS_DIR, f"{lang_code}_proposed"))

    # 5. EXECUTE-Inspired Evaluation
    print("\n--- Running EXECUTE-Inspired Evaluation ---")
    from src.evaluation.execute_tasks import run_all_evaluations
    run_all_evaluations()

    # Save metrics
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "tables", "tokenization_metrics.json"), "w") as f:
        json.dump(metrics_results, f, indent=4)
        
    print("\nSmoke test pipeline completed successfully.")
    print("Run the Web app with: uvicorn app.api:app --reload")

if __name__ == "__main__":
    main()
