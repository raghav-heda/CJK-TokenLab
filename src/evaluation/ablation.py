"""
Standalone Ablation Study Runner for CJK-TokenLab.

Run explicitly:
    python src/evaluation/ablation.py

Tests the effect of decomposition parameters on tokenizer performance.
"""
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import LANGUAGES, RESULTS_DIR, SPLITS_DIR
from src.tokenization.cjk_tokenizer import CJKHybridTokenizer
from src.evaluation.metrics import calculate_tokenization_metrics
from src.preprocessing.prep import load_split


ABLATION_VARIANTS = [
    {
        "name": "all_decomposed",
        "min_frequency": 0,
        "disable_decomposition": False,
        "description": "Decompose ALL CJK characters regardless of frequency"
    },
    {
        "name": "default",
        "min_frequency": 5,
        "disable_decomposition": False,
        "description": "Default setting: decompose characters with frequency < 5"
    },
    {
        "name": "conservative",
        "min_frequency": 20,
        "disable_decomposition": False,
        "description": "Conservative: only decompose very rare characters (freq < 20)"
    },
    {
        "name": "no_decomposition",
        "min_frequency": 5,
        "disable_decomposition": True,
        "description": "Decomposition disabled: plain BPE on raw text, no [DEC] markers"
    },
]


def run_ablation(lang_codes=None, vocab_size=128):
    """Run ablation study across all variants for specified languages."""
    if lang_codes is None:
        lang_codes = list(LANGUAGES.keys())
    
    results = {}
    
    for lang_code in lang_codes:
        print(f"\n{'='*50}")
        print(f"Ablation Study: {LANGUAGES.get(lang_code, lang_code)}")
        print(f"{'='*50}")
        
        train_texts = load_split(lang_code, "train")
        val_texts = load_split(lang_code, "val")
        
        if not train_texts or not val_texts:
            print(f"  Skipping {lang_code}: no training/validation data found.")
            continue
        
        lang_results = {}
        
        for variant in ABLATION_VARIANTS:
            name = variant["name"]
            print(f"\n  Variant: {name} — {variant['description']}")
            
            try:
                tok = CJKHybridTokenizer(
                    lang_code,
                    vocab_size=vocab_size,
                    min_frequency_for_normal_char=variant["min_frequency"],
                    disable_decomposition=variant["disable_decomposition"]
                )
                tok.fit_on_texts(train_texts)
                
                metrics = calculate_tokenization_metrics(tok, val_texts)
                metrics["variant"] = name
                metrics["description"] = variant["description"]
                metrics["min_frequency"] = variant["min_frequency"]
                metrics["disable_decomposition"] = variant["disable_decomposition"]
                
                lang_results[name] = metrics
                
                print(f"    Tokens: {metrics['Total Tokens']}, "
                      f"Chars/Token: {metrics['Chars per Token']:.3f}, "
                      f"UNK: {metrics['Unknown Token Rate']:.4f}, "
                      f"UEC: {metrics['Uniform Encoding Cost (bits/char)']:.3f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                lang_results[name] = {"error": str(e)}
        
        results[lang_code] = lang_results
    
    # Save results
    out_dir = os.path.join(RESULTS_DIR, "tables")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_results.json")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\nAblation results saved to {out_path}")
    return results


if __name__ == "__main__":
    run_ablation()
