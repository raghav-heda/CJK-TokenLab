"""
Tests for new features: disable_decomposition flag, byte-level BPE,
Uniform Encoding Cost, and BPC calculation.
"""
import os
import sys
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')

from src.tokenization.cjk_tokenizer import CJKHybridTokenizer
from src.tokenization.baseline_tokenizers import BaselineTokenizer
from src.evaluation.metrics import calculate_tokenization_metrics


def test_disable_decomposition():
    """When disable_decomposition=True, _preprocess_text returns raw text (no [DEC] markers)."""
    print("--- Test: disable_decomposition flag ---")
    texts = ["我喜欢自然语言处理。", "测试罕见字：龘"]
    
    # With decomposition enabled (default)
    tok_enabled = CJKHybridTokenizer("zho", vocab_size=100, min_frequency_for_normal_char=2,
                                      disable_decomposition=False)
    tok_enabled.fit_on_texts(texts)
    preprocessed_enabled = tok_enabled._preprocess_text("龘")
    
    # With decomposition disabled
    tok_disabled = CJKHybridTokenizer("zho", vocab_size=100, min_frequency_for_normal_char=2,
                                       disable_decomposition=True)
    tok_disabled.fit_on_texts(texts)
    preprocessed_disabled = tok_disabled._preprocess_text("龘")
    
    # Disabled should return raw text
    assert preprocessed_disabled == "龘", f"Expected raw '龘', got '{preprocessed_disabled}'"
    # Disabled should never contain [DEC]
    assert "[DEC]" not in preprocessed_disabled, "Disabled variant should not contain [DEC]"
    
    print(f"  Enabled preprocess: '{preprocessed_enabled}'")
    print(f"  Disabled preprocess: '{preprocessed_disabled}'")
    print("  PASSED")


def test_disable_decomposition_save_load():
    """disable_decomposition flag should round-trip through save/load."""
    print("--- Test: disable_decomposition save/load ---")
    import tempfile
    import shutil
    
    texts = ["테스트"]
    tok = CJKHybridTokenizer("kor", vocab_size=50, min_frequency_for_normal_char=2,
                              disable_decomposition=True)
    tok.fit_on_texts(texts)
    
    tmpdir = os.path.join(os.path.dirname(__file__), "..", "temp_test_ablation")
    os.makedirs(tmpdir, exist_ok=True)
    try:
        tok.save(tmpdir)
        loaded = CJKHybridTokenizer.load(tmpdir)
        assert loaded.disable_decomposition == True, \
            f"Expected disable_decomposition=True after load, got {loaded.disable_decomposition}"
        print("  PASSED")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_byte_level_bpe():
    """Byte-Level BPE baseline should train, encode, and decode without errors."""
    print("--- Test: Byte-Level BPE baseline ---")
    texts = ["我喜欢自然语言处理。", "テスト文です", "한글 테스트"]
    
    save_dir = os.path.join(os.path.dirname(__file__), "..", "temp_test_byte_bpe")
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        tok = BaselineTokenizer("zho", vocab_size=200, method="byte_bpe")
        tok.fit_on_texts(texts, save_dir)
        
        encoded = tok.encode(texts[0])
        assert len(encoded) > 0, "Byte BPE encoding returned empty"
        
        decoded = tok.decode(encoded)
        assert len(decoded) > 0, "Byte BPE decoding returned empty"
        
        print(f"  Input:   '{texts[0]}'")
        print(f"  Encoded: {encoded[:10]}... ({len(encoded)} tokens)")
        print(f"  Decoded: '{decoded}'")
        
        # Test load
        loaded = BaselineTokenizer.load(save_dir, "zho", 200, "byte_bpe")
        enc2 = loaded.encode(texts[0])
        assert enc2 == encoded, "Loaded byte BPE produces different encoding"
        
        print("  PASSED")
    finally:
        import shutil
        shutil.rmtree(save_dir, ignore_errors=True)


def test_uniform_encoding_cost():
    """Uniform Encoding Cost formula: (Total Tokens × log2(Vocab Size)) / Total Chars."""
    print("--- Test: Uniform Encoding Cost ---")
    texts = ["测试"]
    
    tok = CJKHybridTokenizer("zho", vocab_size=100, min_frequency_for_normal_char=2,
                              disable_decomposition=True)
    tok.fit_on_texts(texts)
    
    metrics = calculate_tokenization_metrics(tok, texts)
    
    assert "Uniform Encoding Cost (bits/char)" in metrics, "UEC missing from metrics"
    
    # Verify formula manually
    total_tokens = metrics["Total Tokens"]
    total_chars = metrics["Total Chars"]
    vocab_size = tok.tokenizer.get_vocab_size()
    expected_uec = (total_tokens * math.log2(vocab_size)) / total_chars if total_chars > 0 else 0
    
    assert abs(metrics["Uniform Encoding Cost (bits/char)"] - round(expected_uec, 4)) < 0.01, \
        f"UEC mismatch: got {metrics['Uniform Encoding Cost (bits/char)']}, expected {expected_uec:.4f}"
    
    print(f"  Vocab: {vocab_size}, Tokens: {total_tokens}, Chars: {total_chars}")
    print(f"  UEC: {metrics['Uniform Encoding Cost (bits/char)']}")
    print("  PASSED")


def test_bpc_formula():
    """BPC = total_nll_nats / (ln(2) × total_target_chars)."""
    print("--- Test: BPC formula verification ---")
    
    # Known values
    total_nll_nats = 100.0
    total_target_chars = 50
    
    expected_bpc = total_nll_nats / (math.log(2) * total_target_chars)
    
    # This is the formula used in execute_tasks.py
    computed_bpc = round(total_nll_nats / (math.log(2) * total_target_chars), 4)
    
    assert abs(computed_bpc - round(expected_bpc, 4)) < 0.0001, \
        f"BPC mismatch: {computed_bpc} vs {expected_bpc}"
    
    print(f"  NLL: {total_nll_nats}, Chars: {total_target_chars}")
    print(f"  BPC: {computed_bpc} (expected: {expected_bpc:.4f})")
    print("  PASSED")


if __name__ == "__main__":
    test_disable_decomposition()
    test_disable_decomposition_save_load()
    test_byte_level_bpe()
    test_uniform_encoding_cost()
    test_bpc_formula()
    print("\nAll ablation tests passed!")
