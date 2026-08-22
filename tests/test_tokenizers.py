import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tokenization.baseline_tokenizers import BaselineTokenizer
from src.tokenization.cjk_tokenizer import CJKHybridTokenizer
from src.cjk.decomposition import identify_script, decompose_hangul, decompose_han
from src.evaluation.execute_tasks import evaluate_structural_tasks
from src.training.model import SimpleTransformerLM

def test_decomposition():
    print("--- Testing Script Detection & Decomposition ---")
    # Chinese Han
    assert identify_script("語") == "Han"
    # Korean Hangul
    assert identify_script("한") == "Hangul"
    # Japanese Kana
    assert identify_script("あ") == "Hiragana"
    
    # Hangul -> Jamo
    jamo = decompose_hangul("뀶")
    assert jamo != "뀶" and len(jamo) > 1, "Hangul decomposition failed"
    
    # Han -> Components (if Unihan IDS is available, it should split. Otherwise it falls back)
    han_comp = decompose_han("語")
    print(f"語 decomposed to: {han_comp}")

def test_round_trip_proposed():
    texts = {
        "zho": ["我喜欢自然语言处理。", "测试罕见字：龘"],
        "jpn": ["私は自然言語処理が好きです。", "漢字のテスト：鬱"],
        "kor": ["나는 자연어 처리를 좋아한다.", "한글 테스트: 뀶"]
    }
    
    # Train dummy tokenizers
    for lang, sents in texts.items():
        print(f"--- Testing Proposed Tokenizer {lang} ---")
        
        cjk_tok = CJKHybridTokenizer(lang, vocab_size=100, min_frequency_for_normal_char=2)
        cjk_tok.fit_on_texts(sents)
        
        for text in sents:
            enc = cjk_tok.encode(text)
            if hasattr(enc, 'ids'): enc = enc.ids
            dec = cjk_tok.decode(enc)
            print(f"Original: {text} | Decoded: {dec}")
            assert text.replace(" ", "") == dec.replace(" ", ""), f"Proposed Round-trip failed for {lang}"

def test_nagisa_baseline():
    print("--- Testing Nagisa Baseline (Japanese) ---")
    jpn_sents = ["私は自然言語処理が好きです。"]
    base_tok = BaselineTokenizer("jpn", vocab_size=100, method="bpe")
    # This should trigger nagisa pretokenization without crashing
    base_tok.fit_on_texts(jpn_sents, "tokenizers/test_jpn_baseline")
    enc = base_tok.encode(jpn_sents[0])
    dec = base_tok.decode(enc)
    print(f"Original: {jpn_sents[0]} | Decoded: {dec}")

def test_evaluation_framework():
    print("--- Testing Evaluation Code Execution ---")
    # Create a dummy model
    model = SimpleTransformerLM(vocab_size=100, d_model=32, nhead=2, num_layers=2, dim_feedforward=64, max_seq_len=64)
    # Use dummy tokenizer
    cjk_tok = CJKHybridTokenizer("zho", vocab_size=100, min_frequency_for_normal_char=2)
    cjk_tok.fit_on_texts(["测试"])
    
    # Run eval (it should run without crashing, returning a dict of tasks if TSV files exist)
    results = evaluate_structural_tasks(model, cjk_tok, "zho", "cpu", "test_zho")
    print("Evaluation executed successfully:", type(results))

if __name__ == "__main__":
    test_decomposition()
    test_round_trip_proposed()
    test_nagisa_baseline()
    test_evaluation_framework()
    print("All tests passed!")
