import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding='utf-8')
from src.tokenization.baseline_tokenizers import BaselineTokenizer
from src.tokenization.cjk_tokenizer import CJKHybridTokenizer

def run_diagnostics():
    test_texts_zho = ["这是一个测试句子", "人工智能很有趣"]
    test_texts_jpn = ["これはテスト文です", "人工知能は面白いです"]
    
    save_dir = "temp_test_tokenizers"
    os.makedirs(save_dir, exist_ok=True)
    
    print("Testing Character Tokenizer...")
    t_char = BaselineTokenizer("zho", 2000, "char")
    t_char.fit_on_texts(test_texts_zho, save_dir)
    encoded = t_char.encode(test_texts_zho[0])
    decoded = t_char.decode(encoded)
    print(f"Char encoded length: {len(encoded)}, text: {decoded}")
    assert decoded == test_texts_zho[0]
    
    print("\nTesting Jieba Tokenizer...")
    t_jieba = BaselineTokenizer("zho", 2000, "jieba")
    t_jieba.fit_on_texts(test_texts_zho, save_dir)
    encoded_jieba = t_jieba.encode(test_texts_zho[0])
    decoded_jieba = t_jieba.decode(encoded_jieba)
    print(f"Jieba encoded length: {len(encoded_jieba)}, text: {decoded_jieba}")
    assert decoded_jieba == test_texts_zho[0]
    
    print("\nTesting BPE Tokenizer (Chinese)...")
    t_bpe_zho = BaselineTokenizer("zho", 2000, "bpe")
    t_bpe_zho.fit_on_texts(test_texts_zho, save_dir)
    encoded_bpe_zho = t_bpe_zho.encode(test_texts_zho[0])
    decoded_bpe_zho = t_bpe_zho.decode(encoded_bpe_zho)
    print(f"BPE(Zho) encoded length: {len(encoded_bpe_zho)}, text: {decoded_bpe_zho}")
    assert decoded_bpe_zho == test_texts_zho[0]
    
    print("\nTesting Nagisa Tokenizer...")
    t_nagisa = BaselineTokenizer("jpn", 2000, "nagisa")
    t_nagisa.fit_on_texts(test_texts_jpn, save_dir)
    encoded_nagisa = t_nagisa.encode(test_texts_jpn[0])
    decoded_nagisa = t_nagisa.decode(encoded_nagisa)
    print(f"Nagisa encoded length: {len(encoded_nagisa)}, text: {decoded_nagisa}")
    assert decoded_nagisa == test_texts_jpn[0]
    
    print("\nTesting BPE Tokenizer (Japanese)...")
    t_bpe_jpn = BaselineTokenizer("jpn", 2000, "bpe")
    t_bpe_jpn.fit_on_texts(test_texts_jpn, save_dir)
    encoded_bpe_jpn = t_bpe_jpn.encode(test_texts_jpn[0])
    decoded_bpe_jpn = t_bpe_jpn.decode(encoded_bpe_jpn)
    print(f"BPE(Jpn) encoded length: {len(encoded_bpe_jpn)}, text: {decoded_bpe_jpn}")
    assert decoded_bpe_jpn == test_texts_jpn[0]
    
    print("\nTesting SentencePiece Tokenizer...")
    t_spm = BaselineTokenizer("zho", 19, "spm")
    t_spm.fit_on_texts(test_texts_zho, save_dir)
    encoded_spm = t_spm.encode(test_texts_zho[0])
    decoded_spm = t_spm.decode(encoded_spm)
    print(f"SPM encoded length: {len(encoded_spm)}, text: {decoded_spm}")
    assert decoded_spm == test_texts_zho[0]
    
    print("\nTesting Proposed Hybrid Tokenizer...")
    t_hybrid = CJKHybridTokenizer("zho", 19)
    t_hybrid.fit_on_texts(test_texts_zho)
    encoded_hyb = t_hybrid.encode(test_texts_zho[0])
    decoded_hyb = t_hybrid.decode(encoded_hyb)
    print(f"Hybrid encoded length: {len(encoded_hyb)}, text: {decoded_hyb}")
    assert decoded_hyb == test_texts_zho[0]
    
    print("\nVerification Results:")
    print(f"Jieba != BPE (Zho): {encoded_jieba != encoded_bpe_zho}")
    print(f"Nagisa != BPE (Jpn): {encoded_nagisa != encoded_bpe_jpn}")
    print(f"Char is character level (length matches string length): {len(encoded) == len(test_texts_zho[0])}")

if __name__ == '__main__':
    run_diagnostics()
