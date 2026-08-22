import os
import json
from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from src.cjk.decomposition import decompose_han, decompose_hangul, identify_script

class CJKHybridTokenizer:
    def __init__(self, lang, vocab_size, min_frequency_for_normal_char=5, disable_decomposition=False):
        self.lang = lang
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency_for_normal_char
        self.disable_decomposition = disable_decomposition
        self.char_freqs = Counter()
        self.frequent_chars = set()
        
        # We use BPE for the underlying tokenization
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[DEC]", "[/DEC]"]
        
    def _decompose_char(self, char):
        script = identify_script(char)
        if script == "Han":
            return decompose_han(char)
        elif script == "Hangul":
            return decompose_hangul(char)
        return char

    def fit_on_texts(self, texts):
        # 1. Compute frequencies
        print("Computing character frequencies...")
        for text in texts:
            self.char_freqs.update(text)
            
        # 2. Determine frequent chars
        self.frequent_chars = {char for char, freq in self.char_freqs.items() if freq >= self.min_frequency}
        
        # 3. Create augmented training data
        print("Creating decomposed training data...")
        augmented_texts = []
        for text in texts:
            aug_text = self._preprocess_text(text)
            augmented_texts.append(aug_text)
            
        # 4. Train underlying BPE
        print("Training underlying BPE...")
        trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=self.special_tokens)
        self.tokenizer.train_from_iterator(augmented_texts, trainer=trainer)
        
    def _preprocess_text(self, text):
        """Replaces rare CJK characters with their decompositions wrapped in markers.
        If disable_decomposition is True, returns the raw text unchanged."""
        if self.disable_decomposition:
            return text
        result = []
        for char in text:
            script = identify_script(char)
            # We only decompose Han and Hangul if they are rare
            if (script in ["Han", "Hangul"]) and (char not in self.frequent_chars):
                decomp = self._decompose_char(char)
                if decomp != char:
                    # Space out the components so BPE can learn them, or just let BPE merge them.
                    # We wrap them so we know to try recomposing them later.
                    result.append(f" [DEC] {decomp} [/DEC] ")
                else:
                    result.append(char)
            else:
                result.append(char)
        
        # Join and clean up extra spaces
        return "".join(result).replace("  ", " ").strip()

    def encode(self, text):
        preprocessed = self._preprocess_text(text)
        return self.tokenizer.encode(preprocessed).ids
        
    def decode(self, ids):
        decoded_text = self.tokenizer.decode(ids, skip_special_tokens=False)
        return self._postprocess_text(decoded_text)
        
    def _postprocess_text(self, text):
        """Attempt to recompose anything between [DEC] and [/DEC]."""
        # A simple approach is using regex, but since we have spaces around [DEC] from our preprocess,
        # we need to handle it carefully.
        # tokenizers.decode might have removed spaces depending on how BPE merged it.
        # If [DEC] is a special token, decode(skip_special_tokens=False) keeps it.
        # We will manually replace them.
        import re
        from src.cjk.decomposition import recompose_han, recompose_hangul, identify_script
        
        def replacer(match):
            components = match.group(1).replace(" ", "")
            # Try both. recompose_han and recompose_hangul return original if fail.
            recomp = recompose_hangul(components)
            if recomp != components:
                return recomp
            recomp = recompose_han(components)
            return recomp
            
        # Find [DEC] followed by anything non-greedily, then [/DEC]
        recomposed = re.sub(r'\[DEC\]\s*(.*?)\s*\[/DEC\]', replacer, text)
        
        import regex as re_ext
        # Remove spaces adjacent to CJK characters, preserving English/Latin spacing
        cjk_pattern = r'[\p{Han}\p{Hiragana}\p{Katakana}\p{Hangul}]'
        # Space followed by CJK
        recomposed = re_ext.sub(r'\s+(' + cjk_pattern + r')', r'\1', recomposed)
        # CJK followed by space
        recomposed = re_ext.sub(r'(' + cjk_pattern + r')\s+', r'\1', recomposed)
        
        return recomposed.strip()
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save(os.path.join(path, "tokenizer.json"))
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "lang": self.lang,
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
                "disable_decomposition": self.disable_decomposition,
                "frequent_chars": list(self.frequent_chars)
            }, f, ensure_ascii=False)
            
    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        obj = cls(config["lang"], config["vocab_size"], config.get("min_frequency", 2),
                 disable_decomposition=config.get("disable_decomposition", False))
        obj.frequent_chars = set(config["frequent_chars"])
        obj.tokenizer = Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
        return obj
