import os
import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordLevel
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, ByteLevel as ByteLevelPreTokenizer
import jieba
import pykakasi

class BaselineTokenizer:
    def __init__(self, lang, vocab_size, method="bpe"):
        self.lang = lang
        self.vocab_size = vocab_size
        self.method = method # "bpe", "spm"
        self.tokenizer = None
        self.spm_processor = None
        


    def fit_on_texts(self, texts, save_path):
        os.makedirs(save_path, exist_ok=True)
        if self.method in ["char", "jieba", "nagisa"]:
            self.tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()
            trainer = WordLevelTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
            
            pretokenized = []
            for t in texts:
                if self.method == "char":
                    pretokenized.append(" ".join(list(t)))
                elif self.method == "jieba":
                    pretokenized.append(" ".join(jieba.cut(t)))
                elif self.method == "nagisa":
                    try:
                        import nagisa
                        pretokenized.append(" ".join(nagisa.tagging(t).words))
                    except:
                        pretokenized.append(" ".join(list(t)))
                        
            self.tokenizer.train_from_iterator(pretokenized, trainer=trainer)
            self.tokenizer.save(os.path.join(save_path, f"{self.method}.json"))
            
        elif self.method == "bpe":
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()
            trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
            self.tokenizer.train_from_iterator(texts, trainer=trainer)
            self.tokenizer.save(os.path.join(save_path, f"{self.method}.json"))
            
        elif self.method == "spm":
            # Save texts to a temp file for SPM training
            temp_file = os.path.join(save_path, "temp_train.txt")
            with open(temp_file, "w", encoding="utf-8") as f:
                for text in texts:
                    f.write(text + "\n")
            
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=os.path.join(save_path, "spm"),
                vocab_size=self.vocab_size,
                character_coverage=0.9995 if self.lang in ["zho", "jpn", "kor"] else 1.0,
                model_type="unigram",
                hard_vocab_limit=False
            )
            os.remove(temp_file)
            self.spm_processor = spm.SentencePieceProcessor()
            self.spm_processor.load(os.path.join(save_path, "spm.model"))
            
        elif self.method == "byte_bpe":
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = ByteLevelPreTokenizer()
            trainer = BpeTrainer(vocab_size=self.vocab_size, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"])
            self.tokenizer.train_from_iterator(texts, trainer=trainer)
            self.tokenizer.save(os.path.join(save_path, "byte_bpe.json"))

    def encode(self, text):
        if self.method in ["char", "jieba", "nagisa"]:
            if self.method == "char":
                prep = " ".join(list(text))
            elif self.method == "jieba":
                prep = " ".join(jieba.cut(text))
            elif self.method == "nagisa":
                try:
                    import nagisa
                    prep = " ".join(nagisa.tagging(text).words)
                except:
                    prep = " ".join(list(text))
            return self.tokenizer.encode(prep).ids
        elif self.method == "bpe":
            return self.tokenizer.encode(text).ids
        elif self.method == "byte_bpe":
            return self.tokenizer.encode(text).ids
        elif self.method == "spm":
            return self.spm_processor.encode_as_ids(text)
            
    def decode(self, ids):
        if self.method in ["bpe", "char", "jieba", "nagisa"]:
            return self.tokenizer.decode(ids).replace(" ", "")
        elif self.method == "byte_bpe":
            return self.tokenizer.decode(ids)
        elif self.method == "spm":
            return self.spm_processor.decode_ids(ids)

    def save(self, path):
        pass # Already saved during fit
        
    @classmethod
    def load(cls, path, lang, vocab_size, method="bpe"):
        obj = cls(lang, vocab_size, method)
        if method in ["bpe", "char", "jieba", "nagisa"]:
            obj.tokenizer = Tokenizer.from_file(os.path.join(path, f"{method}.json"))
        elif method == "byte_bpe":
            obj.tokenizer = Tokenizer.from_file(os.path.join(path, "byte_bpe.json"))
        elif method == "spm":
            obj.spm_processor = spm.SentencePieceProcessor()
            obj.spm_processor.load(os.path.join(path, "spm.model"))
        return obj
