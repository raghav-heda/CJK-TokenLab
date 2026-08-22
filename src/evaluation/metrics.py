import time
import math

def calculate_tokenization_metrics(tokenizer, texts):
    start_time = time.time()
    
    total_tokens = 0
    total_chars = 0
    total_unks = 0
    
    # Determine vocabulary size for Uniform Encoding Cost
    vocab_size = 0
    if hasattr(tokenizer, 'tokenizer') and tokenizer.tokenizer is not None:
        vocab_size = tokenizer.tokenizer.get_vocab_size()
    elif hasattr(tokenizer, 'spm_processor') and tokenizer.spm_processor is not None:
        vocab_size = tokenizer.spm_processor.get_piece_size()
    
    for text in texts:
        if hasattr(tokenizer, 'spm_processor') and tokenizer.spm_processor:
            ids = tokenizer.spm_processor.encode_as_ids(text)
        elif hasattr(tokenizer, 'encode'):
            encoded = tokenizer.encode(text)
            ids = encoded.ids if hasattr(encoded, 'ids') else encoded
        else:
            ids = []
            
        total_tokens += len(ids)
        total_chars += len(text)
        
        # Check for UNK token (usually ID 0 or explicitly handled)
        # BPE from huggingface tokenizers uses a specific UNK ID
        unk_id = None
        if hasattr(tokenizer, 'tokenizer') and tokenizer.tokenizer is not None:
            unk_id = tokenizer.tokenizer.token_to_id("[UNK]")
        elif hasattr(tokenizer, 'spm_processor') and tokenizer.spm_processor is not None:
            unk_id = tokenizer.spm_processor.unk_id()
            
        if unk_id is not None:
            total_unks += sum(1 for idx in ids if idx == unk_id)

    encoding_time = time.time() - start_time
    
    # Uniform Encoding Cost (bits/char): theoretical cost under uniform vocabulary distribution
    # Formula: (Total Tokens × log2(Vocab Size)) / Total Chars
    uec = 0.0
    if total_chars > 0 and vocab_size > 1:
        uec = (total_tokens * math.log2(vocab_size)) / total_chars
    
    metrics = {
        "Total Tokens": total_tokens,
        "Total Chars": total_chars,
        "Chars per Token": total_chars / total_tokens if total_tokens > 0 else 0,
        "Tokens per Sentence": total_tokens / len(texts) if len(texts) > 0 else 0,
        "Unknown Token Rate": total_unks / total_tokens if total_tokens > 0 else 0,
        "Encoding Time (s)": encoding_time,
        "Uniform Encoding Cost (bits/char)": round(uec, 4)
    }
    
    return metrics
