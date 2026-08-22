import os
import sys
import json
import random
import time
from sentence_splitter import SentenceSplitter
from deep_translator import GoogleTranslator

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import RAW_DATA_DIR, EXECUTE_REPO_DIR, LANGUAGES, RANDOM_SEED

# Google Translate target language codes
GT_CODES = {
    "zho": "zh-CN",
    "jpn": "ja",
    "kor": "ko"
}

def load_english_source(source_path):
    print(f"Loading English source from: {source_path}")
    with open(source_path, 'r', encoding='utf-8') as file:
        text = file.read().split("<|endoftext|>")[1:] # first story is cut off (as per EXECUTE)

    splitter = SentenceSplitter(language="en")
    split = []
    for i in range(len(text)):
        tokens = splitter.split(text[i])
        split.extend(tokens)
        
    print(f"Total English sentences before filtering: {len(split)}")
    
    filtered_split = []
    for s in split:
        words = s.split()
        if 3 <= len(words) <= 10:
            if s[-1] in ['.', '!', '?']:
                if '"' not in s:
                    filtered_split.append(s)
                    
    print(f"Total English sentences after length/punctuation filtering: {len(filtered_split)}")
    
    random.seed(0) # As per EXECUTE repo: split_ds.shuffle(seed=0)
    random.shuffle(filtered_split)
    
    # EXECUTE cuts dataset to 5k examples
    final_dataset = filtered_split[:5000]
    print(f"Selected {len(final_dataset)} sentences for translation.")
    return final_dataset

def translate_sentences(sentences, target_lang, out_path, max_sentences=None):
    if max_sentences:
        sentences = sentences[:max_sentences]
        
    done = 0
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    existing_lines = 0
    if os.path.exists(out_path):
        with open(out_path, 'r', encoding='utf-8') as f:
            existing_lines = len(f.readlines())
            
    print(f"Found {existing_lines} existing translations. Resuming...")
    
    with open(out_path, 'a', encoding='utf-8') as out_file:
        for i, src_text in enumerate(sentences):
            if i < existing_lines:
                continue
                
            retries = 10
            while retries > 0:
                try:
                    translator = GoogleTranslator(source='en', target=target_lang)
                    translation = translator.translate(src_text)
                    if translation:
                        out_file.write(translation.replace("\n", " ") + '\n')
                        out_file.flush()
                        done += 1
                    if done % 100 == 0 and done > 0:
                        print(f"[{target_lang}] Translated {i+1}/{len(sentences)}")
                    time.sleep(0.5) # Small delay to respect limits
                    break
                except Exception as e:
                    print(f"[{target_lang}] Error translating: {e}. Retrying in {2**(11-retries)}s...")
                    time.sleep(2 ** (11 - retries))
                    retries -= 1
                    
            if retries == 0:
                print(f"[{target_lang}] Failed to translate after retries. Stopping.")
                break
                
    print(f"[{target_lang}] Finished translation. File saved at {out_path}")

def prepare_execute_dataset(max_sentences=None):
    source_path = os.path.join(EXECUTE_REPO_DIR, "data", "TinyStoriesV2-GPT4-valid.txt")
    if not os.path.exists(source_path):
        print(f"ERROR: Cannot find EXECUTE TinyStories data at {source_path}")
        return False
        
    english_sentences = load_english_source(source_path)
    
    metadata = {
        "source_file": source_path,
        "source_description": "TinyStoriesV2-GPT4-valid.txt from EXECUTE benchmark",
        "english_sample_size": len(english_sentences),
        "random_seed": 0,
        "translation_method": "deep_translator (Google API) - deviated from googletrans due to Python compatibility",
        "dataset_sizes": {}
    }
    
    expected_lines = max_sentences if max_sentences else len(english_sentences)
    
    for lang_code in LANGUAGES.keys():
        target_lang = GT_CODES[lang_code]
        out_path = os.path.join(RAW_DATA_DIR, f"{lang_code}_corpus.txt")
        print(f"Translating for {LANGUAGES[lang_code]} ({target_lang}) to {out_path}")
        
        translate_sentences(english_sentences, target_lang, out_path, max_sentences)
        
        with open(out_path, 'r', encoding='utf-8') as f:
            metadata["dataset_sizes"][lang_code] = len(f.readlines())
            
    with open(os.path.join(RAW_DATA_DIR, "dataset_metadata.json"), "w", encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
        
    # Verify alignment
    all_aligned = True
    for lang_code, count in metadata["dataset_sizes"].items():
        if count != expected_lines:
            print(f"ERROR: Alignment mismatch for {lang_code}. Expected {expected_lines}, got {count}.")
            all_aligned = False
            
    if not all_aligned:
        print("Dataset generation halted due to alignment mismatch. Please run again to resume.")
        return False
        
    print("Dataset preparation complete and metadata saved.")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sentences", type=int, default=None, help="Limit number of sentences for smoke testing.")
    args = parser.parse_args()
    
    prepare_execute_dataset(args.max_sentences)
