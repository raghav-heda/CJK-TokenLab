import os
import urllib.request
from jamo import h2j, j2hcj, j2h
import pykakasi
import regex

# The standard CJKVI IDS dataset
IDS_URL = "https://raw.githubusercontent.com/cjkvi/cjkvi-ids/master/ids.txt"
IDS_PATH = os.path.join(os.path.dirname(__file__), "ids.txt")

_ids_dict = None
_rev_ids_dict = None
_kakasi = None

def _load_ids():
    global _ids_dict, _rev_ids_dict
    if _ids_dict is not None:
        return
        
    if not os.path.exists(IDS_PATH):
        print("Downloading IDS dataset for Han decomposition...")
        urllib.request.urlretrieve(IDS_URL, IDS_PATH)
        
    _ids_dict = {}
    _rev_ids_dict = {}
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                char = parts[1]
                decomp = parts[2]
                
                # filter out ambiguous markers and get clean decomposition
                # IDS format often has multiple decompositions separated by tabs, or ^ symbols
                # We'll just take the first one and clean it
                clean_decomp = regex.sub(r'\[\d\]', '', decomp) # remove variations like [1]
                if clean_decomp != char and "?" not in clean_decomp:
                    _ids_dict[char] = clean_decomp
                    # Only map back if we don't have a collision, or map back to the first we see
                    if clean_decomp not in _rev_ids_dict:
                        _rev_ids_dict[clean_decomp] = char

def decompose_han(char):
    """Decompose a Chinese/Japanese Kanji character into components."""
    _load_ids()
    # Return decomposition if available, otherwise return the char itself
    return _ids_dict.get(char, char)

def recompose_han(components):
    """Attempt to recompose components back to a single Han character."""
    _load_ids()
    return _rev_ids_dict.get(components, components)

def decompose_hangul(char):
    """Decompose a Korean Hangul syllable into Jamo."""
    # h2j converts '한' to '한' (Choseong, Jungseong, Jongseong Jamo)
    # j2hcj converts them to compatibility jamo 'ㅎㅏㄴ' for display/tokenization
    decomp = j2hcj(h2j(char))
    return decomp

def recompose_hangul(jamo_str):
    """Recompose compatibility Jamo back to Hangul."""
    # This requires converting compatibility Jamo back to Choseong, Jungseong, Jongseong
    # j2h works on individual separated jamo, but to handle a string we can use a simpler approach
    # The jamo library might be tricky with j2h(compatibility_jamo).
    # Since we only recompose what we decomposed, we can implement a simple mapping
    # Actually, we can use jamo.hcj2j to convert compatibility to standard jamo, then j2h.
    # We will implement it safely in tokenization by ensuring we only apply this to valid sequences.
    try:
        from jamo import hcj_to_jamo, j2h
        if len(jamo_str) == 2:
            return j2h(hcj_to_jamo(jamo_str[0], "lead"), hcj_to_jamo(jamo_str[1], "vowel"))
        elif len(jamo_str) == 3:
            return j2h(hcj_to_jamo(jamo_str[0], "lead"), hcj_to_jamo(jamo_str[1], "vowel"), hcj_to_jamo(jamo_str[2], "tail"))
        return jamo_str
    except Exception as e:
        return jamo_str

def kanji_to_hiragana(text):
    global _kakasi
    if _kakasi is None:
        _kakasi = pykakasi.kakasi()
    
    result = []
    for item in _kakasi.convert(text):
        result.append(item['hira'])
    return "".join(result)

def identify_script(char):
    """Identify the script of a character."""
    # Simple regex based identification
    if regex.match(r'[\p{Han}]', char):
        return "Han"
    elif regex.match(r'[\p{Hiragana}]', char):
        return "Hiragana"
    elif regex.match(r'[\p{Katakana}]', char):
        return "Katakana"
    elif regex.match(r'[\p{Hangul}]', char):
        return "Hangul"
    else:
        return "Other"

if __name__ == "__main__":
    # Smoke tests
    print("Han '語' ->", decompose_han('語'))
    print("Recompose ->", recompose_han(decompose_han('語')))
    
    print("Hangul '한' ->", decompose_hangul('한'))
    print("Recompose ->", recompose_hangul(decompose_hangul('한')))
    
    print("Kanji '語' to Hiragana ->", kanji_to_hiragana('語'))
