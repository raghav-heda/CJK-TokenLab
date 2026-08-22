# gen_vocab.py
import argparse
import datasets
import pycountry
import random
import regex
from sacremoses import MosesTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, default="eng", help="Language to use for tokenization.")

SYMBOLS = set(" .,?!:;'\"‘’“”‚‛„()[]{}<>«»‹›--–—―…'’/\|•·⁝⁞­@#&*~§¶^،؟¿¡。、「」『』`​！（），？।")
NUMBERS = set("0123456789")

LATIN_ALPHA = set("abcdefghijklmnopqrstuvwxyz")

def lowercase(char, args):
    if hasattr(args, "cipher") and args.cipher:
        return char.translate(args.cipher_tolower)
    else:
        return char.lower()


def get_charset(dataset, args):
    charset = {}

    for example in dataset['text']:
        for char in regex.findall(r'\X', example):
            char_lower = lowercase(char, args)
            if char_lower in charset:
                charset[char_lower] += 1
            else:
                charset[char_lower] = 1

    # remove symbols and numbers
    for symbol in SYMBOLS:
        if symbol in charset:
            del charset[symbol]

    for number in NUMBERS:
        if number in charset:
            del charset[number]

    if args.cipher:
        for char in args.cipher_punct:
            if char in charset:
                del charset[char]
        for char in args.cipher_nums:
            if char in charset:
                del charset[char]

    # get most frequent char
    # remove latin alphabet for langs with non-latin alphabet
    most_freq_char = max(charset, key=charset.get)
    args.script = "Latin" if most_freq_char in LATIN_ALPHA else "Non-Latin"
    print(f"Detected {args.script} script.")
    
    if args.script != "Latin":
        print(f"Removing Latin characters from charset.")
        for char in LATIN_ALPHA:
            if char in charset:
                del charset[char]

    # remove rare chars
    new_charset = set()
    for char in charset:
        if charset[char] > 8:
            new_charset.add(char)
    charset = new_charset


    # sort by utf8 order
    charset = sorted(list(charset))

    if hasattr(args, 'charset_preprocessor') and args.charset_preprocessor:
        charset = args.charset_preprocessor(charset)

    with open(args.dataset_path + 'charset.txt', 'w', encoding='utf8') as file:
        file.write("".join(charset))

def get_vocab(dataset, args):
    try:
        lang_iso2 = pycountry.languages.get(alpha_3=args.lang).alpha_2
        tokenizer = MosesTokenizer(lang=lang_iso2)
    except AttributeError: # no iso2
        tokenizer = MosesTokenizer(lang=args.lang)
        
    vocab = {}
    for example in dataset['text']:
        # for word in example.split(" "):
        for word in tokenizer.tokenize(example):
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1

    # remove symbols and numbers
    for symbol in SYMBOLS:
        if symbol in vocab:
            del vocab[symbol]

    for number in NUMBERS:
        if number in vocab:
            del vocab[number]

    if args.cipher:
        for char in args.cipher_punct:
            if char in vocab:
                del vocab[char]
        for char in args.cipher_nums:
            if char in vocab:
                del vocab[char]


    # remove words shorter than 3 chars, except for chinese
    if args.lang not in ['jpn', 'zho', 'kor']: # keep CJK short words
        for word in list(vocab):
            if len(word) < 3:
                del vocab[word]

    # remove latin alphabet for langs with non-latin alphabet
    if args.script != "Latin":
        print(f"Removing Latin characters from vocab.")
        for word in list(vocab):
            for char in LATIN_ALPHA:
                if char in word.lower():
                    del vocab[word]
                    break


    # sort by frequency
    sorted_vocab = []
    for word, freq in vocab.items():
        sorted_vocab.append((word, freq))
    sorted_vocab.sort(reverse=True, key=lambda x: x[1])

    if hasattr(args, 'vocab_preprocessor') and args.vocab_preprocessor:
        sorted_vocab = args.vocab_preprocessor(sorted_vocab)

    with open(args.dataset_path + 'vocab.csv', 'w', encoding='utf8') as file:
        file.write("word,count\n")
        for word, freq in sorted_vocab:
            # print(word,freq)
            file.write(f"{word},{freq}\n")

def main(args):
    random.seed(0)
    args.dataset_path = f"data/translated/{args.lang}/"
    if not args.dataset:
        args.dataset = datasets.load_from_disk(args.dataset_path)
    get_charset(args.dataset, args)
    get_vocab(args.dataset, args)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)