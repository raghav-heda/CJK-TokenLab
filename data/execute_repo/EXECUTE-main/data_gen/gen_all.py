import regex
import argparse
import datasets
from gen_vocab import main as get_vocab
from gen_prompts import main as gen_prompts
from gen_char_tasks import main as gen_char_tasks
from gen_word_tasks import main as gen_word_tasks

# first performs get_vocab to generate the charset and vocab 
# then generates char tasks and 4 random char examples
# then generates word tasks and 4 random word examples
# then generates prompts
parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, default="eng", help="Language ISO 3 code")
parser.add_argument("--cipher", type=str, default="", help="Cipherize lang to other lang")

def prep_japanese_dataset(dataset):
    import nagisa
    split_examples = {'text': []}
    for example in dataset['text']:
        split_example = " ".join(nagisa.tagging(example).words)
        split_examples['text'].append(split_example)

    return split_examples


def prep_chinese_dataset(dataset):
    import jieba
    split_examples = {'text': []}
    for example in dataset['text']:
        split_example = " ".join(jieba.lcut(example))
        split_examples['text'].append(split_example)

    return split_examples

def cipherize(args):
    import random
    random.seed(0)

    # get all chars in dataset
    src_charset = set()

    for example in args.dataset['text']:
        for char in regex.findall(r'\X', example):
            src_charset.add(char)

    with open(f"./data/translated/{args.cipher}/charset.txt", 'r') as file:
        cipher_charset = file.read()

    src_charset = "".join(sorted(list(src_charset)))

    cipher_charset = list(cipher_charset)
    random.shuffle(cipher_charset)

    cipher_charset = cipher_charset[:len(src_charset)-1]
    cipher_charset = " " + "".join(cipher_charset)

    cipher = str.maketrans(src_charset, cipher_charset)

    # cipherize dataset
    cipher_text = {'text': []}
    for example in args.dataset['text']:
        for char in regex.findall(r'\X', example):
            cipher_text['text'].append(example.translate(cipher))

    SYMBOLS = set(".,?!:;'\"‘’“”‚‛„()[]{}<>«»‹›--–—―…'’/\|•·⁝⁞­@#&*~§¶^،؟¿¡。、「」『』`​！（），？।")
    NUMBERS = set("0123456789")

    # set punct and capitals for removal in char task
    args.cipher_punct = []
    args.cipher_nums = []
    cipher_uppers = "" 
    cipher_lowers = ""

    for i in range(len(src_charset)):
        if src_charset[i] in SYMBOLS:
            args.cipher_punct.append(cipher_charset[i])
        elif src_charset[i] in NUMBERS:
            args.cipher_nums.append(cipher_charset[i])
        elif src_charset[i].isupper():
            lower_idx = src_charset.index(src_charset[i].lower())
            cipher_uppers += cipher_charset[i]
            cipher_lowers += cipher_charset[lower_idx]

    args.cipher_tolower = str.maketrans(cipher_uppers, cipher_lowers)

    random.seed(0)
    return datasets.Dataset.from_dict(cipher_text)




def main(args):
    args.task = "all"
    args.prompt_exs = {}

    args.dataset_path = f"data/translated/{args.lang}/"
    args.dataset = datasets.load_from_disk(args.dataset_path)
    if args.lang == "jpn":
        args.dataset = args.dataset.map(prep_japanese_dataset, batched=True, batch_size=1000)
    elif args.lang == "zho":
        args.dataset = args.dataset.map(prep_chinese_dataset, batched=True, batch_size=1000)

    if args.cipher:
        args.dataset = cipherize(args)
        args.lang = f"{args.lang}_{args.cipher}"
        args.dataset.save_to_disk(f"./data/translated/{args.lang}")


    get_vocab(args)
    gen_char_tasks(args)
    gen_word_tasks(args)
    gen_prompts(args)




if __name__ == "__main__":
    args = parser.parse_args()
    main(args)