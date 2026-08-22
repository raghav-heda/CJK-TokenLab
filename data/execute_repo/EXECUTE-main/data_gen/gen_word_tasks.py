# gen_word_tasks.py
# Adapted from https://github.com/Leukas/CUTE/blob/main/data_gen/gen_word_tasks.py
import os
from sentence_splitter import SentenceSplitter
from datasets import load_dataset, Dataset
import random
import argparse
import pycountry
from sacremoses import MosesTokenizer, MosesDetokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="dupe_word", help="Benchmark task. Possible values: \
                    {contains_word, ins_word, del_word, swap_word, sub_word}.")
parser.add_argument("--lang", type=str, default="eng", help="Language ISO 3 code")

def load_vocab(args): # get 1000 most used words longer than 2 chars
    vocab = []
    with open(f"./data/translated/{args.lang}/vocab.csv", 'r') as f:
        for line in f:
            word, _ = line.split(",")
            vocab.append(word.strip())
    return set(vocab)


def prep_data(dataset):
    splitter = SentenceSplitter(language="en")

    split = []
    for i in range(len(dataset)):
        tokens = splitter.split(dataset[i]['text'])
        split.extend(tokens)
        
    split_ds = Dataset.from_dict({'text': split})

    # filter sentences longer than 10 words, shorter than 3 words
    split_ds = split_ds.filter(lambda x: len(x['text'].split()) <= 10)
    split_ds = split_ds.filter(lambda x: len(x['text'].split()) >= 3)

    # filter out sentences with ", to avoid confusion with prompts
    split_ds = split_ds.filter(lambda x: '"' not in x['text'])

    assert len(split_ds) >= 1004

    # shuffle
    split_ds = split_ds.shuffle(seed=0)
    return split_ds

def main(args):
    if args.dataset is None:
        args.dataset = f"./data/translated/{args.lang}/"
    dataset = prep_data(args.dataset)
    vocab = load_vocab(args)
    try:
        lang_iso2 = pycountry.languages.get(alpha_3=args.lang).alpha_2
        tokenizer = MosesTokenizer(lang=lang_iso2)
        detokenizer = MosesDetokenizer(lang=lang_iso2)
    except AttributeError: # no iso2
        tokenizer = MosesTokenizer(lang=args.lang)
        detokenizer = MosesDetokenizer(lang=args.lang)

    if args.task == "all":
        tasks = ["contains_word", "ins_word", "del_word", "swap_word", "sub_word"]
    elif ',' in args.task:
        tasks = args.task.split(",")
    else:
        tasks = [args.task]

    for task in tasks:
        random.seed(0)
        inputs = []
        labels = []
        contains_dupes = []
        uses_dupes = []
        for i, sentence in enumerate(dataset):
            if i == 1004:
                break
            # tokenize
            words = tokenizer.tokenize(sentence['text'], escape=False, protected_patterns="'")
            contains_dupe = any([x for x in words if words.count(x) > 1])
            input_all = []

            # need to apply sorted() so the seeding actually works (set doesn't have consistent order)
            random_word = random.choice(sorted(list(set(words) - set(".,;:!?\'\""))))
            
            # remove spaces for chinese and japanese
            sent = sentence['text'] if args.lang not in ["zho", "jpn"] else sentence['text'].strip().replace(" ", "")

            if task == "contains_word":
                label = "Yes" if i < 502 else "No" # 50% yes, 50% no
                words_not_in_sent = vocab - set(words) 
                new_random_word = random.choice(sorted(list(words_not_in_sent)))
                
                if label == "No":
                    input_all = [new_random_word, sent]
                    random_word = None # for tracking usage
                else:
                    input_all = [random_word, sent]
            elif task == "ins_word":
                random_word_from_vocab = random.choice(sorted(list(vocab)))
                inserted = []
                for w in words:
                    inserted.append(w)
                    if w == random_word:
                        inserted.append(random_word_from_vocab)

                label = detokenizer.detokenize(inserted, unescape=False)
                input_all = [random_word_from_vocab, random_word, sent]
            elif task == "del_word":
                deleted = []
                for w in words:
                    if w != random_word:
                        deleted.append(w)

                label = detokenizer.detokenize(deleted, unescape=False)
                input_all = [random_word, sent]
            elif task == "sub_word":
                random_word_from_vocab = random.choice(sorted(list(vocab)))
                subbed = []
                for w in words:
                    if w == random_word:
                        subbed.append(random_word_from_vocab)
                    else:
                        subbed.append(w)
                label = detokenizer.detokenize(subbed, unescape=False)
                input_all = [random_word, random_word_from_vocab, sent]
            elif task == "swap_word":
                unique_words_in_sent = [x for x in set(words) if words.count(x) == 1]
                unique_words_in_sent = list(set(unique_words_in_sent) - set(".,;:!?\'\""))
                try:
                    w1, w2 = random.sample(sorted(unique_words_in_sent), k=2)
                except ValueError:
                    continue # not 2 unique words in sentence
                
                swapped = []
                for w in words:
                    if w == w1:
                        swapped.append(w2)
                    elif w == w2:
                        swapped.append(w1)
                    else:
                        swapped.append(w)
                label = detokenizer.detokenize(swapped, unescape=False)
                input_all = [w1, w2, sent]
                    
            inputs.append(input_all)
            labels.append(label)
            contains_dupes.append(contains_dupe)

            use_dupes = random_word is None or words.count(random_word) > 1
            uses_dupes.append(use_dupes)

        # randomly remove 4 inputs and labels
        random.seed(0) # reseed so it's consistent across task selection
        random_word_exs = []
        i = 0
        relax_constraints = False
        while i < 4:
            perm = list(range((len(inputs)+1)//2))
            random.shuffle(perm)
            curi = i
            for p in perm:
                random_idx = perm[p]
                if i == 1 or i == 2: # make sure contains_char is "yes no no yes" 
                    random_idx += len(inputs)//2
                if i > 0 or uses_dupes[random_idx] or relax_constraints:
                    if contains_dupes[random_idx] or i >= 2:
                        random_word_exs.extend(inputs[random_idx])
                        random_word_exs.append(labels[random_idx])
                        inputs.pop(random_idx)
                        labels.pop(random_idx)
                        i += 1
                        relax_constraints = False
                        break

            if curi == i:
                relax_constraints = True

        if args.prompt_exs is not None:
            args.prompt_exs[task] = random_word_exs

        num_inputs = len(inputs[0])

        with open(f"./data/tasks/{args.lang}/{task}.tsv", 'w') as f:
            header = "{}label\n".format("".join([f"input{i}\t" for i in range(1, num_inputs+1)]))
            f.write(header)
            for inp, label in zip(inputs, labels):
                inputs_str = "\t".join(inp)
                f.write(f"{inputs_str}\t{label}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)