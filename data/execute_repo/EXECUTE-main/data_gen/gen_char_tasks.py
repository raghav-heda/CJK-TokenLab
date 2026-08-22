# gen_char_tasks.py
# Adapted from https://github.com/Leukas/CUTE/blob/main/data_gen/gen_char_tasks.py
import os
import argparse
import random
import regex

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="all", help="Benchmark task. Possible values: \
                    {all, spell, spell_inverse, contains_char, ins_char, del_char, swap_char, sub_char}.")
parser.add_argument("--lang", type=str, default="eng", help="Language ISO 3 code")

TASKS = ['spell', 'spell_inverse', 'contains_char', 'ins_char', 'del_char', 'swap_char', 'sub_char']


def split_word(word):
    # works for Devanagari
    return regex.findall(r'\X', word)

def word_replace(charlist, from_char, to_char):
    return [char if char != from_char else to_char for char in charlist]


def main(args):
    charset = open(f"./data/translated/{args.lang}/charset.txt", 'r').read()
    ALPHABET = set(split_word(charset))

    path = f"./data/tasks/{args.lang}/"
    os.makedirs(path, exist_ok=True)

    if args.task == "all":
        tasks = TASKS
    elif ',' in args.task:
        tasks = args.task.split(",")
    else:
        tasks = [args.task]

    filepath = f"./data/translated/{args.lang}/vocab.csv"
    lines = open(filepath, 'r').readlines()
    lines = lines[1:] # skip header

    for task in tasks:
        random.seed(0)
        inputs = []
        labels = []
        contains_dupes = []
        uses_dupes = []
        for line in lines:
            if len(inputs) == 1004:
                break

            word = line.split(",")[0]
            word = split_word(word)
            contains_dupe = any([x for x in word if word.count(x) > 1])

            input_all = []
            random_char_in_word = random.choice(sorted(word))

            if "spell" in task and "inverse" not in task:
                label = " ".join(word)
                input_all = [word]
                random_char_in_word = None
            elif "spell_inverse" in task:
                input_all = [" ".join(word)]
                label = word
                random_char_in_word = None
            elif "contains_char" in task:
                label = "Yes" if len(inputs) < 502 else "No" # first half yes, second half no
                
                letters_in_word = set(word)
                if label == "No":
                    try:
                        random_letter = random.choice(sorted(list(ALPHABET - letters_in_word)))
                        random_char_in_word = None
                    except IndexError:
                        print("The word is", word)
                        raise IndexError
                else:
                    random_letter = random_char_in_word

                input_all = [random_letter, word]
            
            elif "ins_char" in task:
                random_letter = random.choice(sorted(list(ALPHABET)))
                label = word_replace(word, random_char_in_word, random_char_in_word+random_letter)
                input_all = [random_letter, random_char_in_word, word]
            elif "del_char" in task:
                if len(set(word)) == 1:
                    continue # results in empty string
                label = word_replace(word, random_char_in_word, "")
                input_all = [random_char_in_word, word]
            elif "sub_char" in task:
                random_letter = random.choice(sorted(list(ALPHABET)))
                label = word_replace(word, random_char_in_word, random_letter)
                input_all = [random_char_in_word, random_letter, word]
            elif "swap_char" in task:
                unique_chars_in_word = sorted([x for x in set(word) if word.count(x) == 1])
                try:
                    c1, c2 = random.sample(unique_chars_in_word, k=2)
                except ValueError: # not enough unique chars in word
                    continue
                label = list(word)
                label[word.index(c1)] = c2
                label[word.index(c2)] = c1
                input_all = [c1, c2, word]
                    
            use_dupes = random_char_in_word is None or word.count(random_char_in_word) > 1 or task == "swap_char"
            
            label = "".join(label)
            inputs.append(input_all)
            labels.append(label)
            contains_dupes.append(contains_dupe)
            uses_dupes.append(use_dupes)

        random.seed(0) # reseed so it's consistent across task selection
        random_char_exs = []
        i = 0
        while i < 4:
            perm = list(range((len(inputs)+1)//2)) # 502 502 501 501
            random.shuffle(perm)
            for p in perm:
                random_idx = perm[p]
                if i == 1 or i == 2: # make sure contains_char is "yes no no yes" 
                    random_idx += len(inputs)//2 # 501 501
                if i > 0 or uses_dupes[random_idx] or args.lang == "zho":
                    if i >= 2 or contains_dupes[random_idx] or args.lang == "zho":
                        ex = ["".join(x) for x in inputs[random_idx]]
                        label = "".join(labels[random_idx])
                        random_char_exs.extend(ex)
                        random_char_exs.append(label)
                        inputs.pop(random_idx)
                        labels.pop(random_idx)
                        i += 1
                        break

        if args.prompt_exs is not None:
            args.prompt_exs[task] = random_char_exs

        num_inputs = len(inputs[0])

        path_name = path+f"{task}.tsv"

        with open(path_name, 'w') as f:
            header = "{}label\n".format("".join([f"input{i}\t" for i in range(1, num_inputs+1)]))
            f.write(header)
            for inp, label in zip(inputs[:1000], labels[:1000]):
                inputs_str = "\t".join(["".join(x) for x in inp])
                f.write(f"{inputs_str}\t{label}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)