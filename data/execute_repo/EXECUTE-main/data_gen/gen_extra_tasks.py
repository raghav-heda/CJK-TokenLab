# Must be run with python==3.9 to support cjklib
import os
import argparse
import regex
import random
from copy import deepcopy
from jamo import j2hcj, h2j
import pykakasi
from cjklib.characterlookup import CharacterLookup
from gen_prompts import main_extra as gen_prompts

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, help="Benchmark task. Possible values: \
    {kor, jpn, zho}.")

COMB_CHARS = "⿰⿱⿲⿳⿴⿵⿶⿸⿹⿺⿻"

def split_word(word):
    return regex.findall(r'\X', word)

def load_charset(args):
    charset = open(f"./data/translated/{args.lang}/charset.txt", 'r').read()
    return sorted(set(split_word(charset)))

def save_spell_tasks(chars, graphs, lang, task_name="g2c", do_reverse=False):
    header ="input1\tlabel\n"
    # g2c
    with open(f"./data/extra_tasks/{lang}/{task_name}.tsv", 'w') as f:
        f.write(header)
        for inp, label in zip(graphs, chars):
            f.write(f"{inp}\t{label}\n")
    if do_reverse:
        # c2g
        with open(f"./data/extra_tasks/{lang}/{task_name[::-1]}.tsv", 'w') as f:
            f.write(header)
            for inp, label in zip(chars, graphs):
                f.write(f"{inp}\t{label}\n")

def save_contains_tasks(inputs, labels, lang):
    header ="input1\tinput2\tlabel\n"
    with open(f"./data/extra_tasks/{lang}/contains.tsv", 'w') as f:
        f.write(header)
        for inp, label in zip(inputs, labels):
            char, graph = inp
            f.write(f"{char}\t{graph}\t{label}\n")

def split_for_prompt(inputs, labels, do_reverse=False):
    random.seed(0) # reseed so it's consistent across task selection
    total_exs = len(labels)
    new_inputs = deepcopy(inputs)
    new_labels = deepcopy(labels)
    exs = []
    if do_reverse:
        rev_exs = []  
    i = 0
    while i < 4:
        half_pt = (total_exs - i) // 2
        perm = list(range(half_pt))
        random.shuffle(perm)
        for p in perm:
            random_idx = perm[p]
            if i == 1 or i == 2: # make sure contains_char is "yes no no yes" 
                random_idx += half_pt # because we removed 1
            ex = ["".join(x) for x in new_inputs[random_idx]]
            label = "".join(new_labels[random_idx])
            exs.extend(ex)
            exs.append(label)
            if do_reverse:
                rev_exs.append(label)
                rev_exs.extend(ex)
            new_inputs.pop(random_idx)
            new_labels.pop(random_idx)
            i += 1
            break

    if do_reverse:
        return exs, rev_exs, new_inputs, new_labels
    return exs, new_inputs, new_labels

def contains_task(charset, chars, graphs):
    num_graphs = len(graphs)

    inputs = []
    labels = []
    for i in range(num_graphs):
        graph = graphs[i]
        char = chars[i]

        label = "Yes" if i < num_graphs//2 else "No" # first half yes, second half no
        
        chars_in_graph = set(char) - set(COMB_CHARS)
        random_char_in_graph = random.choice(sorted(chars_in_graph))

        if label == "No":
            try:
                other_chars = sorted(set(charset) - set(COMB_CHARS) - chars_in_graph)
                random_char = random.choice(other_chars)
                # random_char_in_graph = None
            except IndexError:
                print("The graph is", graph)
                raise IndexError
        else:
            random_char = random_char_in_graph

        input_all = [random_char, graph]

        inputs.append(input_all)
        labels.append(label)
            
    return inputs, labels

def prep_chinese_tasks(args):
    charset = load_charset(args)
    lookup = CharacterLookup('C')

    chars = []
    rads = []
    for char in charset:
        decomps = lookup.getDecompositionEntries(char)
        if len(decomps) == 0: # stuff like "一" cant be decomposed
            continue
        # remove ambiguous decompositions
        if len(decomps) > 1:
            continue

        torads = lookup.decompositionToString(decomps[0])
        torads = regex.sub("\[\d\]", "", torads)
        if torads != char: # filter any already radicals
            if "？" not in torads: # filter any unknowns
                chars.append(char)
                rads.append(torads)

    args.prompt_exs = {}
    g2c_exs, c2g_exs, new_chars, new_rads = split_for_prompt(chars, rads, do_reverse=True)
    save_spell_tasks(new_rads, new_chars, args.lang, do_reverse=True)
    args.prompt_exs[f"{args.lang}_g2c"] = g2c_exs
    args.prompt_exs[f"{args.lang}_c2g"] = c2g_exs

    inputs, labels = contains_task(charset, rads, chars)
    cont_exs, new_inputs, new_labels = split_for_prompt(inputs, labels)
    save_contains_tasks(new_inputs, new_labels, args.lang)
    args.prompt_exs[f"{args.lang}_contains"] = cont_exs


def prep_japanese_tasks(args):
    charset = load_charset(args)
    kks = pykakasi.kakasi()
    lookup = CharacterLookup('J')

    k2h_kanji = []
    hiragana = []

    k2r_kanji = []
    rads = []
    for char in charset:
        kana = kks.convert(char)[0]
        if char == kana['kana']: # filter any katakana
            continue
        tohiragana = kana['hira']
        if tohiragana != char: # filter any already hiragana or punct
            k2h_kanji.append(char)
            hiragana.append(tohiragana)

    for kan in k2h_kanji:
        decomps = lookup.getDecompositionEntries(kan)

        if len(decomps) == 0: # stuff like "一" cant be decomposed
            continue

        # remove ambiguous decompositions
        if len(decomps) > 1:
            continue

        torads = lookup.decompositionToString(decomps[0])
        torads = regex.sub("\[\d\]", "", torads)
        if torads != kan: # filter any already radicals
            if "？" not in torads: # filter any unknowns
                k2r_kanji.append(kan)
                rads.append(torads)

    args.prompt_exs = {}
    g2c_exs, c2g_exs, new_chars, new_rads = split_for_prompt(k2r_kanji, rads, do_reverse=True)
    save_spell_tasks(new_rads, new_chars, args.lang, do_reverse=True)
    args.prompt_exs[f"{args.lang}_g2c"] = g2c_exs
    args.prompt_exs[f"{args.lang}_c2g"] = c2g_exs

    k2h_exs, new_chars, new_rads = split_for_prompt(k2h_kanji, hiragana, do_reverse=False)
    save_spell_tasks(new_rads, new_chars, args.lang, "k2h")
    args.prompt_exs[f"{args.lang}_k2h"] = k2h_exs

    inputs, labels = contains_task(charset, rads, k2r_kanji)
    cont_exs, new_inputs, new_labels = split_for_prompt(inputs, labels)
    save_contains_tasks(new_inputs, new_labels, args.lang)
    args.prompt_exs[f"{args.lang}_contains"] = cont_exs



def prep_korean_tasks(args):
    charset = load_charset(args)

    hangul = []
    jamo = []
    for char in charset:
        tojamo = j2hcj(h2j(char))
        if tojamo != char: # filter any already jamo
            hangul.append(char)
            jamo.append(tojamo)

    args.prompt_exs = {}
    g2c_exs, c2g_exs, new_chars, new_rads = split_for_prompt(hangul, jamo, do_reverse=True)
    save_spell_tasks(new_rads, new_chars, args.lang, do_reverse=True)
    args.prompt_exs[f"{args.lang}_g2c"] = g2c_exs
    args.prompt_exs[f"{args.lang}_c2g"] = c2g_exs

    inputs, labels = contains_task(charset, jamo, hangul)
    cont_exs, new_inputs, new_labels = split_for_prompt(inputs, labels)
    save_contains_tasks(new_inputs, new_labels, args.lang)
    args.prompt_exs[f"{args.lang}_contains"] = cont_exs




def main(args):
    os.makedirs(f"data/extra_tasks/{args.lang}", exist_ok=True)
    args.extra = True
    if args.lang == "zho":
        prep_chinese_tasks(args)
    elif args.lang == "jpn":
        prep_japanese_tasks(args)
    elif args.lang == "kor":
        prep_korean_tasks(args)

    gen_prompts(args)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)