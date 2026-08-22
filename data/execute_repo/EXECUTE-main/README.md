<!-- README.md -->
# EXECUTE 🔪

## EXECUTE: Expandable X(cross)-lingual Extension of CUTE

This work is an extension of the [CUTE](https://github.com/Leukas/CUTE) benchmark for multiple languages.

The languages we cover and have verified the translations of are:
Amharic, Arabic, Chinese, English, German, Korean, Japanese, Russian, and Spanish

We also have unverified datasets for:
Santali, Tamazight, and Xhosa

It is easy to add a new dataset, just follow the steps in [Add a new dataset](#Add-a-new-dataset).

Our corresponding [paper](https://openreview.net/forum?id=m955VLJvcx) has been accepted to Findings of ACL2025! See you in Vienna! 

## Installation

To install dependencies, cd into the repo folder and run:
```bash
pip install -e .
```

## Running

To evaluate, run for example:
```bash
python execute.py --model llama31-8b --task contains_char --lang eng
```

This will evaluate Llama3.1 8B on the Contains Character task for English. All flags are explained here:
```
--model             Name of the model. If it is one of the supported models, it will automatically download the model from the Hub. Otherwise this only serves as a name for the outputs.
--model_path        Model path. Necessary if you use an unsupported model or your model is in a custom location.
--big_model         Supports loading a model across multiple gpus.
--batch_size        Batch size.
--lang              ISO-3 language code.
--task              Benchmark task. Can provide a comma-separated list of the tasks, or one of the keywords below.
--extra_task        Extra tasks. Possible values: all, none, g2c, c2g, k2h (jpn only).
--byte_tokenizer    For the byte-level English experiment only. Path to a byte-level version of a tokenizer.
```
There are also predefined subsets to run:
```
"all"           All tasks.
"none"          No tasks. Useful if you want to run only extra tasks.
"comp"          Composition tasks, i.e. spell, spell_inverse, contains_char, contains_word
"manip"         Manipulation tasks, i.e. ins_char, ins_word, and the same for del, sub, and swap.
"char"          Character-level tasks, i.e. contains_char, ins_char, del_char, sub_char, swap_char
"word"          Word-level tasks, i.e. contains_word, ins_word, del_word, sub_word, swap_word
"other"         The rest. Just spelling and inverse_spelling.
```

The outputs, labels, and scores will be stored in:
```python
f"model_outputs/outputs.{model}.{task}.txt"
f"model_outputs/labels.{model}.{task}.txt"
f"model_outputs/score.{model}.{task}.txt"
```

The outputs of our experiments can already be found there.

## Add a new dataset

#### 1. Translate the TinyStories Dataset
First you need a translated version of TinyStories. You can use our script to translate via Google Translate:
```bash
python ./data_gen/translate_data.py --lang deu --resume
```
Due to the `googletrans` library accessing the web interface, the script will sometimes stop before translating everything. Just run the script again with the `--resume` flag to pick up where you left off.

#### 2. Generate EXECUTE Dataset
The `gen_all.py` script will do the rest.
```bash
python ./data_gen/gen_all.py --lang deu
```

#### 3. Try it out!
```bash
python execute.py --model llama31-8b --task all --lang deu
```

## Extras
### Sub-character tasks:
Run the extra tasks like this: 
```bash
python execute.py --model llama31-8b --task none --extra_task all --lang kor
```

The data is already provided in the repo. It is possible to also generate it, but you need a separate python environment with python3.9, along with these libraries:
```bash
pip install jamo # for hangul to jamo
pip install pykakasi # for kanji to hiragana
pip install cjklib # for chinese chars and japanese kanji to radicals
pip install regex # better regex library
```
After that, but you can generate it like this:
```bash
python ./data_gen/gen_extra_tasks.py --lang kor
```

### Cipher experiment:
You can create your own ciphered text like this:
```bash
python ./data_gen/gen_all.py --lang eng --cipher amh
```
NOTE: This will NOT work for all combinations. The cipher language must have at least as many characters as the original language. Currently known to be bugged with Chinese as a cipher.

You can then run the ciphered text like:
```bash
python execute.py --model llama31-8b --task all --lang eng_amh
```

### Byte-level experiment:
Currently we only support this for Llama 3 (we had to make the tokenizer manually):
```bash
python execute.py --model llama31-8b --task all --lang eng --byte_tokenizer data/byte_tokenizers/llama3/
```
