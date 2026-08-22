# execute.py
# Adapted from https://github.com/Leukas/CUTE/blob/main/main.py
import torch
import os
import argparse
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm import tqdm

from data_gen.prompts import PROMPTS as EMPTY_PROMPTS


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llama31-8b", help="Model name.")
parser.add_argument("--model_path", type=str, default="", help="Model path. Only necessary if \
                    model path is different from that specified in MODELS.")
parser.add_argument("--big_model", action="store_true", help="Load a model across multiple gpus. \
                    Likely necessary for command-r-plus and dbrx.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--task", type=str, default="none", help="Benchmark task. Possible values: \
                    {all, none, comp, manip, char, word, other, \
                    spell, spell_inverse, contains_char, contains_word, ins_char, ins_word, \
                    del_char, del_word, swap_char, swap_word, sub_char, sub_word}.")
parser.add_argument("--extra_task", type=str, default="none", help="Extra task. Possible values: \
                    {all, none, g2c, c2g, k2h}.")
parser.add_argument("--lang", type=str, default="eng", help="Language ISO 3 code.")
parser.add_argument("--byte_tokenizer", type=str, default="", help="2nd tokenizer used for byte-tokenizing inputs.")


PROMPTS = None

TASKS = [
    'spell',
    'spell_inverse',
    'contains_char',
    'contains_word',
    'ins_char',
    'ins_word',
    'del_char',
    'del_word',
    'sub_char',
    'sub_word',
    'swap_char',
    'swap_word',
]

EXTRA_TASKS = {
    'zho': ['zho_g2c', 'zho_c2g', 'zho_contains'], 
    'kor': ['kor_g2c', 'kor_c2g', 'kor_contains'],
    'jpn': ['jpn_g2c', 'jpn_c2g', 'jpn_contains', 'jpn_k2h'],
}

TASK2SUBSET = {
    'all': TASKS,
    'comp': ['spell', 'spell_inverse', 'contains_char', 'contains_word'],
    'manip': ['ins_char', 'ins_word', 'del_char', 'del_word', 'swap_char', 'swap_word', 'sub_char', 'sub_word'],
    'char': ['ins_char', 'del_char', 'swap_char', 'sub_char', 'contains_char'],
    'word': ['ins_word', 'del_word', 'swap_word', 'sub_word', 'contains_word'],
    'other': ['spell', 'spell_inverse'],
}

WORD_TASKS = TASK2SUBSET['word']

MODELS = {
    'aya-expanse-8b': "CohereForAI/aya-expanse-8b",
    'aya-expanse-32b': "CohereForAI/aya-expanse-32b",
    'qwen-7b': "Qwen/Qwen2.5-7B-Instruct",
    'qwen-32b': "Qwen/Qwen2.5-32B-Instruct",
    'llama31-8b': "meta-llama/Llama-3.1-8B-Instruct",
    'llama31-70b': "meta-llama/Llama-3.1-70B-Instruct",
    'llama31-405b': "meta-llama/Meta-Llama-3.1-405B-Instruct-bnb-4bit",
    'llama33-70b': "meta-llama/Llama-3.3-70B-Instruct",
    'mistral-8b': "mistralai/Ministral-8B-Instruct-2410",
    'mistral-24b': "mistralai/Mistral-Small-24B-Instruct-2501",
    'gemma2-9b': "gemma/gemma-2-9b-it",
    'gemma2-27b': "gemma/gemma-2-27b-it"
}


MODEL_SIZES = {
    'big': ['llama31-70b', 'llama33-70b'],
    'medium': ['aya-expanse-32b', 'mistral-24b', 'gemma2-27b', 'qwen-32b'],
    'small': ['aya-expanse-8b', 'mistral-8b', 'gemma2-9b', 'llama31-8b', 'qwen-7b']
}


def byte_tokenize(tokenizer, byte_tokenizer, task, text):
    messages = [
        {"role": "user", "content": EMPTY_PROMPTS[task]}
    ]
    empty_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    empty_prompt = empty_prompt + " Answer: \""
    task_chunks = empty_prompt.split("{}")

    byte_parts = text
    tok_chunks = []
    for i, chunk in enumerate(task_chunks):
        byte_parts = byte_parts.replace(chunk, "@@")
        tok_chunks.append(tokenizer.encode(chunk, return_tensors="pt", add_special_tokens=i==0)[0])

    byte_parts = byte_parts.split("@@")[1:-1]
    tok_byte_parts = []
    for i, part in enumerate(byte_parts):
        if "contains" in task and i % 3 == 2: # keep yes/no on token level 
            tok_byte_parts.append(tokenizer.encode(part, return_tensors="pt", add_special_tokens=False)[0])
        else:
            tok_byte_parts.append(byte_tokenizer.encode(part, return_tensors="pt", add_special_tokens=False)[0])


    final_text = []
    for i in range(len(tok_byte_parts)):
        final_text += tok_chunks[i]
        final_text += tok_byte_parts[i]

    final_text += tok_chunks[-1]

    return torch.LongTensor(final_text).unsqueeze(0)


def padding_collate_fn(batch, tokenizer, max_len=1024):
    """ 
        Pads each list with the tokenizer pad token id and concatenates by key.
        Input: List[{key: List[], ...}]
        Output: {key: LongTensor(), ...}
    """
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer pad_token_id is not set.")

    padded_batch = {}
    for key in batch[0]:
        if key in ["input1", "input2", "input3", "label"]:
            padded_batch[key] = []
            continue
        largest = min(max_len, max([len(b[key]) for b in batch]))

        pad_id = pad_token_id if "labels" not in key else -100
        padded_batch[key] = torch.full((len(batch), largest), pad_id, dtype=torch.long)
    
    for i, sample in enumerate(batch):
        for key in padded_batch:
            if key in ["input1", "input2", "input3", "label"]: 
                padded_batch[key].append(batch[i][key]) 
                continue
            key_len = min(max_len, len(sample[key]))
            padded_batch[key][i, -key_len:] = torch.LongTensor(sample[key][:key_len])

    return padded_batch

def add_prompt(examples, tokenizer, task, byte_tokenizer=None):
    batch = {
        'input_ids':[],
        'attention_mask': [],
        'labels': []
    }

    for i in range(len(examples['input1'])):
        all_inputs = [examples['input1'][i]]
        if 'input2' in examples:
            all_inputs.append(examples['input2'][i])
        if 'input3' in examples:
            all_inputs.append(examples['input3'][i])

        messages = [
            {"role": "user", "content": PROMPTS[task].format(*all_inputs)}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if byte_tokenizer:
            src = byte_tokenize(tokenizer, byte_tokenizer, task, prompt + " Answer: \"")
        else:
            src = tokenizer.encode(prompt + " Answer: \"", return_tensors="pt")
        batch['input_ids'] += src
        tgt = tokenizer.encode(examples['label'][i], return_tensors="pt")
        batch['labels'] += tgt.to(torch.long)
        batch['attention_mask'] += torch.ones_like(src)


    return batch

def clean_output(outs):
    final_out = []
    for i in range(len(outs)):
        try:
            out = outs[i].split("Answer:")[-1] # remove everything before "Answer:"
            out = out.split("<|eot_id|>")[0] # remove </s> and everything after
            out = out.split("</s>")[0] # remove </s> and everything after
            out = out.split("\"")[1].strip() # remove quotes, and stuff like "I hope this answer helped!"
        except: # something wasn't generated properly, discard
            final_out.append("")
            continue
        final_out.append(out)

    return final_out


def execute(args):
    os.makedirs("model_outputs", exist_ok=True)

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )

    assert args.model in MODELS or args.model_path, f"Must specify model path or use model in: {MODELS.keys()}"
    model_path = args.model_path if args.model_path else MODELS[args.model]

    if args.big_model:
        model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config, 
                                                     device_map='auto', trust_remote_code=args.model not in ['command-r', 'command-r-plus'])
    else:
        if args.model == "gemma2-27b":
            model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config, torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=nf4_config)
    model = model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.model not in ['command-r', 'command-r-plus'])

    if args.task == args.extra_task:
        task_wo_lang = args.task.split("_")[1]
        data_folder = f"./data/extra_tasks/{args.lang}/{task_wo_lang}.tsv"
    else:
        data_folder = f"./data/tasks/{args.lang}/{args.task}.tsv"

    dataset = load_dataset('csv', data_files=data_folder, split='train', sep="\t", keep_default_na=False)

    global PROMPTS
    PROMPTS = pickle.load(open(f"./data/tasks/{args.lang}/prompts.pkl", "rb"))

    if args.task == args.extra_task:
        PROMPTS = pickle.load(open(f"./data/extra_tasks/{args.lang}/prompts.pkl", "rb"))

    if args.byte_tokenizer and isinstance(args.byte_tokenizer, str):
        args.byte_tokenizer = AutoTokenizer.from_pretrained(args.byte_tokenizer)

    tokenize = partial(add_prompt, tokenizer=tokenizer, task=args.task, byte_tokenizer=args.byte_tokenizer)
    remove_columns = set(dataset.column_names) - set(['input_ids', 'labels', 'input1', 'input2', 'input3', 'label'])
    dataset = dataset.map(tokenize, batched=True, num_proc=10, remove_columns=remove_columns)

    # make labelset lowercase in case the first word is capitalized
    label_ds = dataset['label']
    if args.task in WORD_TASKS:
        label_ds = [label.lower() for label in label_ds]

    dl = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=partial(padding_collate_fn, tokenizer=tokenizer),
    )

    with torch.no_grad():
        outputs = []
        for batch in tqdm(dl):
            out = model.generate(batch['input_ids'].cuda(), 
                                 attention_mask=batch['attention_mask'].cuda(), 
                                 max_new_tokens=500, 
                                 do_sample=False, 
                                 top_p=1, 
                                 temperature=0,
                                 suppress_tokens=list(range(256,128000)) if args.output_bytes else None
                                )
            outs = tokenizer.batch_decode(out)
            
            true_outs = clean_output(outs)

            print(true_outs[0], f'\033[92m {batch["label"][0]} \033[0m')

            if args.task in WORD_TASKS:
                true_outs = [x.lower() for x in true_outs]
            outputs.extend(true_outs)

    # write outputs and labels to file
    with open(f"model_outputs/outputs.{args.model}.{args.task}.{args.lang}.txt", "w") as f:
        for out in outputs:
            f.write(out + "\n")

    with open(f"model_outputs/labels.{args.model}.{args.task}.{args.lang}.txt", "w") as f:
        for label in label_ds:
            f.write(label + "\n")

    correct = sum(x == y for x, y in zip(outputs, label_ds))
    print(correct / len(outputs))

    # write score to file
    with open(f"model_outputs/score.{args.model}.{args.task}.{args.lang}.txt", "w") as f:
        f.write(f"{correct / len(outputs):.4f}")

def main(args):
    if "," in args.task:
        all_tasks = args.task.split(",")
        for task in all_tasks:
            args.task = task
            execute(args)
    elif args.task in ['all', 'fix', 'char', 'word', 'other', 'comp', 'manip']:
        subset = TASK2SUBSET[args.task]
        for task in subset:
            args.task = task
            execute(args)
    elif args.task == "none":
        pass
    else:
        execute(args)

    if "," in args.extra_task:
        all_tasks = args.extra_task.split(",")
        for task in all_tasks:
            args.task = task
            args.extra_task = task
            execute(args)
    elif args.extra_task == 'all':
        for task in EXTRA_TASKS[args.lang]:
            args.task = task
            args.extra_task = task
            execute(args)
    elif args.extra_task == 'none':
        pass
    else:
        args.task = args.extra_task
        execute(args)


if __name__ == "__main__":
    args = parser.parse_args()
    if "," in args.model:
        models = args.model.split(",")
        task = deepcopy(args.task)
        extra_task = deepcopy(args.extra_task)
        for model in models:
            args.model = model
            args.task = task
            args.extra_task = extra_task
            main(args)
    elif args.model in ['big', 'medium', 'small']:
        models = MODEL_SIZES[args.model]
        task = deepcopy(args.task)
        extra_task = deepcopy(args.extra_task)
        for model in models:
            args.model = model
            args.task = task
            args.extra_task = extra_task
            main(args)
    else:
        main(args)

