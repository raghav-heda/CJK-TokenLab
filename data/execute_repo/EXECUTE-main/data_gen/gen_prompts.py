# gen_prompts.py
import os
import argparse
import pickle

from prompts import PROMPTS, EXTRA_PROMPTS

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, default="eng", help="Language ISO 3 code")

def main_extra(args):
    TASKS = EXTRA_PROMPTS.keys()
    save_path = f"./data/extra_tasks/{args.lang}/prompts.pkl"

    prompts = {}
    for task in TASKS:
        if args.lang not in task:
            continue
        exs = args.prompt_exs[task]
        exs.append("{}")
        if len(exs) >= 13: # 12 + the 1 "{}"
            exs.append("{}")
        if len(exs) == 18: # 16 + the 2 "{}"
            exs.append("{}")
        prompts[task] = EXTRA_PROMPTS[task].format(*exs)
        print(prompts[task])

    # save prompts as pkl
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(prompts, f)

def main(args):
    TASKS = PROMPTS.keys()
    save_path = f"./data/tasks/{args.lang}/prompts.pkl"

    prompts = {}
    for task in TASKS:
        exs = args.prompt_exs[task]
        exs.append("{}")
        if len(exs) >= 13: # 12 + the 1 "{}"
            exs.append("{}")
        if len(exs) == 18: # 16 + the 2 "{}"
            exs.append("{}")
        prompts[task] = PROMPTS[task].format(*exs)
        print(prompts[task])

    # save prompts as pkl
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(prompts, f)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)