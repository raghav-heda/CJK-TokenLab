import os
import sys
sys.path.append(os.getcwd())
import json
import pandas as pd
import torch
import torch.nn.functional as F
import math
import pickle
from config.config import LANGUAGES, RESULTS_DIR, MODELS_DIR, TOKENIZERS_DIR, SMOKE_TEST_PARAMS, FULL_TRAINING_PARAMS
from src.training.model import SimpleTransformerLM
from src.tokenization.baseline_tokenizers import BaselineTokenizer
from src.tokenization.cjk_tokenizer import CJKHybridTokenizer

def get_model_params(model_dir):
    try:
        return FULL_TRAINING_PARAMS
    except:
        return SMOKE_TEST_PARAMS

def evaluate_structural_tasks(model, tokenizer, lang_code, device, save_prefix):
    """
    Scientifically Valid EXECUTE-Inspired CJK Structural Evaluation.
    """
    model.eval()
    model.to(device)
    tasks_dir = os.path.join("data", "execute_repo", "EXECUTE-main", "data", "tasks", lang_code)
    extra_tasks_dir = os.path.join("data", "execute_repo", "EXECUTE-main", "data", "extra_tasks", lang_code)
    
    results = {}
    
    def get_logprob(context_str, target_str):
        text_full = f"{context_str}{target_str}"
        context_only = context_str
        
        if hasattr(tokenizer, 'spm_processor') and tokenizer.spm_processor:
            tokens = tokenizer.spm_processor.encode_as_ids(text_full)
            ctx_tokens = tokenizer.spm_processor.encode_as_ids(context_only)
        elif hasattr(tokenizer, 'encode'):
            enc = tokenizer.encode(text_full)
            tokens = enc.ids if hasattr(enc, 'ids') else enc
            enc_ctx = tokenizer.encode(context_only)
            ctx_tokens = enc_ctx.ids if hasattr(enc_ctx, 'ids') else enc_ctx
        else:
            return float('-inf'), 0
            
        max_len = model.pos_encoder.pe.size(1)
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        if len(ctx_tokens) > max_len:
            ctx_tokens = ctx_tokens[:max_len]
            
        if len(tokens) <= len(ctx_tokens):
            return float('-inf'), 0
            
        input_tensor = torch.tensor([tokens[:-1]], dtype=torch.long).to(device)
        target_tensor = torch.tensor([tokens[1:]], dtype=torch.long).to(device)
        
        if input_tensor.size(1) == 0:
            return float('-inf'), 0
            
        logits = model(input_tensor)
        
        shift = len(ctx_tokens) - 1
        if shift >= logits.size(1) or shift < 0:
            shift = 0
            
        target_logits = logits[0, shift:, :]
        target_labels = target_tensor[0, shift:]
        
        if target_labels.size(0) > 0:
            loss = F.cross_entropy(target_logits, target_labels, reduction='sum')
            log_prob = -loss.item()
            return log_prob, target_labels.size(0)
        else:
            return float('-inf'), 0

    def evaluate_tsv(filepath, task_name, prompts_dict):
        if not os.path.exists(filepath):
            return None
            
        df = pd.read_csv(filepath, sep='\t').dropna()
        if len(df) == 0:
            return None
            
        total = min(100, len(df))
        predictions = []
        
        is_binary_classification = False
        candidates = []
        if 'label' in df.columns:
            unique_labels = set(df['label'].astype(str).unique())
            if unique_labels.issubset({'Yes', 'No'}):
                is_binary_classification = True
                candidates = ['Yes', 'No']
            elif unique_labels.issubset({'True', 'False'}):
                is_binary_classification = True
                candidates = ['True', 'False']

        prompt_template = None
        if prompts_dict and task_name in prompts_dict:
            prompt_template = prompts_dict[task_name]
        elif prompts_dict and task_name.replace("extra_", "") in prompts_dict:
            prompt_template = prompts_dict[task_name.replace("extra_", "")]
            
        correct_count = 0
        sum_normalized_nll = 0.0
        # BPC accumulators: total NLL in nats and total target characters
        total_nll_nats = 0.0
        total_target_chars = 0
        
        with torch.no_grad():
            for i, row in df.head(total).iterrows():
                cols = row.index.tolist()
                
                in_args = [str(row.iloc[j]) for j in range(len(cols)-1)]
                true_target = str(row.iloc[-1])
                if prompt_template:
                    # In python string formatting, if we have more args than {} it ignores the extra if we use numbered formatting, but with {} it requires exact match.
                    # Some prompts might have a bug. Let's wrap in try/except or just pass exactly what's needed.
                    try:
                        context = prompt_template.format(*in_args) + " Answer: \""
                        target_full = f" {true_target} \""
                    except IndexError:
                        context = " ".join(in_args) + "\nTarget: "
                        target_full = true_target
                else:
                    context = "\n".join(in_args) + "\nTarget: "
                    target_full = true_target
                        
                pred_label = None
                is_correct = None
                
                if is_binary_classification:
                    best_score = float('-inf')
                    best_cand = None
                    scores = {}
                    for cand in candidates:
                        cand_target_full = f" {cand} \"" if prompt_template else cand
                        lp, t_len = get_logprob(context, cand_target_full)
                        norm_score = lp / t_len if t_len > 0 else float('-inf')
                        scores[cand] = norm_score
                        if norm_score > best_score:
                            best_score = norm_score
                            best_cand = cand
                            
                    pred_label = best_cand
                    is_correct = (pred_label == true_target)
                    if is_correct:
                        correct_count += 1
                        
                    predictions.append({
                        "context": context,
                        "true_target": true_target,
                        "pred_label": pred_label,
                        "is_correct": is_correct,
                        "scores": scores
                    })
                else:
                    lp, t_len = get_logprob(context, target_full)
                    norm_score = lp / t_len if t_len > 0 else float('-inf')
                    sum_normalized_nll += norm_score
                    # Accumulate for true model BPC
                    if lp != float('-inf') and t_len > 0:
                        total_nll_nats += (-lp)  # lp is negative log-prob, so -lp is positive NLL
                        total_target_chars += len(true_target)
                    
                    predictions.append({
                        "context": context,
                        "true_target": true_target,
                        "mean_token_logprob": norm_score,
                        "token_count": t_len
                    })

        os.makedirs(os.path.join(RESULTS_DIR, "evaluation"), exist_ok=True)
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(os.path.join(RESULTS_DIR, "evaluation", f"{save_prefix}_{task_name}_predictions.csv"), index=False)
        
        if is_binary_classification:
            acc = correct_count / total if total > 0 else 0.0
            # Wilson score 95% confidence interval
            ci_low, ci_high = 0.0, 0.0
            if total > 0:
                z = 1.96  # 95% CI
                n = total
                p_hat = acc
                denom = 1 + z**2 / n
                center = (p_hat + z**2 / (2 * n)) / denom
                margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
                ci_low = max(0.0, center - margin)
                ci_high = min(1.0, center + margin)
            cm_path = os.path.join(RESULTS_DIR, "evaluation", f"{save_prefix}_{task_name}_confusion_matrix.json")
            try:
                cm = {}
                for p in predictions:
                    t = str(p["true_target"])
                    pr = str(p["pred_label"])
                    if t not in cm:
                        cm[t] = {}
                    cm[t][pr] = cm[t].get(pr, 0) + 1
                with open(cm_path, "w") as f:
                    json.dump(cm, f, indent=4)
            except Exception as e:
                pass
                
            return {
                "accuracy": acc,
                "ci_95": [round(ci_low, 4), round(ci_high, 4)],
                "n_samples": total,
                "metric_used": "accuracy (contrastive length-normalized logprob)"
            }
        else:
            mean_nll = sum_normalized_nll / total if total > 0 else float('-inf')
            perplexity = math.exp(-mean_nll) if mean_nll != float('-inf') and mean_nll > -500 else float('inf')
            # True model BPC: total_nll_nats / (ln(2) × total_target_chars)
            bpc = None
            if total_target_chars > 0 and total_nll_nats > 0:
                bpc = round(total_nll_nats / (math.log(2) * total_target_chars), 4)
            return {
                "perplexity": perplexity,
                "bpc": bpc,
                "mean_token_logprob": mean_nll,
                "total_target_chars": total_target_chars,
                "metric_used": "perplexity (generative task, no candidates provided)"
            }

    prompts_path = os.path.join(tasks_dir, "prompts.pkl")
    prompts_dict = {}
    if os.path.exists(prompts_path):
        try:
            with open(prompts_path, 'rb') as f:
                prompts_dict = pickle.load(f)
        except Exception as e:
            pass

    if os.path.exists(tasks_dir):
        for file in os.listdir(tasks_dir):
            if file.endswith(".tsv"):
                task_name = file.replace(".tsv", "")
                res = evaluate_tsv(os.path.join(tasks_dir, file), task_name, prompts_dict)
                if res:
                    results[task_name] = res

    if os.path.exists(extra_tasks_dir):
        extra_prompts_path = os.path.join(extra_tasks_dir, "prompts.pkl")
        extra_prompts_dict = prompts_dict.copy()
        if os.path.exists(extra_prompts_path):
            try:
                with open(extra_prompts_path, 'rb') as f:
                    ep = pickle.load(f)
                    extra_prompts_dict.update(ep)
            except Exception:
                pass
                
        for file in os.listdir(extra_tasks_dir):
            if file.endswith(".tsv"):
                task_name = "extra_" + file.replace(".tsv", "")
                res = evaluate_tsv(os.path.join(extra_tasks_dir, file), task_name, extra_prompts_dict)
                if res:
                    results[task_name] = res
                    
    return results

def load_model_safely(model_path, vocab_size, device):
    params = SMOKE_TEST_PARAMS 
    if "FULL" in os.environ.get("TRAINING_MODE", ""):
        params = FULL_TRAINING_PARAMS
        
    model = SimpleTransformerLM(
        vocab_size=vocab_size,
        d_model=params["d_model"],
        nhead=params["nhead"],
        num_layers=params["num_layers"],
        dim_feedforward=params["dim_feedforward"],
        max_seq_len=params["max_seq_len"]
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Failed to load model {model_path} with {params['d_model']} dims. Trying alternate...")
        alt_params = FULL_TRAINING_PARAMS if params == SMOKE_TEST_PARAMS else SMOKE_TEST_PARAMS
        model = SimpleTransformerLM(
            vocab_size=vocab_size,
            d_model=alt_params["d_model"],
            nhead=alt_params["nhead"],
            num_layers=alt_params["num_layers"],
            dim_feedforward=alt_params["dim_feedforward"],
            max_seq_len=alt_params["max_seq_len"]
        )
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except:
            print(f"Could not load model at {model_path}")
            return None
            
    return model

def run_all_evaluations():
    print("Running EXECUTE-Inspired CJK Structural Evaluation...")
    os.makedirs(os.path.join(RESULTS_DIR, "evaluation"), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = {}
    
    for lang_code in LANGUAGES.keys():
        print(f"Evaluating {LANGUAGES[lang_code]} models...")
        lang_res = {}
        
        base_tok_path = os.path.join(TOKENIZERS_DIR, f"{lang_code}_baseline_bpe")
        base_model_path = os.path.join(MODELS_DIR, f"{lang_code}_baseline", "model.pt")
        
        if os.path.exists(base_tok_path) and os.path.exists(base_model_path):
            try:
                base_tok = BaselineTokenizer.load(base_tok_path, lang_code, vocab_size=0, method="bpe")
                vocab_size = base_tok.tokenizer.get_vocab_size() if hasattr(base_tok, 'tokenizer') else base_tok.vocab_size
                model_base = load_model_safely(base_model_path, vocab_size, device)
                if model_base:
                    lang_res["baseline"] = evaluate_structural_tasks(model_base, base_tok, lang_code, device, f"{lang_code}_baseline")
                else:
                    lang_res["baseline"] = {"error": "Model load failed"}
            except Exception as e:
                lang_res["baseline"] = {"error": str(e)}
        else:
            lang_res["baseline"] = {"status": "missing_files"}
            
        prop_tok_path = os.path.join(TOKENIZERS_DIR, f"{lang_code}_proposed")
        prop_model_path = os.path.join(MODELS_DIR, f"{lang_code}_proposed", "model.pt")
        
        if os.path.exists(prop_tok_path) and os.path.exists(prop_model_path):
            try:
                prop_tok = CJKHybridTokenizer.load(prop_tok_path)
                vocab_size = prop_tok.tokenizer.get_vocab_size()
                model_prop = load_model_safely(prop_model_path, vocab_size, device)
                if model_prop:
                    lang_res["proposed"] = evaluate_structural_tasks(model_prop, prop_tok, lang_code, device, f"{lang_code}_proposed")
                else:
                    lang_res["proposed"] = {"error": "Model load failed"}
            except Exception as e:
                lang_res["proposed"] = {"error": str(e)}
        else:
            lang_res["proposed"] = {"status": "missing_files"}
            
        all_results[lang_code] = lang_res
        
    out_path = os.path.join(RESULTS_DIR, "evaluation", "execute_structural_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Evaluation completed. Results saved to {out_path}")

if __name__ == "__main__":
    run_all_evaluations()


