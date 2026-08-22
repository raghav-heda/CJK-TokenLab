import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LANGUAGES, RESULTS_DIR
from src.cjk.decomposition import decompose_han, decompose_hangul, kanji_to_hiragana, identify_script
from src.tokenization.cjk_tokenizer import CJKHybridTokenizer
from src.tokenization.baseline_tokenizers import BaselineTokenizer

app = FastAPI(title="CJK TokenLab API")

# Ensure static directory exists
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

class TokenizeRequest(BaseModel):
    text: str
    lang_code: str

@app.get("/api/config")
def get_config():
    return {"languages": LANGUAGES}

@app.get("/api/metrics")
def get_metrics():
    metrics_path = os.path.join(RESULTS_DIR, "tables", "tokenization_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.get("/api/eval/{lang_code}/{model_type}")
def get_eval_history(lang_code: str, model_type: str):
    hist_path = os.path.join("models", f"{lang_code}_{model_type}", "history.json")
    if os.path.exists(hist_path):
        with open(hist_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.get("/api/execute_results", response_class=Response)
def get_execute_results():
    execute_path = os.path.join(RESULTS_DIR, "evaluation", "execute_structural_results.json")
    if os.path.exists(execute_path):
        with open(execute_path, "r", encoding="utf-8") as f:
            text = f.read()
            text = text.replace("Infinity", "null").replace("-null", "null")
            return Response(content=text, media_type="application/json")
    return {}

@app.get("/api/status")
def get_status(lang_code: str = "zho"):
    return {
        "dataset": os.path.exists(os.path.join("data", "raw", f"{lang_code}_corpus.txt")),
        "tokenizer": os.path.exists(os.path.join("tokenizers", f"{lang_code}_proposed", "tokenizer.json")),
        "model": os.path.exists(os.path.join("models", f"{lang_code}_proposed", "model.pt")),
        "metrics": os.path.exists(os.path.join(RESULTS_DIR, "tables", "tokenization_metrics.json")),
        "evals": os.path.exists(os.path.join(RESULTS_DIR, "evaluation", "execute_structural_results.json"))
    }

@app.get("/api/ablation")
def get_ablation():
    ablation_path = os.path.join(RESULTS_DIR, "tables", "ablation_results.json")
    if os.path.exists(ablation_path):
        with open(ablation_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@app.post("/api/tokenize")
def tokenize_text(req: TokenizeRequest):
    lang_code = req.lang_code
    text = req.text
    
    baseline_tokens = []
    proposed_tokens = []
    
    # Baseline
    try:
        baseline_path = os.path.join("tokenizers", f"{lang_code}_baseline_bpe")
        baseline_tok = BaselineTokenizer.load(baseline_path, lang_code, 8000, "bpe")
        enc = baseline_tok.tokenizer.encode(text)
        if hasattr(enc, 'ids') and hasattr(enc, 'tokens'):
            baseline_tokens = [{"id": int(i), "text": t.replace('</w>', '').replace(' ', ' ').strip()} for i, t in zip(enc.ids, enc.tokens)]
        else:
            baseline_tokens = [{"id": int(x), "text": "?"} for x in (enc.ids if hasattr(enc, 'ids') else enc)]
    except Exception as e:
        pass
        
    # Proposed
    try:
        proposed_path = os.path.join("tokenizers", f"{lang_code}_proposed")
        proposed_tok = CJKHybridTokenizer.load(proposed_path)
        preprocessed = proposed_tok._preprocess_text(text)
        enc = proposed_tok.tokenizer.encode(preprocessed)
        if hasattr(enc, 'ids') and hasattr(enc, 'tokens'):
            proposed_tokens = [{"id": int(i), "text": t.replace('</w>', '').replace(' ', ' ').strip()} for i, t in zip(enc.ids, enc.tokens)]
        else:
            proposed_tokens = [{"id": int(x), "text": "?"} for x in (enc.ids if hasattr(enc, 'ids') else enc)]
    except Exception as e:
        pass
        
    return {
        "baseline": baseline_tokens,
        "proposed": proposed_tokens
    }

@app.get("/api/analyze_chars")
def analyze_chars(chars: str):
    results = []
    for char in chars[:10]: # limit to 10
        script = identify_script(char)
        data = {"char": char, "script": script}
        if script == "Han":
            data["decomposition"] = decompose_han(char)
            data["hiragana"] = kanji_to_hiragana(char)
        elif script == "Hangul":
            data["decomposition"] = decompose_hangul(char)
        results.append(data)
    return results

# Mount static files for the frontend
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
