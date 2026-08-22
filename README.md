# CJK-TokenLab: A Sub-Character-Aware Tokenization System

This research project investigates whether CJK-specific sub-character information (Chinese/Japanese radicals and components, and Korean Jamo) can improve tokenization efficiency and downstream NLP performance compared with standard subword tokenization (BPE/SentencePiece).

It builds upon the foundational research in the ACL 2025 paper:
**“EXECUTE: A Multilingual Benchmark for LLM Token Understanding”** by Lukas Edman, Helmut Schmid, Alexander Fraser.

## Research Objectives
- **Baseline**: Standard BPE and SentencePiece tokenization.
- **Proposed Method**: CJK Sub-Character-Aware Hybrid Tokenization (Radical/Jamo-Augmented).
- **Core Question**: Does structural fallback for rare characters yield a more efficient token vocabulary and better language modeling performance?

## Methodology & Dataset
To accurately replicate the conditions of the base paper while remaining reproducible without access to the authors' private translated corpora:
1. **Source Data**: The official English `TinyStoriesV2-GPT4-valid.txt` from the EXECUTE repository.
2. **Translation Pipeline**: We replicate the exact sentence-selection methodology and translate the identical subset into Chinese (zh-CN), Japanese (ja), and Korean (ko) using the Google Translate API (via `deep_translator`).
3. **CJK Sub-Character Mappings**: 
   - **Chinese/Japanese**: Uses standard Unihan Ideographic Description Sequences (IDS) for robust radical decomposition (deviating from `cjklib` solely due to modern Python compatibility issues).
   - **Japanese Kana**: Uses `pykakasi`.
   - **Korean Jamo**: Uses `jamo` for Hangul decomposition.

## Project Structure
```
CJK-TokenLab/
├── data/
│   ├── raw/             # Generated translated TinyStories
│   ├── execute_repo/    # The cloned official EXECUTE benchmark repo
│   └── splits/          # Train/Val/Test splits
├── src/
│   ├── data/            # Dataset translation pipeline
│   ├── preprocessing/   # Corpus formatting and splits
│   ├── tokenization/    # Baseline and Proposed tokenizers
│   ├── cjk/             # CJK structural analysis (radicals, jamo)
│   ├── training/        # PyTorch Transformer model and loops
│   └── evaluation/      # Token metrics calculation
├── models/              # Saved PyTorch checkpoints
├── tokenizers/          # Saved tokenizer configs/vocabularies
├── results/             # Evaluation metrics and tables
├── app/                 # Streamlit UI
├── tests/               # Validation scripts
├── config/              # Constants and hyperparameters
└── main.py              # Master pipeline script
```

## Setup and Installation

1. Create a Python 3.9+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Pipeline

### Local Smoke Test
To verify the entire pipeline (data generation, tokenization, model training, and evaluation) on a tiny subset of 100 sentences:
```bash
python main.py
```
*(Note: Ensure you have internet access for the initial `deep_translator` API calls and IDS download).*

### Full Experiment (Colab / GPU Recommended)
To run the full 5000-sentence experiment:
1. Edit `main.py` and remove `max_sentences=100` from `prepare_execute_dataset(max_sentences=100)`.
2. Update `config.py` to use `FULL_TRAINING_PARAMS` instead of `SMOKE_TEST_PARAMS`.
3. Run `python main.py`.

## Web Application
To launch the interactive research UI for analyzing CJK characters, visualizing tokenizer outputs, and viewing experiment results:
```bash
uvicorn app.api:app --reload
```

## Scientific Reproducibility & Limitations
- **Tokenization Impact**: Metrics like perplexity can be artificially skewed when vocabulary sizes or sequence lengths differ drastically. We control for this by explicitly measuring Characters per Token, Unknown Token Rates, and computational encoding time.
- **Translation Fidelity**: Our dataset relies on machine translation (Google API) of English TinyStories. While this mimics EXECUTE's original methodology, the resulting CJK text may contain unnatural phrasing.
- **Decomposition**: Recomposing Han characters from sequential IDS components is intrinsically difficult without an exhaustive reverse-lookup index. Our proposed tokenizer uses a deterministic reverse-map restricted to characters encountered during training.

## Limitations & Known Constraints

| Item | Detail |
|---|---|
| **Dataset source** | CJK corpora are machine-translated from English TinyStories via the Google Translate API, not natively authored CJK text. Machine translation normalizes text and may destroy structural linguistic phenomena that EXECUTE tasks are designed to test (e.g., spelling errors, character boundaries). |
| **Smoke-test dataset** | Local smoke tests use **1,000 sentences per language** with a minimal model (2 layers, d_model=64, 1 epoch). Results from smoke tests are **not** scientifically meaningful and must not be reported as research findings. |
| **Full experiment dataset** | The full experiment uses **5,000 sentences per language** with a larger model (4 layers, d_model=256, 10 epochs). This is the configuration intended for reported results. Set `FULL_TRAINING_PARAMS` in `config/config.py` and `max_sentences=5000` in `main.py` to run the full experiment. |
| **Model scale** | Even the full model is a small Transformer (~500K parameters). Results reflect relative trends between tokenizers, not absolute language modeling quality. |
| **Perplexity comparison caveat** | Raw perplexity values are **not** directly comparable across tokenizers with different vocabulary sizes. **True model BPC** (bits per character) is provided as the fair cross-tokenizer metric: `total_NLL_nats / (ln(2) × total_target_chars)`. |
| **Uniform Encoding Cost** | The tokenizer-only metric "Uniform Encoding Cost (bits/char)" assumes a uniform distribution over the vocabulary. It is a theoretical lower-bound estimate, not a learned model metric. |

## Running the Ablation Study

The ablation study tests the effect of decomposition parameters (min_frequency thresholds and decomposition on/off) on tokenizer performance. It runs **separately** from the main pipeline:

```bash
python src/evaluation/ablation.py
```

Results are saved to `results/tables/ablation_results.json` and displayed in the web UI under the Performance tab.
