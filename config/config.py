import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.environ.get("SPLITS_DIR", os.path.join(DATA_DIR, "splits"))
EXECUTE_REPO_DIR = os.path.join(DATA_DIR, "execute_repo", "EXECUTE-main")

# Models and tokenizers
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(BASE_DIR, "models"))
TOKENIZERS_DIR = os.environ.get("TOKENIZERS_DIR", os.path.join(BASE_DIR, "tokenizers"))

# Results
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(BASE_DIR, "results"))
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

# Languages
LANGUAGES = {
    "zho": "Chinese",
    "jpn": "Japanese",
    "kor": "Korean"
}

# Ensure directories exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, 
             MODELS_DIR, TOKENIZERS_DIR, TABLES_DIR, FIGURES_DIR, LOGS_DIR]:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass

# Training/Tokenizer settings
RANDOM_SEED = 42
VOCAB_SIZE_BASELINE = 8000
VOCAB_SIZE_PROPOSED = 8000
MIN_FREQUENCY_PROPOSED = 5 # Characters less frequent than this get decomposed in proposed method

# Small Model Hyperparameters (for local smoke tests)
SMOKE_TEST_PARAMS = {
    "d_model": 64,
    "nhead": 2,
    "num_layers": 2,
    "dim_feedforward": 128,
    "batch_size": 4,
    "epochs": 1,
    "lr": 1e-3,
    "max_seq_len": 64
}

# Colab / Full training params
FULL_TRAINING_PARAMS = {
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 1024,
    "batch_size": 32,
    "epochs": 10,
    "lr": 5e-4,
    "max_seq_len": 256
}
