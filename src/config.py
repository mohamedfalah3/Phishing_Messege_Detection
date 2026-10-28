"""Central configuration: paths, hyperparameters, and plotting colours."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_URL = ("https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/"
            "resolve/main/Phishing_Email.csv")
RAW_CSV = ROOT / "data" / "raw" / "Phishing_Email.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Text / vocabulary
MAX_LEN = 300          # tokens kept per message (median email is ~160 tokens)
MIN_FREQ = 2           # drop tokens seen fewer than this many times in training
MAX_VOCAB = 30000

# Training
BATCH_SIZE = 64
EPOCHS = 12
PATIENCE = 3           # early stopping on validation loss
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

MODEL_PARAMS = {
    "cnn": {"embed_dim": 128, "num_filters": 128, "kernel_sizes": (3, 4, 5), "dropout": 0.5},
    "bilstm": {"embed_dim": 128, "hidden_dim": 128, "num_layers": 1, "dropout": 0.4},
}

# Error costs used to pick a decision threshold on the validation set.
# A missed phishing message (FN) is treated as 3x worse than a false alarm (FP).
COST_FN = 3.0
COST_FP = 1.0

# Plot colours (validated colourblind-safe categorical order)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e4e3df"
MODEL_COLORS = {"baseline": "#2a78d6", "cnn": "#eb6834", "bilstm": "#1baf7a"}
MODEL_NAMES = {"baseline": "TF-IDF + LogReg (baseline)",
               "cnn": "TextCNN (deep learning)",
               "bilstm": "BiLSTM + Attention (deep learning)"}
