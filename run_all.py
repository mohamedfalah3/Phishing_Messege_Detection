"""Run the full project pipeline end to end:
data preparation -> baseline -> TextCNN -> BiLSTM -> evaluation -> demo predictions."""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = [
    ["data_prep.py"],
    ["baseline.py"],
    ["train.py", "--model", "cnn"],
    ["train.py", "--model", "bilstm"],
    ["evaluate.py"],
    ["predict.py", "--demo"],
]

for step in STEPS:
    print(f"\n{'=' * 70}\n>>> python {' '.join(step)}\n{'=' * 70}", flush=True)
    subprocess.run([sys.executable, *step], cwd=SRC, check=True)
