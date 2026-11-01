# Desktop app

A native Windows desktop application (PySide6 / Qt) for the phishing detection models. It uses the
trained models in `../models` and the shared code in `../src`, so it gives exactly the same predictions
as the scripts and the web UI.

## Run it

```bash
pip install PySide6            # once
python desktop/main.py
```

On Windows you can also double-click **`desktop/run_desktop.bat`** (starts without a console window).

The models must be trained first (`python run_all.py`). The first launch scores the test set in the
background (about a minute on a CPU) and caches it in `desktop/.cache/`; the *Model performance* and
*Error explorer* screens unlock when that finishes.

## Screens

| Screen | What it does |
|---|---|
| **Overview** | Headline test-set results, comparison of all models, how the project works, and all saved evaluation figures in tabs |
| **Analyze a message** | Type or paste a message, pick an example, or **open a `.txt` / `.eml` e-mail file**. Shows the verdict, each model's probability, the BiLSTM attention highlighting, the most-attended words, and common phishing signals |
| **Batch scan** | Open a CSV or TXT file (or scan a test-set sample). Sortable results table, filter by flagged or wrong, accuracy and errors if the file has labels, **export to CSV** |
| **Model performance** | Metrics, confusion matrix and FP/FN trade-off chart that update live as the threshold moves; one click for the cost-tuned threshold; all models compared at the current threshold |
| **Error explorer** | Every false positive or false negative at the current threshold, searchable and sortable; select one to read it with every model's probability and the attention highlighting |

The **decision threshold** slider in the sidebar applies to every screen.

**Shortcuts:** `Ctrl+1` … `Ctrl+5` switch screens · `Ctrl+Enter` analyzes the message.

## Files

```
desktop/
├── main.py           # entry point: splash screen, loads models, starts the window
├── window.py         # main window: sidebar navigation + threshold slider
├── state.py          # shared state (models, threshold, test predictions) + background tasks
├── widgets.py        # tiles, probability cards, verdict banner, charts, table model
├── theme.py          # colours and Qt stylesheet
├── pages/            # one file per screen
└── run_desktop.bat   # Windows launcher
```

## Optional: build a standalone .exe

```bash
pip install pyinstaller
pyinstaller --noconfirm --windowed --name PhishingDetector ^
  --paths src --add-data "models;models" --add-data "results;results" ^
  --add-data "data/processed/test.csv;data/processed" desktop/main.py
```

Because the app bundles PyTorch, the resulting `dist/PhishingDetector` folder is large (around 1 GB).
For a course demo, `run_desktop.bat` is simpler.
