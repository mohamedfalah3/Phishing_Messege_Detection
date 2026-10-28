# Web UI

A Streamlit web app for the phishing detection models. It uses the trained models in `../models` and the
project code in `../src`, so it gives exactly the same predictions as the scripts.

## Run it

```bash
pip install streamlit          # once
cd ui
streamlit run app.py
```

On Windows you can also double-click **`run_ui.bat`**. The app opens at <http://localhost:8501>.

The models must be trained first (`python run_all.py` from the project folder). The first time you open
*Model performance* or *Error explorer*, the app scores the test set with every model (about a minute
on a CPU) and caches the result in `ui/.cache/`.

## Pages

| Page | What it does |
|---|---|
| **Overview** | Headline test-set results and a comparison of all models |
| **Analyze a message** | Paste a message (or pick an example). Shows the verdict, each model's probability, the words the BiLSTM paid attention to, and common phishing signals |
| **Batch scan** | Upload a CSV or TXT of messages (or sample the test set), classify all of them, filter flagged or wrong ones, download the results |
| **Model performance** | Live metrics, confusion matrix and FP/FN trade-off chart that update as you move the threshold; all saved evaluation figures |
| **Error explorer** | Browse every false positive and false negative, search them, click one to inspect it with attention highlighting |
| **Dataset & method** | Dataset, preprocessing, model architectures, training and evaluation, for presenting |

The **decision threshold** in the sidebar applies to every page.

## Files

```
ui/
├── app.py            # entry point: navigation + sidebar threshold
├── backend.py        # loads models, scores messages, attention, cached test predictions
├── components.py     # verdict card, attention highlighting, charts
├── views/            # one file per page
├── .streamlit/       # theme
└── run_ui.bat        # Windows launcher
```
