"""Classical baseline: TF-IDF word/bigram features + logistic regression.
The deep learning models should be compared against this."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from config import MAX_VOCAB, MODELS_DIR, PROCESSED_DIR, SEED
from text_utils import tokenize


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                  ngram_range=(1, 2), min_df=2, max_features=MAX_VOCAB * 2,
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(C=4.0, max_iter=2000, random_state=SEED)),
    ])


def main():
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    pipe = build_pipeline().fit(train.text, train.label)
    print(f"Baseline validation F1: {f1_score(val.label, pipe.predict(val.text)):.4f}")
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "baseline.joblib")
    print(f"Saved to {MODELS_DIR / 'baseline.joblib'}")


if __name__ == "__main__":
    main()
