"""TF-IDF + Logistic Regression baseline. Run: python -m src.train_baseline"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.data import load_banking77


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="artifacts/baseline")
    ap.add_argument("--C", type=float, default=20.0)
    args = ap.parse_args()

    train_df, test_df = load_banking77(args.data_dir)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=args.C, max_iter=1000)),
    ])
    t0 = time.time()
    pipe.fit(train_df["text"], train_df["label"])
    train_s = time.time() - t0
    pred = pipe.predict(test_df["text"])

    metrics = {
        "model": "tfidf_logreg",
        "train_size": len(train_df),
        "test_size": len(test_df),
        "num_classes": int(train_df["label"].nunique()),
        "test_accuracy": round(float(accuracy_score(test_df["label"], pred)), 4),
        "test_macro_f1": round(float(f1_score(test_df["label"], pred, average="macro")), 4),
        "train_seconds": round(train_s, 1),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
