"""Banking77 loading (PolyAI task-specific-datasets, CSV on GitHub; no HF `datasets` needed)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
FILES = {"train": f"{BASE}/train.csv", "test": f"{BASE}/test.csv"}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def load_banking77(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) with columns [text, label]."""
    data_dir = Path(data_dir)
    out = {}
    for split, url in FILES.items():
        path = data_dir / f"banking77_{split}.csv"
        if not path.exists():
            _download(url, path)
        df = pd.read_csv(path).rename(columns={"category": "label"})
        out[split] = df[["text", "label"]].dropna().reset_index(drop=True)
    return out["train"], out["test"]


def make_label_maps(labels) -> tuple[dict[str, int], dict[int, str]]:
    classes = sorted(set(labels))
    label2id = {c: i for i, c in enumerate(classes)}
    return label2id, {i: c for c, i in label2id.items()}


def train_val_split(df: pd.DataFrame, val_size: float = 0.1, seed: int = 42):
    tr, va = train_test_split(df, test_size=val_size, stratify=df["label"], random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True)
