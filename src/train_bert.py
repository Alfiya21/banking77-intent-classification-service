"""Fine-tune DistilBERT (or any HF encoder) for intent classification.

Run:  python -m src.train_bert --model_name distilbert-base-uncased --epochs 5
Quick smoke run:  python -m src.train_bert --epochs 1 --max_train_samples 500
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from src.data import load_banking77, make_label_maps, train_val_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(df, label2id, tokenizer, max_len, batch_size, shuffle):
    items = list(zip(df["text"].tolist(), [label2id[l] for l in df["label"]]))

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        golds.extend(batch["labels"].tolist())
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        preds.extend(model(**inputs).logits.argmax(-1).cpu().tolist())
    return np.array(preds), np.array(golds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="artifacts/bert")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_train_samples", type=int, default=None, help="subset for quick smoke runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"device={device}")

    train_df, test_df = load_banking77(args.data_dir)
    label2id, id2label = make_label_maps(train_df["label"])
    train_df, val_df = train_val_split(train_df, 0.1, args.seed)
    if args.max_train_samples:
        train_df = train_df.sample(min(args.max_train_samples, len(train_df)),
                                   random_state=args.seed).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label2id), id2label=id2label, label2id=label2id
    ).to(device)

    train_loader = make_loader(train_df, label2id, tokenizer, args.max_len, args.batch_size, True)
    val_loader = make_loader(val_df, label2id, tokenizer, args.max_len, 64, False)
    test_loader = make_loader(test_df, label2id, tokenizer, args.max_len, 64, False)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(optim, int(total_steps * args.warmup_ratio), total_steps)

    out = Path(args.out)
    model_dir = out / "model"
    out.mkdir(parents=True, exist_ok=True)
    best_f1, history = -1.0, []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            running += loss.item()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {running / step:.4f}")
        vp, vg = predict(model, val_loader, device)
        val_acc = accuracy_score(vg, vp)
        val_f1 = f1_score(vg, vp, average="macro")
        history.append({"epoch": epoch, "train_loss": round(running / len(train_loader), 4),
                        "val_accuracy": round(float(val_acc), 4), "val_macro_f1": round(float(val_f1), 4)})
        print(history[-1])
        if val_f1 > best_f1:  # keep the best epoch by validation macro-F1
            best_f1 = val_f1
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

    train_seconds = time.time() - t0

    # Final evaluation on the official held-out test split using the best checkpoint
    best = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tp, tg = predict(best, test_loader, device)
    names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(tg, tp, labels=list(range(len(names))), target_names=names,
                                   output_dict=True, zero_division=0)
    confusions = Counter((id2label[g], id2label[p]) for g, p in zip(tg, tp) if g != p)

    metrics = {
        "model_name": args.model_name,
        "device": str(device),
        "train_size": len(train_df), "val_size": len(val_df), "test_size": len(test_df),
        "num_classes": len(label2id),
        "hyperparameters": {k: getattr(args, k) for k in
                            ("epochs", "batch_size", "lr", "max_len", "warmup_ratio", "weight_decay", "seed")},
        "history": history,
        "test_accuracy": round(float(accuracy_score(tg, tp)), 4),
        "test_macro_f1": round(float(f1_score(tg, tp, average="macro")), 4),
        "test_weighted_f1": round(float(f1_score(tg, tp, average="weighted")), 4),
        "train_seconds": round(train_seconds, 1),
        "top_confusions": [{"true": t, "pred": p, "count": c} for (t, p), c in confusions.most_common(15)],
        "per_class": {k: {m: round(v, 4) for m, v in report[k].items()} for k in names},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("test_accuracy", "test_macro_f1", "train_seconds")}, indent=2))


if __name__ == "__main__":
    main()
