"""FastAPI intent-classification service.

Run:  uvicorn src.app:app --host 0.0.0.0 --port 8000
Env:  MODEL_BACKEND=auto|bert|sklearn   MODEL_DIR=artifacts/bert/model
      BASELINE_PATH=artifacts/baseline/model.joblib   CONF_THRESHOLD=0.5
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("intent_service")
if not logger.handlers:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")

MAX_CHARS = 512


# --------------------------------------------------------------------------- predictors
class Predictor(Protocol):
    name: str

    def predict(self, texts: list[str], top_k: int) -> list[list[tuple[str, float]]]: ...


class BertPredictor:
    """HF sequence-classification model (e.g. fine-tuned DistilBERT)."""

    def __init__(self, model_dir: str | Path):
        import torch  # imported lazily so the tests / baseline need no torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.name = f"bert:{getattr(self.model.config, 'model_type', 'encoder')}"

    def predict(self, texts: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        torch = self._torch
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.inference_mode():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
        top_p, top_i = probs.topk(min(top_k, probs.shape[-1]), dim=-1)
        return [[(self.id2label[int(i)], float(p)) for i, p in zip(ri, rp)]
                for ri, rp in zip(top_i.tolist(), top_p.tolist())]


class SklearnPredictor:
    """TF-IDF + LogisticRegression pipeline (baseline / fallback)."""

    def __init__(self, path: str | Path):
        import joblib

        self.pipe = joblib.load(path)
        self.classes = list(self.pipe.classes_)
        self.name = "sklearn:tfidf_logreg"

    def predict(self, texts: list[str], top_k: int) -> list[list[tuple[str, float]]]:
        probs = self.pipe.predict_proba(texts)
        k = min(top_k, probs.shape[1])
        idx = np.argsort(-probs, axis=1)[:, :k]
        return [[(self.classes[j], float(probs[r, j])) for j in row] for r, row in enumerate(idx)]


def load_predictor() -> Predictor:
    backend = os.getenv("MODEL_BACKEND", "auto").lower()
    bert_dir = Path(os.getenv("MODEL_DIR", "artifacts/bert/model"))
    sk_path = Path(os.getenv("BASELINE_PATH", "artifacts/baseline/model.joblib"))
    if backend in ("auto", "bert") and (bert_dir / "config.json").exists():
        return BertPredictor(bert_dir)
    if backend in ("auto", "sklearn") and sk_path.exists():
        return SklearnPredictor(sk_path)
    raise RuntimeError(f"No model found (looked for {bert_dir} and {sk_path}). Train one first.")


# --------------------------------------------------------------------------- schemas
class PredictRequest(BaseModel):
    text: str = Field(..., max_length=MAX_CHARS)
    top_k: int = Field(3, ge=1, le=10)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text must not be blank")
        return v


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)
    top_k: int = Field(3, ge=1, le=10)

    @field_validator("texts")
    @classmethod
    def clean(cls, v: list[str]) -> list[str]:
        v = [t.strip() for t in v]
        if any((not t) or len(t) > MAX_CHARS for t in v):
            raise ValueError(f"each text must be 1-{MAX_CHARS} characters")
        return v


# --------------------------------------------------------------------------- app
class Stats:
    def __init__(self) -> None:
        self.started = time.time()
        self.requests = 0
        self.errors = 0
        self.latencies_ms: deque[float] = deque(maxlen=5000)  # server-side inference time

    def snapshot(self) -> dict:
        lat = np.array(self.latencies_ms) if self.latencies_ms else None
        pct = (lambda q: round(float(np.percentile(lat, q)), 2)) if lat is not None else (lambda q: None)
        return {
            "uptime_seconds": round(time.time() - self.started, 1),
            "requests": self.requests,
            "errors": self.errors,
            "inference_latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99),
                                     "samples": len(self.latencies_ms)},
        }


def create_app(predictor: Predictor | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor or load_predictor()
        app.state.stats = Stats()
        app.state.threshold = float(os.getenv("CONF_THRESHOLD", "0.5"))
        logger.info(json.dumps({"event": "startup", "model": app.state.predictor.name}))
        yield

    app = FastAPI(title="Intent Classification Service", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            ms = round((time.perf_counter() - t0) * 1000, 2)
            stats = getattr(request.app.state, "stats", None)
            if stats is not None and request.url.path != "/health":
                stats.requests += 1
                stats.errors += status >= 500
            logger.info(json.dumps({"event": "request", "id": rid, "method": request.method,
                                    "path": request.url.path, "status": status, "latency_ms": ms}))

    def _run(texts: list[str], top_k: int) -> list[dict]:
        t0 = time.perf_counter()
        results = app.state.predictor.predict(texts, top_k)
        per_item_ms = (time.perf_counter() - t0) * 1000 / len(texts)
        app.state.stats.latencies_ms.append(per_item_ms)
        out = []
        for res in results:
            label, conf = res[0]
            out.append({
                "intent": label,
                "confidence": round(conf, 4),
                "low_confidence": conf < app.state.threshold,
                "top_k": [{"intent": l, "confidence": round(p, 4)} for l, p in res],
            })
        return out

    @app.get("/health")
    def health():
        return {"status": "ok", "model": app.state.predictor.name}

    @app.post("/predict")
    def predict(req: PredictRequest):
        t0 = time.perf_counter()
        result = _run([req.text], req.top_k)[0]
        result["model"] = app.state.predictor.name
        result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return result

    @app.post("/predict/batch")
    def predict_batch(req: BatchRequest):
        return {"model": app.state.predictor.name, "results": _run(req.texts, req.top_k)}

    @app.get("/metrics")
    def metrics():
        return app.state.stats.snapshot()

    return app


app = create_app()
