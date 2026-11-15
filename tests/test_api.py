import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.app import SklearnPredictor, create_app
from src.data import make_label_maps, train_val_split

import pandas as pd


class StubPredictor:
    name = "stub"

    def predict(self, texts, top_k):
        return [[("card_arrival", 0.9), ("lost_or_stolen_card", 0.07), ("top_up_failed", 0.03)][:top_k]
                for _ in texts]


@pytest.fixture()
def client():
    with TestClient(create_app(StubPredictor())) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model": "stub"}


def test_predict_shape(client):
    r = client.post("/predict", json={"text": "Where is my card?", "top_k": 2})
    body = r.json()
    assert r.status_code == 200
    assert body["intent"] == "card_arrival"
    assert body["confidence"] == 0.9 and body["low_confidence"] is False
    assert len(body["top_k"]) == 2
    assert "x-request-id" in r.headers


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {"text": "x" * 513}, {"text": "hi", "top_k": 0}, {}])
def test_predict_validation(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"texts": ["a", "b", "c"]})
    assert r.status_code == 200 and len(r.json()["results"]) == 3
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_metrics_after_traffic(client):
    for _ in range(5):
        client.post("/predict", json={"text": "hello"})
    m = client.get("/metrics").json()
    assert m["requests"] >= 5 and m["errors"] == 0
    assert m["inference_latency_ms"]["samples"] == 5
    assert m["inference_latency_ms"]["p95"] is not None


def test_sklearn_predictor_roundtrip(tmp_path):
    df = pd.DataFrame({"text": ["lost my card", "card lost", "stolen card", "top up failed", "topup did not work", "failed top up"] * 3,
                       "label": ["lost", "lost", "lost", "topup", "topup", "topup"] * 3})
    pipe = Pipeline([("t", TfidfVectorizer()), ("c", LogisticRegression(max_iter=200))]).fit(df.text, df.label)
    path = tmp_path / "m.joblib"
    joblib.dump(pipe, path)
    pred = SklearnPredictor(path).predict(["my card was stolen"], 2)
    assert pred[0][0][0] == "lost" and len(pred[0]) == 2


def test_label_maps_and_split():
    df = pd.DataFrame({"text": [f"t{i}" for i in range(40)], "label": ["a", "b"] * 20})
    l2i, i2l = make_label_maps(df.label)
    assert l2i == {"a": 0, "b": 1} and i2l[1] == "b"
    tr, va = train_val_split(df, 0.25, seed=1)
    assert len(tr) == 30 and len(va) == 10 and set(va.label) == {"a", "b"}
