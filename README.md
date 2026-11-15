# Intent Classification Service: DistilBERT, FastAPI, Docker, AWS

End-to-end NLP/NLU project: fine-tune a transformer for **intent classification** on
[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (77 banking intents), compare it with a
TF-IDF + Logistic Regression baseline, serve it through a **FastAPI** REST API with request logging and a live
`/metrics` endpoint, containerize it with **Docker**, deploy it to the cloud, and **load-test** the live endpoint.

## Results
> Fill these in from your own runs (`artifacts/*/metrics.json`, `artifacts/load_test.json`). Only publish numbers you measured.

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | _run `train_baseline`_ | _..._ |
| DistilBERT (fine-tuned) | _run `train_bert`_ | _..._ |

| Deployment | Value |
|---|---|
| Cloud / service | _e.g. AWS App Runner, 1 vCPU / 2 GB_ |
| Live URL | _..._ |
| p50 / p95 / p99 latency | _... / ... / ... ms_ (concurrency _N_) |
| Throughput | _... requests/sec_ |

## Project layout
```
src/data.py            Banking77 loader (CSV from GitHub), label maps, stratified validation split
src/train_baseline.py  TF-IDF + Logistic Regression baseline
src/train_bert.py      DistilBERT fine-tuning (AdamW, warmup, best epoch by val macro-F1, confusion analysis)
src/app.py             FastAPI service: /predict, /predict/batch, /health, /metrics
scripts/load_test.py   Async load test -> p50/p95/p99 + requests/sec
tests/                 pytest suite (API, validation, predictors)
Dockerfile             CPU-only image, non-root user, healthcheck
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or regular torch for GPU
pip install -r requirements.txt

python -m src.train_baseline            # ~20 s, writes artifacts/baseline/
python -m src.train_bert --epochs 5     # writes artifacts/bert/model + metrics.json
python -m pytest -q
```
Train on a GPU (a free Colab/Kaggle T4 works). On CPU, try `--max_train_samples 2000 --epochs 2` first.

## Run the API
```bash
uvicorn src.app:app --port 8000
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"text":"I lost my card, what should I do?","top_k":3}'
curl localhost:8000/metrics
```
The service loads the fine-tuned model from `artifacts/bert/model` if present, otherwise the baseline.
Responses include top-k intents, a confidence score and a `low_confidence` flag (threshold via `CONF_THRESHOLD`)
so a downstream system can route uncertain requests to a fallback. Each request is logged as one JSON line with a
request id and latency; `/metrics` reports server-side inference latency percentiles.

## Docker
```bash
docker build -t intent-service .
docker run -p 8000:8000 intent-service
```

## Deploy to AWS (App Runner from ECR)
```bash
AWS_REGION=ap-south-1; ACCOUNT=<your-account-id>
aws ecr create-repository --repository-name intent-service --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
docker tag intent-service:latest $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/intent-service:latest
docker push $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/intent-service:latest
```
Then in the AWS console: **App Runner -> Create service -> Container registry (Amazon ECR)**, choose the image,
set port **8000**, health check path **/health**, 1 vCPU / 2 GB. (Azure alternative: Azure Container Apps with the same image.)
Delete the service when finished to avoid charges.

## Load test (against the live URL)
```bash
python scripts/load_test.py --url https://<service-url> --requests 2000 --concurrency 16
```
Reports success rate, requests/sec and mean/p50/p95/p99 latency, and saves `artifacts/load_test.json`.

## Limitations / next steps
- Single-turn, English-only, 77 banking intents; no out-of-scope class (the `low_confidence` flag is a simple fallback).
- CPU serving; ONNX export or quantization would cut latency further.
