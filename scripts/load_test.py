"""Load-test a running /predict endpoint and report latency percentiles + throughput.

Usage:
  python scripts/load_test.py --url http://localhost:8000 --requests 2000 --concurrency 16
  python scripts/load_test.py --url https://<your-service-url> --requests 2000 --concurrency 16
Writes artifacts/load_test.json (copy these real numbers into your README / resume).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np

FALLBACK = [
    "I lost my card, what should I do?", "How do I top up my account?",
    "Why was my transfer declined?", "My card payment is pending for days",
    "How can I change my PIN?", "Where can I find my exchange rate?",
    "I was charged twice for one purchase", "Can I get a refund for a transaction?",
]


def load_texts(n: int) -> list[str]:
    path = Path("data/banking77_test.csv")
    if path.exists():
        import pandas as pd
        texts = pd.read_csv(path)["text"].dropna().tolist()
        return random.sample(texts, min(n, len(texts)))
    return FALLBACK


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="artifacts/load_test.json")
    args = ap.parse_args()

    texts = load_texts(args.requests)
    endpoint = args.url.rstrip("/") + "/predict"
    sem = asyncio.Semaphore(args.concurrency)
    latencies: list[float] = []
    failures = 0

    async with httpx.AsyncClient(timeout=30) as client:
        for i in range(args.warmup):  # warm-up (not measured)
            await client.post(endpoint, json={"text": texts[i % len(texts)]})

        async def one(i: int) -> None:
            nonlocal failures
            async with sem:
                t0 = time.perf_counter()
                try:
                    r = await client.post(endpoint, json={"text": texts[i % len(texts)]})
                    ok = r.status_code == 200
                except httpx.HTTPError:
                    ok = False
                dt = (time.perf_counter() - t0) * 1000
                if ok:
                    latencies.append(dt)
                else:
                    failures += 1

        start = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(args.requests)))
        wall = time.perf_counter() - start

    lat = np.array(latencies)
    result = {
        "url": endpoint,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "success": len(latencies),
        "failures": failures,
        "wall_seconds": round(wall, 2),
        "requests_per_second": round(len(latencies) / wall, 1),
        "latency_ms": {
            "mean": round(float(lat.mean()), 2) if len(lat) else None,
            "p50": round(float(np.percentile(lat, 50)), 2) if len(lat) else None,
            "p95": round(float(np.percentile(lat, 95)), 2) if len(lat) else None,
            "p99": round(float(np.percentile(lat, 99)), 2) if len(lat) else None,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
