from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

from rentsense.config import TrainConfig
from rentsense.preprocess import prepare_xy
from rentsense.scoring import score_dataframe


REQUESTS = Counter("rentsense_requests_total", "Total requests", ["endpoint", "status"])
LATENCY = Histogram("rentsense_request_latency_seconds", "Request latency", ["endpoint"])

app = FastAPI(title="RentSense API", version="0.1.0")


class ScoreRequest(BaseModel):
    records: list[dict[str, Any]]
    model_path: str = "artifacts/best_model.joblib"
    target: str = "baseRent"
    th_pct: float = 0.20
    th_eur: float = 150.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/score")
def score(req: ScoreRequest):
    t0 = time.time()
    endpoint = "/score"

    try:
        mp = Path(req.model_path)
        if not mp.exists():
            REQUESTS.labels(endpoint=endpoint, status="error").inc()
            raise HTTPException(status_code=400, detail=f"Model not found: {req.model_path}")

        df = pd.DataFrame(req.records)
        cfg = TrainConfig(target=req.target, th_pct=req.th_pct, th_eur=req.th_eur)
        prepared = prepare_xy(df, cfg)
        X = prepared.X

        y_true = None
        if req.target in df.columns:
            y_true = pd.to_numeric(df.loc[X.index, req.target], errors="coerce")

        est = joblib.load(mp)
        scored = score_dataframe(est, X, y_true=y_true, th_pct=req.th_pct, th_eur=req.th_eur)

        REQUESTS.labels(endpoint=endpoint, status="ok").inc()
        return {
            "rows": int(len(scored)),
            "dropped": prepared.dropped,
            "result": scored.to_dict(orient="records"),
        }

    finally:
        LATENCY.labels(endpoint=endpoint).observe(time.time() - t0)
