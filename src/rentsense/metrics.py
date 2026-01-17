from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def evaluate_regression(name: str, y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mp = float(mape(y_true, y_pred))
    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mp, "R2": r2}


def segment_metrics(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    bins = [0, 400, 800, 1200, 10_000]
    labels = ["<400", "400-800", "800-1200", ">1200"]
    seg = pd.cut(y_true, bins=bins, labels=labels, include_lowest=True)

    rows = []
    for lab in labels:
        m = seg == lab
        if int(m.sum()) == 0:
            continue
        yt = y_true[m]
        yp = y_pred[m]
        rows.append(
            {
                "seg": str(lab),
                "n": int(m.sum()),
                "MAE": float(mean_absolute_error(yt, yp)),
                "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
                "MAPE": float(mape(yt, yp)),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["n", "MAE", "RMSE", "MAPE"]).set_index(
            pd.Index([], name="seg")
        )
    return pd.DataFrame(rows).set_index("seg")
