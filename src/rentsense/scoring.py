from __future__ import annotations

import numpy as np
import pandas as pd


def score_dataframe(
    model,
    X: pd.DataFrame,
    y_true: pd.Series | None = None,
    th_pct: float = 0.20,
    th_eur: float = 150.0,
    pred_clip: tuple[float, float] = (1e-3, 25_000.0),
) -> pd.DataFrame:
    preds = model.predict(X)
    preds = np.clip(preds, pred_clip[0], pred_clip[1])

    out_df = X.copy()
    out_df["y_pred"] = preds

    if y_true is not None:
        out_df["y_true"] = y_true.values
        out_df["ratio"] = out_df["y_true"] / out_df["y_pred"]
        out_df["pct_diff"] = out_df["ratio"] - 1.0
        out_df["eur_diff"] = out_df["y_true"] - out_df["y_pred"]

        over_mask = (out_df["pct_diff"] > th_pct) & (out_df["eur_diff"] > th_eur)
        under_mask = (out_df["pct_diff"] < -th_pct) & (out_df["eur_diff"] < -th_eur)

        out_df["flag"] = np.where(
            over_mask, "overpriced",
            np.where(under_mask, "underpriced", "ok")
        )
    else:
        out_df["flag"] = "unknown"

    return out_df


def top_examples(scored_df: pd.DataFrame, n: int = 10):
    if "flag" not in scored_df.columns:
        return scored_df.iloc[0:0], scored_df.iloc[0:0]

    over_mask = scored_df["flag"] == "overpriced"
    under_mask = scored_df["flag"] == "underpriced"

    over = (
        scored_df[over_mask]
        .sort_values(["pct_diff", "eur_diff"], ascending=False)
        .head(n)
    )

    under = (
        scored_df[under_mask]
        .sort_values(["pct_diff", "eur_diff"], ascending=True)
        .head(n)
    )

    return over, under
