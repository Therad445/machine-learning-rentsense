from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def save_missing_top30(df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    na = (df.isna().mean() * 100).sort_values(ascending=False)
    na_tbl = na.to_frame("missing_%").head(30)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    na_tbl.to_csv(out_csv, index=True)
    return na_tbl


def plot_target_hist(df: pd.DataFrame, target: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(df[target].dropna(), bins=60)
    plt.title(f"Target distribution: {target}")
    plt.xlabel(target)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_scatter_area_vs_target(df: pd.DataFrame, area_col: str, target: str, out_png: Path) -> None:
    if area_col not in df.columns:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    x = pd.to_numeric(df[area_col], errors="coerce")
    y = df[target]
    mask = x.notna() & y.notna()

    plt.figure()
    plt.scatter(x[mask], y[mask], s=8, alpha=0.3)
    plt.title(f"{target} vs {area_col}")
    plt.xlabel(area_col)
    plt.ylabel(target)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_top_locations(df: pd.DataFrame, geo_col: str, out_png: Path, topn: int = 15) -> None:
    if geo_col not in df.columns:
        return
    out_png.parent.mkdir(parents=True, exist_ok=True)
    top = df[geo_col].astype("string").str.zfill(5).value_counts().head(topn)

    plt.figure()
    plt.bar(top.index.astype(str), top.values)
    plt.title(f"Top {len(top)}: {geo_col}")
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_pred_vs_true(y_true, y_pred, out_png: Path) -> None:
    import numpy as np

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=0.35)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([mn, mx], [mn, mx])
    plt.title("Prediction vs True")
    plt.xlabel("y_true")
    plt.ylabel("y_pred")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
