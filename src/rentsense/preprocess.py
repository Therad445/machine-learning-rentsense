from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import TrainConfig


@dataclass
class PreparedData:
    X: pd.DataFrame
    y: pd.Series
    numeric_cols: list[str]
    categorical_cols: list[str]
    dropped: dict[str, list[str]]


def _drop_id_like_cols(df: pd.DataFrame) -> list[str]:
    drop_like_id: list[str] = []
    for c in df.columns:
        lc = c.lower()
        if lc in {"id", "uid", "listing_id", "ad_id"} or lc.endswith("_id") or "scoutid" in lc:
            drop_like_id.append(c)
    return drop_like_id


def _get_leak_cols(target: str) -> list[str]:
    if target == "baseRent":
        return ["totalRent", "baseRentRange"]
    if target == "totalRent":
        return ["baseRent", "baseRentRange"]
    return []


def prepare_xy(df: pd.DataFrame, cfg: TrainConfig) -> PreparedData:
    data = df.copy()

    # geo_plz: numeric -> string zfill(5)
    if "geo_plz" in data.columns:
        s = pd.to_numeric(data["geo_plz"], errors="coerce")
        data["geo_plz"] = s.round().astype("Int64").astype("string")
        data["geo_plz"] = data["geo_plz"].str.zfill(5)

    if cfg.target not in data.columns:
        raise ValueError(f"Target '{cfg.target}' not found in columns")

    # target numeric + drop NaN target
    data = data.dropna(subset=[cfg.target]).copy()
    data[cfg.target] = pd.to_numeric(data[cfg.target], errors="coerce")
    data = data.dropna(subset=[cfg.target]).copy()

    # livingSpace filter
    if cfg.living_space_col in data.columns:
        ls = pd.to_numeric(data[cfg.living_space_col], errors="coerce")
        data = data[ls.between(cfg.living_space_min, cfg.living_space_max)].copy()

    y = data[cfg.target].copy()
    X = data.drop(columns=[cfg.target]).copy()

    dropped: dict[str, list[str]] = {}

    # anti-leakage
    leak_cols = [c for c in _get_leak_cols(cfg.target) if c in X.columns]
    if leak_cols:
        X = X.drop(columns=leak_cols)
    dropped["leakage"] = leak_cols

    # drop heavy text
    drop_heavy = [c for c in cfg.drop_heavy if c in X.columns]
    if drop_heavy:
        X = X.drop(columns=drop_heavy)
    dropped["heavy"] = drop_heavy

    # drop id-like
    id_like = [c for c in _drop_id_like_cols(X) if c in X.columns]
    if id_like:
        X = X.drop(columns=id_like)
    dropped["id_like"] = id_like

    # numeric/categorical
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # drop high-cardinality categories (protect geo_plz)
    protect = set(cfg.protect_cols)
    high_card = [
        c
        for c in categorical_cols
        if c not in protect and X[c].nunique(dropna=True) > cfg.high_card_th
    ]
    if high_card:
        X = X.drop(columns=high_card)
    dropped["high_card"] = high_card

    # recompute
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    # ensure y numeric
    y = pd.to_numeric(y, errors="coerce")
    mask = np.isfinite(y.to_numpy())
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    return PreparedData(
        X=X,
        y=y,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        dropped=dropped,
    )
