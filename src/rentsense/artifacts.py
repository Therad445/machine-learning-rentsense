from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import TrainConfig


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(model, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    joblib.dump(model, out_path)


def save_df(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)


def save_metadata(out_path: Path, cfg: TrainConfig, extra: dict[str, Any]) -> None:
    ensure_dir(out_path.parent)
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": asdict(cfg),
        "extra": extra,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
