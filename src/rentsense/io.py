from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def read_csv_robust(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    '''
    Reads CSV trying multiple encodings. Falls back to replace invalid chars.
    Mirrors notebook approach.
    '''

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_err: Exception | None = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows)
        except UnicodeDecodeError as e:
            last_err = e

    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", nrows=nrows)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV {path}. Last unicode error: {last_err}") from e
