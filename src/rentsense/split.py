from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import TrainConfig


def split_data(X: pd.DataFrame, y: pd.Series, full_df: pd.DataFrame, cfg: TrainConfig):
    '''
    Split options:
    - random (stratify regio1 if exists)
    - geo_holdout (group by geo_plz)
    - time (sort by parsed date and holdout last test_size fraction)
    '''
    if cfg.split_mode == "geo_holdout" and "geo_plz" in X.columns:
        gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
        tr_idx, te_idx = next(gss.split(X, y, groups=X["geo_plz"]))
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
        return X_train, X_test, y_train, y_test

    if cfg.split_mode == "time" and cfg.date_col in full_df.columns:
        d = pd.to_datetime(full_df[cfg.date_col], format=cfg.date_format, errors="coerce")
        mask = d.notna()
        X2 = X.loc[mask].copy()
        y2 = y.loc[mask].copy()
        d2 = d.loc[mask].copy()

        order = np.argsort(d2.values)
        X2 = X2.iloc[order]
        y2 = y2.iloc[order]

        cut = int(len(X2) * (1.0 - cfg.test_size))
        X_train, X_test = X2.iloc[:cut], X2.iloc[cut:]
        y_train, y_test = y2.iloc[:cut], y2.iloc[cut:]
        return X_train, X_test, y_train, y_test

    strat = X["regio1"] if "regio1" in X.columns else None
    return train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed, stratify=strat
    )
