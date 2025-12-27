from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def _wrap_log_target(pipe: Pipeline) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=pipe,
        func=np.log1p,
        inverse_func=np.expm1,
    )


def build_ridge_model(numeric_cols: list[str], categorical_cols: list[str]) -> TransformedTargetRegressor:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20)),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", Ridge(alpha=50.0)),
    ])
    return _wrap_log_target(pipe)


def build_histgbr_model(
    numeric_cols: list[str],
    categorical_cols: list[str],
    seed: int = 42,
    fast_mode: bool = True,
) -> TransformedTargetRegressor:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(
        random_state=seed,
        learning_rate=0.08,
        max_iter=(120 if fast_mode else 300),
        early_stopping=True,
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model),
    ])
    return _wrap_log_target(pipe)


def safe_clip_preds(pred: np.ndarray, lo: float = 0.0, hi: float = 25_000.0) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    return np.clip(pred, lo, hi)
