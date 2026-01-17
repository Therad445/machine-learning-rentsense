from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance


def permutation_importance_topk(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    out_csv: Path,
    out_png: Path,
    n_sample: int,
    n_repeats: int,
    seed: int = 42,
    k: int = 20,
) -> pd.DataFrame:
    n_sample = min(n_sample, len(X_test))
    X_s = X_test.sample(n=n_sample, random_state=seed)
    y_s = y_test.loc[X_s.index]

    r = permutation_importance(
        model,
        X_s,
        y_s,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    imp = pd.Series(r.importances_mean, index=X_s.columns).sort_values(ascending=False)
    imp_tbl = imp.head(k).to_frame("perm_importance_mean")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    imp_tbl.to_csv(out_csv, index=True)

    plt.figure()
    plt.bar(imp_tbl.index.astype(str), imp_tbl["perm_importance_mean"].values)
    plt.title("Top permutation importance (MAE)")
    plt.xticks(rotation=75, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

    return imp_tbl
