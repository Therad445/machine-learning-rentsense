from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SplitMode = Literal["random", "geo_holdout", "time"]


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42

    target: str = "baseRent"

    # fast-mode knobs
    fast_mode: bool = True
    fast_nrows: int = 80_000
    fast_plot_n: int = 20_000

    # feature selection / cleaning
    high_card_th: int = 300
    protect_cols: tuple[str, ...] = ("geo_plz",)

    # drop heavy text-like cols
    drop_heavy: tuple[str, ...] = (
        "description",
        "facilities",
        "street",
        "streetPlain",
        "houseNumber",
    )

    # filter weird outliers
    living_space_col: str = "livingSpace"
    living_space_min: float = 5.0
    living_space_max: float = 300.0

    # target clipping based on train quantiles
    clip_q_low: float = 0.01
    clip_q_high: float = 0.99

    # split
    split_mode: SplitMode = "random"
    test_size: float = 0.2

    # for "time" split: date parsing format (e.g. "May19")
    date_col: str = "date"
    date_format: str = "%b%y"

    # thresholds for overpriced/underpriced flags
    th_pct: float = 0.20
    th_eur: float = 150.0

    # permutation importance
    perm_n_sample_fast: int = 800
    perm_n_sample_full: int = 4000
    perm_n_repeats_fast: int = 3
    perm_n_repeats_full: int = 10
