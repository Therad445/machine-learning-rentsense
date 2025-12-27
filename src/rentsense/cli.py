from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich import print as rprint

from .artifacts import save_df, save_metadata, save_model
from .config import TrainConfig
from .eda import (
    plot_pred_vs_true,
    plot_scatter_area_vs_target,
    plot_target_hist,
    plot_top_locations,
    save_missing_top30,
)
from .importance import permutation_importance_topk
from .io import read_csv_robust
from .metrics import evaluate_regression, segment_metrics
from .models import build_histgbr_model, build_ridge_model, safe_clip_preds
from .preprocess import prepare_xy
from .scoring import score_dataframe, top_examples
from .split import split_data

app = typer.Typer(add_completion=False)


@app.command()
def train(
    data: Path = typer.Option(..., help="Path to immo_data.csv"),
    artifacts: Path = typer.Option(Path("artifacts"), help="Artifacts output directory"),
    split: str = typer.Option("random", help="random|geo_holdout|time"),
    fast: bool = typer.Option(True, help="Fast mode: limit rows and faster importance"),
    fast_nrows: int = typer.Option(80_000, help="nrows in fast mode"),
    fast_plot_n: int = typer.Option(20_000, help="n points in plots (sample)"),
    high_card_th: int = typer.Option(300, help="Drop categorical columns with > this unique values"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    cfg = TrainConfig(
        seed=seed,
        fast_mode=fast,
        fast_nrows=fast_nrows,
        fast_plot_n=fast_plot_n,
        high_card_th=high_card_th,
        split_mode=split,  # type: ignore[arg-type]
    )

    artifacts.mkdir(parents=True, exist_ok=True)

    df = read_csv_robust(data, nrows=(cfg.fast_nrows if cfg.fast_mode else None))
    rprint(f"[bold]Loaded:[/bold] {df.shape} from {data}")

    # profile: missing
    na_tbl = save_missing_top30(df, artifacts / "missing_top30.csv")
    rprint(f"[green]Saved[/green] missing_top30.csv (rows: {len(na_tbl)})")

    # EDA (sample for speed)
    plot_df = df.sample(min(cfg.fast_plot_n, len(df)), random_state=cfg.seed) if cfg.fast_mode else df
    if cfg.target in plot_df.columns:
        plot_target_hist(plot_df, cfg.target, artifacts / "eda_target_hist.png")
        plot_scatter_area_vs_target(
            plot_df, cfg.living_space_col, cfg.target, artifacts / "eda_scatter_area_vs_target.png"
        )
        plot_top_locations(plot_df, "geo_plz", artifacts / "eda_top_locations.png")

    prepared = prepare_xy(df, cfg)
    X, y = prepared.X, prepared.y

    # split
    X_train, X_test, y_train, y_test = split_data(X, y, df, cfg)
    rprint(f"[bold]Split:[/bold] {cfg.split_mode} | train {X_train.shape} | test {X_test.shape}")

    # train-based target clipping (quantile)
    q_low, q_high = y_train.quantile([cfg.clip_q_low, cfg.clip_q_high])
    tr_mask = y_train.between(q_low, q_high)
    te_mask = y_test.between(q_low, q_high)
    X_train, y_train = X_train.loc[tr_mask], y_train.loc[tr_mask]
    X_test, y_test = X_test.loc[te_mask], y_test.loc[te_mask]
    rprint(f"[bold]Clipping:[/bold] q{cfg.clip_q_low}/{cfg.clip_q_high} = {float(q_low):.2f}/{float(q_high):.2f}")

    ridge = build_ridge_model(prepared.numeric_cols, prepared.categorical_cols)
    hgb = build_histgbr_model(
        prepared.numeric_cols, prepared.categorical_cols, seed=cfg.seed, fast_mode=cfg.fast_mode
    )

    fitted = {}
    results = []

    for name, est in [("Ridge (baseline)", ridge), ("HistGBR (strong)", hgb)]:
        est.fit(X_train, y_train)
        pred = safe_clip_preds(est.predict(X_test))
        results.append(evaluate_regression(name, y_test, pred))
        fitted[name] = est
        rprint(f"[cyan]{name}[/cyan] done")

    res_df = pd.DataFrame(results).sort_values("MAE")
    save_df(res_df, artifacts / "metrics.csv")
    rprint("[green]Saved[/green] metrics.csv")

    best_name = str(res_df.iloc[0]["model"])
    best_model = fitted[best_name]
    save_model(best_model, artifacts / "best_model.joblib")
    rprint(f"[bold green]Best[/bold green]: {best_name} → saved best_model.joblib")

    # segment metrics for best
    best_pred = safe_clip_preds(best_model.predict(X_test))
    seg_tbl = segment_metrics(y_test, best_pred)
    (artifacts / "metrics_by_segment.csv").write_text(seg_tbl.to_csv(index=True), encoding="utf-8")
    rprint("[green]Saved[/green] metrics_by_segment.csv")

    # importance
    n_sample = cfg.perm_n_sample_fast if cfg.fast_mode else cfg.perm_n_sample_full
    n_repeats = cfg.perm_n_repeats_fast if cfg.fast_mode else cfg.perm_n_repeats_full
    permutation_importance_topk(
        best_model,
        X_test,
        y_test,
        out_csv=artifacts / "feature_importance_top20.csv",
        out_png=artifacts / "feature_importance_top20.png",
        n_sample=n_sample,
        n_repeats=n_repeats,
        seed=cfg.seed,
        k=20,
    )
    rprint("[green]Saved[/green] feature_importance_top20.*")

    # scoring / examples (test set)
    scored = score_dataframe(best_model, X_test, y_true=y_test, th_pct=cfg.th_pct, th_eur=cfg.th_eur)
    over, under = top_examples(scored, n=10)
    over.to_csv(artifacts / "examples_overpriced_top10.csv", index=False)
    under.to_csv(artifacts / "examples_underpriced_top10.csv", index=False)
    rprint("[green]Saved[/green] examples_overpriced_top10.csv & examples_underpriced_top10.csv")

    plot_pred_vs_true(scored["y_true"].to_numpy(), scored["y_pred"].to_numpy(), artifacts / "pred_vs_true.png")
    rprint("[green]Saved[/green] pred_vs_true.png")

    save_metadata(
        artifacts / "run_metadata.json",
        cfg,
        extra={
            "dropped": prepared.dropped,
            "columns": {"numeric": prepared.numeric_cols, "categorical": prepared.categorical_cols},
            "best_model": best_name,
            "metrics": res_df.to_dict(orient="records"),
            "flags_count": scored["flag"].value_counts().to_dict(),
        },
    )
    rprint("[green]Saved[/green] run_metadata.json")


@app.command()
def score(
    model: Path = typer.Option(..., help="Path to trained model .joblib"),
    input: Path = typer.Option(..., help="CSV to score (must contain features, optionally target)"),
    out: Path = typer.Option(Path("artifacts/scored.csv"), help="Output CSV"),
    target: str = typer.Option("baseRent", help="Target col name (optional in input)"),
    th_pct: float = typer.Option(0.20, help="Pct threshold for over/under"),
    th_eur: float = typer.Option(150.0, help="EUR threshold for over/under"),
    nrows: int = typer.Option(0, help="If >0, read only first n rows"),
) -> None:
    import joblib

    df = read_csv_robust(input, nrows=(nrows if nrows > 0 else None))

    y_true = None
    if target in df.columns:
        y_true = pd.to_numeric(df[target], errors="coerce")

    cfg = TrainConfig(target=target, th_pct=th_pct, th_eur=th_eur)
    prepared = prepare_xy(df, cfg)
    X = prepared.X

    est = joblib.load(model)
    scored = score_dataframe(est, X, y_true=y_true, th_pct=th_pct, th_eur=th_eur)

    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    rprint(f"[green]Saved[/green] {out}")
