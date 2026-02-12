# src/feature_engineering.py
# Build supervised features from weekly demand (lags, rolling stats, calendar features)
#
# Run (from project root):
#   python src/feature_engineering.py --horizon 13
#
# Defaults:
#   input : data/processed/weekly_demand.parquet
#   output: data/processed/features.parquet
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate lag/rolling features from weekly demand panel")
    p.add_argument("--in_path", type=str, default=str(project_root() / "data" / "processed" / "weekly_demand.parquet"))
    p.add_argument("--out_path", type=str, default=str(project_root() / "data" / "processed" / "features.parquet"))
    p.add_argument("--week_freq", type=str, default="W-SUN")
    p.add_argument("--horizon", type=int, default=13)

    p.add_argument("--min_weeks", type=int, default=80)
    p.add_argument("--lags", type=str, default="1,2,4,8,13,26,52")
    p.add_argument("--roll_windows", type=str, default="4,8,13,26")
    return p.parse_args()


def _parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def make_complete_weekly_panel(weekly: pd.DataFrame, week_freq: str) -> pd.DataFrame:
    weekly = weekly.copy()
    weekly["week"] = pd.to_datetime(weekly["week"])
    weekly = weekly.sort_values(["product_card_id", "week"])

    out = []
    for pid, g in weekly.groupby("product_card_id", sort=False):
        start = g["week"].min()
        end = g["week"].max()
        idx = pd.date_range(start, end, freq=week_freq)

        s = (
            g.set_index("week")["order_item_quantity"]
            .reindex(idx)
            .fillna(0.0)
            .rename("demand")
            .to_frame()
        )
        s["product_card_id"] = int(pid)
        s["week"] = s.index
        out.append(s.reset_index(drop=True))

    panel = pd.concat(out, ignore_index=True)
    return panel[["product_card_id", "week", "demand"]]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["week"] = pd.to_datetime(df["week"])
    df["weekofyear"] = df["week"].dt.isocalendar().week.astype(int)
    df["year"] = df["week"].dt.year.astype(int)

    w = df["weekofyear"].astype(float)
    df["woy_sin"] = np.sin(2 * np.pi * w / 52.0)
    df["woy_cos"] = np.cos(2 * np.pi * w / 52.0)
    df["month"] = df["week"].dt.month.astype(int)
    return df


def add_lag_and_rolling_features(df: pd.DataFrame, lags: List[int], roll_windows: List[int]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["product_card_id", "week"])
    grp = df.groupby("product_card_id", sort=False)

    for lag in lags:
        df[f"lag_{lag}"] = grp["demand"].shift(lag)

    for win in roll_windows:
        r = grp["demand"].shift(1).rolling(win)
        df[f"roll_mean_{win}"] = r.mean()
        df[f"roll_std_{win}"] = r.std()
        df[f"roll_min_{win}"] = r.min()
        df[f"roll_max_{win}"] = r.max()

    df["zero_rate_13"] = grp["demand"].shift(1).rolling(13).apply(
        lambda x: float(np.mean(np.asarray(x) == 0.0)),
        raw=False,
    )

    def last_nonzero(arr: np.ndarray) -> float:
        arr = np.asarray(arr)
        nz = arr[arr > 0]
        return float(nz[-1]) if nz.size else 0.0

    df["last_nonzero_13"] = grp["demand"].shift(1).rolling(13).apply(last_nonzero, raw=True)
    return df


def add_target(df: pd.DataFrame, H: int) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["product_card_id", "week"])
    df["y"] = df.groupby("product_card_id", sort=False)["demand"].shift(-int(H))
    return df


def filter_products_by_history(df: pd.DataFrame, min_weeks: int) -> pd.DataFrame:
    counts = df.groupby("product_card_id")["week"].count()
    keep = counts[counts >= int(min_weeks)].index
    return df[df["product_card_id"].isin(keep)].copy()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}. Run data_prep.py first.")

    weekly = pd.read_parquet(in_path)
    required = {"product_card_id", "week", "order_item_quantity"}
    missing = required - set(weekly.columns)
    if missing:
        raise KeyError(f"weekly_demand.parquet missing columns: {missing}")

    panel = make_complete_weekly_panel(weekly, week_freq=args.week_freq)
    panel = filter_products_by_history(panel, min_weeks=args.min_weeks)
    panel = add_calendar_features(panel)

    lags = _parse_int_list(args.lags)
    wins = _parse_int_list(args.roll_windows)
    panel = add_lag_and_rolling_features(panel, lags=lags, roll_windows=wins)
    panel = add_target(panel, H=args.horizon)

    out = panel.dropna().copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    print("Saved features to:", out_path.resolve())
    print("Shape:", out.shape)
    print("Products:", out["product_card_id"].nunique())
    print("Date range:", out["week"].min(), "→", out["week"].max())
    print("Target horizon H =", int(args.horizon))


if __name__ == "__main__":
    main()
