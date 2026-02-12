# inventory_features.py
# Build inventory decision features from:
#   - weekly demand history (data/processed/weekly_demand.parquet or .csv)
#   - D-Linear forecast CSVs (outputs/forecasts/product_<pid>_forecast.csv)
#
# Run examples:
#   python inventory_features.py
#   python inventory_features.py --weekly data/processed/weekly_demand.csv --forecasts_dir outputs/forecasts --out outputs/inventory_features.csv
#   python inventory_features.py --forecast_weeks 52 --lead_time_weeks 2
#
# Requires:
#   pip install pandas numpy
#   (pyarrow only needed if you use parquet)

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


WEEK_FREQ_DEFAULT = "W-SUN"


# -----------------------
# Helpers
# -----------------------
def load_weekly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Weekly file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    needed = {"product_card_id", "week", "order_item_quantity"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Weekly file is missing columns: {missing}")

    df["week"] = pd.to_datetime(df["week"])
    df["product_card_id"] = df["product_card_id"].astype(int)
    df["order_item_quantity"] = pd.to_numeric(df["order_item_quantity"], errors="coerce").fillna(0.0)
    return df


def read_forecast_file(fp: Path) -> Tuple[int, pd.DataFrame]:
    """
    Expected forecast CSV columns:
      product_card_id, week, forecast_units
    """
    df = pd.read_csv(fp)
    needed = {"product_card_id", "week", "forecast_units"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Forecast file {fp.name} missing columns: {missing}")

    df["week"] = pd.to_datetime(df["week"])
    df["product_card_id"] = df["product_card_id"].astype(int)
    df["forecast_units"] = pd.to_numeric(df["forecast_units"], errors="coerce").fillna(0.0)

    pid = int(df["product_card_id"].iloc[0])
    return pid, df


def linear_slope(y: np.ndarray) -> float:
    """Slope of y vs time index (0..n-1)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    # least squares slope
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom <= 1e-12:
        return 0.0
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    return float(slope)


def last_nonzero(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    nz = arr[arr > 0]
    return float(nz[-1]) if len(nz) else 0.0


def zero_rate(arr: np.ndarray) -> float:
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr == 0.0))


def build_full_weekly_series(
    weekly: pd.DataFrame,
    pid: int,
    week_freq: str
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    g = weekly[weekly["product_card_id"] == pid].copy()
    if g.empty:
        return np.array([], dtype="float32"), pd.DatetimeIndex([])

    g = g.sort_values("week")
    start = g["week"].min()
    end = g["week"].max()
    idx = pd.date_range(start, end, freq=week_freq)

    s = (
        g.set_index("week")["order_item_quantity"]
        .reindex(idx)
        .fillna(0.0)
        .astype("float32")
        .values
    )
    return s, idx


def trim_trailing_zeros(series: np.ndarray, idx: pd.DatetimeIndex) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    nz = np.where(series > 0)[0]
    if len(nz) == 0:
        return series, idx
    last_nz = int(nz[-1])
    return series[: last_nz + 1], idx[: last_nz + 1]


# -----------------------
# Feature construction
# -----------------------
def history_features(series: np.ndarray) -> Dict[str, float]:
    """
    Compute history-based features from a weekly demand series
    (assumed to already be trimmed to last real demand point).
    """
    s = np.asarray(series, dtype=float)

    def window(arr: np.ndarray, k: int) -> np.ndarray:
        if len(arr) == 0:
            return arr
        return arr[-k:] if len(arr) >= k else arr

    last4 = window(s, 4)
    last13 = window(s, 13)
    last26 = window(s, 26)
    last52 = window(s, 52)

    feats = {
        # levels
        "hist_last": float(s[-1]) if len(s) else 0.0,
        "hist_mean_4": float(np.mean(last4)) if len(last4) else 0.0,
        "hist_mean_13": float(np.mean(last13)) if len(last13) else 0.0,
        "hist_mean_26": float(np.mean(last26)) if len(last26) else 0.0,
        "hist_mean_52": float(np.mean(last52)) if len(last52) else 0.0,

        # volatility
        "hist_std_13": float(np.std(last13)) if len(last13) else 0.0,
        "hist_std_26": float(np.std(last26)) if len(last26) else 0.0,

        # intermittency
        "hist_zero_rate_13": zero_rate(last13),
        "hist_zero_rate_52": zero_rate(last52),

        # last non-zero
        "hist_last_nonzero_13": last_nonzero(last13),
        "hist_last_nonzero_52": last_nonzero(last52),

        # simple recent trend proxy
        "hist_slope_13": linear_slope(last13) if len(last13) else 0.0,
    }
    return feats


def forecast_features(forecast: np.ndarray) -> Dict[str, float]:
    f = np.asarray(forecast, dtype=float)
    if len(f) == 0:
        return {
            "fc_mean": 0.0,
            "fc_std": 0.0,
            "fc_min": 0.0,
            "fc_max": 0.0,
            "fc_sum": 0.0,
            "fc_slope": 0.0,
        }

    return {
        "fc_mean": float(np.mean(f)),
        "fc_std": float(np.std(f)),
        "fc_min": float(np.min(f)),
        "fc_max": float(np.max(f)),
        "fc_sum": float(np.sum(f)),
        "fc_slope": linear_slope(f),
    }


def build_inventory_features(
    weekly: pd.DataFrame,
    forecasts_dir: Path,
    week_freq: str,
    forecast_weeks: int,
    lead_time_weeks: int,
    mode: str = "future",
    snapshots_per_product: int = 20,
    anchor_stride: int = 1,
    history_window: int = 52,
    forecast_source: str = "auto",
) -> pd.DataFrame:
    """
    Build an inventory-feature table.

    Modes
    -----
    future (default):
        One snapshot per product anchored at the last historical week, using D-Linear forecast files:
        outputs/forecasts/product_<pid>_forecast.csv

    rolling:
        Multiple historical snapshots per product anchored at recent weeks in the demand history.
        Target expected_demand_lead_time is computed from realized demand AFTER the anchor (supervised learning).

        Forecast-like features are computed from:
          - actual future demand (proxy forecast) [default in rolling]
          - dlinear (only if you have backtest forecasts per anchor saved; otherwise falls back to actual)

    Why rolling mode?
        It increases training rows for XGBoost (e.g., 54 products × 20 anchors ≈ 1080 rows),
        reducing overfitting and making cross-validation more reliable.
    """
    mode = str(mode).lower().strip()
    if mode not in {"future", "rolling"}:
        raise ValueError("mode must be 'future' or 'rolling'")

    forecasts_dir = Path(forecasts_dir)
    rows: List[Dict[str, object]] = []

    if mode == "future":
        if not forecasts_dir.exists():
            raise FileNotFoundError(f"Forecasts directory not found: {forecasts_dir}")

        files = sorted(forecasts_dir.glob("product_*_forecast.csv"))
        if not files:
            raise FileNotFoundError(
                f"No forecast files found in {forecasts_dir}. Expected files like product_191_forecast.csv"
            )

        for fp in files:
            try:
                pid, fdf = read_forecast_file(fp)
                f = fdf["forecast_units"].values.astype("float32")[:forecast_weeks]

                s_full, idx_full = build_full_weekly_series(weekly, pid, week_freq=week_freq)
                if len(s_full) == 0:
                    continue

                s_trim, idx_trim = trim_trailing_zeros(s_full, idx_full)
                if len(s_trim) == 0:
                    continue

                anchor_week = idx_trim[-1]
                woy = int(anchor_week.isocalendar().week)
                month = int(anchor_week.month)
                year = int(anchor_week.year)

                h_feats = history_features(s_trim)
                f_feats = forecast_features(f)

                expected_lt_demand = float(np.sum(f[:lead_time_weeks])) if len(f) >= 1 else 0.0

                row = {
                    "product_card_id": pid,
                    "last_hist_week": anchor_week,
                    "anchor_year": year,
                    "anchor_month": month,
                    "anchor_woy": woy,
                    "lead_time_weeks": int(lead_time_weeks),
                    "expected_demand_lead_time": expected_lt_demand,
                    "snapshot_mode": "future",
                }
                row.update(h_feats)
                row.update(f_feats)
                rows.append(row)

            except Exception as e:
                print(f"[WARN] Skipping {fp.name}: {e}")

    else:
        # rolling mode
        if forecast_source == "auto":
            forecast_source_eff = "actual"
        else:
            forecast_source_eff = str(forecast_source).lower().strip()

        stride = max(1, int(anchor_stride))
        k = max(1, int(snapshots_per_product))
        hw = max(1, int(history_window))

        pids = sorted(weekly["product_card_id"].dropna().unique().tolist())
        for pid in pids:
            try:
                pid = int(pid)
                s_full, idx_full = build_full_weekly_series(weekly, pid, week_freq=week_freq)
                if len(s_full) == 0:
                    continue
                s_trim, idx_trim = trim_trailing_zeros(s_full, idx_full)
                n = len(s_trim)

                # Need enough history and enough future to compute lead-time target at anchors
                if n < (hw + lead_time_weeks + 1):
                    continue

                # latest anchor index that still has lead_time weeks of realized future demand available
                last_anchor = n - lead_time_weeks - 1

                anchors: List[int] = []
                a = last_anchor
                while a >= hw and len(anchors) < k:
                    anchors.append(a)
                    a -= stride
                anchors = sorted(anchors)

                for a in anchors:
                    anchor_week = idx_trim[a]
                    woy = int(anchor_week.isocalendar().week)
                    month = int(anchor_week.month)
                    year = int(anchor_week.year)

                    hist = s_trim[: a + 1].astype("float32")
                    future_real = s_trim[a + 1 : a + 1 + forecast_weeks].astype("float32")

                    expected_lt_demand = float(np.sum(s_trim[a + 1 : a + 1 + lead_time_weeks]))

                    # forecast-like array for forecast_features
                    if forecast_source_eff == "actual":
                        f = future_real
                    elif forecast_source_eff == "dlinear":
                        # Historical backtest forecasts are not available by default.
                        # Fall back to actual future to avoid empty features.
                        f = future_real
                    else:
                        f = future_real

                    if len(f) < forecast_weeks:
                        f = np.pad(f, (0, forecast_weeks - len(f)), mode="constant", constant_values=0.0)

                    h_feats = history_features(hist)
                    f_feats = forecast_features(f[:forecast_weeks])

                    row = {
                        "product_card_id": pid,
                        "last_hist_week": anchor_week,
                        "anchor_year": year,
                        "anchor_month": month,
                        "anchor_woy": woy,
                        "lead_time_weeks": int(lead_time_weeks),
                        "expected_demand_lead_time": expected_lt_demand,
                        "snapshot_mode": "rolling",
                        "anchor_index": int(a),
                    }
                    row.update(h_feats)
                    row.update(f_feats)
                    rows.append(row)

            except Exception as e:
                print(f"[WARN] rolling: skipping pid={pid}: {e}")

    if not rows:
        raise ValueError("No feature rows produced. Check inputs and mode settings.")

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build inventory decision features from D-Linear forecasts")

    p.add_argument("--weekly", type=str, default="data/processed/weekly_demand.parquet",
                   help="Weekly demand file (parquet or csv) with columns: product_card_id, week, order_item_quantity")

    p.add_argument("--forecasts_dir", type=str, default="outputs/forecasts",
                   help="Directory containing product_<pid>_forecast.csv files")

    p.add_argument("--out", type=str, default="outputs/inventory_features.csv",
                   help="Output file path (.csv or .parquet)")

    p.add_argument("--week_freq", type=str, default=WEEK_FREQ_DEFAULT)
    p.add_argument("--forecast_weeks", type=int, default=52)
    p.add_argument("--lead_time_weeks", type=int, default=2)

    # Snapshot generation
    p.add_argument(
        "--mode",
        type=str,
        default="future",
        choices=["future", "rolling"],
        help="future: one snapshot per product using future forecast; rolling: multiple historical snapshots per product",
    )
    p.add_argument(
        "--snapshots_per_product",
        type=int,
        default=20,
        help="(rolling) number of recent anchor weeks per product to generate",
    )
    p.add_argument(
        "--anchor_stride",
        type=int,
        default=1,
        help="(rolling) step size between anchors when moving backward in time",
    )
    p.add_argument(
        "--history_window",
        type=int,
        default=52,
        help="(rolling) minimum history length before anchors start",
    )
    p.add_argument(
        "--forecast_source",
        type=str,
        default="auto",
        choices=["auto", "actual", "dlinear"],
        help="(rolling) forecast features source: auto->actual; dlinear requires historical backtest forecasts",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    weekly_path = Path(args.weekly)
    forecasts_dir = Path(args.forecasts_dir)
    out_path = Path(args.out)

    weekly = load_weekly(weekly_path)

    feat_df = build_inventory_features(
        weekly=weekly,
        forecasts_dir=forecasts_dir,
        week_freq=args.week_freq,
        forecast_weeks=args.forecast_weeks,
        lead_time_weeks=args.lead_time_weeks,
        mode=args.mode,
        snapshots_per_product=args.snapshots_per_product,
        anchor_stride=args.anchor_stride,
        history_window=args.history_window,
        forecast_source=args.forecast_source,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".parquet":
        feat_df.to_parquet(out_path, index=False)
    else:
        feat_df.to_csv(out_path, index=False)

    print("Saved inventory features:", out_path)
    print("Shape:", feat_df.shape)
    print("Products:", feat_df["product_card_id"].nunique())
    print("\nColumns:")
    print(feat_df.columns.tolist())


if __name__ == "__main__":
    main()
