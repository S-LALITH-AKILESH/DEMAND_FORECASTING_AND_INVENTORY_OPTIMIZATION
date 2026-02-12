# src/dlinear_predict.py
# Predict + plot D-Linear forecasts for selected product IDs OR all products
#
# Examples:
#   python src/dlinear_predict.py --data data/processed/weekly_demand.parquet --ckpt models/dlinear_model.pt --pid 191
#   python src/dlinear_predict.py --pid 191 --pid 957 --forecast_weeks 52 --use_rolling
#   python src/dlinear_predict.py --all_products --forecast_weeks 52 --use_rolling
#
# Requires:
#   pip install torch pandas pyarrow numpy matplotlib

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt


WEEK_FREQ_DEFAULT = "W-SUN"


# -----------------------
# D-Linear model (must match training)
# -----------------------
class DLinear(nn.Module):
    def __init__(self, L: int, H: int, trend_k: int = 7):
        super().__init__()
        self.L = L
        self.H = H
        self.trend_k = max(1, int(trend_k))
        self.linear_trend = nn.Linear(L, H, bias=True)
        self.linear_resid = nn.Linear(L, H, bias=True)

    def moving_average(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        k = self.trend_k
        if k == 1:
            return x
        pad = k // 2
        x1 = x.unsqueeze(1)
        xpad = torch.nn.functional.pad(x1, (pad, pad), mode="reflect")
        w = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / float(k)
        trend = torch.nn.functional.conv1d(xpad, w)
        return trend.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.moving_average(x)
        resid = x - trend
        return self.linear_trend(trend) + self.linear_resid(resid)


# -----------------------
# Data helpers
# -----------------------
def load_weekly(data_path: Path) -> pd.DataFrame:
    if data_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    # expected columns: product_card_id, week, order_item_quantity
    needed = {"product_card_id", "week", "order_item_quantity"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Input weekly file is missing columns: {missing}")

    df["week"] = pd.to_datetime(df["week"])
    return df


def build_product_series(
    weekly: pd.DataFrame,
    pid: int,
    week_freq: str,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Return full weekly series (with missing weeks filled as 0) and its index."""
    g = weekly[weekly["product_card_id"] == pid].copy()
    if g.empty:
        raise ValueError(f"No rows found for product_card_id={pid}")

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
    """
    Trim trailing padded zeros so that:
    - plot ends at last real non-zero demand (no fake dip)
    - model input doesn't include padded zeros (prevents forecast collapse)
    """
    nz = np.where(series > 0)[0]
    if len(nz) == 0:
        return series, idx
    last_nz = int(nz[-1])
    return series[: last_nz + 1], idx[: last_nz + 1]


# -----------------------
# Forecast helpers
# -----------------------
def direct_forecast(
    model: nn.Module,
    hist_input: np.ndarray,
    mu: float,
    sd: float,
    device: torch.device,
) -> np.ndarray:
    """One-shot forecast using model output horizon H."""
    x = (hist_input - mu) / sd
    xb = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_scaled = model(xb).cpu().numpy().ravel()
    pred = pred_scaled * sd + mu
    pred = np.maximum(0.0, pred)
    return pred.astype("float32")


def rolling_forecast_1step(
    model: nn.Module,
    hist_input: np.ndarray,
    mu: float,
    sd: float,
    steps: int,
    device: torch.device,
) -> np.ndarray:
    """
    Produce 'steps' weeks using rolling 1-step-ahead prediction.
    Uses the first element of the H-step output each time.
    """
    buf = hist_input.astype("float32").copy()
    out: List[float] = []

    for _ in range(steps):
        x = (buf - mu) / sd
        xb = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_scaled = model(xb).cpu().numpy().ravel()

        y1 = float(pred_scaled[0] * sd + mu)
        y1 = max(0.0, y1)
        out.append(y1)

        buf = np.roll(buf, -1)
        buf[-1] = y1

    return np.array(out, dtype="float32")


# -----------------------
# Plotting
# -----------------------
def plot_forecast(
    pid: int,
    hist_series: np.ndarray,
    hist_idx: pd.DatetimeIndex,
    forecast: np.ndarray,
    week_freq: str,
    out_png: Path,
    plot_weeks: int,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plot_weeks = int(min(plot_weeks, len(hist_series)))
    y_hist = hist_series[-plot_weeks:]
    x_hist = hist_idx[-plot_weeks:]

    roll4 = pd.Series(y_hist).rolling(4, min_periods=1).mean().values

    last = x_hist[-1]
    f_idx = pd.date_range(last + pd.Timedelta(days=7), periods=len(forecast), freq=week_freq)

    resid = y_hist - roll4
    sigma = float(np.std(resid)) if len(resid) > 5 else 1.0
    lo = np.maximum(0.0, forecast - 1.96 * sigma)
    hi = forecast + 1.96 * sigma

    plt.figure(figsize=(12, 5))
    plt.axvspan(f_idx[0], f_idx[-1], alpha=0.06)

    plt.step(x_hist, y_hist, where="post", label="historical (weekly)")
    plt.plot(x_hist, roll4, label="rolling 4W")
    plt.fill_between(f_idx, lo, hi, alpha=0.2, label="95% band")
    plt.plot(f_idx, forecast, label=f"forecast ({len(forecast)}w)")

    plt.title(f"Intermittent Demand Forecast — product_card_id={pid}")
    plt.xlabel("week")
    plt.ylabel("units")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def save_forecast_csv(
    pid: int,
    last_hist_date: pd.Timestamp,
    forecast: np.ndarray,
    week_freq: str,
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f_idx = pd.date_range(last_hist_date + pd.Timedelta(days=7), periods=len(forecast), freq=week_freq)

    df_out = pd.DataFrame(
        {
            "product_card_id": pid,
            "week": f_idx,
            "forecast_units": forecast,
        }
    )
    df_out.to_csv(out_csv, index=False)


# -----------------------
# Main
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict and plot D-Linear forecasts")

    p.add_argument("--data", type=str, default="data/processed/weekly_demand.parquet")
    p.add_argument("--ckpt", type=str, default="models/dlinear_model.pt")

    p.add_argument("--out_plots", type=str, default="outputs/plots")
    p.add_argument("--out_forecasts", type=str, default="outputs/forecasts")

    # product selection
    p.add_argument("--pid", type=int, action="append", help="product_card_id (repeatable)")
    p.add_argument("--all_products", action="store_true", help="Forecast all product_card_id values")

    # display + forecast params
    p.add_argument("--plot_weeks", type=int, default=156, help="weeks of history to plot (default 3 years)")
    p.add_argument("--forecast_weeks", type=int, default=52, help="weeks to forecast")
    p.add_argument("--week_freq", type=str, default=WEEK_FREQ_DEFAULT)

    p.add_argument("--use_rolling", action="store_true",
                   help="Use rolling 1-step forecast to reach forecast_weeks even if checkpoint H is smaller.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    ckpt_path = Path(args.ckpt)

    out_plots = Path(args.out_plots)
    out_forecasts = Path(args.out_forecasts)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data: {data_path}. Run data_prep.py first.")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Train with dlinear.py first.")

    weekly = load_weekly(data_path)

    # Determine product IDs to run
    if args.all_products:
        product_ids = sorted(weekly["product_card_id"].unique().tolist())
    else:
        if not args.pid:
            raise ValueError("Provide --pid <id> (repeatable) OR use --all_products.")
        product_ids = args.pid

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")

    L = int(ckpt["L"])
    H = int(ckpt["H"])
    trend_k = int(ckpt.get("trend_k", 7))
    scalers: Dict[int, Tuple[float, float]] = ckpt.get("scalers", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DLinear(L=L, H=H, trend_k=trend_k).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print("Loaded model:", ckpt_path)
    print(f"Config: L={L}, H={H}, trend_k={trend_k}, device={device}")
    print(f"Week freq: {args.week_freq}")
    print(f"Forecast weeks: {args.forecast_weeks} (use_rolling={args.use_rolling})")
    print(f"Products to process: {len(product_ids)}")

    ok, skipped, failed = 0, 0, 0

    for pid in product_ids:
        try:
            series, idx = build_product_series(weekly, pid, args.week_freq)

            # Trim trailing zeros (prevents dips/jumps + improves continuity)
            series_trim, idx_trim = trim_trailing_zeros(series, idx)

            if len(series_trim) < L:
                skipped += 1
                continue

            hist_input = series_trim[-L:].astype("float32")

            # per-product scaler (fallback to trimmed stats if missing)
            if int(pid) in scalers:
                mu, sd = scalers[int(pid)]
            else:
                mu = float(np.mean(series_trim))
                sd = float(np.std(series_trim))
                if not np.isfinite(sd) or sd < 1e-6:
                    sd = 1.0

            # Forecast generation
            if args.use_rolling or args.forecast_weeks != H:
                forecast = rolling_forecast_1step(
                    model=model,
                    hist_input=hist_input,
                    mu=mu,
                    sd=sd,
                    steps=args.forecast_weeks,
                    device=device,
                )
            else:
                forecast = direct_forecast(
                    model=model,
                    hist_input=hist_input,
                    mu=mu,
                    sd=sd,
                    device=device,
                )

            # Save plot
            out_png = out_plots / f"product_{pid}_dlinear_forecast.png"
            plot_forecast(
                pid=int(pid),
                hist_series=series_trim,
                hist_idx=idx_trim,
                forecast=forecast,
                week_freq=args.week_freq,
                out_png=out_png,
                plot_weeks=args.plot_weeks,
            )

            # Save forecast CSV
            out_csv = out_forecasts / f"product_{pid}_forecast.csv"
            save_forecast_csv(
                pid=int(pid),
                last_hist_date=idx_trim[-1],
                forecast=forecast,
                week_freq=args.week_freq,
                out_csv=out_csv,
            )

            ok += 1

        except Exception as e:
            failed += 1
            print(f"[FAIL] pid={pid}: {e}")

    print("\nDone.")
    print(f"OK={ok}  SKIPPED={skipped}  FAILED={failed}")
    print(f"Plots: {out_plots.resolve()}")
    print(f"Forecast CSVs: {out_forecasts.resolve()}")


if __name__ == "__main__":
    main()
