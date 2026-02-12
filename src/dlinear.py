
# src/dlinear.py
# Train a global D-Linear model on weekly demand WITH early stopping (patience)
#
# Run example:
#   python src/dlinear.py --data data/processed/weekly_demand.parquet --out models/dlinear_model.pt --L 52 --H 13 --epochs 200 --patience 20
#
# Requires:
#   pip install torch pandas pyarrow numpy

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


WEEK_FREQ_DEFAULT = "W-SUN"


# -----------------------
# Utilities
# -----------------------
def seed_all(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_weekly_panel(
    weekly: pd.DataFrame,
    week_freq: str = WEEK_FREQ_DEFAULT,
) -> pd.DataFrame:
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
            .astype("float32")
        )
        dfp = pd.DataFrame(
            {
                "product_card_id": pid,
                "week": idx,
                "demand": s.values,
            }
        )
        out.append(dfp)

    panel = pd.concat(out, ignore_index=True)
    return panel


def per_product_scalers(panel: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    scalers: Dict[int, Tuple[float, float]] = {}
    for pid, g in panel.groupby("product_card_id"):
        x = g["demand"].values.astype("float32")
        mu = float(np.mean(x))
        sd = float(np.std(x))
        if not np.isfinite(sd) or sd < 1e-6:
            sd = 1.0
        scalers[int(pid)] = (mu, sd)
    return scalers


# -----------------------
# D-Linear model
# -----------------------
class DLinear(nn.Module):
    def __init__(self, L: int, H: int, trend_k: int = 7):
        super().__init__()
        self.L = L
        self.H = H
        self.trend_k = max(1, int(trend_k))

        self.linear_trend = nn.Linear(L, H, bias=True)
        self.linear_resid = nn.Linear(L, H, bias=True)

        nn.init.xavier_uniform_(self.linear_trend.weight)
        nn.init.zeros_(self.linear_trend.bias)
        nn.init.xavier_uniform_(self.linear_resid.weight)
        nn.init.zeros_(self.linear_resid.bias)

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
# Dataset
# -----------------------
@dataclass
class WindowSpec:
    L: int
    H: int


class GlobalWindowDataset(Dataset):
    def __init__(
        self,
        panel: pd.DataFrame,
        scalers: Dict[int, Tuple[float, float]],
        spec: WindowSpec,
        min_weeks: int = 80,
    ):
        super().__init__()
        self.panel = panel
        self.scalers = scalers
        self.spec = spec
        self.min_weeks = int(min_weeks)

        self.products: List[int] = []
        self.series: List[np.ndarray] = []
        self.index_map: List[Tuple[int, int]] = []

        for pid, g in panel.groupby("product_card_id", sort=False):
            pid = int(pid)
            x = g["demand"].values.astype("float32")
            if len(x) < max(self.min_weeks, spec.L + spec.H):
                continue
            self.products.append(pid)
            self.series.append(x)

        for si, x in enumerate(self.series):
            max_start = len(x) - (spec.L + spec.H) + 1
            for t in range(max_start):
                self.index_map.append((si, t))

        if len(self.index_map) == 0:
            raise ValueError("No training windows available. Reduce L/H or min_weeks, or check data.")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        si, t = self.index_map[idx]
        x = self.series[si]
        pid = self.products[si]
        mu, sd = self.scalers[pid]

        L, H = self.spec.L, self.spec.H
        x_in = x[t : t + L]
        y_out = x[t + L : t + L + H]

        x_in = (x_in - mu) / sd
        y_out = (y_out - mu) / sd

        return torch.from_numpy(x_in), torch.from_numpy(y_out)


# -----------------------
# Split
# -----------------------
def time_split_dataset(dataset: GlobalWindowDataset, val_ratio: float = 0.1):
    val_ratio = float(val_ratio)
    train_idx, val_idx = [], []
    by_series: Dict[int, List[int]] = {}

    for i, (si, _) in enumerate(dataset.index_map):
        by_series.setdefault(si, []).append(i)

    for idxs in by_series.values():
        n = len(idxs)
        n_val = max(1, int(round(n * val_ratio)))
        train_idx.extend(idxs[:-n_val])
        val_idx.extend(idxs[-n_val:])

    return (
        torch.utils.data.Subset(dataset, train_idx),
        torch.utils.data.Subset(dataset, val_idx),
    )


# -----------------------
# Training with patience
# -----------------------
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    patience_ctr = 0

    history = {"train_mae": [], "val_mae": []}

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_losses.append(loss_fn(pred, yb).item())

        train_mae = float(np.mean(train_losses))
        val_mae = float(np.mean(val_losses))
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        print(f"Epoch {ep:03d}/{epochs}  train_MAE={train_mae:.4f}  val_MAE={val_mae:.4f}")

        if val_mae < best_val:
            best_val = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {ep} (best val_MAE={best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


# -----------------------
# Main
# -----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train global D-Linear on weekly demand")
    p.add_argument("--data", type=str, default="data/processed/weekly_demand.parquet")
    p.add_argument("--out", type=str, default="models/dlinear_model.pt")
    p.add_argument("--week_freq", type=str, default=WEEK_FREQ_DEFAULT)

    p.add_argument("--L", type=int, default=52)
    p.add_argument("--H", type=int, default=13)
    p.add_argument("--trend_k", type=int, default=7)
    p.add_argument("--min_weeks", type=int, default=80)

    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.1)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    weekly = pd.read_parquet(args.data)
    panel = to_weekly_panel(weekly, week_freq=args.week_freq)
    scalers = per_product_scalers(panel)

    spec = WindowSpec(L=args.L, H=args.H)
    dataset = GlobalWindowDataset(panel, scalers, spec, min_weeks=args.min_weeks)
    train_ds, val_ds = time_split_dataset(dataset, val_ratio=args.val_ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(f"Training windows: {len(train_ds)}  Validation windows: {len(val_ds)}")
    print(f"L={args.L}, H={args.H}, trend_k={args.trend_k}, week_freq={args.week_freq}")

    model = DLinear(L=args.L, H=args.H, trend_k=args.trend_k).to(device)

    history = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    out = Path(args.out)
    ensure_dirs(out)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "L": args.L,
            "H": args.H,
            "trend_k": args.trend_k,
            "week_freq": args.week_freq,
            "scalers": scalers,
            "history": history,
        },
        out,
    )
    print("Saved model to:", out)


if __name__ == "__main__":
    main()
