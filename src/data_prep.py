# src/data_prep.py
# Prepare DataCoSupplyChainDataset.csv -> clean_orders_minimal.parquet + weekly_demand.parquet
#
# Run (from project root):
#   python src/data_prep.py
#   python src/data_prep.py --csv data/raw/DataCoSupplyChainDataset.csv
#
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def project_root() -> Path:
    # repo_root/src/data_prep.py -> repo_root
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare DataCo dataset into clean + weekly demand parquet files")
    p.add_argument(
        "--csv",
        type=str,
        default=str(project_root() / "data" / "raw" / "DataCoSupplyChainDataset.csv"),
        help="Path to DataCoSupplyChainDataset.csv",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=str(project_root() / "data" / "processed"),
        help="Output directory for parquet files",
    )
    p.add_argument("--week_freq", type=str, default="W-SUN", help="Weekly resample frequency (default week ending Sunday)")
    p.add_argument("--encoding", type=str, default="latin1", help="CSV encoding (DataCo often needs latin1)")
    return p.parse_args()


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first matching column name (case/space-insensitive)."""
    norm = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in norm:
            return norm[key]
    raise KeyError(f"None of the candidate columns were found: {candidates}")


def parse_mixed_dates(s: pd.Series) -> pd.Series:
    """
    Parse mixed date formats safely:
    - First try month-first (MM/DD/YYYY, MM-DD-YYYY)
    - Then fallback day-first for rows that failed
    """
    s = s.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})

    d1 = pd.to_datetime(s, errors="coerce", dayfirst=False)
    mask = d1.isna() & s.notna()
    if mask.any():
        d2 = pd.to_datetime(s[mask], errors="coerce", dayfirst=True)
        d1.loc[mask] = d2
    return d1


def main() -> None:
    args = parse_args()
    raw_csv = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"CSV not found at: {raw_csv}\n"
            "Place the file at data/raw/DataCoSupplyChainDataset.csv or pass --csv <path>."
        )

    df = pd.read_csv(raw_csv, encoding=args.encoding)
    print("Loaded:", df.shape)

    product_col = find_col(df, ["Product Card Id", "product_card_id"])
    qty_col = find_col(df, ["Order Item Quantity", "order_item_quantity"])
    order_date_col = find_col(
        df,
        [
            "order date (DateOrders)",
            "Order date (DateOrders)",
            "order_date (DateOrders)",
            "order_date_dateorders",
            "order_date",
        ],
    )

    print("Using columns:")
    print("  product_col   =", product_col)
    print("  qty_col       =", qty_col)
    print("  order_date_col=", order_date_col)

    df["order_date"] = parse_mixed_dates(df[order_date_col])

    before = len(df)
    df = df[df["order_date"].notna()].copy()
    after = len(df)
    print(f"Dropped invalid dates: {before - after} rows")

    df[qty_col] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)

    clean = df[[product_col, qty_col, "order_date"]].copy().rename(
        columns={product_col: "product_card_id", qty_col: "order_item_quantity"}
    )
    clean["product_card_id"] = clean["product_card_id"].astype(int)

    weekly = (
        clean.set_index("order_date")
        .groupby("product_card_id")["order_item_quantity"]
        .resample(args.week_freq)
        .sum()
        .reset_index()
        .rename(columns={"order_date": "week"})
    )

    clean_path = out_dir / "clean_orders_minimal.parquet"
    weekly_path = out_dir / "weekly_demand.parquet"
    clean.to_parquet(clean_path, index=False)
    weekly.to_parquet(weekly_path, index=False)

    print("Saved:")
    print(" ", clean_path)
    print(" ", weekly_path)

    print("\nSanity checks:")
    print("Weekly rows:", weekly.shape[0])
    print("Products:", weekly["product_card_id"].nunique())
    print("Date range:", weekly["week"].min(), "→", weekly["week"].max())

    top = weekly.groupby("product_card_id")["order_item_quantity"].sum().sort_values(ascending=False).head(5)
    print("\nTop 5 products by total demand (weekly summed):")
    print(top)


if __name__ == "__main__":
    main()
