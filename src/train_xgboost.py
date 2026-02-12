# src/train_xgboost.py
# Train an XGBoost regression model for inventory decision-making.
#
# Fixes included:
# - Robust dtype handling for pandas StringDtype / nullable dtypes (no np.issubdtype crash)
# - Prints TRAIN vs TEST metrics (MAE/RMSE/R²)
# - Optional K-fold cross validation via --cv_folds (prints mean±std)
# - Supports CSV or Parquet inputs
# - Saves scaler + feature names for reproducible inference
#
# Example:
#   python src/train_xgboost.py --data outputs/inventory_features.csv --target_col expected_demand_lead_time --cv_folds 5
#
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from pandas.api.types import is_numeric_dtype

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost for inventory optimization (regression)")

    p.add_argument(
        "--data",
        type=str,
        default="outputs/inventory_features.csv",
        help="Path to feature table (CSV or Parquet).",
    )
    p.add_argument(
        "--target_col",
        type=str,
        default="expected_demand_lead_time",
        help="Target column to predict.",
    )

    # Outputs
    p.add_argument("--out_model", type=str, default="models/xgboost_inventory_model.json")
    p.add_argument("--out_scaler", type=str, default="models/xgb_scaler.joblib")
    p.add_argument("--out_features", type=str, default="models/xgb_feature_names.json")

    # Evaluation
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--cv_folds", type=int, default=0, help="If >=2, run K-fold CV and report mean±std.")

    # Plotting
    p.add_argument("--plot_topk", type=int, default=15, help="Top-k feature importances to plot (0 disables).")

    # XGBoost hyperparams (exposed for tuning)
    p.add_argument("--n_estimators", type=int, default=800)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)

    return p.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def build_model(args: argparse.Namespace) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        objective="reg:squarederror",
        random_state=args.random_state,
        n_jobs=-1,
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def print_metrics_block(title: str, m: Dict[str, float]) -> None:
    print(title)
    print(f"  MAE  : {m['MAE']:.3f}")
    print(f"  RMSE : {m['RMSE']:.3f}")
    print(f"  R²   : {m['R2']:.3f}")


def run_kfold_cv(X: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> Dict[str, Tuple[float, float]]:
    k = int(args.cv_folds)
    kf = KFold(n_splits=k, shuffle=True, random_state=args.random_state)

    maes, rmses, r2s = [], [], []

    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = build_model(args)
        model.fit(X_tr_s, y_tr)

        pred = model.predict(X_te_s)
        m = metrics(y_te, pred)
        maes.append(m["MAE"])
        rmses.append(m["RMSE"])
        r2s.append(m["R2"])

        print(f"[CV] Fold {fold}/{k}  MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  R²={m['R2']:.3f}")

    def mean_std(vals: List[float]) -> Tuple[float, float]:
        arr = np.array(vals, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=0))

    return {"MAE": mean_std(maes), "RMSE": mean_std(rmses), "R2": mean_std(r2s)}


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    df = load_table(data_path)

    # Target
    target_col = args.target_col
    if target_col not in df.columns:
        if "y" in df.columns:
            print(f"[WARN] Target column '{target_col}' not found. Falling back to 'y'.")
            target_col = "y"
        else:
            raise KeyError(
                f"Target column '{target_col}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

    y = pd.to_numeric(df[target_col], errors="coerce").astype(float).values

    # Features
    drop_cols = ["product_card_id", "week", "last_hist_week", target_col]
    X_df = df.drop(columns=drop_cols, errors="ignore").copy()

    # Robust coercion: handle pandas StringDtype/nullable dtypes cleanly
    for c in list(X_df.columns):
        if not is_numeric_dtype(X_df[c]):
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")

    # Drop all-NaN columns
    all_nan = [c for c in X_df.columns if X_df[c].isna().all()]
    if all_nan:
        print(f"[INFO] Dropping all-NaN columns after coercion: {all_nan}")
        X_df = X_df.drop(columns=all_nan)

    X_df = X_df.fillna(0.0)

    feature_names = X_df.columns.tolist()
    X = X_df.astype(float).values

    # Train/test evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    scaler_tt = StandardScaler()
    X_train_s = scaler_tt.fit_transform(X_train)
    X_test_s = scaler_tt.transform(X_test)

    model_tt = build_model(args)
    model_tt.fit(X_train_s, y_train)

    pred_train = model_tt.predict(X_train_s)
    pred_test = model_tt.predict(X_test_s)

    m_train = metrics(y_train, pred_train)
    m_test = metrics(y_test, pred_test)

    print("\nXGBoost Inventory Decision Model Performance")
    print("--------------------------------------------")
    print(f"Data file : {data_path}")
    print(f"Target    : {target_col}")
    print(f"Rows      : {len(df)}")
    print(f"Features  : {len(feature_names)}")
    print_metrics_block("Train metrics:", m_train)
    print_metrics_block("Test  metrics:", m_test)

    # Optional CV
    if int(args.cv_folds) >= 2:
        print(f"\nK-Fold CV (k={int(args.cv_folds)})")
        print("--------------------------------------------")
        cv = run_kfold_cv(X, y, args)
        print(
            f"[CV] MAE : {cv['MAE'][0]:.3f} ± {cv['MAE'][1]:.3f}\n"
            f"[CV] RMSE: {cv['RMSE'][0]:.3f} ± {cv['RMSE'][1]:.3f}\n"
            f"[CV] R²  : {cv['R2'][0]:.3f} ± {cv['R2'][1]:.3f}"
        )

    # Fit final model on all data and save
    scaler = StandardScaler()
    X_all_s = scaler.fit_transform(X)

    model = build_model(args)
    model.fit(X_all_s, y)

    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(out_model)

    out_scaler = Path(args.out_scaler)
    out_scaler.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_scaler)

    out_features = Path(args.out_features)
    out_features.parent.mkdir(parents=True, exist_ok=True)
    out_features.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    print(f"\nSaved XGBoost model to: {out_model}")
    print(f"Saved scaler to      : {out_scaler}")
    print(f"Saved feature names to: {out_features}")

    # Feature importance plot
    if args.plot_topk and args.plot_topk > 0:
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][: int(args.plot_topk)]

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(idx)), importances[idx])
        plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=45, ha="right")
        plt.title("Top Feature Importances (XGBoost)")
        plt.tight_layout()

        plot_path = out_model.with_suffix(".feature_importance.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Saved feature importance plot to: {plot_path}")


if __name__ == "__main__":
    main()
