# src/eval_sramo.py
# Evaluate SRAMO policy vs baselines (supports residual OU checkpoints)
#
# Usage:
#   python src/eval_sramo.py --pid 191 --episodes 30
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

from sramo_env import InventoryEnv, InventoryCosts, EnvConfig


class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.tensor([-0.3], dtype=torch.float32))

    def forward(self, s: torch.Tensor):
        h = self.net(s)
        mu = self.mu(h)
        std = torch.exp(self.log_std).clamp(1e-3, 50.0)
        return mu, std


def load_forecast(pid: int, forecasts_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    fp = forecasts_dir / f"product_{pid}_forecast.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing forecast file: {fp}")
    df = pd.read_csv(fp)
    if "forecast_units" not in df.columns:
        raise KeyError(f"{fp} must contain column 'forecast_units'")
    mu = df["forecast_units"].astype(float).to_numpy().astype(np.float32)
    if "sigma_units" in df.columns:
        sig = df["sigma_units"].astype(float).to_numpy().astype(np.float32)
    else:
        sig = (0.1 * mu + 1.0).astype(np.float32)
    return mu, sig


def base_stock_target(mu: np.ndarray, sig: np.ndarray, t: int, lead_time: int, safety_z: float) -> float:
    T = len(mu)
    lt = max(1, int(lead_time))
    t0 = int(np.clip(t, 0, T - 1))
    t1 = min(T, t0 + lt)
    mu_lt = float(np.sum(mu[t0:t1]))
    sig_lt = float(np.sqrt(np.sum(np.square(sig[t0:t1]))))
    return max(0.0, mu_lt + float(safety_z) * sig_lt)


def inv_position(env: InventoryEnv) -> float:
    pipe = float(np.sum(getattr(env, "pipeline", 0.0)))
    return float(env.on_hand + pipe - env.backlog)


def policy_forecast(mu_t: float, lead_time_weeks: int) -> float:
    return float(mu_t) * float(max(1, lead_time_weeks))


def policy_order_up_to(env: InventoryEnv, S: float) -> float:
    ip = inv_position(env)
    return float(max(0.0, S - ip))


@torch.no_grad()
def policy_sramo_action(ckpt: Dict, env: InventoryEnv, state: np.ndarray, device: torch.device, mu: np.ndarray, sig: np.ndarray) -> float:
    if "actor_state" not in ckpt:
        raise KeyError("Checkpoint missing 'actor_state' (continuous PPO expected).")

    state_dim = int(len(state))
    actor = Actor(state_dim).to(device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()

    st = torch.from_numpy(state).float().to(device)
    delta_mu, _ = actor.forward(st)
    delta = float(delta_mu.item())

    max_order = float(env.cfg.max_order)

    if str(ckpt.get("policy_type", "")).lower().startswith("ppo_residual"):
        cfg = ckpt.get("config", {}) or {}
        safety_z = float(cfg.get("safety_z", 1.0))
        delta_clip_frac = float(cfg.get("delta_clip_frac", 0.5))
        max_delta = float(delta_clip_frac) * max_order
        delta = float(np.clip(delta, -max_delta, max_delta))

        S = base_stock_target(mu, sig, int(env.t), int(env.cfg.lead_time_weeks), safety_z)
        q_base = max(0.0, S - inv_position(env))
        q = q_base + delta
    else:
        q = delta

    return float(np.clip(q, 0.0, max_order))


def run_episode(env: InventoryEnv, policy_fn, max_order: float) -> Dict[str, float]:
    s = env.reset(init_inventory=0.0)
    done = False

    total_cost = 0.0
    unmet = 0.0
    hold = 0.0
    stockout = 0.0
    ordering = 0.0

    while not done:
        q = float(policy_fn(env, s))
        q = float(np.clip(q, 0.0, max_order))
        s, r, done, info = env.step(q)

        total_cost += float(info["total_cost"])
        unmet += float(info["unmet"])
        hold += float(info["holding_cost"])
        stockout += float(info["stockout_cost"])
        ordering += float(info["ordering_cost"])

    return {
        "total_cost": total_cost,
        "unmet": unmet,
        "hold": hold,
        "stockout": stockout,
        "ordering": ordering,
    }


def summarize(arr: np.ndarray) -> Tuple[float, float]:
    return float(arr.mean()), float(arr.std(ddof=0))


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SRAMO policy vs baselines")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--model_path", type=str, default="models/sramo_policy_pid_{pid}.pt")
    p.add_argument("--forecasts_dir", type=str, default="outputs/forecasts")

    p.add_argument("--episode_weeks", type=int, default=52)
    p.add_argument("--lead_time_weeks", type=int, default=1)
    p.add_argument("--max_order", type=float, default=500.0)
    p.add_argument("--max_inventory", type=float, default=2000.0)

    p.add_argument("--cost_holding", type=float, default=0.5)
    p.add_argument("--cost_stockout", type=float, default=5.0)
    p.add_argument("--cost_ordering", type=float, default=0.2)
    p.add_argument("--cost_fixed_order", type=float, default=2.0)

    p.add_argument("--order_up_to_mult", type=float, default=1.0, help="S = mult * lead-time mean demand")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mu, sig = load_forecast(args.pid, Path(args.forecasts_dir))
    T = min(int(args.episode_weeks), len(mu))
    mu = mu[:T]
    sig = sig[:T]

    costs = InventoryCosts(
        holding=args.cost_holding,
        stockout=args.cost_stockout,
        ordering=args.cost_ordering,
        fixed_order=args.cost_fixed_order,
    )
    cfg = EnvConfig(
        lead_time_weeks=args.lead_time_weeks,
        max_order=args.max_order,
        max_inventory=args.max_inventory,
        episode_weeks=T,
        seed=42,
    )
    env = InventoryEnv(mu, sig, costs=costs, cfg=cfg)

    model_path = Path(args.model_path.format(pid=args.pid))
    print(f"[EVAL] Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    print(f"[EVAL] Checkpoint keys: {list(ckpt.keys())}")
    print(f"[EVAL] policy_type: {ckpt.get('policy_type')}")

    lt = max(1, int(args.lead_time_weeks))
    S = float(args.order_up_to_mult) * float(np.sum(mu[:lt]))

    def pol_sramo(_env: InventoryEnv, state: np.ndarray) -> float:
        return policy_sramo_action(ckpt, _env, state, device, mu, sig)

    def pol_forecast(_env: InventoryEnv, state: np.ndarray) -> float:
        t = int(_env.t)
        mu_t = float(mu[min(t, T - 1)])
        return policy_forecast(mu_t, lt)

    def pol_ou(_env: InventoryEnv, state: np.ndarray) -> float:
        return policy_order_up_to(_env, S)

    results = {}
    for name, pol in [("sramo", pol_sramo), ("forecast", pol_forecast), ("order_up_to", pol_ou)]:
        totals, unmets = [], []
        holds, stockouts, orderings = [], [], []
        for _ in range(int(args.episodes)):
            ep = run_episode(env, pol, args.max_order)
            totals.append(ep["total_cost"])
            unmets.append(ep["unmet"])
            holds.append(ep["hold"])
            stockouts.append(ep["stockout"])
            orderings.append(ep["ordering"])
        results[name] = {
            "total_cost": np.array(totals, dtype=np.float32),
            "unmet": np.array(unmets, dtype=np.float32),
            "hold": float(np.mean(holds)),
            "stockout": float(np.mean(stockouts)),
            "ordering": float(np.mean(orderings)),
        }

    print(f"\nEvaluation for product_id={args.pid} over {args.episodes} episodes\n")
    print("Policy         Total Cost (mean±std)        Unmet (mean±std)       Hold       Stockout   Ordering")
    print("-" * 100)

    for name in ["sramo", "forecast", "order_up_to"]:
        tc_m, tc_s = summarize(results[name]["total_cost"])
        um_m, um_s = summarize(results[name]["unmet"])
        print(
            f"{name:<13} {tc_m:10.2f} ± {tc_s:7.2f}   "
            f"{um_m:8.2f} ± {um_s:6.2f}   "
            f"{results[name]['hold']:8.2f}   {results[name]['stockout']:8.2f}   {results[name]['ordering']:8.2f}"
        )


if __name__ == "__main__":
    main()
