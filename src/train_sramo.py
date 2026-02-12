# src/train_sramo.py
# Residual SRAMO (PPO) trainer:
#   q_t = clip(q_base_t + delta_t, 0, max_order)
# where q_base_t is an Order-Up-To style baseline computed from forecasts + current inventory position.
#
# This avoids the "do-nothing / under-ordering" local optimum by giving PPO a strong prior,
# and letting it learn only corrections (delta).
#
# Run:
#   python src/train_sramo.py --pid 191 --iters 6000 --episodes_per_iter 8 --reward_scale 1000 --safety_z 1.0 --lambda_delta 1.0
#
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch import nn

from sramo_env import InventoryEnv, InventoryCosts, EnvConfig


def load_forecast(pid: int, forecasts_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expects CSV: outputs/forecasts/product_<pid>_forecast.csv
    with columns: week, forecast_units, (optional) sigma_units
    """
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
    """S_t = sum_{i=t..t+L-1} mu_i + z * sqrt(sum sig_i^2)"""
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


def order_up_to_baseline(env: InventoryEnv, mu: np.ndarray, sig: np.ndarray, safety_z: float) -> Tuple[float, float]:
    """Returns (q_base, S_t)."""
    S = base_stock_target(mu, sig, int(env.t), int(env.cfg.lead_time_weeks), float(safety_z))
    ip = inv_position(env)
    q_base = max(0.0, S - ip)
    return float(q_base), float(S)


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

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(s)
        mu = self.mu(h)
        std = torch.exp(self.log_std).clamp(1e-3, 50.0)
        return mu, std

    def dist(self, s: torch.Tensor):
        mu, std = self.forward(s)
        return torch.distributions.Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


def compute_gae(rews: np.ndarray, vals: np.ndarray, dones: np.ndarray, gamma: float, lam: float):
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for t in reversed(range(T)):
        next_val = 0.0 if dones[t] else vals[t + 1]
        delta = rews[t] + gamma * next_val - vals[t]
        last = delta + gamma * lam * (0.0 if dones[t] else 1.0) * last
        adv[t] = last
    returns = adv + vals[:-1]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, returns


@torch.no_grad()
def rollout_episode(
    env: InventoryEnv,
    actor: Actor,
    device: torch.device,
    reward_scale: float,
    mu: np.ndarray,
    sig: np.ndarray,
    safety_z: float,
    lambda_delta: float,
    delta_clip_frac: float,
    p_baseline_only: float,
):
    s = env.reset(init_inventory=0.0)
    states, deltas, logps, rews, dones = [], [], [], [], []
    done = False

    while not done:
        q_base, _S = order_up_to_baseline(env, mu, sig, safety_z)

        use_baseline = (np.random.rand() < float(p_baseline_only))

        st = torch.from_numpy(s).float().to(device)
        if use_baseline:
            delta_raw = 0.0
            logp = 0.0
        else:
            dist = actor.dist(st)
            delta = dist.sample()
            logp = float(dist.log_prob(delta).sum(-1).item())
            delta_raw = float(delta.item())

        # clip delta
        max_delta = float(delta_clip_frac) * float(env.cfg.max_order)
        delta_clipped = float(np.clip(delta_raw, -max_delta, max_delta))

        # final order
        q = float(np.clip(q_base + delta_clipped, 0.0, float(env.cfg.max_order)))

        ns, r, done, info = env.step(q)

        delta_pen = float(lambda_delta) * float((delta_clipped / float(env.cfg.max_order)) ** 2)
        r_adj = (float(r) - delta_pen) / float(reward_scale)

        states.append(s)
        deltas.append(delta_clipped)
        logps.append(logp)
        rews.append(r_adj)
        dones.append(done)

        s = ns

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(deltas, dtype=np.float32),
        np.asarray(logps, dtype=np.float32),
        np.asarray(rews, dtype=np.float32),
        np.asarray(dones, dtype=np.bool_),
    )


def parse_args():
    p = argparse.ArgumentParser(description="Residual SRAMO PPO: Order-Up-To baseline + learned delta")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--forecasts_dir", type=str, default="outputs/forecasts")

    # env
    p.add_argument("--episode_weeks", type=int, default=52)
    p.add_argument("--lead_time_weeks", type=int, default=1)
    p.add_argument("--max_order", type=float, default=500.0)
    p.add_argument("--max_inventory", type=float, default=2000.0)

    # costs
    p.add_argument("--cost_holding", type=float, default=0.5)
    p.add_argument("--cost_stockout", type=float, default=5.0)
    p.add_argument("--cost_ordering", type=float, default=0.2)
    p.add_argument("--cost_fixed_order", type=float, default=2.0)

    # residual baseline settings
    p.add_argument("--safety_z", type=float, default=1.0)
    p.add_argument("--lambda_delta", type=float, default=1.0)
    p.add_argument("--delta_clip_frac", type=float, default=0.5)

    # curriculum
    p.add_argument("--baseline_p_start", type=float, default=0.8)
    p.add_argument("--baseline_p_mid", type=float, default=0.5)
    p.add_argument("--baseline_p_end", type=float, default=0.2)
    p.add_argument("--baseline_warmup1", type=int, default=500)
    p.add_argument("--baseline_warmup2", type=int, default=1500)

    # PPO
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--episodes_per_iter", type=int, default=8)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--minibatch_size", type=int, default=512)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.02)
    p.add_argument("--value_coef", type=float, default=0.5)

    p.add_argument("--reward_scale", type=float, default=1000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="models/sramo_policy_pid_{pid}.pt")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        seed=args.seed,
    )
    env = InventoryEnv(mu, sig, costs=costs, cfg=cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)
    state_dim = int(len(env.reset()))
    actor = Actor(state_dim).to(device).train()
    print("Actor device:", next(actor.parameters()).device)
    critic = Critic(state_dim).to(device).train()
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=args.lr)

    print(f"[TRAIN] Residual SRAMO (OU + delta) pid={args.pid} device={device}")
    print(f"[TRAIN] iters={args.iters} episodes_per_iter={args.episodes_per_iter} reward_scale={args.reward_scale}")

    for it in range(1, int(args.iters) + 1):
        if it <= int(args.baseline_warmup1):
            p_base = float(args.baseline_p_start)
        elif it <= int(args.baseline_warmup2):
            p_base = float(args.baseline_p_mid)
        else:
            p_base = float(args.baseline_p_end)

        S_all, D_all, LP_all, R_all, Done_all = [], [], [], [], []
        ep_costs = []

        for _ in range(int(args.episodes_per_iter)):
            S, D, LP, R, Done = rollout_episode(
                env=env,
                actor=actor,
                device=device,
                reward_scale=args.reward_scale,
                mu=mu,
                sig=sig,
                safety_z=args.safety_z,
                lambda_delta=args.lambda_delta,
                delta_clip_frac=args.delta_clip_frac,
                p_baseline_only=p_base,
            )
            S_all.append(S); D_all.append(D); LP_all.append(LP); R_all.append(R); Done_all.append(Done)

            # approximate cost from reward (unscale + remove penalty roughly). Just for monitoring.
            ep_costs.append(float(-np.sum(R) * args.reward_scale))

        states = np.concatenate(S_all, axis=0)
        deltas = np.concatenate(D_all, axis=0)
        old_logps = np.concatenate(LP_all, axis=0)
        rewards = np.concatenate(R_all, axis=0)
        dones = np.concatenate(Done_all, axis=0)

        with torch.no_grad():
            v = critic(torch.from_numpy(states).float().to(device)).cpu().numpy().astype(np.float32)
        vals = np.concatenate([v, np.zeros(1, dtype=np.float32)], axis=0)

        adv, returns = compute_gae(rewards, vals, dones, args.gamma, args.lam)

        st = torch.from_numpy(states).float().to(device)
        dt = torch.from_numpy(deltas).float().unsqueeze(-1).to(device)
        old_lp = torch.from_numpy(old_logps).float().to(device)
        adv_t = torch.from_numpy(adv).float().to(device)
        ret_t = torch.from_numpy(returns).float().to(device)

        n = len(states)
        idxs = np.arange(n)

        for _ in range(int(args.update_epochs)):
            np.random.shuffle(idxs)
            for start in range(0, n, int(args.minibatch_size)):
                mb = idxs[start : start + int(args.minibatch_size)]
                if len(mb) < 32:
                    continue

                s_mb = st[mb]
                d_mb = dt[mb]
                oldlp_mb = old_lp[mb]
                adv_mb = adv_t[mb]
                ret_mb = ret_t[mb]

                dist = actor.dist(s_mb)
                newlp = dist.log_prob(d_mb).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = torch.exp(newlp - oldlp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value = critic(s_mb)
                value_loss = ((value - ret_mb) ** 2).mean()

                loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
                opt.step()

        if it % 50 == 0 or it == 1:
            mean_cost = float(np.mean(ep_costs)) if ep_costs else float("nan")
            print(
                f"iter={it:04d}  policy_loss={float(policy_loss.item()):.4f}  "
                f"value_loss={float(value_loss.item()):.4f}  entropy={float(entropy.item()):.4f}  "
                f"mean_cost={mean_cost:.2f}  p_base={p_base:.2f}"
            )

    out_path = Path(args.out.format(pid=args.pid))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "pid": int(args.pid),
            "actor_state": actor.state_dict(),
            "critic_state": critic.state_dict(),
            "config": vars(args),
            "policy_type": "ppo_residual_ou",
        },
        out_path,
    )
    print(f"Saved SRAMO policy to: {out_path}")


if __name__ == "__main__":
    main()
