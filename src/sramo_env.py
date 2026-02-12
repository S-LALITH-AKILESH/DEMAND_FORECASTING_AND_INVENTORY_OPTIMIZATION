# src/sramo_env.py
# Inventory control environment for SRAMO/RL (single product, weekly)
#
# Key features for stability:
# - Safe indexing at terminal step (never out-of-bounds)
# - Configurable demand distribution: normal or poisson
# - Optional reward shaping toward a base-stock target S_t (helps PPO learn)
#
# State (6 dims):
#   [ on_hand/max_inv,
#     backlog/max_inv,
#     mu_t/500,
#     sigma_t/200,
#     woy_norm,
#     base_stock_target/max_order ]   (acts like a "suggestion" signal)
#
# Action:
#   reorder quantity (float) -> clipped to [0, max_order]
#
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class InventoryCosts:
    holding: float = 0.5        # cost per unit held per week
    stockout: float = 5.0       # penalty per unit unmet demand
    ordering: float = 0.2       # cost per unit ordered
    fixed_order: float = 2.0    # fixed cost per order if order > 0


@dataclass
class EnvConfig:
    lead_time_weeks: int = 1
    max_order: float = 500.0
    max_inventory: float = 2000.0
    episode_weeks: int = 52
    demand_noise_scale: float = 1.0
    seed: int = 42

    # demand model
    demand_dist: str = "normal"   # "normal" or "poisson"

    # optional shaping toward a base-stock target
    shaping_beta: float = 0.0     # 0 disables
    safety_z: float = 0.0         # safety factor for target S


class InventoryEnv:
    """
    Single-product inventory environment.
    forecast_mu[t]    = expected demand at week t
    forecast_sigma[t] = uncertainty (std) at week t
    """

    def __init__(
        self,
        forecast_mu: np.ndarray,
        forecast_sigma: np.ndarray,
        costs: InventoryCosts = InventoryCosts(),
        cfg: EnvConfig = EnvConfig(),
    ):
        self.forecast_mu = np.asarray(forecast_mu, dtype=np.float32)
        self.forecast_sigma = np.asarray(forecast_sigma, dtype=np.float32)
        if self.forecast_mu.shape != self.forecast_sigma.shape:
            raise ValueError("forecast_mu and forecast_sigma must have same shape")

        self.costs = costs
        self.cfg = cfg

        self.rng = np.random.default_rng(int(cfg.seed))

        self.t = 0
        self.on_hand = 0.0
        self.backlog = 0.0
        self.pipeline = np.zeros(int(cfg.lead_time_weeks), dtype=np.float32)

    @property
    def episode_weeks(self) -> int:
        return int(self.cfg.episode_weeks)

    def reset(self, init_inventory: float = 0.0) -> np.ndarray:
        self.t = 0
        self.on_hand = float(np.clip(init_inventory, 0.0, self.cfg.max_inventory))
        self.backlog = 0.0
        self.pipeline = np.zeros(int(self.cfg.lead_time_weeks), dtype=np.float32)
        return self._get_state()

    # -----------------
    # Internals
    # -----------------
    def _receive_pipeline(self) -> None:
        if len(self.pipeline) == 0:
            return
        arrived = float(self.pipeline[0])
        self.on_hand = float(np.clip(self.on_hand + arrived, 0.0, self.cfg.max_inventory))
        if len(self.pipeline) > 1:
            self.pipeline[:-1] = self.pipeline[1:]
        self.pipeline[-1] = 0.0

    def _place_order(self, action: float) -> float:
        q = float(action)
        if not np.isfinite(q):
            q = 0.0
        q = float(np.clip(q, 0.0, self.cfg.max_order))
        if len(self.pipeline) == 0:
            self.on_hand = float(np.clip(self.on_hand + q, 0.0, self.cfg.max_inventory))
        else:
            self.pipeline[-1] += q
        return q

    def _sample_demand(self, mu: float, sigma: float) -> float:
        mu = max(0.0, float(mu))
        sigma = max(0.0, float(sigma)) * float(self.cfg.demand_noise_scale)

        dist = str(self.cfg.demand_dist).lower().strip()
        if dist == "poisson":
            # Poisson uses mean; sigma is ignored here (kept for state + target computation)
            lam = mu
            return float(self.rng.poisson(lam=lam))
        # normal
        d = self.rng.normal(loc=mu, scale=sigma) if sigma > 0 else mu
        return float(max(0.0, d))

    def _fulfill(self, demand: float) -> Tuple[float, float]:
        demand = float(max(0.0, demand))

        # Serve backlog first
        if self.backlog > 0:
            serve = min(self.on_hand, self.backlog)
            self.on_hand -= serve
            self.backlog -= serve

        # Serve current demand
        fulfilled = min(self.on_hand, demand)
        self.on_hand -= fulfilled
        unmet = demand - fulfilled
        if unmet > 0:
            self.backlog += unmet

        return float(fulfilled), float(unmet)

    def _inventory_position(self) -> float:
        return float(self.on_hand + float(np.sum(self.pipeline)) - self.backlog)

    def _base_stock_target(self, t_idx: int) -> float:
        # Target S = lead-time mean + z * lead-time std, clipped
        lt = max(1, int(self.cfg.lead_time_weeks))
        t0 = int(np.clip(t_idx, 0, self.episode_weeks - 1))
        t1 = min(self.episode_weeks, t0 + lt)
        mu_lt = float(np.sum(self.forecast_mu[t0:t1]))
        sig_lt = float(np.sqrt(np.sum(np.square(self.forecast_sigma[t0:t1]))))
        S = mu_lt + float(self.cfg.safety_z) * sig_lt
        return float(np.clip(S, 0.0, self.cfg.max_inventory))

    def _get_state(self) -> np.ndarray:
        # clamp index for terminal-safe access
        t_idx = int(np.clip(self.t, 0, self.episode_weeks - 1))

        mu = float(self.forecast_mu[t_idx])
        sig = float(self.forecast_sigma[t_idx])
        woy = float((t_idx % 52) + 1) / 52.0

        # include base-stock target as a "suggestion" channel (scaled)
        S = self._base_stock_target(t_idx)

        return np.array(
            [
                self.on_hand / self.cfg.max_inventory,
                self.backlog / self.cfg.max_inventory,
                mu / 500.0,
                sig / 200.0,
                woy,
                S / self.cfg.max_order,  # scaled similarly to order magnitude
            ],
            dtype=np.float32,
        )

    # -----------------
    # Step
    # -----------------
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.t >= self.episode_weeks:
            raise RuntimeError("Episode already finished. Call reset().")

        # 1) receive arriving stock
        self._receive_pipeline()

        # 2) place order
        q = self._place_order(action)

        # 3) demand
        mu = float(self.forecast_mu[self.t])
        sig = float(self.forecast_sigma[self.t])
        demand = self._sample_demand(mu, sig)

        # 4) fulfill
        fulfilled, unmet = self._fulfill(demand)

        # 5) costs
        holding_cost = float(self.costs.holding * self.on_hand)
        stockout_cost = float(self.costs.stockout * unmet)
        ordering_cost = float(self.costs.ordering * q + (self.costs.fixed_order if q > 0 else 0.0))

        base_cost = holding_cost + stockout_cost + ordering_cost

        # optional shaping toward base-stock target (helps PPO not collapse to "order 0")
        shaping = 0.0
        if float(self.cfg.shaping_beta) > 0:
            S = self._base_stock_target(self.t)
            inv_pos = self._inventory_position()
            shaping = float(self.cfg.shaping_beta) * float((inv_pos - S) ** 2)

        reward = -(base_cost + shaping)

        info = {
            "t": int(self.t),
            "order": float(q),
            "demand": float(demand),
            "fulfilled": float(fulfilled),
            "unmet": float(unmet),
            "on_hand": float(self.on_hand),
            "backlog": float(self.backlog),
            "inv_pos": float(self._inventory_position()),
            "base_stock_target": float(self._base_stock_target(self.t)),
            "holding_cost": float(holding_cost),
            "stockout_cost": float(stockout_cost),
            "ordering_cost": float(ordering_cost),
            "shaping": float(shaping),
            "total_cost": float(base_cost + shaping),
        }

        self.t += 1
        done = self.t >= self.episode_weeks
        return self._get_state(), float(reward), bool(done), info
