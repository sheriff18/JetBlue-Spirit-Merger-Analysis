"""Step 4: Merger simulation - predicting the fare increase.

FROM SCREEN TO PREDICTION
-------------------------
GUPPI says "there is upward pressure." Simulation says "fares rise by
X%." It is the most informative tool and the most assumption-heavy,
which is why it comes last and why its assumptions are stated openly.

THE THREE STEPS
---------------
1. CALIBRATE demand. We observe shares and fares but not the demand
   curve. Using a logit model plus one assumed margin, the pre-merger
   Bertrand first-order condition pins down the price sensitivity:

        m = 1 / (alpha * p * (1 - s))   =>   alpha = 1 / (m * p * (1-s))

   Then mean utilities follow from observed shares.

2. RECOVER marginal costs. Under Nash-Bertrand pricing, each firm's
   observed price satisfies its FOC. Inverting those conditions gives
   the marginal costs consistent with observed behavior. (This is the
   same logic as backing implied volatility out of an option price:
   assume the model, invert to find the unobserved input.)

3. RE-SOLVE the equilibrium with the merged ownership structure, and
   optionally an efficiency credit. The difference in prices is the
   predicted merger effect.

WHAT TO BE HONEST ABOUT
-----------------------
Logit imposes proportional diversion (the "independence of irrelevant
alternatives" property), which is exactly the assumption DOJ disputed
in this case. So the simulation, run plainly, is CONSERVATIVE relative
to DOJ's theory. Saying that out loud is stronger than pretending the
model is neutral.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LogitMarket:
    """Calibrated logit demand for a single route."""

    def __init__(self, fares: np.ndarray, shares: np.ndarray,
                 carriers: list[str], margin: float = 0.25,
                 outside_share: float = 0.20):
        # Renormalize inside shares to leave room for an outside option
        # (not flying, driving, connecting itineraries)
        s = np.asarray(shares, dtype=float)
        s = s / s.sum() * (1.0 - outside_share)

        self.p = np.asarray(fares, dtype=float)
        self.s = s
        self.carriers = list(carriers)
        self.s0 = 1.0 - s.sum()

        # Calibrate price coefficient from the largest carrier's FOC
        i = int(np.argmax(s))
        self.alpha = 1.0 / (margin * self.p[i] * (1.0 - s[i]))
        self.delta = np.log(s / self.s0) + self.alpha * self.p

    def shares_at(self, p: np.ndarray) -> np.ndarray:
        u = self.delta - self.alpha * np.asarray(p, dtype=float)
        e = np.exp(u)
        return e / (1.0 + e.sum())

    def jacobian(self, p: np.ndarray) -> np.ndarray:
        s = self.shares_at(p)
        J = self.alpha * np.outer(s, s)
        np.fill_diagonal(J, -self.alpha * s * (1.0 - s))
        return J


def _omega(mkt: LogitMarket, p: np.ndarray, own: np.ndarray) -> np.ndarray:
    return -own * mkt.jacobian(p).T


def recover_costs(mkt: LogitMarket, owners: list[int]) -> np.ndarray:
    o = np.asarray(owners)
    own = (o[:, None] == o[None, :]).astype(float)
    s = mkt.shares_at(mkt.p)
    return mkt.p - np.linalg.solve(_omega(mkt, mkt.p, own), s)


def solve_prices(mkt: LogitMarket, costs: np.ndarray, owners: list[int],
                 p0: np.ndarray | None = None, damping: float = 0.4,
                 tol: float = 1e-10, max_iter: int = 5000) -> np.ndarray:
    o = np.asarray(owners)
    own = (o[:, None] == o[None, :]).astype(float)
    p = mkt.p.copy() if p0 is None else np.asarray(p0, float)
    for _ in range(max_iter):
        s = mkt.shares_at(p)
        p_new = costs + np.linalg.solve(_omega(mkt, p, own), s)
        if np.max(np.abs(p_new - p)) < tol:
            return p_new
        p = damping * p_new + (1 - damping) * p
    raise RuntimeError("Simulation did not converge")


def simulate_route(panel: pd.DataFrame, market: str,
                   firm_a: str, firm_b: str, margin: float = 0.25,
                   efficiency: float = 0.0,
                   outside_share: float = 0.20) -> dict:
    """Simulate the merger on one route."""
    r = (panel[panel["market"] == market]
         .sort_values("share", ascending=False).reset_index(drop=True))
    if firm_a not in set(r["carrier"]) or firm_b not in set(r["carrier"]):
        return {}

    mkt = LogitMarket(r["fare"].values, r["share"].values,
                      list(r["carrier"]), margin, outside_share)

    owners_pre = list(range(len(r)))
    ia = list(r["carrier"]).index(firm_a)
    ib = list(r["carrier"]).index(firm_b)
    owners_post = owners_pre.copy()
    owners_post[ib] = owners_post[ia]

    c_pre = recover_costs(mkt, owners_pre)
    c_post = c_pre.copy()
    if efficiency > 0:
        c_post[[ia, ib]] *= (1.0 - efficiency)

    p_pre = solve_prices(mkt, c_pre, owners_pre)
    p_post = solve_prices(mkt, c_post, owners_post, p0=p_pre)

    pct = 100.0 * (p_post - p_pre) / p_pre
    pax = float(r["pax"].sum())
    s_pre = mkt.shares_at(p_pre)

    # Passenger-weighted average fare effect across the route
    avg_effect = float(np.sum(pct * s_pre) / np.sum(s_pre))

    return {"market": market, "carriers": list(r["carrier"]),
            "fares_pre": p_pre, "fares_post": p_post,
            "pct_change": pct,
            "merging_effect": float(np.mean(pct[[ia, ib]])),
            "avg_route_effect": avg_effect,
            "route_pax": pax,
            "efficiency": efficiency}


def simulate_all(panel: pd.DataFrame, routes: list[str],
                 firm_a: str, firm_b: str, margin: float = 0.25,
                 efficiency: float = 0.0) -> pd.DataFrame:
    rows = []
    for m in routes:
        try:
            res = simulate_route(panel, m, firm_a, firm_b, margin,
                                 efficiency)
        except Exception:
            continue
        if res:
            rows.append({"market": m,
                         "merging_carriers_pct": res["merging_effect"],
                         "route_avg_pct": res["avg_route_effect"],
                         "route_pax": res["route_pax"]})
    df = pd.DataFrame(rows)
    return (df.sort_values("merging_carriers_pct", ascending=False)
            .reset_index(drop=True)) if len(df) else df
