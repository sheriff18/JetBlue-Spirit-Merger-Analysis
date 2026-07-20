"""Step 1: Structural screens - HHI under the 2023 Merger Guidelines.

THE LOGIC
---------
Before any modeling, agencies ask a simple question: how concentrated
is this market, and how much more concentrated does the merger make it?

HHI (Herfindahl-Hirschman Index) = sum of squared percentage shares.
A monopoly scores 10,000. Four equal firms score 2,500.

The 2023 DOJ/FTC Merger Guidelines set a structural presumption of
illegality when EITHER:

  (a) post-merger HHI > 1,800 AND the change > 100, or
  (b) the merged firm's share > 30% AND the change > 100

The change in HHI has a convenient closed form: for two merging firms
with percentage shares s1 and s2, delta HHI = 2 * s1 * s2. That is
why overlap matters so much - if the parties don't both fly a route,
the delta is zero and the route raises no concern.

WHAT THIS BUYS YOU
------------------
The screen doesn't prove harm. It identifies which of ~2,000 routes
deserve real analysis. In the JetBlue/Spirit case, DOJ focused on
overlap routes precisely because that is where the delta was large.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HHI_UNCONCENTRATED = 1000
HHI_CONCENTRATED = 1800
DELTA_THRESHOLD = 100
SHARE_THRESHOLD = 0.30


def route_hhi(panel: pd.DataFrame) -> pd.DataFrame:
    """HHI for every route in the panel."""
    p = panel.copy()
    p["share_pct"] = 100.0 * p["share"]
    p["sq"] = p["share_pct"] ** 2
    out = p.groupby("market", as_index=False).agg(
        hhi=("sq", "sum"), n_carriers=("carrier", "nunique"),
        route_pax=("pax", "sum"))
    return out


def merger_screen(panel: pd.DataFrame, firm_a: str, firm_b: str
                  ) -> pd.DataFrame:
    """Pre/post HHI and the 2023 Guidelines presumption, route by route.

    Only routes where BOTH carriers operate produce a delta; those are
    the "overlap routes" that decide airline merger cases.
    """
    base = route_hhi(panel)

    shares = panel.pivot_table(index="market", columns="carrier",
                               values="share", fill_value=0.0)
    for f in (firm_a, firm_b):
        if f not in shares:
            shares[f] = 0.0

    s_a = shares[firm_a]
    s_b = shares[firm_b]

    out = base.set_index("market").copy()
    out["share_a"] = s_a
    out["share_b"] = s_b
    out["combined_share"] = s_a + s_b
    out["overlap"] = (s_a > 0) & (s_b > 0)

    # delta HHI = 2 * s1 * s2 (in percentage-point units)
    out["hhi_delta"] = 2.0 * (100 * s_a) * (100 * s_b)
    out["hhi_post"] = out["hhi"] + out["hhi_delta"]

    out["presumption"] = (
        ((out["hhi_post"] > HHI_CONCENTRATED) &
         (out["hhi_delta"] > DELTA_THRESHOLD)) |
        ((out["combined_share"] > SHARE_THRESHOLD) &
         (out["hhi_delta"] > DELTA_THRESHOLD))
    )
    return out.reset_index().rename(columns={"hhi": "hhi_pre"})


def summarize_screen(screen: pd.DataFrame) -> dict:
    """Headline numbers for the screening exhibit."""
    ov = screen[screen["overlap"]]
    flagged = ov[ov["presumption"]]
    return {
        "routes_total": len(screen),
        "routes_overlap": len(ov),
        "routes_flagged": len(flagged),
        "pax_on_flagged": float(flagged["route_pax"].sum()),
        "pax_share_flagged": float(
            flagged["route_pax"].sum() / screen["route_pax"].sum()),
        "mean_delta_on_overlap": float(ov["hhi_delta"].mean()),
        "mean_post_hhi_flagged": float(flagged["hhi_post"].mean()),
    }
