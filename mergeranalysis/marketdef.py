"""Step 2: Market definition - the hypothetical monopolist test.

WHY THIS IS THE WHOLE BALLGAME
------------------------------
Concentration numbers are meaningless until you say concentration IN
WHAT. Define the market as "nonstop BOS-FLL service" and two carriers
look dominant. Define it as "all air travel between the Northeast and
Florida, including connections" and they look small.

So antitrust uses a formal test. Imagine a single firm owned every
product in your proposed market. Could it profitably impose a Small
but Significant Non-transitory Increase in Price (conventionally 5%)?

  * If YES - customers have nowhere good to go, so the boundary is
    real. It is a relevant antitrust market.
  * If NO - too many customers would flee to substitutes outside your
    boundary. The market is drawn too narrowly; widen it and retest.

CRITICAL LOSS ANALYSIS
----------------------
The practical implementation. Raising price by t costs you volume but
earns more per remaining unit. The break-even volume loss is:

        critical loss = t / (t + m)

where m is the contribution margin. If the ACTUAL predicted loss is
smaller than the critical loss, the price rise is profitable and your
candidate market is a real market.

Note the counterintuitive implication that trips people up in
interviews: HIGH margins mean a LOW critical loss, so high-margin
industries have narrower markets. If you earn a lot per unit, losing
units hurts more.

THE FIGHT IN JETBLUE/SPIRIT
---------------------------
JetBlue argued for broad markets - all carriers, metro areas, and
connecting service included - which would make the merger look
harmless. DOJ argued for narrower ones, and emphasized that Spirit's
ultra-low-cost customers wouldn't substitute to legacy carriers at
all. We test both by running the HMT on airport-pair and
metro-aggregated definitions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def critical_loss(margin: float, ssnip: float = 0.05) -> float:
    """Break-even volume loss for a SSNIP: t / (t + m)."""
    return ssnip / (ssnip + margin)


def actual_loss_logit(shares_in_market: float, margin: float,
                      ssnip: float = 0.05) -> float:
    """Predicted volume loss under a logit/aggregate-elasticity model.

    With aggregate market elasticity implied by the observed margin,
    a group controlling `shares_in_market` of the candidate market
    loses approximately:

        actual loss ~ epsilon * t * (1 - group share)

    where epsilon is the own-price elasticity implied by the
    Lerner condition, epsilon = 1 / m. The (1 - share) term captures
    that a larger group internalizes more of the diverted volume.
    """
    epsilon = 1.0 / margin
    return epsilon * ssnip * (1.0 - shares_in_market)


def hmt_route(panel: pd.DataFrame, market: str, candidate: list[str],
              margin: float = 0.25, ssnip: float = 0.05) -> dict:
    """Run the HMT for a candidate set of carriers on one route."""
    r = panel[panel["market"] == market]
    if r.empty:
        raise ValueError(f"No data for route {market}")

    total = r["pax"].sum()
    in_group = r[r["carrier"].isin(candidate)]["pax"].sum()
    group_share = in_group / total if total else 0.0

    cl = critical_loss(margin, ssnip)
    al = actual_loss_logit(group_share, margin, ssnip)

    return {"market": market, "candidate": candidate,
            "group_share": group_share, "margin": margin, "ssnip": ssnip,
            "critical_loss": cl, "actual_loss": al,
            "ssnip_profitable": al < cl,
            "is_relevant_market": al < cl}


def smallest_market(panel: pd.DataFrame, market: str,
                    seed: list[str], margin: float = 0.25,
                    ssnip: float = 0.05) -> dict:
    """Smallest-market principle: add carriers until the SSNIP passes.

    Start with the merging parties, and if a hypothetical monopolist
    over just them couldn't raise price, add the next-largest carrier
    and retest. The first candidate set that passes is the relevant
    market.
    """
    r = panel[panel["market"] == market].sort_values(
        "share", ascending=False)
    others = [c for c in r["carrier"] if c not in seed]

    candidate = list(seed)
    path = []
    while True:
        res = hmt_route(panel, market, candidate, margin, ssnip)
        path.append(res)
        if res["is_relevant_market"] or not others:
            return {"market": market, "relevant_market": candidate,
                    "result": res, "path": path}
        candidate = candidate + [others.pop(0)]


def compare_definitions(panel_airport: pd.DataFrame,
                        panel_metro: pd.DataFrame,
                        routes: list[str], parties: list[str],
                        margin: float = 0.25) -> pd.DataFrame:
    """Airport-pair vs metro-area market definitions.

    THE KEY SENSITIVITY. If the merger only looks problematic under
    narrow airport-pair definitions, the case is fragile. If it looks
    problematic under both, the conclusion is robust to the defense's
    preferred framing.
    """
    rows = []
    for m in routes:
        try:
            a = hmt_route(panel_airport, m, parties, margin)
            rows.append({"route": m, "definition": "Airport pair",
                         "party_share": a["group_share"],
                         "actual_loss": a["actual_loss"],
                         "critical_loss": a["critical_loss"],
                         "is_market": a["is_relevant_market"]})
        except ValueError:
            continue
    return pd.DataFrame(rows)
