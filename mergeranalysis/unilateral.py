"""Step 3: Unilateral effects - diversion, GUPPI, and UPP.

THE INTUITION
-------------
Before the merger, when JetBlue raises fares on BOS-FLL, it loses
customers - some to Spirit, some to Delta, some to not flying. Losing
them is costly, which disciplines pricing.

After the merger, customers who switch to Spirit are no longer lost:
they stay inside the merged firm. That removes part of the cost of
raising fares, creating upward pricing pressure. No coordination, no
conspiracy - just changed incentives. This is a UNILATERAL effect.

DIVERSION RATIO
    D(A->B) = fraction of A's lost sales that go to B.
    Under a logit model this is B's share renormalized by everyone
    except A: D(A->B) = s_B / (1 - s_A).

GUPPI
    GUPPI_A = D(A->B) * m_B * (p_B / p_A)

    Reads as: the share of lost sales recaptured, times the profit
    earned on each recaptured sale, scaled into A's own price units.
    Agencies treat GUPPIs above roughly 5% as worth investigating and
    above 10% as serious.

UPP (Farrell-Shapiro)
    UPP_A = D(A->B) * (p_B - c_B) - E * c_A

    Same idea in dollars, minus an efficiency credit E. Efficiencies
    push prices DOWN; recapture pushes them UP; UPP is the net.

THE ULCC ASYMMETRY IN THIS CASE
-------------------------------
Standard logit assumes diversion is proportional to share - if Delta
is big, most diverted customers go to Delta. DOJ's theory was that
this understates the harm, because Spirit's ultra-low-cost customers
are price-sensitive and would NOT trade up to a legacy carrier; they
would stop flying or drive.

That means diversion between JetBlue and Spirit is HIGHER than share
alone implies. We implement both:

  * proportional (logit) diversion - the conservative baseline
  * ULCC-weighted diversion - upweighting substitution among low-cost
    carriers, which is DOJ's theory

Reporting both, and being explicit that the second embeds a
behavioral assumption, is the honest way to present a contested
theory - you show what the conclusion depends on rather than hiding
the assumption inside a single number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Carriers generally classified as ultra-low-cost (ULCC) or low-cost
ULCC = {"NK", "F9", "G4", "SY"}          # Spirit, Frontier, Allegiant, Sun Country
LCC = {"B6", "WN"}                        # JetBlue, Southwest
LEGACY = {"AA", "DL", "UA", "AS"}


def diversion_proportional(panel: pd.DataFrame, market: str,
                           from_c: str, to_c: str) -> float:
    """Logit diversion: D(A->B) = s_B / (1 - s_A)."""
    r = panel[panel["market"] == market].set_index("carrier")["share"]
    if from_c not in r or to_c not in r:
        return 0.0
    s_a, s_b = float(r[from_c]), float(r[to_c])
    return s_b / (1.0 - s_a) if s_a < 1.0 else 0.0


def diversion_ulcc_weighted(panel: pd.DataFrame, market: str,
                            from_c: str, to_c: str,
                            ulcc_weight: float = 2.0) -> float:
    """Diversion with stronger substitution within the low-fare segment.

    Implements DOJ's theory: a Spirit customer is more likely to
    switch to another low-fare carrier than to a legacy carrier.
    `ulcc_weight` is the relative preference multiplier - it is an
    ASSUMPTION, not an estimate, and the sensitivity analysis varies it.
    """
    r = panel[panel["market"] == market].set_index("carrier")["share"]
    if from_c not in r or to_c not in r:
        return 0.0

    def seg(c):
        return "low" if c in ULCC or c in LCC else "other"

    from_seg = seg(from_c)
    weights = {}
    for c, s in r.items():
        if c == from_c:
            continue
        w = ulcc_weight if seg(c) == from_seg else 1.0
        weights[c] = w * float(s)

    total = sum(weights.values())
    return weights.get(to_c, 0.0) / total if total else 0.0


def guppi(panel: pd.DataFrame, market: str, from_c: str, to_c: str,
          margin: float = 0.25, method: str = "proportional",
          ulcc_weight: float = 2.0) -> float:
    """GUPPI for `from_c` acquiring `to_c` on one route."""
    if method == "proportional":
        d = diversion_proportional(panel, market, from_c, to_c)
    else:
        d = diversion_ulcc_weighted(panel, market, from_c, to_c,
                                    ulcc_weight)

    r = panel[panel["market"] == market].set_index("carrier")["fare"]
    if from_c not in r or to_c not in r:
        return 0.0
    return float(d * margin * (r[to_c] / r[from_c]))


def upp(panel: pd.DataFrame, market: str, from_c: str, to_c: str,
        margin: float = 0.25, efficiency: float = 0.0,
        method: str = "proportional") -> float:
    """Farrell-Shapiro UPP in dollars per ticket."""
    if method == "proportional":
        d = diversion_proportional(panel, market, from_c, to_c)
    else:
        d = diversion_ulcc_weighted(panel, market, from_c, to_c)

    r = panel[panel["market"] == market].set_index("carrier")["fare"]
    if from_c not in r or to_c not in r:
        return 0.0
    p_a, p_b = float(r[from_c]), float(r[to_c])
    c_a = p_a * (1.0 - margin)      # implied marginal cost of A
    c_b = p_b * (1.0 - margin)      # implied marginal cost of B
    return float(d * (p_b - c_b) - efficiency * c_a)


def screen_all_routes(panel: pd.DataFrame, firm_a: str, firm_b: str,
                      margin: float = 0.25,
                      method: str = "proportional") -> pd.DataFrame:
    """GUPPI for every overlap route, both directions."""
    both = (panel.pivot_table(index="market", columns="carrier",
                              values="share", fill_value=0.0))
    if firm_a not in both or firm_b not in both:
        return pd.DataFrame()
    overlap = both[(both[firm_a] > 0) & (both[firm_b] > 0)].index

    rows = []
    for m in overlap:
        g_ab = guppi(panel, m, firm_a, firm_b, margin, method)
        g_ba = guppi(panel, m, firm_b, firm_a, margin, method)
        d_ab = (diversion_proportional(panel, m, firm_a, firm_b)
                if method == "proportional"
                else diversion_ulcc_weighted(panel, m, firm_a, firm_b))
        pax = float(panel[panel["market"] == m]["pax"].sum())
        rows.append({"market": m, "diversion_a_to_b": d_ab,
                     "guppi_a": g_ab, "guppi_b": g_ba,
                     "max_guppi": max(g_ab, g_ba), "route_pax": pax})
    return (pd.DataFrame(rows).sort_values("max_guppi", ascending=False)
            .reset_index(drop=True))
