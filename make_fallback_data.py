"""Generate the offline fallback route panel.

IMPORTANT - READ THIS
---------------------
This builds a SYNTHETIC route panel calibrated to publicly known
features of the JetBlue and Spirit networks as of 2022:

  * JetBlue hubs: BOS, JFK, FLL, MCO, LGA
  * Spirit hubs:  FLL, MCO, LAS, DTW, ORD
  * Heavy head-to-head overlap in Northeast-Florida markets, which is
    where DOJ concentrated its case
  * ULCC fares roughly 30-40% below legacy fares on the same route
  * Legacy carriers (AA, DL, UA) present on most large routes

It exists ONLY so the repository runs end to end without downloading
~800MB of DB1B files. It is NOT real DOT data, and no number produced
from it should be presented as a real estimate.

For the real thing:
    python -m mergeranalysis.db1b --build
which downloads the four 2022 quarters from BTS and rebuilds
data/route_panel.csv from actual ticket records.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)

JB_CITIES = ["BOS", "JFK", "FLL", "MCO", "LGA", "TPA", "RSW", "PBI",
             "SJU", "LAX", "SFO", "AUS", "RDU", "BWI", "DCA"]
NK_CITIES = ["FLL", "MCO", "LAS", "DTW", "ORD", "TPA", "ATL", "BWI",
             "DFW", "LAX", "EWR", "PHL", "MSP", "IAH", "CLE"]
OTHER = ["ATL", "DEN", "PHX", "SEA", "MSY", "CLT", "SAN", "MIA",
         "PIT", "STL", "CMH", "IND", "MKE", "SAT", "BNA"]

ALL_CITIES = sorted(set(JB_CITIES + NK_CITIES + OTHER))

# Base one-way fares by carrier type
FARE_BASE = {"B6": 175, "NK": 105, "AA": 235, "DL": 245, "UA": 240,
             "WN": 190, "F9": 110, "AS": 225, "G4": 100}

rows = []
markets = set()

# Build a route universe
for i, a in enumerate(ALL_CITIES):
    for b in ALL_CITIES[i + 1:]:
        if rng.random() > 0.72:
            continue
        markets.add(f"{min(a,b)}-{max(a,b)}")

for m in sorted(markets):
    a, b = m.split("-")
    dist = float(rng.uniform(300, 2600))

    carriers = []
    # JetBlue present if both endpoints in its network (mostly)
    if a in JB_CITIES and b in JB_CITIES and rng.random() < 0.88:
        carriers.append("B6")
    elif rng.random() < 0.16:
        carriers.append("B6")
    # Spirit similarly
    if a in NK_CITIES and b in NK_CITIES and rng.random() < 0.85:
        carriers.append("NK")
    elif rng.random() < 0.14:
        carriers.append("NK")
    # Legacies and others
    for c, prob in [("AA", 0.62), ("DL", 0.60), ("UA", 0.48),
                    ("WN", 0.42), ("F9", 0.18), ("AS", 0.10)]:
        if rng.random() < prob:
            carriers.append(c)

    if len(carriers) < 2:
        continue

    route_pax = float(rng.lognormal(np.log(28000), 0.85))
    raw_w = rng.dirichlet(np.ones(len(carriers)) * rng.uniform(1.2, 3.5))

    for c, w in zip(carriers, raw_w):
        pax = route_pax * w
        if pax < 500:
            continue
        base = FARE_BASE[c]
        # Distance and route-level noise
        fare = base * (0.55 + 0.00042 * dist) * rng.lognormal(0, 0.13)
        rows.append({"market": m, "carrier": c, "pax": round(pax),
                     "fare": round(fare, 2), "distance": round(dist, 1)})

panel = pd.DataFrame(rows)
panel["route_pax"] = panel.groupby("market")["pax"].transform("sum")
panel["share"] = panel["pax"] / panel["route_pax"]
panel = panel[panel["share"] >= 0.01].copy()
panel["route_pax"] = panel.groupby("market")["pax"].transform("sum")
panel["share"] = panel["pax"] / panel["route_pax"]

panel.to_csv("data/route_panel.csv", index=False)

both = panel.pivot_table(index="market", columns="carrier",
                         values="share", fill_value=0.0)
overlap = both[(both.get("B6", 0) > 0) & (both.get("NK", 0) > 0)]

print(f"Wrote data/route_panel.csv")
print(f"  {panel['market'].nunique():,} routes, "
      f"{len(panel):,} route-carrier rows")
print(f"  B6/NK overlap routes: {len(overlap):,}")
print(f"  Mean B6 fare ${panel[panel.carrier=='B6'].fare.mean():.0f}, "
      f"NK fare ${panel[panel.carrier=='NK'].fare.mean():.0f}, "
      f"AA fare ${panel[panel.carrier=='AA'].fare.mean():.0f}")
print("  NOTE: synthetic fallback - see docstring")
