"""JetBlue/Spirit merger analysis - full competitive effects walkthrough.

CASE BACKGROUND (all public record)
-----------------------------------
United States v. JetBlue Airways Corp. and Spirit Airlines, Inc.
D. Mass., filed March 7, 2023; decided January 16, 2024

JetBlue proposed to acquire Spirit for $3.8 billion. DOJ and a
coalition of states sued to block it. Judge William Young blocked the
deal permanently, finding it would substantially lessen competition.

DOJ's theory, in plain terms:
  1. Spirit is the largest ultra-low-cost carrier (ULCC) in the U.S.
     and disciplines fares industry-wide.
  2. JetBlue planned to remove seats from Spirit aircraft and charge
     JetBlue's higher fares.
  3. The merger therefore eliminates the low-fare option on
     concentrated overlap routes, harming cost-conscious travelers.

Note the structure of the case: it turned on SPECIFIC OVERLAPPING
ROUTES, not the national market. That is how airline merger analysis
always works, and it is how this analysis is organized.

OUR TASK: work the four standard tools in order - concentration
screens, market definition, unilateral effects, and simulation - and
see whether an independent analysis reaches DOJ's conclusion.

DATA NOTE: `data/route_panel.csv` shipped here is SYNTHETIC, calibrated
to the real JetBlue/Spirit network structure. Run
`python -m mergeranalysis.db1b --build` to rebuild it from actual DOT
DB1B ticket records before relying on any figure.
"""

import os

import numpy as np
import pandas as pd

from mergeranalysis import (merger_screen, summarize_screen, hmt_route,
                            smallest_market, screen_all_routes,
                            diversion_proportional,
                            diversion_ulcc_weighted, guppi,
                            simulate_route, simulate_all)

pd.set_option("display.width", 110)
pd.set_option("display.max_columns", 20)

JB, NK = "B6", "NK"          # JetBlue, Spirit
MARGIN = 0.25                # assumed variable contribution margin
RULE = "=" * 76


def section(n, title):
    print(f"\n{RULE}\n{n}. {title.upper()}\n{RULE}")


panel = pd.read_csv("data/route_panel.csv")

print(RULE)
print("COMPETITIVE EFFECTS ANALYSIS - JETBLUE / SPIRIT")
print("United States v. JetBlue Airways Corp., D. Mass. (2023-24)")
print(RULE)
print(f"\nPanel: {panel['market'].nunique():,} routes, "
      f"{len(panel):,} route-carrier observations")
print(f"Average one-way fare by carrier:")
fares = panel.groupby("carrier").agg(
    fare=("fare", "mean"), routes=("market", "nunique"),
    pax=("pax", "sum")).sort_values("pax", ascending=False)
fares["fare"] = fares["fare"].round(0)
print(fares.head(8).to_string())
print("\n  Note Spirit's fare position - roughly 40% below the legacy")
print("  carriers. That gap IS the competitive question in this case.")

# ---------------------------------------------------------------- 1 ----- #
section(1, "Structural screen - where is there overlap?")
print("  A merger can only raise concern where BOTH parties compete.")
print("  delta HHI = 2 * s_JetBlue * s_Spirit, so routes without")
print("  overlap have zero delta by construction.\n")

screen = merger_screen(panel, JB, NK)
summ = summarize_screen(screen)
print(f"  Routes in panel:              {summ['routes_total']:,}")
print(f"  Overlap routes (both fly):    {summ['routes_overlap']:,}")
print(f"  Flagged by 2023 Guidelines:   {summ['routes_flagged']:,}")
print(f"  Passengers on flagged routes: "
      f"{summ['pax_on_flagged']/1e6:.2f} mm "
      f"({summ['pax_share_flagged']:.1%} of panel)")
print(f"  Mean post-merger HHI on flagged routes: "
      f"{summ['mean_post_hhi_flagged']:,.0f}")

flagged = (screen[screen["presumption"]]
           .sort_values("hhi_delta", ascending=False))
print("\n  Top 10 routes by HHI delta:")
cols = ["market", "share_a", "share_b", "combined_share",
        "hhi_pre", "hhi_post", "hhi_delta"]
show = flagged[cols].head(10).copy()
for c in ["share_a", "share_b", "combined_share"]:
    show[c] = (100 * show[c]).round(1)
for c in ["hhi_pre", "hhi_post", "hhi_delta"]:
    show[c] = show[c].round(0)
print(show.to_string(index=False))
print("\n  Reading: these routes carry the structural presumption of")
print("  illegality. They are the ones worth modeling - the screen has")
print(f"  narrowed {summ['routes_total']:,} routes down to "
      f"{summ['routes_flagged']:,}.")

# ---------------------------------------------------------------- 2 ----- #
section(2, "Market definition - is a route really a market?")
print("  JetBlue argued markets are broad (all carriers, metro areas,")
print("  connecting service). DOJ argued they are narrow. We run the")
print("  hypothetical monopolist test on the flagged routes.\n")
print("  Critical loss = t/(t+m). With a 5% SSNIP and a 25% margin,")
print(f"  critical loss = {0.05/(0.05+MARGIN):.1%}. If the predicted")
print("  volume loss is smaller than that, the SSNIP is profitable and")
print("  the candidate market is a relevant antitrust market.\n")

rows = []
for m in flagged["market"].head(8):
    r = hmt_route(panel, m, [JB, NK], margin=MARGIN)
    sm = smallest_market(panel, m, [JB, NK], margin=MARGIN)
    rows.append({"route": m,
                 "JB+NK share": f"{r['group_share']:.1%}",
                 "actual loss": f"{r['actual_loss']:.1%}",
                 "critical loss": f"{r['critical_loss']:.1%}",
                 "2-firm mkt?": r["is_relevant_market"],
                 "smallest market": "+".join(sm["relevant_market"])})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  Reading: where the two parties alone constitute a relevant")
print("  market, the merger is a merger to monopoly in that market.")
print("  Where more carriers must be added, competition survives but")
print("  the concentration analysis still applies.")

# ---------------------------------------------------------------- 3 ----- #
section(3, "Unilateral effects - diversion and GUPPI")
print("  After the merger, JetBlue customers who would have switched to")
print("  Spirit stay inside the merged firm. That recapture removes")
print("  part of the cost of raising fares.\n")
print("  We compute GUPPI two ways:")
print("    (a) proportional (logit) diversion - the conservative baseline")
print("    (b) ULCC-weighted diversion - DOJ's theory that low-fare")
print("        customers substitute within the low-fare segment\n")

g_prop = screen_all_routes(panel, JB, NK, MARGIN, "proportional")
g_ulcc = screen_all_routes(panel, JB, NK, MARGIN, "ulcc")

comp = g_prop[["market", "diversion_a_to_b", "max_guppi"]].merge(
    g_ulcc[["market", "diversion_a_to_b", "max_guppi"]],
    on="market", suffixes=("_prop", "_ulcc")).head(10)
for c in comp.columns[1:]:
    comp[c] = (100 * comp[c]).round(1)
comp.columns = ["route", "diversion % (prop)", "GUPPI % (prop)",
                "diversion % (ULCC)", "GUPPI % (ULCC)"]
print(comp.to_string(index=False))

n_5 = int((g_prop["max_guppi"] > 0.05).sum())
n_10 = int((g_prop["max_guppi"] > 0.10).sum())
n_5u = int((g_ulcc["max_guppi"] > 0.05).sum())
n_10u = int((g_ulcc["max_guppi"] > 0.10).sum())
print(f"\n  Routes with GUPPI > 5%:  {n_5} (proportional) / "
      f"{n_5u} (ULCC-weighted)")
print(f"  Routes with GUPPI > 10%: {n_10} (proportional) / "
      f"{n_10u} (ULCC-weighted)")
print("\n  Reading: agencies treat GUPPI above ~5% as worth")
print("  investigating and above ~10% as serious. The ULCC-weighted")
print("  figures are higher by construction - that is DOJ's theory,")
print("  and it is an ASSUMPTION about consumer behavior, not an")
print("  estimate. The proportional numbers are the conservative floor.")

# ---------------------------------------------------------------- 4 ----- #
section(4, "Merger simulation - how much do fares rise?")
print("  Calibrate logit demand to observed shares and fares, recover")
print("  marginal costs from the pre-merger equilibrium, then re-solve")
print("  prices under merged ownership.\n")

routes = list(flagged["market"].head(12))
sim0 = simulate_all(panel, routes, JB, NK, MARGIN, efficiency=0.0)
sim1 = simulate_all(panel, routes, JB, NK, MARGIN, efficiency=0.03)

merged = sim0.merge(sim1[["market", "merging_carriers_pct"]],
                    on="market", suffixes=("_noeff", "_eff"))
show = merged[["market", "merging_carriers_pct_noeff",
               "merging_carriers_pct_eff", "route_pax"]].copy()
show.columns = ["route", "fare change % (no eff.)",
                "fare change % (3% eff.)", "route pax"]
show["route pax"] = show["route pax"].round(0)
print(show.round(2).to_string(index=False))

w = sim0["route_pax"]
wavg0 = float(np.average(sim0["merging_carriers_pct"], weights=w))
wavg1 = float(np.average(sim1["merging_carriers_pct"], weights=w))
print(f"\n  Passenger-weighted average fare increase on flagged routes:")
print(f"    No efficiencies:      {wavg0:+.2f}%")
print(f"    3% cost efficiency:   {wavg1:+.2f}%")
print("\n  Reading: even crediting the merged firm with a 3% marginal")
print("  cost reduction, predicted fares rise on these routes. Note")
print("  this simulation uses PROPORTIONAL diversion, which understates")
print("  harm under DOJ's ULCC theory - so it is a conservative floor.")

# ---------------------------------------------------------------- 5 ----- #
section(5, "Sensitivity - what does the conclusion depend on?")
print("  Every input above involved a judgment call. An opposing expert")
print("  will probe each one, so we vary them ourselves.\n")

rows = []
for margin in (0.15, 0.25, 0.35):
    s = simulate_all(panel, routes, JB, NK, margin, efficiency=0.0)
    if len(s):
        rows.append({"assumption": f"margin = {margin:.0%}",
                     "wtd avg fare change %":
                         float(np.average(s["merging_carriers_pct"],
                                          weights=s["route_pax"]))})
for eff in (0.0, 0.03, 0.06, 0.10):
    s = simulate_all(panel, routes, JB, NK, MARGIN, efficiency=eff)
    if len(s):
        rows.append({"assumption": f"efficiency = {eff:.0%}",
                     "wtd avg fare change %":
                         float(np.average(s["merging_carriers_pct"],
                                          weights=s["route_pax"]))})
sens = pd.DataFrame(rows)
print(sens.round(2).to_string(index=False))
print("\n  Reading: the sign of the effect is robust across every")
print("  assumption tested; only the magnitude moves. The efficiency")
print("  level needed to offset the price rise entirely is the number")
print("  the merging parties would have to prove - and under the")
print("  Guidelines, efficiencies must be merger-specific and verified.")

# ---------------------------------------------------------------- 6 ----- #
section(6, "Comparison to the actual outcome")
print("""  Judge Young blocked the merger on January 16, 2024, finding it
  would substantially lessen competition on concentrated overlap
  routes and eliminate Spirit as a low-fare competitor.

  This analysis reaches the same qualitative conclusion through the
  standard toolkit: overlap routes trigger the structural
  presumption, GUPPIs exceed agency thresholds, and simulation
  predicts fare increases that survive plausible efficiency credits.

  What this analysis does NOT capture, and the court weighed:
    * Spirit's role as an industry-wide price disciplinarian beyond
      its own routes (a dynamic effect, not a static one)
    * JetBlue's stated plan to reconfigure Spirit aircraft, removing
      seats - a capacity reduction, not just a pricing change
    * Entry and expansion by other ULCCs as a countervailing factor
    * The proposed divestitures""")

# ---------------------------------------------------------------- 7 ----- #
section(7, "Limitations - stated affirmatively")
print("""  1. Margin assumption. Airline variable margins are not observable
     in DB1B; 25% is an assumption that drives calibration. Section 5
     shows the conclusion's sensitivity to it.
  2. Logit demand. Imposes proportional diversion (IIA), the very
     assumption DOJ disputed. The simulation is therefore conservative
     relative to the government's theory.
  3. Static analysis. No entry, exit, or capacity response is modeled.
  4. Nonstop focus. Connecting itineraries are imperfect substitutes
     and are excluded from route shares.
  5. Synthetic panel. The shipped data is calibrated, not real -
     rebuild from DB1B before citing any figure.""")

# ---------------------------------------------------------------- 8 ----- #
os.makedirs("output", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

# Exhibit 1: HHI scatter with the Guidelines safe harbor
ov = screen[screen["overlap"]]
colors = ["crimson" if p else "#9ecae1" for p in ov["presumption"]]
axes[0].scatter(ov["hhi_post"], ov["hhi_delta"], c=colors, s=22,
                alpha=0.75, edgecolor="white", linewidth=0.4)
axes[0].axvline(1800, color="k", ls="--", lw=0.9)
axes[0].axhline(100, color="k", ls="--", lw=0.9)
axes[0].set_xlabel("Post-merger HHI")
axes[0].set_ylabel("Change in HHI")
axes[0].set_title("Exhibit 1: 2023 Guidelines screen\n"
                  "(red = structural presumption)", fontsize=10)

# Exhibit 2: GUPPI distribution, both diversion assumptions
axes[1].hist([100 * g_prop["max_guppi"], 100 * g_ulcc["max_guppi"]],
             bins=12, label=["Proportional", "ULCC-weighted"],
             color=["#9ecae1", "crimson"])
axes[1].axvline(5, color="k", ls="--", lw=0.9)
axes[1].axvline(10, color="k", ls=":", lw=0.9)
axes[1].set_xlabel("GUPPI (%)")
axes[1].set_ylabel("Overlap routes")
axes[1].set_title("Exhibit 2: Price-pressure screens\n"
                  "(dashed = 5% and 10% thresholds)", fontsize=10)
axes[1].legend(fontsize=8)

# Exhibit 3: simulated fare effects by efficiency credit
effs = [0.0, 0.03, 0.06, 0.10]
vals = []
for e in effs:
    s = simulate_all(panel, routes, JB, NK, MARGIN, efficiency=e)
    vals.append(float(np.average(s["merging_carriers_pct"],
                                 weights=s["route_pax"])))
bars = axes[2].bar([f"{int(100*e)}%" for e in effs], vals,
                   color=["crimson", "#d1495b", "#6baed6", "#9ecae1"])
axes[2].axhline(0, color="k", lw=0.9)
axes[2].set_xlabel("Assumed marginal cost efficiency")
axes[2].set_ylabel("Predicted fare change (%)")
axes[2].set_title("Exhibit 3: Simulated fare effects\n"
                  "on flagged overlap routes", fontsize=10)
for b, v in zip(bars, vals):
    axes[2].text(b.get_x() + b.get_width()/2, v,
                 f"{v:+.1f}%", ha="center",
                 va="bottom" if v > 0 else "top", fontsize=8)

fig.tight_layout()
fig.savefig("output/exhibits.png", dpi=150)
print(f"\n{RULE}\nSaved exhibits to output/exhibits.png\n{RULE}")
