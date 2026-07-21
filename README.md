# Merger Competitive Effects Analysis — JetBlue / Spirit

A route-level competitive effects analysis of JetBlue's proposed
acquisition of Spirit Airlines, built on **26 million real DOT ticket
records**, working the four standard tools of merger review —
concentration screens, market definition, unilateral effects, and
merger simulation — against the case DOJ actually brought and won.

![Exhibits](output/exhibits.png)

## The business problem

*United States v. JetBlue Airways Corp.* (D. Mass., filed March 2023;
decided January 16, 2024). JetBlue proposed to acquire Spirit for $3.8
billion. DOJ and a coalition of states sued to block it, and Judge
William Young permanently enjoined the deal.

DOJ's theory: Spirit is the largest ultra-low-cost carrier in the U.S.
and disciplines fares wherever it flies. JetBlue planned to reconfigure
Spirit's aircraft and charge JetBlue's higher fares. On concentrated
routes where both carriers competed, eliminating Spirit would raise
prices for exactly the travelers least able to absorb them.

The case turned on **specific overlapping routes**, not the national
market — two airlines can look small nationally while dominating dozens
of individual city pairs. That is how airline merger analysis works,
and it is how this analysis is structured.

The fare data shows why the theory had force:

| Carrier | Avg. one-way fare | Routes |
|---|---|---|
| Allegiant (G4) | $87 | 439 |
| Frontier (F9) | $121 | 722 |
| **Spirit (NK)** | **$140** | 548 |
| Southwest (WN) | $228 | 2,127 |
| **JetBlue (B6)** | **$247** | 469 |
| American (AA) | $298 | 2,512 |
| United (UA) | $311 | 2,190 |
| Delta (DL) | $321 | 2,525 |

JetBlue sits *between* Southwest and the legacy carriers — it is not a
low-fare carrier in the way Spirit is. That gap is the competitive
question in the case.

## The four tools, in order

| Step | Tool | Question it answers |
|---|---|---|
| 1 | **HHI screens** (2023 DOJ/FTC Guidelines) | Where do the parties overlap, and which routes trigger the structural presumption? |
| 2 | **Hypothetical monopolist test** (SSNIP / critical loss) | Is a single route a relevant antitrust market, or must substitutes be included? |
| 3 | **Diversion ratios, GUPPI, UPP** | How much of JetBlue's lost traffic is recaptured by Spirit, and what pricing incentive does that create? |
| 4 | **Bertrand–Nash merger simulation** | How much do fares rise, and what efficiency credit would offset it? |

## Results

From 3,354 routes with meaningful traffic:

**Screening.** 113 overlap routes; **68 trigger the 2023 Guidelines
structural presumption**, carrying 3.1 million sampled passengers. Mean
post-merger HHI on flagged routes: 4,384.

The most concentrated overlaps are exactly the Northeast–Florida–Puerto
Rico markets at the center of DOJ's case:

| Route | JetBlue | Spirit | Combined | ΔHHI |
|---|---|---|---|---|
| MCO–PSE | 50.2% | 49.8% | **100.0%** | 5,000 |
| BOS–SJU | 61.7% | 28.0% | 89.6% | 3,451 |
| FLL–RIC | 28.9% | 54.3% | 83.1% | 3,132 |
| BQN–MCO | 43.6% | 34.2% | 77.8% | 2,979 |
| FLL–SJU | 53.7% | 27.6% | 81.3% | 2,960 |
| FLL–LGA | 36.4% | 34.4% | 70.8% | 2,507 |
| BOS–FLL | 54.6% | 18.1% | 72.7% | 1,975 |

Orlando–Ponce is a **merger to monopoly** — the two carriers are the
entire market.

**Market definition.** On every top overlap route, the two parties
alone satisfy the hypothetical monopolist test: predicted volume loss
of 0–7% against a 16.7% critical loss. On those routes the merger
combines the only two competitors in a properly defined market.

**Unilateral effects.** 60 of 113 overlap routes exceed the 5% GUPPI
investigation threshold; **39 exceed 10%**. Under ULCC-weighted
diversion reflecting DOJ's substitution theory, 52 exceed 10%. The
highest routes (BQN–FLL, FLL–STT, MCO–PSE) show diversion approaching
100% — JetBlue and Spirit are each other's only meaningful competitor.

**Simulation.** Passenger-weighted average fare increase of **+11.6%**
on flagged routes with no efficiencies. Crediting a 10% marginal cost
reduction — well above what merging parties typically verify — the
predicted increase is still **+9.0%**.

## Being honest about the assumptions

Every step embeds a judgment an opposing expert would probe, so each is
tested rather than buried:

- **Margin (25% assumed).** Airline variable margins aren't observable
  in DB1B. Sensitivity: the predicted fare effect ranges from +7.0% to
  +16.2% across 15–35% margins — the *sign* is robust, the magnitude
  isn't.
- **Logit demand imposes proportional diversion** (the IIA property) —
  precisely the assumption DOJ disputed, arguing Spirit's
  price-sensitive customers wouldn't trade up to legacy carriers. The
  simulation is therefore **conservative relative to the government's
  theory**, and ULCC-weighted diversion is reported alongside it as an
  explicit behavioral assumption rather than folded silently into one
  number.
- **Static analysis** — no entry, capacity response, or the seat
  reconfiguration the court weighed heavily.

## Data and cleaning

**DOT DB1B** (Airline Origin and Destination Survey), 2022 Q1–Q4 — a
free 10% sample of all domestic tickets, ~26 million records before
cleaning. This is the dataset airline merger economists have used for
decades.

Cleaning filters, all documented in `db1b.py` because each is
contestable:

| Filter | Rationale |
|---|---|
| Single-ticket domestic itineraries | Interline/international tickets have fare allocation problems |
| $20–$2,000 fare screen | Excludes award tickets, employee fares, data errors |
| Bulk fares dropped | Opaque wholesale contracts, not consumer prices |
| Non-directional airport pairs | Competition is on the city pair, not direction |
| Drop carrier code `99` | DB1B's unknown/other code — appears on ~20,000 routes, not a real carrier |
| **≥2,000 sampled passengers/year** | **The consequential one.** Median raw airport pair carries 26 passengers/year — a few connecting itineraries, not a market. Takes the panel from ~45,000 nominal routes to 3,354 genuine ones. A share screen can't do this: on a 3-ticket route, one carrier trivially holds 33%. |
| ≥1% within-route carrier share | Drops fringe presence, with shares renormalized after |

Roughly 95% of raw records survive the fare and bulk screens; the route
count falls sharply because most airport pairs carry negligible traffic.

## Limitations

Stated affirmatively rather than buried — the margin assumption, the
IIA property of logit demand, the static framing, the cleaning
thresholds above, the 2022 data vintage, and the fare basis (DB1B
market fares are used as reported; if round-trip itineraries are not
halved, fare *levels* are overstated, though GUPPI depends on fare
*ratios* and the simulation calibrates off margins and shares, so the
direction and significance of results are unaffected).

## Run it

```bash
pip install -r requirements.txt
python -m mergeranalysis.db1b --build    # ~800MB download, 20-40 min
python run_analysis.py
```

## References

- 2023 DOJ/FTC Merger Guidelines (Guideline 1; Appendix 2 on market
  definition)
- Farrell & Shapiro (2010), "Antitrust Evaluation of Horizontal
  Mergers: An Economic Alternative to Market Definition"
- Werden & Froeb, calibrated-demand merger simulation
- *United States v. JetBlue Airways Corp.*, No. 1:23-cv-10511
  (D. Mass. Jan. 16, 2024)
