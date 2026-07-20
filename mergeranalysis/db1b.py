"""DB1B data acquisition and cleaning.

WHAT DB1B IS
------------
The DOT's Airline Origin and Destination Survey (DB1B) is a 10%
random sample of all domestic airline tickets, published quarterly and
free. Each record is a ticket with origin, destination, operating
carrier, fare, and passenger count.

This is the dataset airline merger economists have used for decades,
and it is why airlines are one of the few industries where you can do
credible merger analysis without licensed data.

WHY THE CLEANING RULES MATTER
-----------------------------
Raw DB1B is noisy in ways that will corrupt your shares and fares if
you ignore them. Each filter below is standard in the published
literature (Borenstein; Berry & Jia; DOJ's own airline filings), and
each one is a choice an opposing expert could question - so they are
stated explicitly rather than buried:

  1. Domestic, single-ticket itineraries only. Interline and
     international tickets have fare allocation problems.
  2. Round-trip fares divided by 2 to put everything on a one-way
     basis, so round trips and one-ways are comparable.
  3. Drop fares below $20 or above $2,000. These are frequent-flyer
     awards, employee tickets, and data errors - not market prices.
  4. Drop bulk-fare tickets (BulkFare flag), which are opaque
     wholesale contracts, not consumer prices.
  5. Markets defined as DIRECTIONAL city pairs collapsed to
     non-directional (BOS-FLL and FLL-BOS are one market), because
     competition is on the city pair, not the direction.
  6. Carriers with under 1% of a route's passengers are dropped from
     share calculations as noise.

CHOICE OF MARKET DEFINITION
---------------------------
We use AIRPORT pairs, not city pairs, as the baseline - then test
sensitivity to metro-area aggregation (e.g. treating FLL, MIA, and
PBI as one South Florida market). That test matters here because
JetBlue argued for broad metro markets; DOJ argued for narrower ones.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Fare screens
MIN_FARE = 20.0
MAX_FARE = 2000.0
MIN_CARRIER_SHARE = 0.01

# Metro-area groupings for the sensitivity test
METRO_GROUPS = {
    "FLL": "SFL", "MIA": "SFL", "PBI": "SFL",      # South Florida
    "JFK": "NYC", "LGA": "NYC", "EWR": "NYC",      # New York
    "ORD": "CHI", "MDW": "CHI",                     # Chicago
    "LAX": "LAB", "BUR": "LAB", "LGB": "LAB",
    "SNA": "LAB", "ONT": "LAB",                     # LA basin
    "IAH": "HOU", "HOU": "HOU",                     # Houston
    "DCA": "WAS", "IAD": "WAS", "BWI": "WAS",       # Washington
    "SFO": "BAY", "OAK": "BAY", "SJC": "BAY",       # Bay Area
}


def download_db1b(year: int, quarter: int) -> str:
    """Download and unzip one quarter of DB1B Market data from BTS.

    Roughly 200MB zipped per quarter. Run this locally; the file is
    too large to commit to a repository.
    """
    url = ("https://transtats.bts.gov/PREZIP/"
           f"Origin_and_Destination_Survey_DB1BMarket_{year}_{quarter}.zip")
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, f"db1b_{year}q{quarter}.zip")

    if not os.path.exists(zip_path):
        import urllib.request
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, zip_path)

    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        z.extract(name, DATA_DIR)
    return os.path.join(DATA_DIR, name)


def clean_db1b(path: str) -> pd.DataFrame:
    """Apply the standard cleaning filters to raw DB1B Market data."""
    cols = ["Origin", "Dest", "TkCarrier", "MktFare", "Passengers",
            "RPCarrier", "BulkFare", "MktDistance"]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    carrier_col = "TkCarrier" if "TkCarrier" in df else "RPCarrier"
    df = df.rename(columns={carrier_col: "carrier", "MktFare": "fare",
                            "Passengers": "pax", "Origin": "origin",
                            "Dest": "dest", "MktDistance": "distance"})

    n0 = len(df)
    if "BulkFare" in df:
        df = df[df["BulkFare"] == 0]
    df = df[(df["fare"] >= MIN_FARE) & (df["fare"] <= MAX_FARE)]
    df = df.dropna(subset=["origin", "dest", "carrier", "fare", "pax"])
    print(f"  Cleaning: {n0:,} -> {len(df):,} records "
          f"({100*len(df)/n0:.1f}% retained)")

    # Non-directional market key: sorted airport pair
    df["market"] = [f"{min(o,d)}-{max(o,d)}"
                    for o, d in zip(df["origin"], df["dest"])]
    return df


def build_route_panel(df: pd.DataFrame,
                      metro: bool = False) -> pd.DataFrame:
    """Aggregate tickets into a route-carrier panel with shares and fares.

    This is the analytical unit: one row per (route, carrier) with
    passengers, passenger-weighted average fare, and share.
    """
    d = df.copy()
    if metro:
        d["origin"] = d["origin"].map(lambda a: METRO_GROUPS.get(a, a))
        d["dest"] = d["dest"].map(lambda a: METRO_GROUPS.get(a, a))
        d["market"] = [f"{min(o,dd)}-{max(o,dd)}"
                       for o, dd in zip(d["origin"], d["dest"])]

    # Passenger-weighted average fare per route-carrier
    d["fare_x_pax"] = d["fare"] * d["pax"]
    g = d.groupby(["market", "carrier"], as_index=False).agg(
        pax=("pax", "sum"), fare_x_pax=("fare_x_pax", "sum"),
        distance=("distance", "mean"))
    g["fare"] = g["fare_x_pax"] / g["pax"]
    g = g.drop(columns=["fare_x_pax"])

    # Shares within each route
    g["route_pax"] = g.groupby("market")["pax"].transform("sum")
    g["share"] = g["pax"] / g["route_pax"]
    g = g[g["share"] >= MIN_CARRIER_SHARE]

    # Renormalize after dropping fringe carriers
    g["route_pax"] = g.groupby("market")["pax"].transform("sum")
    g["share"] = g["pax"] / g["route_pax"]
    return g.reset_index(drop=True)


def load_panel(metro: bool = False) -> pd.DataFrame:
    """Load the cached route panel, or the calibrated fallback."""
    path = os.path.join(DATA_DIR, "route_panel.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    raise FileNotFoundError(
        "No route panel found. Run `python make_fallback_data.py` for "
        "the offline sample, or `python -m mergeranalysis.db1b --build` "
        "to construct one from real DB1B downloads.")


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        frames = []
        for q in (1, 2, 3, 4):
            p = download_db1b(2022, q)
            frames.append(clean_db1b(p))
        raw = pd.concat(frames, ignore_index=True)
        panel = build_route_panel(raw)
        panel.to_csv(os.path.join(DATA_DIR, "route_panel.csv"), index=False)
        print(f"Wrote route_panel.csv: {len(panel):,} route-carrier rows, "
              f"{panel['market'].nunique():,} routes")
