"""JetBlue/Spirit merger analysis - the four tools of merger review.

db1b           DOT DB1B acquisition, cleaning, and route-panel construction
concentration  HHI screens under the 2023 DOJ/FTC Merger Guidelines
marketdef      Hypothetical monopolist test (SSNIP / critical loss)
unilateral     Diversion ratios, GUPPI, UPP
simulation     Calibrated logit Bertrand-Nash merger simulation
"""
from .db1b import load_panel, build_route_panel, clean_db1b
from .concentration import route_hhi, merger_screen, summarize_screen
from .marketdef import (critical_loss, hmt_route, smallest_market,
                        compare_definitions)
from .unilateral import (diversion_proportional, diversion_ulcc_weighted,
                         guppi, upp, screen_all_routes)
from .simulation import LogitMarket, simulate_route, simulate_all

__all__ = ["load_panel", "build_route_panel", "clean_db1b", "route_hhi",
           "merger_screen", "summarize_screen", "critical_loss",
           "hmt_route", "smallest_market", "compare_definitions",
           "diversion_proportional", "diversion_ulcc_weighted", "guppi",
           "upp", "screen_all_routes", "LogitMarket", "simulate_route",
           "simulate_all"]
__version__ = "1.0.0"
