"""Prospect trade value from a scouting grade (Future Value, 20-80 scale).

You choose the grade. The app turns it into surplus value with the same
machinery used for major leaguers:

1. **Grade -> WAR if he pans out.** On the FanGraphs FV scale a 50 is an
   average regular (~2 WAR), 60 an all-star-level player (~4 WAR), and
   70 a perennial star (~6 WAR).
2. **Bust risk.** Most prospects never reach their grade. Each grade gets a
   realization factor, the expected share of that WAR actually delivered.
   Pitchers carry extra risk (injuries), and every extra year away from the
   majors adds more. The factors are tuned so a hitter about a year away lands
   near rough public prospect valuations: about $110M for a 70, $75M for a 65,
   $50M for a 60, $30M for a 55, $15M for a 50, $5M for a 45 and ~$1M for a 40.
   Those targets approximate published prospect surplus-value research. They
   aren't exact figures, so adjust REALIZATION here if you prefer others.
3. **Team control.** Six seasons from his debut year (ETA), with salaries from
   the comparable-player contract model (league minimum, then arbitration
   raises) and the same $/WAR, salary growth and 8% discount as everyone else.

The stats shown next to each prospect are a reference for choosing a grade,
not an input to the value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import contracts, value

GRADES = [40, 45, 50, 55, 60, 65, 70, 80]
GRADE_LABELS = {
    40: "40: up-and-down / org depth", 45: "45: bench or middle reliever",
    50: "50: average regular / #4 starter", 55: "55: above-average regular",
    60: "60: all-star level", 65: "65: perennial all-star", 70: "70: superstar", 80: "80: generational",
}
FV_WAR = {40: 0.5, 45: 1.0, 50: 2.0, 55: 3.0, 60: 4.0, 65: 5.0, 70: 6.0, 80: 7.5}
# Expected share of the grade's WAR actually delivered (bust risk), hitters ~1 year out.
REALIZATION = {40: 0.20, 45: 0.22, 50: 0.23, 55: 0.29, 60: 0.36, 65: 0.42, 70: 0.51, 80: 0.58}
PITCHER_RISK = 0.75          # extra multiplier for pitchers
RISK_PER_YEAR = 0.92         # each year of ETA beyond the first
CONTROL_YEARS = 6
FIRST_YEAR_SHARE = 0.5       # debut season is usually a partial season

# Default years until MLB debut, by highest level reached.
DEFAULT_ETA = {"AAA": 1, "AA": 2, "A+": 3, "A": 3, "Rk": 4}
# Rough typical age at each level, used for "age vs level".
TYPICAL_AGE = {"AAA": 26.5, "AA": 24.0, "A+": 22.5, "A": 21.5, "Rk": 19.5}
LEVEL_ORDER = ["AAA", "AA", "A+", "A", "Rk"]


@dataclass
class ProspectValue:
    surplus: float
    production: float
    salary: float
    expected_war: float      # expected WAR per full season (after bust risk)
    war_total: float         # expected WAR over the control years
    salaries: list[float]


def prospect_value(grade: int, eta: int, pitcher: bool, age: float | None, season: int,
                   growth: float | None = None) -> ProspectValue:
    """Surplus value ($M, discounted to today) of a prospect."""
    g = contracts.trend_growth() if growth is None else growth
    eta = max(int(eta), 0)
    risk = REALIZATION[grade] * (PITCHER_RISK if pitcher else 1.0) * RISK_PER_YEAR ** max(eta - 1, 0)
    exp_war = FV_WAR[grade] * risk
    wars = [exp_war * (FIRST_YEAR_SHARE if y == 0 else 1.0) for y in range(CONTROL_YEARS)]
    debut_age = (age if age is not None else 22.0) + eta
    path = contracts.salary_path(debut_age, 0.0, [], wars, pitcher, season + eta, g)
    dpw = value.dollars_per_war(season)
    prod = paid = 0.0
    for y, (w, s) in enumerate(zip(wars, path.salaries)):
        t = eta + y                              # years from now
        disc = 1.0 / (1 + value.DISCOUNT_RATE) ** t
        prod += max(w, 0.0) * dpw * (1 + g) ** t * disc
        paid += s * (FIRST_YEAR_SHARE if y == 0 else 1.0) * disc
    # A team can always release a prospect, so his value can't go below zero.
    return ProspectValue(max(prod - paid, 0.0), prod, paid, exp_war, float(sum(wars)), path.salaries)


# ------------------------------------------------------------------ stat lines

def summarize_stats(milb: pd.DataFrame, player_id: int, age: float | None) -> dict:
    """Highest-level stat line for a prospect, plus age relative to that level."""
    rows = milb[milb.player_id == player_id] if len(milb) else milb
    if rows is None or len(rows) == 0:
        return {"level": None, "line": "No stats this season", "age_vs_level": None, "pitcher_stats": False}
    hit = rows[rows.group == "hitting"]
    pit = rows[rows.group == "pitching"]
    is_p = pit.OUTS.sum() / 3 * 4.3 > hit.PA.sum()
    use = pit if is_p else hit
    use = use.assign(rank=use.level.map({l: i for i, l in enumerate(LEVEL_ORDER)}))
    vol = "BF" if is_p else "PA"
    use = use[use[vol] > 0].sort_values("rank")
    if len(use) == 0:
        return {"level": None, "line": "No stats this season", "age_vs_level": None, "pitcher_stats": bool(is_p)}
    top = use.iloc[0]
    lvl = top.level
    if is_p:
        ip = top.OUTS / 3
        era = top.ER / ip * 9 if ip else float("nan")
        k = top.SO / top.BF if top.BF else 0
        bb = top.BB / top.BF if top.BF else 0
        line = f"{lvl}: {era:.2f} ERA, {k:.0%} K, {bb:.0%} BB in {ip:.0f} IP"
    else:
        ab, pa = top.AB, top.PA
        avg = top.H / ab if ab else 0
        obp = (top.H + top.BB + top.HBP) / max(ab + top.BB + top.HBP + top.SF, 1)
        slg = (top.H + top["2B"] + 2 * top["3B"] + 3 * top.HR) / ab if ab else 0
        line = f"{lvl}: {avg:.3f}/{obp:.3f}/{slg:.3f}, {top.HR} HR, {top.SB} SB in {pa} PA".replace("0.", ".")
    if len(use) > 1:
        line += " (also " + ", ".join(use.level.iloc[1:].tolist()) + ")"
    avl = None if age is None else round(age - TYPICAL_AGE[lvl], 1)
    return {"level": lvl, "line": line, "age_vs_level": avl, "pitcher_stats": bool(is_p)}
