"""Contract predictions from historical comparable players.

The comparison set (``resources/contract_comps.csv``) holds ~13,000 player
seasons from 2000-2016 with known salaries (Lahman database). Each salary is
stored as a multiple of that season's league-average salary, which removes
salary inflation from the comparison. A comparable player who earned 3x the
league average in 2008 is treated like a player earning 3x the average today.

To predict a player's salary for a future season:
  1. Put him in a service class: pre-arbitration (<3 years), arbitration
     (3-5), or free-agent contract (6+). Salary rules differ so much between
     these that comps only come from the same class.
  2. Find the 30 nearest comps by age, MLB service, recent WAR (3/2/1 weighted
     over the prior three seasons) and pitcher/hitter.
  3. Take their median salary multiple and multiply by the league-average
     salary for the target season. Known averages come from the AP opening-day
     salary study. Later seasons grow at the long-run trend rate.

Pre-arb and arbitration players are re-predicted each season, so their raises
show up. A veteran's salary is predicted once and held flat, since a
free-agent contract's annual value is fixed when signed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import data

RES = Path(__file__).resolve().parent / "resources"

# League-average opening-day salary, $M (AP study; 2020 uses the full-season rate).
# 2000-2016 come from the Lahman salary table (same opening-day basis).
PUBLISHED_AVG_SALARY = {
    2017: 4.45, 2018: 4.41, 2019: 4.38, 2020: 4.43, 2021: 4.17,
    2022: 4.42, 2023: 4.91, 2024: 4.98, 2025: 5.16,
}
TREND_YEARS = 20  # window for the long-run growth rate
K_NEIGHBORS = 30


# ----------------------------------------------------------------- shared features

SEASON_DATES = {2020: ("07-23", "09-27")}
DEFAULT_SEASON = ("04-01", "09-30")


def debut_credit(debut: str) -> float:
    """Share of the debut season remaining on the debut date (0-1)."""
    d = pd.Timestamp(debut)
    start, end = SEASON_DATES.get(d.year, DEFAULT_SEASON)
    s, e = pd.Timestamp(f"{d.year}-{start}"), pd.Timestamp(f"{d.year}-{end}")
    return float(np.clip((e - d).days / max((e - s).days, 1), 0.0, 1.0))


def service_years(debut: str | None, mlb_years: set[int], season: int) -> float:
    """Approximate MLB service before ``season``.

    Real service time counts days on the active roster or injured list. As a
    proxy: partial credit for the debut season (by debut date), then a full
    year for every later season in which the player appeared in the majors.
    Used identically for the historical comps and for current players.
    """
    if not debut or pd.isna(debut):
        return 0.0
    dy = pd.Timestamp(debut).year
    if dy >= season:
        return 0.0
    total = debut_credit(debut)
    total += sum(1.0 for y in mlb_years if dy < y < season)
    return float(total)


def recent_war(prev: list[float]) -> float:
    """3/2/1-weighted WAR from the three prior seasons (most recent first)."""
    w = [3, 2, 1]
    vals = (list(prev) + [0.0, 0.0, 0.0])[:3]
    return float(sum(a * b for a, b in zip(w, vals)) / 6)


def service_class(service: float) -> str:
    if service < 3:
        return "pre-arb"
    if service < 6:
        return "arb"
    return "free agent"


# ----------------------------------------------------------------- salary levels

@lru_cache(maxsize=1)
def comps() -> pd.DataFrame:
    df = pd.read_csv(RES / "contract_comps.csv")
    df["cls"] = df.service.map(service_class)
    return df


@lru_cache(maxsize=1)
def avg_salary_series() -> pd.Series:
    lahman = pd.read_csv(RES / "lahman_avg_salary.csv", index_col=0).iloc[:, 0]
    s = pd.concat([lahman[lahman.index >= 2000], pd.Series(PUBLISHED_AVG_SALARY)])
    return s[~s.index.duplicated(keep="last")].sort_index()


def trend_growth() -> float:
    """Long-run annual growth in the average salary (CAGR over TREND_YEARS)."""
    s = avg_salary_series()
    last = s.index.max()
    first = last - TREND_YEARS
    return float((s[last] / s[first]) ** (1 / TREND_YEARS) - 1)


def avg_salary(year: int, growth: float | None = None) -> float:
    s = avg_salary_series()
    if year in s.index:
        return float(s[year])
    g = trend_growth() if growth is None else growth
    last = int(s.index.max())
    return float(s[last] * (1 + g) ** (year - last))


# ----------------------------------------------------------------- prediction

@dataclass
class SalaryPrediction:
    salary: float           # $M for the target season
    multiple: float         # x league average
    cls: str
    comps: pd.DataFrame     # the nearest historical comps


def predict_salary(age: float, service: float, war_recent: float, pitcher: bool, year: int,
                   growth: float | None = None, k: int = K_NEIGHBORS) -> SalaryPrediction:
    cls = service_class(service)
    pool = comps()
    pool = pool[pool.cls == cls]
    dist = np.sqrt(((pool.age - age) / 3.0) ** 2
                   + ((pool.war_recent - war_recent) / 1.0) ** 2
                   + ((pool.service - service) / 1.0) ** 2
                   + ((pool.pitcher - int(pitcher)) * 0.7) ** 2)
    near = pool.assign(distance=dist).nsmallest(k, "distance")
    mult = float(near.salary_index.median())
    sal = max(mult * avg_salary(year, growth), min_salary(year, growth))
    return SalaryPrediction(sal, mult, cls, near)


def min_salary(year: int, growth: float | None = None) -> float:
    known = {2022: 0.70, 2023: 0.72, 2024: 0.74, 2025: 0.76, 2026: 0.78}
    if year in known:
        return known[year]
    g = trend_growth() if growth is None else growth
    return known[2026] * (1 + g) ** (year - 2026) if year > 2026 else 0.5


def default_control_years(service: float, age: float | None) -> int:
    """Seasons of control after this one: from service if under 6 years, else a guess."""
    if service < 6:
        return int(max(0, 5 - np.floor(service)))
    if age is None or age < 30:
        return 3
    if age < 33:
        return 2
    return 1


@dataclass
class ContractPath:
    salaries: list[float]   # $M per season, index 0 = current season
    cls_by_year: list[str]
    first: SalaryPrediction


def salary_path(age: float, service: float, prior_war: list[float], war_by_year: list[float],
                pitcher: bool, season: int, growth: float | None = None) -> ContractPath:
    """Predicted salary for each season in ``war_by_year`` (index 0 = this season).

    ``prior_war``: actual WAR for the seasons before this one, most recent first.
    ``war_by_year``: projected full-season WAR for this and future seasons.
    """
    history = list(prior_war)          # most recent first
    salaries, classes = [], []
    first = None
    fa_salary = None
    for y, w in enumerate(war_by_year):
        svc = service + y
        pred = predict_salary(age + y, svc, recent_war(history), pitcher, season + y, growth)
        if first is None:
            first = pred
        if pred.cls == "free agent":
            if fa_salary is None:
                fa_salary = pred.salary
            salaries.append(fa_salary)
        else:
            salaries.append(pred.salary)
        classes.append(pred.cls)
        history = [w] + history        # this season becomes "prior" for the next
    return ContractPath(salaries, classes, first)


# ----------------------------------------------------------------- current players

_HISTORY: dict[int, dict] = {}


def _store_people(people: list[dict]) -> None:
    for p in people:
        years = set()
        for block in p.get("stats", []):
            for sp in block.get("splits", []):
                if sp.get("sport", {}).get("id", 1) == 1 and "season" in sp:
                    years.add(int(sp["season"]))
        _HISTORY[p["id"]] = {"debut": p.get("mlbDebutDate"), "years": years}


def prefetch_histories(player_ids: list[int]) -> None:
    """Load MLB debut dates and seasons played for many players (100 per request)."""
    need = sorted({int(i) for i in player_ids if int(i) > 0 and int(i) not in _HISTORY})
    for i in range(0, len(need), 100):
        chunk = need[i : i + 100]
        d = data._get("people", {"personIds": ",".join(map(str, chunk)),
                                 "hydrate": "stats(group=[hitting,pitching],type=[yearByYear],sportId=1)"},
                      ttl_days=30)
        _store_people(d.get("people", []))
        for pid in chunk:
            _HISTORY.setdefault(pid, {"debut": None, "years": set()})


def player_history(player_id: int, season: int) -> tuple[float, list[int]]:
    """(approximate service before ``season``, MLB seasons played) from the Stats API."""
    prefetch_histories([player_id])
    h = _HISTORY[int(player_id)]
    return service_years(h["debut"], h["years"], season), sorted(h["years"])


# ----------------------------------------------------------------- free-agent market
#
# Free agents are priced from actual signings rather than from every veteran's
# salary: veterans on pre-free-agency extensions or backloaded deals make the
# general comps run well below the open market. Two sources:
#
# * resources/recent_fa_signings.csv - major free-agent contracts from
#   2022-2025 (hand-entered from public reporting; see recent_fa_contracts.csv)
# * resources/fa_signings.csv - 2000-2016 signings inferred from Lahman
#   (veterans who changed teams with a new salary; annual value = average
#   salary over their first seasons with the new club)
#
# Since 2016 the league-average salary has barely grown while good free agents'
# prices kept rising, so relative to the average a 3+ WAR free agent now costs
# far more than in the Lahman era. Older signings are scaled up by that measured
# gap (full gap at 3+ WAR, none at 1 WAR or below, where recent data is too thin
# and selective to measure it). A free agent's annual value is the
# distance-weighted average (in logs) of his 20 nearest signings by age, recent
# WAR and pitcher/hitter, with recent signings counted triple. His contract
# length comes from his nearest recent signings.

K_FA = 20
RECENT_WEIGHT = 3.0


def _sig_dist(pool: pd.DataFrame, age: float, war: float, pitcher: bool) -> np.ndarray:
    return np.sqrt(((pool.age - age) / 3.0) ** 2 + ((pool.war_recent - war) / 0.8) ** 2
                   + ((pool.pitcher - int(pitcher)) * 0.7) ** 2).to_numpy()


def _knn_log(pool: pd.DataFrame, age, war, pitcher, k, col="log_mult", weight_col=None):
    d = _sig_dist(pool, age, war, pitcher)
    idx = np.argsort(d)[:k]
    w = 1.0 / (d[idx] + 0.3)
    if weight_col is not None:
        w = w * pool[weight_col].to_numpy()[idx]
    return float((pool[col].to_numpy()[idx] * w).sum() / w.sum()), pool.iloc[idx].assign(distance=d[idx])


@lru_cache(maxsize=1)
def _signing_pools() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """(recent signings, era-adjusted Lahman signings, measured era gap in logs)."""
    rec = pd.read_csv(RES / "recent_fa_signings.csv")
    lah = pd.read_csv(RES / "fa_signings.csv")
    lah = lah[lah.war_recent > -1.5].copy()
    rec["log_mult"] = np.log(rec.aav_index)
    lah["log_mult"] = np.log(lah.aav_index)
    # gap = how far recent 3+ WAR signings sit above what the Lahman-era market would pay
    gaps = []
    for r in rec[rec.war_recent >= 3].itertuples():
        pred, _ = _knn_log(lah, r.age, r.war_recent, r.pitcher, K_FA)
        gaps.append(r.log_mult - pred)
    gap = float(np.median(gaps)) if gaps else 0.0
    lah["log_mult"] = lah.log_mult + gap * np.clip((lah.war_recent - 1.0) / 2.0, 0.0, 1.0)
    lah["source"] = "2000–16 (era-adjusted)"
    rec["source"] = "2022–25"
    lah["w"], rec["w"] = 1.0, RECENT_WEIGHT
    return rec, lah, gap


def era_gap() -> float:
    """Measured premium of today's market over 2000-16 for 3+ WAR free agents (multiplier)."""
    return float(np.exp(_signing_pools()[2]))


@dataclass
class FAContract:
    aav: float              # $M per year in the signing season
    years: int
    multiple: float         # x league-average salary
    comps: pd.DataFrame


def predict_fa_contract(age: float, war_recent: float, pitcher: bool, year: int,
                        growth: float | None = None, exclude: tuple[str, int] | None = None) -> FAContract:
    rec, lah, _ = _signing_pools()
    if exclude is not None:
        rec = rec[~((rec.name == exclude[0]) & (rec.year == exclude[1]))]
    pool = pd.concat([rec, lah], ignore_index=True)
    log_mult, near = _knn_log(pool, age, war_recent, pitcher, K_FA, weight_col="w")
    mult = float(np.exp(log_mult))
    # contract length from the nearest recent signings (Lahman tenure isn't contract length)
    # pitchers sign shorter deals than hitters of the same age and value, so a
    # pitcher's length comes from pitchers (hitters can borrow from either)
    d = np.sqrt(((rec.age - age) / 3.0) ** 2 + ((rec.war_recent - war_recent) / 0.8) ** 2
                + ((rec.pitcher - int(pitcher)) * (3.0 if pitcher else 0.7)) ** 2).to_numpy()
    idx = np.argsort(d)[:7]
    w = 1.0 / (d[idx] + 0.3)
    yrs = rec.years.to_numpy()[idx]
    order = np.argsort(yrs)
    cum = np.cumsum(w[order]) / w.sum()
    rec_years = float(yrs[order][np.searchsorted(cum, 0.5)])     # weighted median
    # For lower-tier players the recent list only holds unusually big deals, so
    # blend toward how long comparable 2000-16 signings stayed with their club.
    d_l = _sig_dist(lah, age, war_recent, pitcher)
    old_years = float(np.median(lah.years.to_numpy()[np.argsort(d_l)[:K_FA]]))
    wr = float(np.clip((war_recent - 1.0) / 2.0, 0.0, 1.0))
    years = int(round(wr * rec_years + (1 - wr) * old_years))
    aav = max(mult * avg_salary(year, growth), min_salary(year, growth))
    return FAContract(aav, max(years, 1), mult, near)
