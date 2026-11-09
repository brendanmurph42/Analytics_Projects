"""Automatic depth charts and roster-based team projections.

Every team's roster (40-man, on the chosen date) is ranked by projected value
and given playing time the way a typical club uses its players over a full
season:

  hitters   starting catcher 480 PA, backup catcher 180 PA,
            8 other regulars (incl. DH) 550 PA each, 4 bench players 150 PA each
  pitchers  5 starters 150 IP each, 2 depth starters 35 IP each,
            7 relievers 60 IP each, 3 mop-up arms 20 IP each

The slots deliberately add up to a bit less than a full season (~5,650 of
~6,150 PA and ~1,350 of ~1,450 IP). The rest is injuries and call-ups,
filled at replacement level. Players on the 60-day injured list during the
season get a reduced share.

A team's projection is the sum of its players' runs above replacement
(rate x playing time). Win% is centered so the league averages .500. A trade
or signing re-runs the depth chart, so a star pushes a bench player out of
the lineup rather than simply adding his WAR on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HIT_SLOTS = [("C1", 480)] + [("REG", 550)] * 8 + [("C2", 180)] + [("BENCH", 150)] * 4
PIT_SLOTS = [("SP", 150)] * 5 + [("SPD", 35)] * 2 + [("RP", 60)] * 7 + [("MOP", 20)] * 3
IL_FACTOR = {"Injured 60-Day": 0.4, "Injured 15-Day": 0.85, "Injured 10-Day": 0.9, "Injured 7-Day": 0.9}

# rate columns per model: (offense per PA, defense per PA, pitching per IP)
MODEL_RATES = {
    "war": ("todate_off_per_pa", "def_per_pa", "todate_pit_per_ip"),
    "proj": ("proj_off_per_pa", "def_per_pa", "proj_pit_per_ip"),
    "rd": ("todate_off_per_pa", "def_per_pa", "todate_pit_per_ip"),
}


def allocate(roster: pd.DataFrame, in_season: bool = False) -> pd.DataFrame:
    """Full-season PA and IP for each player on one roster.

    ``roster`` columns: player_id, pos, status, proj_off_per_pa, def_per_pa,
    proj_pit_per_ip, sp_share, and optional fixed_pa / fixed_ip overrides
    (NaN = automatic). Ranking uses the projection rates, the way a manager
    plays his best players; every model then values the same playing time.
    """
    r = roster.copy()
    for c in ("fixed_pa", "fixed_ip"):
        if c not in r:
            r[c] = np.nan
    avail = r.status.map(IL_FACTOR).fillna(1.0) if in_season else pd.Series(1.0, index=r.index)
    r["pa"], r["ip"] = 0.0, 0.0

    # ---- hitters
    hitters = r[(r.pos != "P") & r.fixed_pa.isna()].copy()
    hitters["score"] = hitters.proj_off_per_pa + hitters.def_per_pa
    hitters = hitters.sort_values("score", ascending=False)
    catchers = list(hitters.index[hitters.pos == "C"])
    used: set = set()

    def take(pool):
        for i in pool:
            if i not in used:
                used.add(i)
                return i
        return None

    non_catchers = [i for i in hitters.index if hitters.at[i, "pos"] != "C"]
    for slot, pa in HIT_SLOTS:
        # catchers fill the two catcher slots; the rest of the lineup and bench
        # come from non-catchers (a third catcher rarely plays much)
        i = take(catchers) if slot in ("C1", "C2") else take(non_catchers)
        if i is not None:
            r.at[i, "pa"] = pa * avail[i]

    # ---- pitchers (two-way players can appear in both pools)
    pitchers = r[((r.pos == "P") | (r.pos == "TWP")) & r.fixed_ip.isna()].copy()
    pitchers["score"] = pitchers.proj_pit_per_ip
    pitchers = pitchers.sort_values("score", ascending=False)
    starters = list(pitchers.index[pitchers.sp_share >= 0.5])
    used_p: set = set()

    def take_p(pool):
        for i in pool:
            if i not in used_p:
                used_p.add(i)
                return i
        return None

    for slot, ip in PIT_SLOTS:
        i = take_p(starters) if slot in ("SP", "SPD") else None
        if i is None:
            i = take_p(pitchers.index)
        if i is not None:
            r.at[i, "ip"] = ip * avail[i]

    # ---- user overrides displace the lowest-ranked automatic playing time
    for col, fixed, order in (("pa", "fixed_pa", hitters.index[::-1]), ("ip", "fixed_ip", pitchers.index[::-1])):
        extra = r[fixed].fillna(0).sum()
        mask = r[fixed].notna()
        r.loc[mask, col] = r.loc[mask, fixed]
        for i in order:
            if extra <= 0:
                break
            cut = min(r.at[i, col], extra)
            r.at[i, col] -= cut
            extra -= cut
    return r


@dataclass
class TeamRuns:
    off: dict[str, float]     # model -> offensive runs above replacement
    defense: float            # positional runs (same for every model)
    pit: dict[str, float]     # model -> pitching runs above replacement

    def total(self, model: str) -> float:
        return self.off[model] + self.defense + self.pit[model]


def team_runs(alloc: pd.DataFrame) -> TeamRuns:
    off, pit = {}, {}
    for m, (o, d, p) in MODEL_RATES.items():
        off[m] = float((alloc[o] * alloc.pa).sum())
        pit[m] = float((alloc[p] * alloc.ip).sum())
    defense = float((alloc.def_per_pa * alloc.pa).sum())
    return TeamRuns(off, defense, pit)


def league_win_pct(runs: dict[int, TeamRuns], runs_per_win: float, lg_rpg: float) -> dict[str, pd.Series]:
    """Projected full-season win% per team for each model, centered on .500."""
    from .strength import pythag

    ids = list(runs)
    out = {}
    for m in ("war", "proj"):
        tot = pd.Series({t: runs[t].total(m) for t in ids})
        out[m] = 0.5 + (tot - tot.mean()) / runs_per_win / 162
    off = pd.Series({t: runs[t].off["rd"] for t in ids})
    prev = pd.Series({t: runs[t].defense + runs[t].pit["rd"] for t in ids})
    rs = lg_rpg + (off - off.mean()) / 162
    ra = lg_rpg - (prev - prev.mean()) / 162
    out["rd_rs"], out["rd_ra"] = rs, ra
    out["rd"] = pd.Series(pythag(rs.to_numpy(), ra.to_numpy()), index=ids)
    return out
