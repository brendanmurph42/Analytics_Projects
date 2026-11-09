"""Glue: next-playoffs odds as of any date, before and after a trade or signing.

For a chosen date the app works out which playoffs come next: this season's
if the regular season hasn't ended, otherwise next season's. It then:

1. projects every team from its roster on that date (automatic depth chart x
   player projections, see ``depth.py``),
2. blends that projection with results so far (once games have been played),
3. simulates every remaining game, before and after the move.

A trade or signing re-runs the depth charts of the teams involved, so playing
time shifts realistically (a star pushes a bench player out of the lineup).

Models (they differ in how each player's rate is measured)
------
* ``war``  - recent performance: this season so far, or last season plus this
             one before about a month of games, fWAR-style.
* ``proj`` - Marcel-style projections (3 prior seasons + this season,
             regressed, age-adjusted).
* ``rd``   - recent performance split into runs scored and runs allowed,
             converted to win% with Pythagenpat (diminishing returns).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import contracts, data, depth, players, simulate, strength, value

MODELS = {"war": "WAR (recent)", "proj": "Projection (Marcel)", "rd": "Run differential"}

# Preseason team talent = a x (last season's Pythagorean win% - .500)
#                       + b x (roster projection - .500), per model.
# Fit on opening days 2023-2026 (120 team-seasons). Leave-one-season-out RMSE
# was 9.1-9.4 wins, vs 9.3 for last season alone and 9.9-10.3 for the roster
# projection alone (actual wins have an SD of 12.0).
PRIOR_WEIGHTS = {"proj": (0.50, 0.22), "war": (0.40, 0.28), "rd": (0.41, 0.26)}
RATE_COLS = ["todate_off_per_pa", "def_per_pa", "todate_pit_per_ip", "proj_off_per_pa", "proj_pit_per_ip"]


@dataclass
class Context:
    season: int               # the season whose playoffs are next
    as_of: date
    in_season: bool           # True once that season's games have started
    season_start: date
    season_end: date
    teams: pd.DataFrame
    sched: pd.DataFrame
    setup: simulate.SeasonSetup
    base: pd.DataFrame        # blended talent per team (wp_war, wp_proj, rs, ra, ...)
    proj: dict                # roster projections per model
    pv: players.PlayerValues
    hp: pd.DataFrame
    pp: pd.DataFrame
    rates: pd.DataFrame       # every rostered player: rates + team_id + status
    allocs: dict = field(default_factory=dict)   # team_id -> depth chart
    runs: dict = field(default_factory=dict)     # team_id -> depth.TeamRuns
    lg_rpg: float = 4.4
    rem_games: pd.Series = None
    free_agents: list = field(default_factory=list)

    @property
    def idx(self):
        return {t: i for i, t in enumerate(self.setup.team_ids)}

    @property
    def rem_frac(self) -> pd.Series:
        return self.rem_games / 162.0

    def season_pt(self, player_id: int) -> tuple[float, float]:
        """Full-season (PA, IP) the player gets on his current team's depth chart."""
        if player_id in self.rates.index:
            a = self.allocs.get(int(self.rates.at[player_id, "team_id"]))
            if a is not None and player_id in a.index:
                return float(a.at[player_id, "pa"]), float(a.at[player_id, "ip"])
        return 0.0, 0.0


def load_rosters(team_ids: list[int], as_of: date) -> dict[int, list]:
    return data.league_map(as_of, team_ids)[0]


def load_context(as_of: date) -> Context:
    season = data.target_season(as_of)
    start, end = data.season_dates(season)
    in_season = start <= as_of <= end
    teams = data.get_teams(season)
    sched = data.get_schedule(season)
    setup = simulate.build_setup(teams, sched, as_of)
    tg = float((setup.base_w + setup.base_l).mean())
    pv, hp, pp = players.value_all_players(season, as_of, tg)

    rosters, free_agents = data.league_map(as_of, teams.team_id.tolist())
    ids, pos, team_of, status = [], {}, {}, {}
    for t, lst in rosters.items():
        for p in lst:
            if p.player_id in team_of:
                continue
            ids.append(p.player_id)
            pos[p.player_id] = p.pos
            team_of[p.player_id] = t
            status[p.player_id] = p.status
    rates = players.player_rates(pv, hp, pp, ids, as_of, pos).set_index("player_id", drop=False)
    rates["team_id"] = rates.player_id.map(team_of)
    rates["status"] = rates.player_id.map(status).fillna("")

    allocs = {t: depth.allocate(rates[rates.team_id == t], in_season) for t in teams.team_id}
    runs = {t: depth.team_runs(a) for t, a in allocs.items()}
    lg_rpg = strength.league_rpg(season, as_of, sched)
    proj = depth.league_win_pct(runs, pv.ctx.runs_per_win, lg_rpg)
    prior = blended_prior(season, teams.team_id.tolist(), proj, lg_rpg)
    base = strength.baseline_strength(season, as_of, teams, sched, prior).loc[teams.team_id]
    rem = pd.Series(np.bincount(setup.rem_home, minlength=30) + np.bincount(setup.rem_away, minlength=30),
                    index=setup.team_ids)
    return Context(season, as_of, in_season, start, end, teams, sched, setup, base, proj, pv, hp, pp,
                   rates, allocs, runs, lg_rpg, rem, free_agents)


def blended_prior(season: int, team_ids: list[int], proj: dict, lg_rpg: float) -> pd.DataFrame:
    """Preseason talent per model: last season's results blended with roster projections."""
    prev = data.get_schedule(season - 1)
    prev = strength.team_totals(prev[prev.final], team_ids)
    prs, pra = prev.RS / prev.G, prev.RA / prev.G
    py_prev = pd.Series(strength.pythag(prs.to_numpy(), pra.to_numpy()), index=team_ids)
    out = pd.DataFrame(index=team_ids)
    for m in ("war", "proj"):
        a, b = PRIOR_WEIGHTS[m]
        out[f"wp_{m}"] = 0.5 + a * (py_prev - 0.5) + b * (proj[m] - 0.5)
    a, b = PRIOR_WEIGHTS["rd"]
    out["rs"] = lg_rpg + a * (prs - lg_rpg) + b * (proj["rd_rs"] - lg_rpg)
    out["ra"] = lg_rpg + a * (pra - lg_rpg) + b * (proj["rd_ra"] - lg_rpg)
    return out


# ----------------------------------------------------------------- trades & signings


def trade_effect(ctx: Context, moves: list[dict]) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Run deltas per team from a set of moves, by re-running depth charts.

    Each move: player_id, from_id (None for a signing / outside the deal),
    to_id, optional fixed_pa / fixed_ip (full-season units, NaN = automatic),
    and for players not on any roster (free agents, custom players) the rate
    columns plus pos and sp_share.

    Returns (deltas per team in full-season runs: war_runs, proj_runs, rd_rs,
    rd_ra; new depth charts for the affected teams).
    """
    affected = {m["to_id"] for m in moves} | {m["from_id"] for m in moves if m.get("from_id") is not None}
    rows, allocs_after = [], {}
    for t in affected:
        roster = ctx.rates[ctx.rates.team_id == t].copy()
        leaving = {m["player_id"] for m in moves if m.get("from_id") == t}
        roster = roster[~roster.player_id.isin(leaving)]
        roster["fixed_pa"], roster["fixed_ip"] = np.nan, np.nan
        for m in moves:
            if m["to_id"] != t:
                continue
            pid = m["player_id"]
            if pid in ctx.rates.index:
                row = ctx.rates.loc[[pid]].copy()
            else:
                row = pd.DataFrame([{c: m.get(c, 0.0) for c in RATE_COLS + ["sp_share"]}])
                row["player_id"], row["pos"], row["status"] = pid, m.get("pos", "DH"), ""
                row.index = [pid]
            row["team_id"] = t
            row["fixed_pa"] = m.get("fixed_pa", np.nan)
            row["fixed_ip"] = m.get("fixed_ip", np.nan)
            roster = pd.concat([roster, row])
        a = depth.allocate(roster, ctx.in_season)
        allocs_after[t] = a
        before, after = ctx.runs[t], depth.team_runs(a)
        rows.append({
            "team_id": t,
            "war_runs": after.total("war") - before.total("war"),
            "proj_runs": after.total("proj") - before.total("proj"),
            "rd_rs": after.off["rd"] - before.off["rd"],
            "rd_ra": -((after.defense + after.pit["rd"]) - (before.defense + before.pit["rd"])),
        })
    return pd.DataFrame(rows).set_index("team_id"), allocs_after


def scenario_wp(ctx: Context, deltas: pd.DataFrame | None) -> dict[str, np.ndarray]:
    """Team win% vectors for every model, optionally including trade deltas.

    Deltas are full-season runs; spread over 162 games they change each
    team's per-game talent, which then applies to whatever games remain.
    """
    b = ctx.base
    rpw = ctx.pv.ctx.runs_per_win
    wp_war = b.wp_war.to_numpy().copy()
    wp_proj = b.wp_proj.to_numpy().copy()
    rs = b.rs.to_numpy().copy()
    ra = b.ra.to_numpy().copy()
    if deltas is not None:
        for tid, row in deltas.iterrows():
            i = ctx.idx[tid]
            wp_war[i] += row.war_runs / rpw / 162
            wp_proj[i] += row.proj_runs / rpw / 162
            rs[i] += row.rd_rs / 162
            ra[i] += row.rd_ra / 162
    return {"war": wp_war, "proj": wp_proj, "rd": strength.pythag(rs, ra)}


def run_models(ctx: Context, deltas: pd.DataFrame | None, models: list[str], sims: int, seed: int = 42):
    U = simulate.draw_uniforms(len(ctx.setup.rem_home), sims, seed)
    noise = simulate.draw_talent_noise(strength.talent_sd(ctx.base.G.to_numpy()), sims, seed)
    before = scenario_wp(ctx, None)
    after = scenario_wp(ctx, deltas)
    out = {}
    cache = {}
    for m in models:
        for label, wp in (("before", before[m]), ("after", after[m])):
            key = wp.tobytes()
            if key not in cache:
                cache[key] = simulate.simulate(ctx.setup, wp, U, seed, noise)
            out[(m, label)] = cache[key]
    return out


# ----------------------------------------------------------------- trade value defaults


def player_value_defaults(ctx: Context, rates: pd.DataFrame, growth: float,
                          custom_specs: dict[int, dict] | None = None) -> list[dict]:
    """Default trade-value inputs and surplus for each row of ``rates``.

    ``rates`` rows need ``to_id`` (whose remaining schedule counts this season)
    and may carry ``season_pa`` / ``season_ip`` (full-season playing time on
    the new team). Otherwise playing time comes from the player's current
    depth chart, or a typical role for players without a team. Returns per
    row: war_season, age, service, priors, pitcher, control, path, salary,
    surplus. Shared by the dropdown labels and the trade value table.
    """
    custom_specs = custom_specs or {}
    season = ctx.season
    rpw = ctx.pv.ctx.runs_per_win
    real_ids = [int(p) for p in rates.player_id if int(p) > 0]
    contracts.prefetch_histories(real_ids)
    out = []
    for _, r in rates.iterrows():
        pid = int(r.player_id)
        age = None if pd.isna(r.get("age")) else float(r.age)
        if pid < 0:
            spec = custom_specs.get(pid, {})
            war_s = float(r.get("war_pace", 0.0))
            svc = float(spec.get("service", 1.0))
            priors = [war_s] * 3 if svc >= 1 else []
            pitcher = spec.get("kind") == "Pitcher"
        else:
            pa, ip = (float(r.season_pa), float(r.season_ip)) if "season_pa" in r and pd.notna(r.season_pa) \
                else ctx.season_pt(pid)
            if pa == 0 and ip == 0:          # no depth-chart role: assume a typical one
                if r.pos == "P":
                    ip = 150.0 if r.get("sp_share", 0) >= 0.5 else 60.0
                else:
                    pa = 450.0
            war_s = float(((r.proj_off_per_pa + r.def_per_pa) * pa + r.proj_pit_per_ip * ip) / rpw)
            svc, _ = contracts.player_history(pid, season)
            priors = [float(players.season_war(season - j).get(pid, 0.0)) for j in (1, 2, 3)]
            pitcher = r.pos == "P"
        war_s = round(war_s, 1)
        control = contracts.default_control_years(svc, age)
        path = contracts.salary_path(age if age is not None else 27.0, svc, priors,
                                     value.war_path(war_s, age, control), pitcher, season, growth)
        rem_frac = float(ctx.rem_games.get(int(r.to_id), 0)) / 162
        sv = value.surplus_value(war_s, age, control, path.salaries, rem_frac, season, growth)
        out.append({"war_season": war_s, "age": age, "service": svc, "priors": priors, "pitcher": pitcher,
                    "control": control, "path": path, "salary": round(float(np.mean(path.salaries)), 1),
                    "surplus": sv.total})
    return out
