"""Glue: evaluate a trade under three models and simulate before/after odds.

A trade moves players between two teams. Each player brings a rate of runs
above replacement (per PA for hitting, per IP for pitching) and an expected
amount of remaining playing time. Whatever playing time a departing player
leaves behind is assumed to be filled at replacement level, and an arriving
player's playing time displaces replacement-level time. That's the standard
WAR logic, and it handles multi-player and uneven trades.

Models
------
* ``war``  - season-to-date player value (fWAR-style) -> runs / runs-per-win
             -> added to the team's true-talent win% over remaining games.
* ``proj`` - same, but player value comes from Marcel-style projections
             (3 prior seasons + this season, regressed, age-adjusted).
* ``rd``   - season-to-date runs split into offense (moves runs scored) and
             run prevention (moves runs allowed); team win% re-derived from
             adjusted RS/RA via Pythagenpat, so diminishing returns apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from . import contracts, data, players, simulate, strength, value

MODELS = {"war": "WAR (to date)", "proj": "Projection (Marcel)", "rd": "Run differential"}


@dataclass
class Context:
    season: int
    as_of: date
    teams: pd.DataFrame
    sched: pd.DataFrame
    setup: simulate.SeasonSetup
    base: pd.DataFrame
    pv: players.PlayerValues
    hp: pd.DataFrame
    pp: pd.DataFrame
    rem_games: pd.Series      # remaining games per team_id

    @property
    def idx(self):
        return {t: i for i, t in enumerate(self.setup.team_ids)}


def load_context(season: int, as_of: date) -> Context:
    teams = data.get_teams(season)
    sched = data.get_schedule(season)
    setup = simulate.build_setup(teams, sched, as_of)
    base = strength.baseline_strength(season, as_of, teams, sched).loc[teams.team_id]
    tg = float(base.G.mean())
    pv, hp, pp = players.value_all_players(season, as_of, tg)
    rem = pd.Series(np.bincount(setup.rem_home, minlength=30) + np.bincount(setup.rem_away, minlength=30),
                    index=setup.team_ids)
    return Context(season, as_of, teams, sched, setup, base, pv, hp, pp, rem)


def trade_runs(moves: pd.DataFrame, rem_games: pd.Series) -> pd.DataFrame:
    """Per-team run deltas from a table of moves.

    ``moves`` columns: team_id (receiving team), sign (+1 arriving, -1
    departing), rem_pa, rem_ip, plus per-player rate columns.
    Returns per-team: war_runs, proj_runs, rd_rs, rd_ra.
    """
    m = moves.copy()
    m["war_runs"] = m.sign * ((m.todate_off_per_pa + m.def_per_pa) * m.rem_pa + m.todate_pit_per_ip * m.rem_ip)
    m["proj_runs"] = m.sign * ((m.proj_off_per_pa + m.def_per_pa) * m.rem_pa + m.proj_pit_per_ip * m.rem_ip)
    m["rd_rs"] = m.sign * (m.todate_off_per_pa * m.rem_pa)
    m["rd_ra"] = -m.sign * (m.def_per_pa * m.rem_pa + m.todate_pit_per_ip * m.rem_ip)
    return m.groupby("team_id")[["war_runs", "proj_runs", "rd_rs", "rd_ra"]].sum()


def scenario_wp(ctx: Context, deltas: pd.DataFrame | None) -> dict[str, np.ndarray]:
    """Team win% vectors for every model, optionally including trade deltas."""
    b = ctx.base
    rpw = ctx.pv.ctx.runs_per_win
    wp_war = b.wp.to_numpy().copy()
    wp_proj = b.wp.to_numpy().copy()
    rs = b.rs.to_numpy().copy()
    ra = b.ra.to_numpy().copy()
    if deltas is not None:
        for tid, row in deltas.iterrows():
            i = ctx.idx[tid]
            g = max(float(ctx.rem_games.get(tid, 0)), 1.0)
            wp_war[i] += row.war_runs / rpw / g
            wp_proj[i] += row.proj_runs / rpw / g
            rs[i] += row.rd_rs / g
            ra[i] += row.rd_ra / g
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

    ``rates`` rows come from ``players.player_rates`` (or custom players) and
    need a ``to_id`` column (the team whose remaining schedule counts this
    season). Returns per row: war_season, age, service, priors, pitcher,
    control, path (ContractPath), salary (average predicted $M/yr), surplus.
    Used by both the roster dropdown labels and the trade value table so the
    two always agree.
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
            war_s = float(((r.proj_off_per_pa + r.def_per_pa) * min(r.rem_pa_per_g * 162, 700)
                           + r.proj_pit_per_ip * min(r.rem_ip_per_g * 162, 220)) / rpw)
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
