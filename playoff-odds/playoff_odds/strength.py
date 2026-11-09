"""Team strength estimates as of a date.

Two baselines are produced for every team:

* ``wp``  - a true-talent win% (used by the WAR and projection models):
            half actual record, half Pythagorean record, regressed toward a
            prior (last season's Pythagorean record, itself regressed halfway
            to .500) with 70 games of weight.
* ``rs`` / ``ra`` - true-talent runs scored / allowed per game (used by the
            run-differential model), each regressed toward a prior with 40
            games of weight, converted to win% via Pythagenpat.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import data

WP_REGRESS_GAMES = 70.0
RUNS_REGRESS_GAMES = 40.0
HOME_WIN_PCT = 0.54


def pythag_exp(rs_pg, ra_pg):
    return np.power(np.asarray(rs_pg) + np.asarray(ra_pg), 0.287)


def pythag(rs_pg, ra_pg):
    x = pythag_exp(rs_pg, ra_pg)
    rs = np.power(rs_pg, x)
    return rs / (rs + np.power(ra_pg, x))


def team_totals(games: pd.DataFrame, team_ids) -> pd.DataFrame:
    rows = []
    for tid in team_ids:
        h = games[games.home_id == tid]
        a = games[games.away_id == tid]
        w = int(h.home_win.sum() + (~a.home_win).sum())
        g = len(h) + len(a)
        rs = int(h.home_score.sum() + a.away_score.sum())
        ra = int(h.away_score.sum() + a.home_score.sum())
        rows.append({"team_id": tid, "G": g, "W": w, "L": g - w, "RS": rs, "RA": ra})
    return pd.DataFrame(rows).set_index("team_id")


def league_rpg(season: int, as_of: date, sched: pd.DataFrame) -> float:
    """Runs per team-game: last season, blended with this season once there's a sample."""
    prev = data.get_schedule(season - 1)
    prev = prev[prev.final]
    lg = (prev.home_score.sum() + prev.away_score.sum()) / (2 * len(prev))
    cur = sched[(sched.date <= pd.Timestamp(as_of)) & sched.final]
    if len(cur) > 150:
        lg = 0.5 * lg + 0.5 * (cur.home_score.sum() + cur.away_score.sum()) / (2 * len(cur))
    return float(lg)


def baseline_strength(season: int, as_of: date, teams: pd.DataFrame, sched: pd.DataFrame,
                      prior: pd.DataFrame | None = None) -> pd.DataFrame:
    """True-talent estimates per model, blending results so far with a prior.

    ``prior`` (indexed by team_id) holds roster-based projections: columns
    wp_war, wp_proj (win%) and rs, ra (runs per game) for the
    run-differential model. Without it, the prior is last season's
    Pythagorean record regressed halfway to average.
    """
    ids = teams.team_id.tolist()
    played = sched[(sched.date <= pd.Timestamp(as_of)) & sched.final]
    cur = team_totals(played, ids)

    prev_sched = data.get_schedule(season - 1)
    prev_sched = prev_sched[prev_sched.final]
    prev = team_totals(prev_sched, ids)
    lg_rpg = (prev.RS.sum()) / prev.G.sum()
    if cur.G.sum() > 300:  # blend in this season's scoring environment once available
        lg_rpg = 0.5 * lg_rpg + 0.5 * cur.RS.sum() / cur.G.sum()

    prev_rs = prev.RS / prev.G
    prev_ra = prev.RA / prev.G
    if prior is None:
        pw = 0.5 + 0.5 * (pythag(prev_rs, prev_ra) - 0.5)
        prior_wp = {"war": pw, "proj": pw}
        prior_rs = 0.5 * lg_rpg + 0.5 * prev_rs
        prior_ra = 0.5 * lg_rpg + 0.5 * prev_ra
    else:
        pr = prior.loc[cur.index]
        prior_wp = {"war": pr.wp_war, "proj": pr.wp_proj}
        prior_rs, prior_ra = pr.rs, pr.ra

    g = cur.G.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        act_wp = np.where(g > 0, cur.W / g, 0.5)
        py = np.where(g > 0, pythag(np.maximum(cur.RS, 1) / np.maximum(g, 1), np.maximum(cur.RA, 1) / np.maximum(g, 1)), 0.5)
        obs_wp = 0.5 * act_wp + 0.5 * py
        rs_pg = np.where(g > 0, cur.RS / np.maximum(g, 1), prior_rs)
        ra_pg = np.where(g > 0, cur.RA / np.maximum(g, 1), prior_ra)

    out = pd.DataFrame(index=cur.index)
    out["G"], out["W"], out["L"], out["RS"], out["RA"] = cur.G, cur.W, cur.L, cur.RS, cur.RA
    for m in ("war", "proj"):
        out[f"wp_{m}"] = (g * obs_wp + WP_REGRESS_GAMES * prior_wp[m]) / (g + WP_REGRESS_GAMES)
    out["wp"] = out.wp_proj
    out["rs"] = (g * rs_pg + RUNS_REGRESS_GAMES * prior_rs) / (g + RUNS_REGRESS_GAMES)
    out["ra"] = (g * ra_pg + RUNS_REGRESS_GAMES * prior_ra) / (g + RUNS_REGRESS_GAMES)
    out["wp_rd"] = pythag(out.rs, out.ra)
    return out


def game_probs(home_idx: np.ndarray, away_idx: np.ndarray, wp: np.ndarray) -> np.ndarray:
    """Home-team win probability for each game via log5 with home-field odds.

    ``wp`` may be (T,) or (sims, T); the result is (N,) or (sims, N).
    """
    wp = np.clip(wp, 0.2, 0.8)
    odds = wp / (1 - wp)
    hfa = HOME_WIN_PCT / (1 - HOME_WIN_PCT)
    o = odds[..., home_idx] / odds[..., away_idx] * hfa
    return o / (1 + o)


# Preseason talent uncertainty. The blended preseason estimate missed actual
# records by 9.2 wins (RMSE, 2023-26); removing the ~6.4 wins of pure
# 162-game luck leaves ~6.6 wins (.041) of talent uncertainty.
TALENT_PRIOR_SD = 0.041


def talent_sd(games_played) -> np.ndarray:
    """Posterior SD of a team's true-talent win% after n games (normal approx).

    Prior uncertainty around the preseason estimate is ~.041 (~6.6 wins);
    each game adds binomial information with variance .25.
    """
    n = np.asarray(games_played, float)
    return np.sqrt(1.0 / (1.0 / TALENT_PRIOR_SD**2 + n / 0.25))
