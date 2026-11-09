"""Player valuation: runs above replacement per PA (hitters) and per IP (pitchers).

Two flavours of rate are produced for every player:

* **to-date** rates, built only from this season's stats through the as-of date
  (used by the WAR model and the run-differential model), and
* **projected** rates, a Marcel-style weighted blend of the current season and
  the three prior seasons, regressed to league average and age-adjusted
  (used by the projection model).

Hitting value  = batting runs (wOBA-based wRAA) + baserunning (wSB)
                 + positional adjustment + replacement-level runs.
Pitching value = FIP-based runs saved vs league + replacement-level runs.

Replacement level is calibrated the way FanGraphs does it: 1,000 WAR per
full season across MLB, split 57% hitters / 43% pitchers.

Known simplifications: no park factors and no fielding metrics beyond the
positional adjustment; positions come from each player's primary position.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from . import data

# Linear weights (runs relative to an out) used for the wOBA numerator.
W_UBB, W_HBP, W_1B, W_2B, W_3B, W_HR = 0.69, 0.72, 0.88, 1.25, 1.58, 2.03
WOBA_SCALE = 1.20
SB_RUNS, CS_RUNS = 0.20, -0.42

# Positional adjustment, runs per full season (~680 PA).
POS_ADJ = {"C": 12.5, "SS": 7.5, "2B": 2.5, "3B": 2.5, "CF": 2.5, "LF": -7.5, "RF": -7.5,
           "OF": -2.5, "1B": -12.5, "DH": -17.5, "TWP": -17.5, "P": 0.0}
FULL_SEASON_PA = 680.0

# Marcel weights by season offset (0 = current season to date).
HIT_WEIGHTS = {0: 6, 1: 5, 2: 4, 3: 3}
PIT_WEIGHTS = {0: 4, 1: 3, 2: 2, 3: 1}
HIT_REGRESS_PA = 1200.0  # weighted PA of league-average added
PIT_REGRESS_IP = 400.0   # weighted IP of league-average added

TOTAL_WAR_PER_SEASON = 1000.0
HIT_SHARE = 0.57
TEAM_GAMES_PER_SEASON = 30 * 162


@dataclass
class LeagueContext:
    season: int
    lg_woba: float
    lg_wsb_rate: float      # wSB runs per SB opportunity
    lg_ra9: float
    fip_const: float
    runs_per_win: float
    hit_repl_per_pa: float  # replacement runs added per PA
    pit_repl_scale: float   # multiplier on role-based replacement runs per IP
    lg_r_per_pa: float = 0.118  # league runs per plate appearance


# --------------------------------------------------------------- helpers


def _woba_parts(h: pd.DataFrame):
    singles = h["H"] - h["2B"] - h["3B"] - h["HR"]
    num = (W_UBB * (h["BB"] - h["IBB"]) + W_HBP * h["HBP"] + W_1B * singles
           + W_2B * h["2B"] + W_3B * h["3B"] + W_HR * h["HR"])
    den = h["AB"] + h["BB"] - h["IBB"] + h["SF"] + h["HBP"]
    sb_opp = singles + h["BB"] - h["IBB"] + h["HBP"]
    return num, den, sb_opp


def _ip(p: pd.DataFrame):
    return p["OUTS"] / 3.0


def _fip_raw(p: pd.DataFrame):
    ip = _ip(p).replace(0, np.nan)
    return (13 * p["HR"] + 3 * (p["BB"] - p["IBB"] + p["HBP"]) - 2 * p["SO"]) / ip


def _sp_share(p: pd.DataFrame):
    return (p["GS"] / p["GP"].replace(0, np.nan)).fillna(0.0).clip(0, 1)


def _repl_ra9_mult(sp_share):
    # Replacement-level win% ~.380 for starters, ~.470 for relievers, converted
    # to an RA9 multiple of league average via Pythagorean exponent 1.83.
    sp = (0.62 / 0.38) ** (1 / 1.83)
    rp = (0.53 / 0.47) ** (1 / 1.83)
    return sp_share * sp + (1 - sp_share) * rp


def _batting_raa(h: pd.DataFrame, lg_woba: float, lg_wsb_rate: float) -> pd.DataFrame:
    num, den, sb_opp = _woba_parts(h)
    woba = num / den.replace(0, np.nan)
    wraa = ((woba - lg_woba) / WOBA_SCALE * den).fillna(0.0)
    wsb = SB_RUNS * h["SB"] + CS_RUNS * h["CS"] - lg_wsb_rate * sb_opp
    pos = h["pos"].map(POS_ADJ).fillna(0.0)
    posadj = pos * h["PA"] / FULL_SEASON_PA
    return pd.DataFrame({"wraa": wraa, "wsb": wsb, "posadj": posadj}, index=h.index)


def build_context(season: int, hit: pd.DataFrame, pit: pd.DataFrame) -> LeagueContext:
    """League constants + replacement calibration from a stats sample."""
    num, den, sb_opp = _woba_parts(hit)
    lg_woba = num.sum() / den.sum()
    lg_wsb_rate = (SB_RUNS * hit["SB"].sum() + CS_RUNS * hit["CS"].sum()) / sb_opp.sum()
    ip = _ip(pit)
    lg_ip = ip.sum()
    lg_ra9 = pit["R"].sum() / lg_ip * 9
    lg_fip_raw = (13 * pit["HR"].sum() + 3 * (pit["BB"].sum() - pit["IBB"].sum() + pit["HBP"].sum())
                  - 2 * pit["SO"].sum()) / lg_ip
    fip_const = lg_ra9 - lg_fip_raw
    rpw = 9 * (pit["R"].sum() / lg_ip) * 1.5 + 3

    # Season fraction represented by this sample (team-games ~ IP / 9 / 2 per game)
    team_games = lg_ip / 9.0
    frac = team_games / TEAM_GAMES_PER_SEASON
    target_runs = TOTAL_WAR_PER_SEASON * frac * rpw

    b = _batting_raa(hit, lg_woba, lg_wsb_rate)
    hit_raa = (b["wraa"] + b["wsb"] + b["posadj"]).sum()
    hit_repl_per_pa = (HIT_SHARE * target_runs - hit_raa) / hit["PA"].sum()

    fip = _fip_raw(pit) + fip_const
    pit_raa = ((lg_ra9 - fip) * ip / 9).fillna(0.0).sum()
    base_repl = ((_repl_ra9_mult(_sp_share(pit)) - 1) * lg_ra9 * ip / 9).sum()
    pit_repl_scale = ((1 - HIT_SHARE) * target_runs - pit_raa) / base_repl

    lg_r_per_pa = pit["R"].sum() / max(pit["BF"].sum(), 1)
    return LeagueContext(season, lg_woba, lg_wsb_rate, lg_ra9, fip_const, rpw,
                         hit_repl_per_pa, pit_repl_scale, lg_r_per_pa)


def _hitter_values(h: pd.DataFrame, ctx: LeagueContext) -> pd.DataFrame:
    b = _batting_raa(h, ctx.lg_woba, ctx.lg_wsb_rate)
    out = pd.DataFrame({"player_id": h["player_id"], "PA": h["PA"]})
    out["off_raa"] = b["wraa"] + b["wsb"]
    out["off_rar"] = out["off_raa"] + ctx.hit_repl_per_pa * h["PA"]
    out["def_runs"] = b["posadj"]
    return out


def _pitcher_values(p: pd.DataFrame, ctx: LeagueContext) -> pd.DataFrame:
    ip = _ip(p)
    fip = _fip_raw(p) + ctx.fip_const
    out = pd.DataFrame({"player_id": p["player_id"], "IP": ip, "sp_share": _sp_share(p)})
    out["pit_raa"] = ((ctx.lg_ra9 - fip) * ip / 9).fillna(0.0)
    repl = (_repl_ra9_mult(out["sp_share"]) - 1) * ctx.lg_ra9 / 9 * ctx.pit_repl_scale
    out["pit_repl_per_ip"] = repl
    out["pit_rar"] = out["pit_raa"] + repl * ip
    return out


# --------------------------------------------------------------- main entry


@dataclass
class PlayerValues:
    ctx: LeagueContext
    table: pd.DataFrame  # one row per player with rates
    team_games_played: float
    hit_todate: pd.DataFrame
    pit_todate: pd.DataFrame
    recent_label: str = ""


def _age_on(birth: str | None, on: date) -> float | None:
    if not birth:
        return None
    b = pd.Timestamp(birth).date()
    return (on - b).days / 365.25


def value_all_players(season: int, as_of: date, team_games_played: float
                      ) -> tuple[PlayerValues, pd.DataFrame, pd.DataFrame]:
    hit, pit = data.stats_to_date(season, as_of)

    # League constants: use this season to date once the sample is reasonable,
    # otherwise fall back to the full prior season.
    if hit["PA"].sum() >= 30000 and pit["OUTS"].sum() > 0:
        ctx = build_context(season, hit, pit)
    else:
        ph, pp = data.season_stats(season - 1)
        ctx = build_context(season, ph, pp)

    hv = _hitter_values(hit, ctx) if len(hit) else pd.DataFrame(columns=["player_id"])
    pv = _pitcher_values(pit, ctx) if len(pit) else pd.DataFrame(columns=["player_id"])

    # ---------- Marcel-style projections (rates relative to each season's league)
    hist_h, hist_p = [], []
    for off in (1, 2, 3):
        yr = season - off
        sh, sp = data.season_stats(yr)
        if len(sh) == 0:
            continue
        yctx = build_context(yr, sh, sp)
        hh = _hitter_values(sh, yctx)
        hh["off"] = off
        hist_h.append(hh[["player_id", "PA", "off_raa", "off"]])
        pp_ = _pitcher_values(sp, yctx)
        pp_["off"] = off
        hist_p.append(pp_[["player_id", "IP", "pit_raa", "sp_share", "off"]])
    if len(hv):
        cur_h = hv[["player_id", "PA", "off_raa"]].copy()
        cur_h["off"] = 0
        hist_h.append(cur_h)
    if len(pv):
        cur_p = pv[["player_id", "IP", "pit_raa", "sp_share"]].copy()
        cur_p["off"] = 0
        hist_p.append(cur_p)

    H = pd.concat(hist_h, ignore_index=True) if hist_h else pd.DataFrame(columns=["player_id", "PA", "off_raa", "off"])
    P = pd.concat(hist_p, ignore_index=True) if hist_p else pd.DataFrame(columns=["player_id", "IP", "pit_raa", "sp_share", "off"])
    H["w"] = H["off"].map(HIT_WEIGHTS)
    P["w"] = P["off"].map(PIT_WEIGHTS)

    hp = H.assign(wraa=H["w"] * H["off_raa"], wpa=H["w"] * H["PA"]).groupby("player_id")[["wraa", "wpa"]].sum()
    hp["proj_off_raa_per_pa"] = hp["wraa"] / (hp["wpa"] + HIT_REGRESS_PA)

    pp = P.assign(wraa=P["w"] * P["pit_raa"], wip=P["w"] * P["IP"],
                  wsp=P["w"] * P["IP"] * P["sp_share"]).groupby("player_id")[["wraa", "wip", "wsp"]].sum()
    pp["proj_pit_raa_per_ip"] = pp["wraa"] / (pp["wip"] + PIT_REGRESS_IP)
    pp["proj_sp_share"] = (pp["wsp"] / pp["wip"].replace(0, np.nan)).fillna(0.0)

    # ---------- "recent" stats for the WAR and run-differential models: this
    # season to date, or (before ~a month of games) last season plus this one.
    recent_label = f"{season} so far"
    if hit["PA"].sum() < 30000:
        ph_full, pp_full = data.season_stats(season - 1)
        def _combine(a, b):
            both = pd.concat([x for x in (a, b) if len(x)], ignore_index=True)
            if not len(both):
                return a
            num = [c for c in both.columns if c not in ("player_id", "name", "pos")]
            agg = {c: "sum" for c in num}
            agg.update({"name": "last", "pos": "last"})
            return both.groupby("player_id", as_index=False).agg(agg)
        hit = _combine(ph_full, hit)
        pit = _combine(pp_full, pit)
        recent_label = f"{season - 1}" + (f" + {season} so far" if hv is not None and len(hv) else "")
        hv = _hitter_values(hit, ctx) if len(hit) else pd.DataFrame(columns=["player_id"])
        pv = _pitcher_values(pit, ctx) if len(pit) else pd.DataFrame(columns=["player_id"])

    # ---------- assemble one row per player
    ids = pd.Index(sorted(set(hv.get("player_id", pd.Series(dtype=int)).tolist())
                          | set(pv.get("player_id", pd.Series(dtype=int)).tolist())), name="player_id")
    t = pd.DataFrame(index=ids)
    names = pd.concat([hit[["player_id", "name", "pos"]], pit[["player_id", "name", "pos"]]]).drop_duplicates("player_id")
    t = t.join(names.set_index("player_id"))
    if len(hv):
        t = t.join(hv.set_index("player_id"))
    if len(pv):
        t = t.join(pv.set_index("player_id"))
    for c in ["PA", "off_raa", "off_rar", "def_runs", "IP", "pit_raa", "pit_rar", "sp_share", "pit_repl_per_ip"]:
        if c not in t:
            t[c] = 0.0
    t = t.fillna({"PA": 0, "off_raa": 0, "off_rar": 0, "def_runs": 0, "IP": 0, "pit_raa": 0,
                  "pit_rar": 0, "sp_share": 0})
    tg = max(team_games_played, 1.0)
    t["pa_per_g"] = (t["PA"] / tg).clip(upper=4.6)
    t["ip_per_g"] = (t["IP"] / tg).clip(upper=1.3)
    t["war_todate"] = (t["off_rar"] + t["def_runs"] + t["pit_rar"]) / ctx.runs_per_win
    pvals = PlayerValues(ctx, t, team_games_played, hit, pit)
    pvals.recent_label = recent_label
    return pvals, hp, pp


def player_rates(pv: PlayerValues, hp: pd.DataFrame, pp: pd.DataFrame, player_ids: list[int],
                 as_of: date, positions: dict[int, str] | None = None) -> pd.DataFrame:
    """Per-player rates for the three models (runs above replacement per PA / IP)."""
    ctx = pv.ctx
    people = data.get_people(player_ids)
    rows = []
    for pid in player_ids:
        r = pv.table.loc[pid] if pid in pv.table.index else None
        info = people.get(pid, {})
        pos = (positions or {}).get(pid) or info.get("pos") or (r["pos"] if r is not None else "")
        name = info.get("name") or (r["name"] if r is not None else str(pid))
        pa = float(r["PA"]) if r is not None else 0.0
        ip = float(r["IP"]) if r is not None else 0.0
        pos_rate = POS_ADJ.get(pos, 0.0) / FULL_SEASON_PA
        is_pitcher = pos == "P"

        # to-date rates
        off_rate = float(r["off_rar"]) / pa if pa > 0 else 0.0  # no stats -> replacement
        pit_rate = float(r["pit_rar"]) / ip if ip > 0 else 0.0

        # projected rates
        age = _age_on(info.get("birth_date"), as_of)
        age_adj = 0.0
        if age is not None:
            k = 0.006 * ctx.lg_woba / WOBA_SCALE
            age_adj = k * (29 - age) if age < 29 else -0.5 * k * (age - 29)
        if pid in hp.index:
            proj_off = float(hp.loc[pid, "proj_off_raa_per_pa"]) + age_adj + ctx.hit_repl_per_pa
        else:
            proj_off = 0.0  # unknown hitter -> replacement level
        if pid in pp.index:
            sp_share = float(pp.loc[pid, "proj_sp_share"])
            repl = (_repl_ra9_mult(sp_share) - 1) * ctx.lg_ra9 / 9 * ctx.pit_repl_scale
            proj_pit = float(pp.loc[pid, "proj_pit_raa_per_ip"]) + age_adj * 4.3 + repl
        else:
            proj_pit = 0.0

        rows.append({
            "player_id": pid, "name": name, "pos": pos, "age": age,
            "PA": pa, "IP": round(ip, 1),
            "war_todate": float(r["war_todate"]) if r is not None else 0.0,
            "rem_pa_per_g": 0.0 if is_pitcher and pa < 10 else (float(r["pa_per_g"]) if r is not None else 0.0),
            "rem_ip_per_g": float(r["ip_per_g"]) if r is not None else 0.0,
            "todate_off_per_pa": off_rate,
            "def_per_pa": pos_rate,
            "todate_pit_per_ip": pit_rate,
            "proj_off_per_pa": proj_off,
            "proj_pit_per_ip": proj_pit,
            "sp_share": float(pp.loc[pid, "proj_sp_share"]) if pid in pp.index
            else (float(r["sp_share"]) if r is not None else 0.0),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- past seasons


_SEASON_WAR: dict[int, pd.Series] = {}


def season_war(season: int) -> pd.Series:
    """Full-season WAR by player_id for a completed season (same framework)."""
    if season not in _SEASON_WAR:
        h, p = data.season_stats(season)
        if len(h) == 0:
            _SEASON_WAR[season] = pd.Series(dtype=float)
        else:
            ctx = build_context(season, h, p)
            hv, pv = _hitter_values(h, ctx), _pitcher_values(p, ctx)
            w = pd.concat([
                pd.Series(((hv.off_rar + hv.def_runs) / ctx.runs_per_win).values, index=hv.player_id.values),
                pd.Series((pv.pit_rar / ctx.runs_per_win).values, index=pv.player_id.values),
            ])
            _SEASON_WAR[season] = w.groupby(level=0).sum()
    return _SEASON_WAR[season]


# --------------------------------------------------------------- custom players

# Full-season playing time used to express a "WAR pace" and as default usage.
CUSTOM_PACE = {"Hitter": 650.0, "SP": 180.0, "RP": 65.0}
CUSTOM_PER_GAME = {"Hitter": 4.2, "SP": 1.1, "RP": 0.4}
ERA_TO_RA9 = 1.08  # runs allowed run ~8% higher than earned runs


def custom_player_rates(spec: dict, ctx: LeagueContext) -> dict:
    """Rates for a user-defined player, in the same shape as ``player_rates`` rows.

    ``spec`` keys: name, kind ("Hitter" | "Pitcher"), pos (hitters),
    role ("SP" | "RP", pitchers), mode ("WAR pace" | "Rate stat"), value.
      * Hitter, WAR pace:  WAR per 650 PA.
      * Hitter, Rate stat: wRC+ (100 = league average bat).
      * Pitcher, WAR pace: WAR per 180 IP (SP) or 65 IP (RP).
      * Pitcher, Rate stat: FIP (or ERA) on the ERA scale.
    A custom player is treated as a known talent level, so the to-date and
    projection rates are the same.
    """
    off = def_ = pit = 0.0
    pa_g = ip_g = 0.0
    value = float(spec["value"])
    if spec["kind"] == "Hitter":
        pos = spec.get("pos") or "DH"
        def_ = POS_ADJ.get(pos, 0.0) / FULL_SEASON_PA
        if spec["mode"] == "WAR pace":
            off = value * ctx.runs_per_win / CUSTOM_PACE["Hitter"] - def_
        else:
            off = (value / 100.0 - 1.0) * ctx.lg_r_per_pa + ctx.hit_repl_per_pa
        pa_g = CUSTOM_PER_GAME["Hitter"]
    else:
        role = spec.get("role") or "SP"
        pos = "P"
        if spec["mode"] == "WAR pace":
            pit = value * ctx.runs_per_win / CUSTOM_PACE[role]
        else:
            sp_share = 1.0 if role == "SP" else 0.0
            repl = (_repl_ra9_mult(sp_share) - 1) * ctx.lg_ra9 / 9 * ctx.pit_repl_scale
            pit = (ctx.lg_ra9 - value * ERA_TO_RA9) / 9 + repl
        ip_g = CUSTOM_PER_GAME[role]
    pace = CUSTOM_PACE["Hitter"] if spec["kind"] == "Hitter" else CUSTOM_PACE[spec.get("role") or "SP"]
    war_pace = ((off + def_) * pace if spec["kind"] == "Hitter" else pit * pace) / ctx.runs_per_win
    return {
        "name": spec["name"], "pos": pos, "age": float(spec.get("age") or 27), "PA": 0.0, "IP": 0.0, "war_todate": float("nan"),
        "war_pace": war_pace,
        "rem_pa_per_g": pa_g, "rem_ip_per_g": ip_g,
        "todate_off_per_pa": off, "def_per_pa": def_, "todate_pit_per_ip": pit,
        "proj_off_per_pa": off, "proj_pit_per_ip": pit,
    }
