"""Streamlit app: playoff odds on any date, before and after a trade.

Run with:  streamlit run app.py
"""

from __future__ import annotations

from datetime import date
from io import StringIO

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from playoff_odds import data, players, trade

st.set_page_config(page_title="Trade Playoff Odds", page_icon="⚾", layout="wide")


# ------------------------------------------------------------------ caching

@st.cache_resource(show_spinner="Loading season data…")
def get_ctx(season: int, as_of: date) -> trade.Context:
    return trade.load_context(season, as_of)


@st.cache_data(show_spinner=False)
def season_bounds(season: int):
    s = data.get_schedule(season)
    return s.date.min().date(), s.date.max().date()


@st.cache_data(show_spinner=False)
def roster(team_id: int, as_of: date):
    return data.get_roster(team_id, as_of)


@st.cache_data(show_spinner="Simulating seasons…", max_entries=64)
def simulate_cached(season, as_of, moves_json, models, sims, seed):
    ctx = get_ctx(season, as_of)
    moves = pd.read_json(StringIO(moves_json)) if moves_json else None
    deltas = trade.trade_runs(moves, ctx.rem_games) if moves is not None and len(moves) else None
    res = trade.run_models(ctx, deltas, list(models), sims, seed)
    return {k: v.__dict__ for k, v in res.items()}, (deltas.to_dict() if deltas is not None else None)


# ------------------------------------------------------------------ sidebar

st.sidebar.title("⚾ Settings")
season = st.sidebar.selectbox("Season", data.SUPPORTED_SEASONS[::-1], index=1)
lo, hi = season_bounds(season)
default_day = min(max(date(season, 7, 31), lo), hi)
as_of = st.sidebar.date_input(
    "Odds as of (end of day)", value=default_day, min_value=lo, max_value=hi,
    help="Games on or before this date count as played. Every later game is simulated.",
)
model_keys = st.sidebar.multiselect(
    "Models", list(trade.MODELS), default=list(trade.MODELS), format_func=trade.MODELS.get,
)
sims = st.sidebar.select_slider("Simulations", [1000, 2000, 5000, 10000, 20000], value=5000)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
st.sidebar.caption(
    "The same random draws are used before and after the trade, so the "
    "difference reflects the trade rather than simulation noise."
)

if not model_keys:
    st.warning("Pick at least one model.")
    st.stop()

ctx = get_ctx(season, as_of)
teams = ctx.teams.set_index("team_id")
idx = ctx.idx

st.title("Trade Playoff Odds")
st.caption(
    f"{season} season · odds as of end of {as_of:%b %d, %Y} · "
    f"{len(ctx.setup.rem_home):,} games left to simulate"
)

# ------------------------------------------------------------------ trade builder

team_opts = ctx.teams.sort_values("name").team_id.tolist()
name_of = lambda t: f"{teams.loc[t, 'name']} ({ctx.base.loc[t, 'W']}-{ctx.base.loc[t, 'L']})"

c1, c2 = st.columns(2)
with c1:
    team_a = st.selectbox("Team A", team_opts, format_func=name_of,
                          index=team_opts.index(121) if 121 in team_opts else 0)
with c2:
    opts_b = [t for t in team_opts if t != team_a]
    team_b = st.selectbox("Team B", opts_b, format_func=name_of,
                          index=opts_b.index(116) if 116 in opts_b else 0)


def roster_labels(team_id):
    r = roster(team_id, as_of)
    tbl = ctx.pv.table
    lab, pos = {}, {}
    for p in sorted(r, key=lambda p: -(tbl.loc[p.player_id, "war_todate"] if p.player_id in tbl.index else -9)):
        war = tbl.loc[p.player_id, "war_todate"] if p.player_id in tbl.index else 0.0
        status = "" if p.status in ("Active", "") else f" · {p.status}"
        lab[p.player_id] = f"{p.name} ({p.pos}, {war:.1f} WAR){status}"
        pos[p.player_id] = p.pos
    return lab, pos


lab_a, pos_a = roster_labels(team_a)
lab_b, pos_b = roster_labels(team_b)
c1, c2 = st.columns(2)
with c1:
    send_a = st.multiselect(f"{teams.loc[team_a, 'short']} send", list(lab_a), format_func=lab_a.get,
                            placeholder="Players leaving Team A")
with c2:
    send_b = st.multiselect(f"{teams.loc[team_b, 'short']} send", list(lab_b), format_func=lab_b.get,
                            placeholder="Players leaving Team B")

moves = None
if send_a or send_b:
    positions = {**pos_a, **pos_b}
    rates = players.player_rates(ctx.pv, ctx.hp, ctx.pp, send_a + send_b, as_of, positions)
    rates["from"] = [teams.loc[team_a if p in send_a else team_b, "abbr"] for p in rates.player_id]
    rates["to"] = [teams.loc[team_b if p in send_a else team_a, "abbr"] for p in rates.player_id]
    rates["to_id"] = [team_b if p in send_a else team_a for p in rates.player_id]
    rates["rem_pa"] = (rates.rem_pa_per_g * rates.to_id.map(ctx.rem_games)).round(0)
    rates["rem_ip"] = (rates.rem_ip_per_g * rates.to_id.map(ctx.rem_games)).round(1)

    st.subheader("Players in the deal")
    st.caption(
        "Remaining PA and IP default to each player's season-to-date rate per team game. "
        "Edit them for injuries, role changes, or a hitter you expect to play every day."
    )
    edit = st.data_editor(
        rates[["name", "pos", "from", "to", "PA", "IP", "war_todate", "rem_pa", "rem_ip"]],
        column_config={
            "name": "Player", "pos": "Pos", "from": "From", "to": "To",
            "PA": st.column_config.NumberColumn("PA so far", disabled=True),
            "IP": st.column_config.NumberColumn("IP so far", disabled=True, format="%.1f"),
            "war_todate": st.column_config.NumberColumn("WAR so far", disabled=True, format="%.1f"),
            "rem_pa": st.column_config.NumberColumn("Remaining PA", min_value=0, max_value=450, step=5),
            "rem_ip": st.column_config.NumberColumn("Remaining IP", min_value=0.0, max_value=150.0, step=1.0, format="%.1f"),
        },
        disabled=["name", "pos", "from", "to"], hide_index=True, width="stretch",
    )
    rates["rem_pa"] = edit["rem_pa"].to_numpy()
    rates["rem_ip"] = edit["rem_ip"].to_numpy()

    # Rest-of-season WAR each model credits to each player
    rpw = ctx.pv.ctx.runs_per_win
    show = pd.DataFrame({"Player": rates.name})
    show["WAR (to date) model"] = ((rates.todate_off_per_pa + rates.def_per_pa) * rates.rem_pa
                                   + rates.todate_pit_per_ip * rates.rem_ip) / rpw
    show["Projection model"] = ((rates.proj_off_per_pa + rates.def_per_pa) * rates.rem_pa
                                + rates.proj_pit_per_ip * rates.rem_ip) / rpw
    show["Run diff: runs scored"] = rates.todate_off_per_pa * rates.rem_pa
    show["Run diff: runs prevented"] = rates.def_per_pa * rates.rem_pa + rates.todate_pit_per_ip * rates.rem_ip
    with st.expander("Rest-of-season value by model", expanded=False):
        st.dataframe(show.style.format({c: "{:+.2f}" for c in show.columns[1:]}),
                     hide_index=True, width="stretch")
        st.caption("WAR columns are wins above replacement over the remaining games; run-diff columns are runs.")

    arrive = rates.assign(team_id=rates.to_id, sign=1)
    depart = rates.assign(team_id=[teams.index[teams.abbr == f][0] for f in rates["from"]], sign=-1)
    moves = pd.concat([arrive, depart], ignore_index=True)[
        ["team_id", "sign", "rem_pa", "rem_ip", "todate_off_per_pa", "def_per_pa",
         "todate_pit_per_ip", "proj_off_per_pa", "proj_pit_per_ip"]]

# ------------------------------------------------------------------ simulate

res_raw, deltas = simulate_cached(
    season, as_of, moves.to_json() if moves is not None else None, tuple(model_keys), sims, int(seed)
)
res = {k: pd.Series(v) for k, v in res_raw.items()}


def stat(model, label, key, team_id):
    return res[(model, label)][key][idx[team_id]]


def blend(label, key, team_id):
    return float(np.mean([stat(m, label, key, team_id) for m in model_keys]))


st.divider()
blend_name = "average of selected models" if len(model_keys) > 1 else trade.MODELS[model_keys[0]]
st.subheader("Playoff odds")
st.caption(f"Headline numbers are the {blend_name}.")

cols = st.columns(2)
for col, tid in zip(cols, (team_a, team_b)):
    with col:
        b, a = blend("before", "playoff", tid), blend("after", "playoff", tid)
        st.metric(f"{teams.loc[tid, 'name']}: make playoffs", f"{a:.1%}",
                  f"{(a - b) * 100:+.1f} pts (was {b:.1%})" if moves is not None else None)
        m1, m2, m3 = st.columns(3)
        for mc, key, lbl in ((m1, "division", "Division"), (m2, "bye", "Bye"),
                             (m3, "mean_wins", "Wins")):
            bb, aa = blend("before", key, tid), blend("after", key, tid)
            if key == "mean_wins":
                mc.metric(lbl, f"{aa:.0f}", f"{aa - bb:+.1f}" if moves is not None else None)
            else:
                mc.metric(lbl, f"{aa:.0%}", f"{(aa - bb) * 100:+.1f} pts" if moves is not None else None)

# per-model table
rows = []
for tid in (team_a, team_b):
    for m in model_keys:
        rows.append({
            "Team": teams.loc[tid, "abbr"], "Model": trade.MODELS[m],
            "Playoffs before": stat(m, "before", "playoff", tid),
            "Playoffs after": stat(m, "after", "playoff", tid),
            "Δ pts": (stat(m, "after", "playoff", tid) - stat(m, "before", "playoff", tid)) * 100,
            "Division after": stat(m, "after", "division", tid),
            "Bye after": stat(m, "after", "bye", tid),
            "Wins before": stat(m, "before", "mean_wins", tid),
            "Wins after": stat(m, "after", "mean_wins", tid),
        })
st.dataframe(
    pd.DataFrame(rows).style.format({
        "Playoffs before": "{:.1%}", "Playoffs after": "{:.1%}", "Δ pts": "{:+.1f}",
        "Division after": "{:.1%}", "Bye after": "{:.1%}", "Wins before": "{:.1f}", "Wins after": "{:.1f}",
    }),
    hide_index=True, width="stretch",
)

# seed distribution chart
st.subheader("Seed distribution")
seed_rows = []
for tid in (team_a, team_b):
    for label in ("before", "after"):
        dist = np.mean([np.array(res[(m, label)]["seed_dist"])[idx[tid]] for m in model_keys], axis=0)
        for k, p in enumerate(dist):
            seed_rows.append({"Team": teams.loc[tid, "abbr"], "Scenario": label.title(),
                              "Seed": "Out" if k == 0 else f"#{k}", "Probability": p})
sd = pd.DataFrame(seed_rows)
chart = (
    alt.Chart(sd)
    .mark_bar()
    .encode(
        x=alt.X("Seed:N", sort=["#1", "#2", "#3", "#4", "#5", "#6", "Out"], title=None),
        xOffset=alt.XOffset("Scenario:N", sort=["Before", "After"]),
        y=alt.Y("Probability:Q", axis=alt.Axis(format="%"), title=None),
        color=alt.Color("Scenario:N", sort=["Before", "After"],
                        scale=alt.Scale(range=["#9aa5b1", "#2563eb"]), legend=alt.Legend(orient="top")),
        column=alt.Column("Team:N", title=None),
        tooltip=["Team", "Scenario", "Seed", alt.Tooltip("Probability:Q", format=".1%")],
    )
    .properties(width=320, height=220)
)
st.altair_chart(chart)

# league-wide table
st.subheader("Everyone else")
st.caption("A trade also shifts the odds of the teams competing with the two trade partners for spots.")
base_wp = trade.scenario_wp(ctx, None)
lg_rows = []
for tid in ctx.teams.team_id:
    b, a = blend("before", "playoff", tid), blend("after", "playoff", tid)
    lg_rows.append({
        "Team": teams.loc[tid, "name"], "League": teams.loc[tid, "league"], "Division": teams.loc[tid, "division"],
        "W": int(ctx.base.loc[tid, "W"]), "L": int(ctx.base.loc[tid, "L"]),
        "Talent W%": float(np.mean([base_wp[m][idx[tid]] for m in model_keys])),
        "Playoffs before": b, "Playoffs after": a, "Δ pts": (a - b) * 100,
    })
lg = pd.DataFrame(lg_rows)
leagues = sorted({teams.loc[team_a, "league"], teams.loc[team_b, "league"]})
tabs = st.tabs(leagues + ["All MLB"])
for tab, lgname in zip(tabs, leagues + ["All"]):
    with tab:
        view = lg if lgname == "All" else lg[lg.League == lgname]
        view = view.sort_values("Playoffs after", ascending=False)
        st.dataframe(
            view.style.format({"Talent W%": "{:.3f}", "Playoffs before": "{:.1%}",
                               "Playoffs after": "{:.1%}", "Δ pts": "{:+.1f}"}),
            hide_index=True, width="stretch", height=min(38 * (len(view) + 1), 1100),
        )

# ------------------------------------------------------------------ methodology

with st.expander("How it works"):
    c = ctx.pv.ctx
    st.markdown(f"""
**Simulation.** Every regular-season game after the as-of date is simulated {sims:,} times. Each game's
home-win probability comes from the two teams' talent win% (log5) plus home-field advantage (.540).
Each simulated season also draws its own "true" talent for every team around the estimate. The spread
is about ±.035 before the season and narrows as games are played, so early-season odds aren't
overconfident.
Playoff spots follow the 12-team format: 3 division winners and 3 wild cards per league, with the top
two division winners getting byes.

**Tiebreakers.** These follow MLB's rules (no tiebreaker games): head-to-head record (combined among
all tied clubs when 3 or more are tied), then intradivision record, then intraleague record, then record in
the last half of intraleague games, extended back one game at a time. Run on the full 2022–2025
seasons, the simulator reproduces every actual playoff field and seed.

**Team talent.** For the WAR and projection models, talent is half actual record and half Pythagorean
record, regressed toward a prior with 70 games of weight. The prior is last season's Pythagorean record,
regressed halfway to .500. The run-differential model regresses runs scored and allowed per game
separately (40 games), then converts them with Pythagenpat.

**Player value.** Hitters get wOBA-based batting runs, plus baserunning (SB/CS), plus a positional
adjustment. Pitchers get FIP-based runs saved, with replacement level set separately for starters and
relievers. Replacement level is calibrated to 1,000 WAR per season (57% hitters, 43% pitchers).
Runs per win this season: **{c.runs_per_win:.2f}**.
- *WAR (to date)*: season-to-date runs above replacement per PA or IP, times remaining playing time.
  The total is converted to wins and spread over the team's remaining games.
- *Projection (Marcel)*: weights this season and the three prior seasons 6/5/4/3 for hitters and
  4/3/2/1 for pitchers, regresses to league average, and adjusts for age.
- *Run differential*: offensive runs move the team's runs scored. Pitching and defensive runs move
  runs allowed. Win% is then recomputed with Pythagenpat, so a great player helps a bit less on an
  already-great team.

**Trade logic.** A departing player's playing time is filled at replacement level, and an arriving
player's time displaces replacement level. This is why the remaining PA and IP you enter matter.

**Not modeled.** Park factors, fielding metrics beyond the positional adjustment, reliever leverage,
injuries after the as-of date (edit playing time instead), and postseason series odds.
""")
