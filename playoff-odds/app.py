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

from playoff_odds import contracts, data, players, trade, value

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
st.sidebar.divider()
st.sidebar.subheader("Trade value")
spot_value = st.sidebar.number_input(
    "Value of a playoff spot ($M)", min_value=0, max_value=200, value=40, step=5,
    help="Converts each team's change in playoff odds into dollars, so contenders pay up for help now. "
         "Set to 0 to judge on player value alone.",
)
tol_pct = st.sidebar.slider(
    "Max value a team will give up", 0, 50, 15, step=5, format="%d%%",
    help="A team rejects the deal if it loses more than this share of the deal's size (minimum $3M).",
) / 100
growth_pct = st.sidebar.number_input(
    "Salary growth per year (%)", min_value=0.0, max_value=10.0,
    value=round(contracts.trend_growth() * 100, 1), step=0.1, format="%.1f",
    help="How fast the league-average salary (and the price of a win) rises in future seasons. "
         "The default is the long-run trend over the last 20 years of published averages.",
)
growth = growth_pct / 100
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

# ------------------------------------------------------------------ custom players

if "custom_players" not in st.session_state:
    st.session_state.custom_players = []
library: list[dict] = st.session_state.custom_players
abbr_a, abbr_b = teams.loc[team_a, "abbr"], teams.loc[team_b, "abbr"]
HITTER_POS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH"]


def custom_label(spec: dict) -> str:
    if spec["kind"] == "Hitter":
        what = f"{spec['pos']}, " + (f"{spec['value']:.1f} WAR/650 PA" if spec["mode"] == "WAR pace"
                                     else f"{spec['value']:.0f} wRC+")
    else:
        what = f"{spec['role']}, " + (f"{spec['value']:.1f} WAR/{'180' if spec['role'] == 'SP' else '65'} IP"
                                      if spec["mode"] == "WAR pace" else f"{spec['value']:.2f} FIP")
    return f"{spec['name']} ({what})"


with st.expander("➕ Custom players: free agents, call-ups, or anyone you want to make up"):
    st.caption(
        "Describe a player by talent level instead of looking him up. Custom players are kept for this "
        "browser session. Download the list to keep it, and upload it next time."
    )
    with st.form("new_custom", clear_on_submit=True):
        f1, f2, f3 = st.columns([2, 1, 1])
        cname = f1.text_input("Name", placeholder="e.g. Top prospect call-up")
        ckind = f2.radio("Type", ["Hitter", "Pitcher"], horizontal=True)
        cmode = f3.radio("Describe by", ["WAR pace", "Rate stat"], horizontal=True,
                         help="WAR pace: WAR per 650 PA (hitters) or per 180 IP / 65 IP (starters / relievers). "
                              "Rate stat: wRC+ for hitters, FIP or ERA for pitchers.")
        g1, g2, g4, g5, g3 = st.columns(5)
        cpos = g1.selectbox("Position (hitters)", HITTER_POS, index=HITTER_POS.index("CF"))
        crole = g2.selectbox("Role (pitchers)", ["SP", "RP"])
        cage = g4.number_input("Age", min_value=18, max_value=45, value=27, step=1,
                               help="Used for aging and contract predictions in trade value.")
        csvc = g5.number_input("MLB service (yrs)", min_value=0.0, max_value=25.0, value=1.0, step=0.5,
                               help="Under 3: pre-arbitration. 3 to 6: arbitration. 6+: free agent. "
                                    "Drives the predicted salary and the years of control.")
        cvalue = g3.number_input(
            "Value", value=3.0, step=0.1, format="%.2f",
            help="WAR pace: 2 is average, 5 is all-star, 8+ is MVP-level. wRC+: 100 is average. "
                 "FIP: about 4.10 is average.",
        )
        if st.form_submit_button("Add player", type="primary"):
            nm = cname.strip() or f"Custom {ckind.lower()} {len(library) + 1}"
            if cmode == "Rate stat" and ckind == "Hitter" and cvalue < 10:
                st.error("For hitters, the rate stat is wRC+ (e.g. 120), not WAR.")
            elif cmode == "Rate stat" and ckind == "Pitcher" and not 0.5 <= cvalue <= 9:
                st.error("For pitchers, the rate stat is FIP or ERA (e.g. 3.25).")
            else:
                library.append({"id": -(max([-x["id"] for x in library], default=0) + 1), "name": nm,
                                "kind": ckind, "pos": cpos if ckind == "Hitter" else "P",
                                "role": crole if ckind == "Pitcher" else None,
                                "mode": cmode, "value": float(cvalue), "age": int(cage),
                                "service": float(csvc)})
                st.rerun()

    if library:
        st.dataframe(pd.DataFrame([{"Player": custom_label(x),
                                    "WAR / full season": players.custom_player_rates(x, ctx.pv.ctx)["war_pace"]}
                                   for x in library]).style.format({"WAR / full season": "{:.1f}"}),
                     hide_index=True, width="stretch")
    d1, d2, d3 = st.columns([1, 1, 2])
    d1.download_button("Download list", data=pd.Series(library).to_json(orient="values"),
                       file_name="custom_players.json", mime="application/json", disabled=not library)
    if d2.button("Clear list", disabled=not library):
        st.session_state.custom_players = []
        st.rerun()
    up = d3.file_uploader("Upload a saved list", type="json", label_visibility="collapsed")
    if up is not None and st.session_state.get("_last_upload") != up.file_id:
        import json
        try:
            loaded = json.loads(up.getvalue())
            have = {(x["name"], x["kind"], x["value"]) for x in library}
            for x in loaded:
                if (x["name"], x["kind"], x["value"]) not in have:
                    x["id"] = -(max([-y["id"] for y in library], default=0) + 1)
                    library.append(x)
            st.session_state._last_upload = up.file_id
            st.rerun()
        except (ValueError, KeyError, TypeError):
            st.error("That file isn't a custom player list from this app.")

custom_in = []
if library:
    by_id = {x["id"]: x for x in library}
    c1, c2 = st.columns(2)
    with c1:
        join_a = st.multiselect(f"Custom players joining {teams.loc[team_a, 'short']}", list(by_id),
                                format_func=lambda i: custom_label(by_id[i]), placeholder="None")
    with c2:
        join_b = st.multiselect(f"Custom players joining {teams.loc[team_b, 'short']}",
                                [i for i in by_id if i not in join_a],
                                format_func=lambda i: custom_label(by_id[i]), placeholder="None")
    source = st.radio(
        "Custom players come from", ["Outside the deal (free agent, minors)", "The other team in the deal"],
        horizontal=True,
        help="'The other team' also removes the player from that club, as in a normal trade.",
    )
    from_other = source.startswith("The other")
    for i in join_a:
        custom_in.append((by_id[i], team_a, team_b if from_other else None))
    for i in join_b:
        custom_in.append((by_id[i], team_b, team_a if from_other else None))

moves = None
if send_a or send_b or custom_in:
    positions = {**pos_a, **pos_b}
    real = players.player_rates(ctx.pv, ctx.hp, ctx.pp, send_a + send_b, as_of, positions) \
        if (send_a or send_b) else pd.DataFrame()
    if len(real):
        real["to_id"] = [team_b if p in send_a else team_a for p in real.player_id]
        real["from_id"] = [team_a if p in send_a else team_b for p in real.player_id]
    custom_rows = []
    for spec, to_id, from_id in custom_in:
        row = players.custom_player_rates(spec, ctx.pv.ctx)
        row.update(player_id=spec["id"], to_id=to_id, from_id=from_id)
        custom_rows.append(row)
    rates = pd.concat([real, pd.DataFrame(custom_rows)], ignore_index=True)
    rates["from"] = [teams.loc[f, "abbr"] if pd.notna(f) else "Outside" for f in rates.from_id]
    rates["to"] = [teams.loc[t, "abbr"] for t in rates.to_id]
    rates["rem_pa"] = (rates.rem_pa_per_g * rates.to_id.map(ctx.rem_games)).round(0)
    rates["rem_ip"] = (rates.rem_ip_per_g * rates.to_id.map(ctx.rem_games)).round(1)

    st.subheader("Players in the deal")
    st.caption(
        "Remaining PA and IP default to each player's season-to-date rate per team game. Custom players "
        "default to everyday hitter, rotation starter, or regular reliever usage. Edit them for injuries, "
        "role changes, or a hitter you expect to play every day."
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
        st.caption("WAR columns are wins above replacement over the remaining games; run-diff columns are runs. "
                   "Custom players have the same value in every model.")

    arrive = rates.assign(team_id=rates.to_id, sign=1)
    depart = rates[rates.from_id.notna()].assign(sign=-1)
    depart["team_id"] = depart.from_id.astype(int)
    moves = pd.concat([arrive, depart], ignore_index=True)[
        ["team_id", "sign", "rem_pa", "rem_ip", "todate_off_per_pa", "def_per_pa",
         "todate_pit_per_ip", "proj_off_per_pa", "proj_pit_per_ip"]]
    moves["team_id"] = moves.team_id.astype(int)

    # ---------------------------------------------------------- trade value inputs
    st.subheader("Trade value")
    st.caption(
        "Surplus value = projected WAR × market $/WAR, minus salary, over the years of team control "
        f"(this season's remaining {ctx.rem_games[team_a] / 162:.0%} plus future seasons), discounted 8% a year. "
        "**Salaries are predicted from historical comparable players**: the 30 most similar seasons "
        "by service class, age, service time and recent WAR (2000–2016), scaled to the league-average salary "
        f"of each season (growing {growth:.1%}/yr after 2025). Years of control come from service time. "
        "Edit any of them to match the real contract. "
        f"Market rate this season: ${value.dollars_per_war(season):.1f}M per WAR."
    )
    is_custom = (rates.player_id < 0).to_numpy()
    war_season = np.where(
        is_custom,
        rates.get("war_pace", pd.Series(np.nan, index=rates.index)),
        ((rates.proj_off_per_pa + rates.def_per_pa) * (rates.rem_pa_per_g * 162).clip(upper=700)
         + rates.proj_pit_per_ip * (rates.rem_ip_per_g * 162).clip(upper=220)) / rpw,
    )
    rates["war_season"] = np.round(war_season.astype(float), 1)
    ages = [None if pd.isna(a) else float(a) for a in rates.age]
    custom_by_id = {x["id"]: x for x in library}

    services, priors, pitch_flags = [], [], []
    for k, r in rates.reset_index(drop=True).iterrows():
        pid = int(r.player_id)
        if pid < 0:
            spec = custom_by_id.get(pid, {})
            services.append(float(spec.get("service", 1.0)))
            priors.append([float(r.war_season)] * 3 if spec.get("service", 1.0) >= 1 else [])
            pitch_flags.append(spec.get("kind") == "Pitcher")
        else:
            svc, _ = contracts.player_history(pid, season)
            services.append(svc)
            priors.append([float(players.season_war(season - j).get(pid, 0.0)) for j in (1, 2, 3)])
            pitch_flags.append(r.pos == "P")
    rates["service"] = np.round(services, 1)
    rates["control"] = [contracts.default_control_years(sv, a) for sv, a in zip(services, ages)]

    def predicted_path(k, war_s, control):
        wars = value.war_path(war_s, ages[k], control)
        return contracts.salary_path(ages[k] if ages[k] is not None else 27.0, services[k], priors[k],
                                     wars, pitch_flags[k], season, growth)

    paths = [predicted_path(k, float(rates.war_season.iloc[k]), int(rates.control.iloc[k]))
             for k in range(len(rates))]
    rates["salary"] = [round(float(np.mean(pth.salaries)), 1) for pth in paths]
    rates["cls"] = [pth.first.cls for pth in paths]
    rates["age_disp"] = [None if a is None else int(a) for a in ages]
    vedit = st.data_editor(
        rates[["name", "from", "to", "age_disp", "service", "cls", "war_season", "control", "salary"]],
        column_config={
            "name": "Player", "from": "From", "to": "To",
            "age_disp": st.column_config.NumberColumn("Age", disabled=True),
            "service": st.column_config.NumberColumn("Service", disabled=True, format="%.1f",
                                                     help="Approximate MLB service years before this season."),
            "cls": st.column_config.TextColumn("Status", disabled=True),
            "war_season": st.column_config.NumberColumn(
                "Proj. WAR / season", step=0.1, format="%.1f",
                help="Full-season projection at the player's current playing time (projection model)."),
            "control": st.column_config.NumberColumn(
                "Control (yrs)", min_value=0, max_value=12, step=1,
                help="Seasons the acquiring team controls the player beyond this one."),
            "salary": st.column_config.NumberColumn(
                "Avg salary ($M/yr)", min_value=0.0, max_value=70.0, step=0.5, format="$%.1fM",
                help="Average of the predicted salaries over the control period. Type a number to replace "
                     "the prediction with a flat salary."),
        },
        disabled=["name", "from", "to", "age_disp", "service", "cls"], hide_index=True, width="stretch",
        key="value_editor",
    )
    value_players, contract_rows = [], []
    for k, r in rates.reset_index(drop=True).iterrows():
        war_s, control = float(vedit.war_season.iloc[k]), int(vedit.control.iloc[k])
        edited_war_or_ctl = (war_s != float(r.war_season)) or (control != int(r.control))
        pth = predicted_path(k, war_s, control) if edited_war_or_ctl else paths[k]
        typed = float(vedit.salary.iloc[k])
        salary_used = pth.salaries if abs(typed - float(r.salary)) < 0.05 else typed
        rem_frac = float(ctx.rem_games[r.to_id]) / 162
        sv = value.surplus_value(war_s, ages[k], control, salary_used, rem_frac, season, growth)
        value_players.append({"name": r["name"], "to_id": int(r.to_id),
                              "from_id": None if pd.isna(r.from_id) else int(r.from_id),
                              "surplus": sv.total, "production": sv.production, "salary": sv.salary,
                              "war_total": sv.war_total})
        contract_rows.append((r["name"], pth, salary_used))

    with st.expander("Contract predictions and comparable players"):
        for name, pth, used in contract_rows:
            yrs = [season + y for y in range(len(pth.salaries))]
            st.markdown(f"**{name}**: {pth.first.cls}, about {pth.first.multiple:.2f}× the league-average salary")
            line = " · ".join(f"{y}: ${s_:.1f}M" for y, s_ in zip(yrs, pth.salaries))
            st.caption(("Predicted salary by season: " + line) if isinstance(used, list)
                       else f"Using your flat salary of ${used:.1f}M/yr (prediction was: {line})")
            c = pth.first.comps.head(8).copy()
            c[f"Salary in {season} $"] = c.salary_index * contracts.avg_salary(season, growth)
            st.dataframe(
                c[["name", "year", "age", "service", "war_recent", "salary_m", "salary_index",
                   f"Salary in {season} $"]].rename(columns={
                    "name": "Comparable player", "year": "Year", "age": "Age", "service": "Service",
                    "war_recent": "Recent WAR", "salary_m": "Salary then ($M)", "salary_index": "× avg"}),
                hide_index=True, width="stretch",
                column_config={
                    "Recent WAR": st.column_config.NumberColumn(format="%.1f"),
                    "Salary then ($M)": st.column_config.NumberColumn(format="$%.2fM"),
                    "× avg": st.column_config.NumberColumn(format="%.2f×"),
                    f"Salary in {season} $": st.column_config.NumberColumn(format="$%.1fM"),
                },
            )
        st.caption("Comparable seasons come from the Lahman Baseball Database (SABR, CC BY-SA 3.0), which "
                   "tracked salaries through 2016. Salaries are compared as multiples of each season's "
                   "league-average salary, so 2005 and 2025 dollars line up.")

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

# ------------------------------------------------------------------ trade verdict

if moves is not None:
    pdelta = {t: blend("after", "playoff", t) - blend("before", "playoff", t) for t in (team_a, team_b)}
    verdicts = value.evaluate(value_players, pdelta, (team_a, team_b), spot_value, tol_pct)
    st.subheader("Trade verdict")
    no = [v for v in verdicts.values() if not v.accepts]
    if not no:
        st.success("✅ **Trade accepted.** Both teams come out within their tolerance.")
    else:
        msgs = []
        for v in no:
            short = teams.loc[v.team_id, "short"]
            gap = -v.net - v.tolerance
            msgs.append(f"the **{short}** say no: they'd lose **${-v.net:.1f}M** of value. Their limit is "
                        f"${v.tolerance:.1f}M, so add about **${gap:.1f}M** more for them.")
        st.error("❌ **Trade rejected.** " + " Also, ".join(m[0].upper() + m[1:] for m in msgs)
                 + " The playoff odds below are still shown so you can explore.")
    vc = st.columns(2)
    for col, t in zip(vc, (team_a, team_b)):
        v = verdicts[t]
        with col:
            st.markdown(f"**{teams.loc[t, 'name']}** {'✅ accepts' if v.accepts else '❌ rejects'}")
            st.dataframe(pd.DataFrame({
                "": ["Value received", "Value sent", "Contender premium", "Net"],
                "$M": [round(v.received, 1), round(-v.sent, 1), round(v.premium, 1),
                       round(round(v.received, 1) + round(-v.sent, 1) + round(v.premium, 1), 1)],
            }).style.format({"$M": "{:+.1f}"}), hide_index=True, width="stretch")
    with st.expander("Surplus value by player"):
        st.dataframe(pd.DataFrame([{
            "Player": p["name"], "To": teams.loc[p["to_id"], "abbr"],
            "From": teams.loc[p["from_id"], "abbr"] if p["from_id"] else "Outside",
            "Proj. WAR (control period)": p["war_total"], "Production ($M)": p["production"],
            "Salary owed ($M)": p["salary"], "Surplus ($M)": p["surplus"],
        } for p in value_players]).style.format({
            "Proj. WAR (control period)": "{:.1f}", "Production ($M)": "{:.1f}",
            "Salary owed ($M)": "{:.1f}", "Surplus ($M)": "{:+.1f}"}), hide_index=True, width="stretch")
        st.caption(f"Contender premium = change in playoff odds × ${spot_value}M per playoff spot "
                   "(average of the selected models). Dollars are discounted to today.")
    st.divider()

blend_name = "average of selected models" if len(model_keys) > 1 else trade.MODELS[model_keys[0]]
st.subheader("Playoff odds")
st.caption(f"Headline numbers are the {blend_name}.")



def shown(before: float, after: float, scale: float = 100.0) -> tuple[float, float, float]:
    """Round before/after to one decimal *first*, then subtract.

    Deltas computed from unrounded values can disagree with the numbers on
    screen (88.64 -> 85.96 shows as 88.6 -> 86.0 but "-2.7"), so every
    displayed change is the difference of the displayed values.
    """
    b, a = round(before * scale, 1), round(after * scale, 1)
    d = round(a - b, 1)
    return b, a, (0.0 if d == 0 else d)  # avoid "-0.0"


def delta_text(d: float, unit: str = "") -> str:
    return f"{d:+.1f}{unit}" if d != 0 else f"±0.0{unit}"


# Sub-metrics use a smaller font so one-decimal values fit in narrow columns.
st.markdown(
    "<style>[class*='st-key-submetrics'] [data-testid='stMetricValue']{font-size:1.6rem}</style>",
    unsafe_allow_html=True,
)
cols = st.columns(2)
for col, tid in zip(cols, (team_a, team_b)):
    with col:
        b, a, d = shown(blend("before", "playoff", tid), blend("after", "playoff", tid))
        st.metric(f"{teams.loc[tid, 'name']}: make playoffs", f"{a:.1f}%",
                  f"{delta_text(d, ' pts')} (was {b:.1f}%)" if moves is not None else None,
                  delta_color="normal" if d != 0 else "off")
        with st.container(key=f"submetrics_{tid}"):
            m1, m2, m3 = st.columns(3)
            for mc, key, lbl in ((m1, "division", "Division"), (m2, "bye", "Bye"),
                                 (m3, "mean_wins", "Wins")):
                scale = 1.0 if key == "mean_wins" else 100.0
                bb, aa, dd = shown(blend("before", key, tid), blend("after", key, tid), scale)
                unit = "" if key == "mean_wins" else "%"
                mc.metric(lbl, f"{aa:.1f}{unit}",
                          delta_text(dd, "" if key == "mean_wins" else " pts") if moves is not None else None,
                          delta_color="normal" if dd != 0 else "off")

# per-model table
rows = []
for tid in (team_a, team_b):
    for m in model_keys:
        rows.append({
            "Team": teams.loc[tid, "abbr"], "Model": trade.MODELS[m],
            **dict(zip(("Playoffs before", "Playoffs after", "Δ pts"),
                       shown(stat(m, "before", "playoff", tid), stat(m, "after", "playoff", tid)))),
            "Division after": round(stat(m, "after", "division", tid) * 100, 1),
            "Bye after": round(stat(m, "after", "bye", tid) * 100, 1),
            **dict(zip(("Wins before", "Wins after", "Δ wins"),
                       shown(stat(m, "before", "mean_wins", tid), stat(m, "after", "mean_wins", tid), 1.0))),
        })
st.dataframe(
    pd.DataFrame(rows).style.format({
        "Playoffs before": "{:.1f}%", "Playoffs after": "{:.1f}%", "Δ pts": "{:+.1f}",
        "Division after": "{:.1f}%", "Bye after": "{:.1f}%", "Wins before": "{:.1f}", "Wins after": "{:.1f}",
        "Δ wins": "{:+.1f}",
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
    b, a, d = shown(blend("before", "playoff", tid), blend("after", "playoff", tid))
    lg_rows.append({
        "Team": teams.loc[tid, "name"], "League": teams.loc[tid, "league"], "Division": teams.loc[tid, "division"],
        "W": int(ctx.base.loc[tid, "W"]), "L": int(ctx.base.loc[tid, "L"]),
        "Talent W%": float(np.mean([base_wp[m][idx[tid]] for m in model_keys])),
        "Playoffs before": b, "Playoffs after": a, "Δ pts": d,
    })
lg = pd.DataFrame(lg_rows)
leagues = sorted({teams.loc[team_a, "league"], teams.loc[team_b, "league"]})
tabs = st.tabs(leagues + ["All MLB"])
for tab, lgname in zip(tabs, leagues + ["All"]):
    with tab:
        view = lg if lgname == "All" else lg[lg.League == lgname]
        view = view.sort_values("Playoffs after", ascending=False)
        st.dataframe(
            view.style.format({"Talent W%": "{:.3f}", "Playoffs before": "{:.1f}%",
                               "Playoffs after": "{:.1f}%", "Δ pts": "{:+.1f}"}),
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

**Custom players.** You describe a custom player's talent directly, as a WAR pace (per 650 PA, or per
180 IP for starters and 65 IP for relievers) or as a rate stat (wRC+ for hitters, FIP or ERA for pitchers).
The app converts that to runs per PA or per IP with this season's league constants. The same value is used
in all three models, since there's no track record for them to disagree about.

**Trade value.** Each player's surplus value is projected WAR × market $/WAR, minus salary, summed over
this season's remaining games plus the years of team control, aged (+0.4 WAR/yr before 26, −0.5 from
29–32, −0.7 after) and discounted 8% a year. Control comes from service time. Salaries come from the 30 most similar historical player seasons (2000–2016, Lahman) in the same
service class, by age, service time and recent WAR. Each comp's salary is expressed as a multiple of that
season's league-average salary, then rescaled to the league average for each future season, which grows
with the salary-growth setting. Each team's net is value received − value sent + a contender premium (change in playoff odds ×
the value of a playoff spot). A team rejects if its net loss is bigger than the tolerance you set in the sidebar.

**Not modeled.** Park factors, fielding metrics beyond the positional adjustment, reliever leverage,
injuries after the as-of date (edit playing time instead), and postseason series odds.
""")
