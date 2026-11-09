"""Streamlit app: odds of making the next playoffs as of any date, before and after a move.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from playoff_odds import contracts, data, players, prospects, trade, value

st.set_page_config(page_title="Playoff Odds", page_icon="⚾", layout="wide")
RATE_COLS = trade.RATE_COLS


# ------------------------------------------------------------------ caching

@st.cache_resource(show_spinner="Loading rosters, projections and schedule…", max_entries=8)
def get_ctx(as_of: date) -> trade.Context:
    return trade.load_context(as_of)


@st.cache_data(show_spinner="Estimating contracts and surplus values…")
def roster_surplus(as_of: date, team_id: int, growth: float) -> dict[int, float]:
    """Default surplus value ($M) for everyone on a roster, for the dropdown labels."""
    ctx = get_ctx(as_of)
    r = ctx.rates[ctx.rates.team_id == team_id].copy()
    if not len(r):
        return {}
    r["to_id"] = team_id
    d = trade.player_value_defaults(ctx, r, growth)
    return {int(pid): x["surplus"] for pid, x in zip(r.player_id, d)}


@st.cache_data(show_spinner="Projecting free agents…")
def free_agent_table(as_of: date, growth: float) -> pd.DataFrame:
    """Free agents who played in the majors recently, with projections and predicted contracts."""
    ctx = get_ctx(as_of)
    recent = set(ctx.pv.table.index) | set(ctx.hp.index[ctx.hp.wpa > 0]) | set(ctx.pp.index[ctx.pp.wip > 0])
    fas = [f for f in ctx.free_agents if f.player_id in recent]
    if not fas:
        return pd.DataFrame()
    r = players.player_rates(ctx.pv, ctx.hp, ctx.pp, [f.player_id for f in fas], ctx.as_of,
                             {f.player_id: f.pos for f in fas})
    r["last_team"] = [f.last_team for f in fas]
    r["to_id"] = ctx.teams.team_id.iloc[0]   # placeholder; value doesn't depend on the club here
    d = trade.player_value_defaults(ctx, r, growth)
    r["war_season"] = [x["war_season"] for x in d]
    fac = [trade.free_agent_contract(row, ctx.season, growth) for _, row in r.iterrows()]
    r["fa_salary"] = [round(c.aav, 1) for c in fac]
    r["fa_years"] = [c.years for c in fac]
    return r.sort_values("war_season", ascending=False).set_index("player_id", drop=False)


@st.cache_data(show_spinner="Loading minor-league rosters…")
def org_prospects(team_id: int, as_of: date, exclude: tuple[int, ...]):
    return data.get_org_prospects(team_id, as_of, set(exclude))


@st.cache_data(show_spinner="Loading minor-league stats (first time for a date takes ~15s)…")
def milb_stats(season: int, as_of: date):
    return data.milb_stats_to_date(season, as_of)


@st.cache_data(show_spinner="Estimating prospect grades from historical comps…")
def prospect_estimates(season: int, as_of: date) -> pd.DataFrame:
    return prospects.estimate_grades(milb_stats(season, as_of), season)


@st.cache_data(show_spinner="Simulating seasons…", max_entries=64)
def simulate_cached(as_of, moves_json, models, sims, seed):
    ctx = get_ctx(as_of)
    moves = json.loads(moves_json) if moves_json else []
    deltas = trade.trade_effect(ctx, moves)[0] if moves else None
    res = trade.run_models(ctx, deltas, list(models), sims, seed)
    return {k: v.__dict__ for k, v in res.items()}, (deltas.to_dict() if deltas is not None else None)


# ------------------------------------------------------------------ sidebar

st.sidebar.title("⚾ Settings")
today = date.today()
as_of = st.sidebar.date_input(
    "Odds as of (end of day)", value=today, min_value=date(data.FIRST_SEASON, 3, 1), max_value=today,
    help="Odds are for the next playoffs after this date: this season's until its regular season ends, "
         "then next season's. Games through this date count as played, and everything after is simulated.",
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
    "The same random draws are used before and after a move, so the "
    "difference reflects the move rather than simulation noise."
)

if not model_keys:
    st.warning("Pick at least one model.")
    st.stop()

ctx = get_ctx(as_of)
season = ctx.season
teams = ctx.teams.set_index("team_id")
idx = ctx.idx
rem_frac = ctx.rem_frac
offseason = not ctx.in_season

st.title(f"{season} Playoff Odds")
if offseason:
    phase = "spring training" if as_of.year == season else "offseason"
    st.caption(
        f"As of the end of {as_of:%b %d, %Y} ({phase}). The full {season} season "
        f"({len(ctx.setup.rem_home):,} games, starting {ctx.season_start:%b %d}) is projected from each club's "
        "current roster, blended with last season's results."
    )
else:
    st.caption(f"As of the end of {as_of:%b %d, %Y} · {len(ctx.setup.rem_home):,} games left to simulate "
               f"in the {season} regular season (ends {ctx.season_end:%b %d}).")
if season >= 2027:
    st.info("MLB's labor agreement expired December 1, 2026. These odds assume the 2027 season keeps the "
            "current schedule and 12-team playoff format.", icon="ℹ️")

# ------------------------------------------------------------------ trade builder

team_opts = ctx.teams.sort_values("name").team_id.tolist()


def name_of(t):
    if offseason:
        w = round(ctx.base.loc[t, "wp_proj"] * 162)
        return f"{teams.loc[t, 'name']} (proj. {w}-{162 - w})"
    return f"{teams.loc[t, 'name']} ({ctx.base.loc[t, 'W']}-{ctx.base.loc[t, 'L']})"


c1, c2 = st.columns(2)
with c1:
    team_a = st.selectbox("Team A", team_opts, format_func=name_of,
                          index=team_opts.index(111) if 111 in team_opts else 0)
with c2:
    opts_b = [t for t in team_opts if t != team_a]
    team_b = st.selectbox("Team B", opts_b, format_func=name_of,
                          index=opts_b.index(158) if 158 in opts_b else 0)


def money(m: float) -> str:
    """$M figure for labels: $55M, -$12M, $0.4M."""
    a = abs(m)
    txt = f"${a:.1f}M" if a < 1 else f"${a:.0f}M"
    return ("-" if m < 0 else "") + txt


def roster_labels(team_id):
    r = ctx.rates[ctx.rates.team_id == team_id]
    tbl = ctx.pv.table
    sv = roster_surplus(as_of, team_id, round(growth, 4))
    war = {pid: (tbl.loc[pid, "war_todate"] if pid in tbl.index else 0.0) for pid in r.player_id}
    lab, pos = {}, {}
    for pid in sorted(r.player_id, key=lambda p: -war[p]):
        row = r.loc[pid]
        status = "" if row.status in ("Active", "") else f" · {row.status}"
        val = f", {money(sv[pid])}" if pid in sv else ""
        lab[pid] = f"{row['name']} ({row.pos}, {war[pid]:.1f} WAR{val}){status}"
        pos[pid] = row.pos
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
st.caption(f"Each player shows WAR for {ctx.pv.recent_label} and his default **surplus value**: projected "
           "production minus predicted salary over his years of control. You can edit the contract under Trade value.")

# ------------------------------------------------------------------ free agents

fa_a, fa_b = [], []
fat = free_agent_table(as_of, round(growth, 4))
if len(fat):
    def fa_label(pid):
        r = fat.loc[pid]
        age = f"{r.age:.0f}" if pd.notna(r.age) else "?"
        return (f"{r['name']} ({r.pos}, {age}) · proj. {r.war_season:.1f} WAR · predicted "
                f"{money(r.fa_salary)}/yr × {int(r.fa_years)} · last with {r.last_team}")
    with st.expander(f"✍️ Sign free agents ({len(fat)} unsigned players who played in the majors recently)",
                     expanded=offseason):
        st.caption(
            "Sorted by projected WAR for a full season. Contracts are predicted from his 20 most similar actual "
            "free-agent signings: major 2022–25 deals, plus 2000–16 signings scaled up "
            f"{contracts.era_gap():.2f}× for today's market (the measured gap for 3+ WAR players). "
            "Edit them under Trade value. Signings change the signing team's depth chart and odds, and the "
            "other team isn't affected.")
        c1, c2 = st.columns(2)
        with c1:
            fa_a = st.multiselect(f"{teams.loc[team_a, 'short']} sign", list(fat.index), format_func=fa_label,
                                  placeholder="Free agents joining Team A")
        with c2:
            fa_b = st.multiselect(f"{teams.loc[team_b, 'short']} sign", [i for i in fat.index if i not in fa_a],
                                  format_func=fa_label, placeholder="Free agents joining Team B")

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

# ------------------------------------------------------------------ prospects

prospect_values: list[dict] = []
pros_a, pros_b = [], []
include_prospects = st.toggle(
    "Include prospects", value=False,
    help="Adds each organization's minor leaguers (off the 40-man roster on this date). "
         "You grade each prospect, and the app turns the grade into trade value.",
)
if include_prospects:
    # Before a season starts, prospects are judged on last season's full minor-league stats.
    if offseason and as_of < date(season, 2, 1):
        milb_season, milb_date = season - 1, data.season_dates(season - 1)[1]
    else:
        milb_season, milb_date = season, as_of
    milb = milb_stats(milb_season, milb_date)
    p_info: dict[int, dict] = {}

    est = prospect_estimates(milb_season, milb_date)

    def default_eta(level):
        # ETA counts seasons from the target season; over the winter a season of development has passed
        return max(prospects.DEFAULT_ETA.get(level, 3) - (1 if offseason else 0), 0)

    def prospect_labels(team_id, forty_ids):
        lst = org_prospects(team_id, as_of, tuple(sorted(forty_ids)))
        lab, rows = {}, []
        for pr in lst:
            age = players._age_on(pr.birth_date, as_of)
            sm = prospects.summarize_stats(milb, pr.player_id, age)
            if offseason and as_of < date(season, 2, 1):
                sm["line"] = sm["line"].replace(": ", f" ({milb_season}): ", 1) if sm["level"] else sm["line"]
            e = est.loc[pr.player_id] if pr.player_id in est.index else None
            grade = int(e.est_grade) if e is not None else 40
            rank = int(e["rank"]) if e is not None else None
            p_info[pr.player_id] = {"name": pr.name, "pos": pr.pos, "age": age, **sm, "org": team_id,
                                    "est_grade": grade, "est_rank": rank,
                                    "exp_war": float(e.exp_war) if e is not None else 0.0}
            rows.append((rank if rank is not None else 10**6, pr.player_id))
        for _, pid in sorted(rows):
            i = p_info[pid]
            agetxt = f"{i['age']:.0f}" if i["age"] is not None else "?"
            eta = default_eta(i["level"])
            pv = prospects.prospect_value(i["est_grade"], eta, i["pitcher_stats"] or i["pos"] == "P",
                                          i["age"], season, growth).surplus
            rk = f"#{i['est_rank']}, " if i["est_rank"] else ""
            lab[pid] = (f"{i['name']} ({i['pos']}, {agetxt}) · est. FV {i['est_grade']} ({rk}{money(pv)}) · "
                        f"{i['line']}")
        return lab

    plab_a = prospect_labels(team_a, set(lab_a))
    plab_b = prospect_labels(team_b, set(lab_b))
    st.caption(
        "Sorted by **estimated grade**. Each prospect is matched with the 60 most similar minor-league seasons "
        "from 2015–2019, by level, age relative to the level, performance relative to the level, and playing "
        "time. His expected MLB WAR is what those players produced over the next six years, busts included. "
        "Grades follow his rank among all eligible minor leaguers (#1–4 → 70, through #15 → 65, #45 → 60, "
        "#120 → 55, #330 → 50, #900 → 45). It's a stats-only estimate: no scouting, injuries or makeup. "
        "Override the grade below when you know better."
    )
    c1, c2 = st.columns(2)
    with c1:
        pros_a = st.multiselect(f"{teams.loc[team_a, 'short']} prospects sent", list(plab_a),
                                format_func=plab_a.get, placeholder="Prospects leaving Team A")
    with c2:
        pros_b = st.multiselect(f"{teams.loc[team_b, 'short']} prospects sent", list(plab_b),
                                format_func=plab_b.get, placeholder="Prospects leaving Team B")

    if pros_a or pros_b:
        st.subheader("Prospects in the deal")
        st.caption(
            "Each prospect starts at his **estimated Future Value grade** (20–80 scouting scale). Change it if "
            "you disagree, for example to match a published prospect ranking. ETA is the number of seasons until he's expected to reach the majors. Value accounts "
            "for bust risk (higher for pitchers and for prospects further away), six years of team control, "
            "and predicted pre-arbitration and arbitration salaries. Prospects don't change the "
            f"{season} playoff odds."
        )
        prow = []
        for pid in pros_a + pros_b:
            i = p_info[pid]
            from_id = team_a if pid in pros_a else team_b
            pitcher = i["pitcher_stats"] or i["pos"] == "P"
            prow.append({
                "pid": pid, "Prospect": i["name"],
                "Grade": prospects.GRADE_LABELS[i["est_grade"]],
                "ETA (yrs)": default_eta(i["level"]),
                "Type": "Pitcher" if pitcher else "Hitter",
                "Pos": i["pos"],
                "Age": None if i["age"] is None else round(i["age"], 1),
                "Age vs level": i["age_vs_level"], "This season": i["line"],
                "From": teams.loc[from_id, "abbr"], "To": teams.loc[team_b if from_id == team_a else team_a, "abbr"],
            })
        pdf = pd.DataFrame(prow)
        pedit = st.data_editor(
            pdf.drop(columns=["pid"]),
            column_config={
                "Grade": st.column_config.SelectboxColumn(
                    "FV grade", options=list(prospects.GRADE_LABELS.values()), required=True,
                    help="FanGraphs-style Future Value. 50 = average regular, 60 = all-star level, "
                         "70 = superstar. Most organizations have only a few prospects at 55+."),
                "ETA (yrs)": st.column_config.NumberColumn(min_value=0, max_value=6, step=1),
                "Type": st.column_config.SelectboxColumn(options=["Hitter", "Pitcher"], required=True),
                "Age": st.column_config.NumberColumn(format="%.1f"),
                "Age vs level": st.column_config.NumberColumn(
                    format="%+.1f", help="Years older (+) or younger (−) than a typical player at his level."),
            },
            disabled=["Prospect", "Pos", "From", "To", "Age", "Age vs level", "This season"],
            hide_index=True, width="stretch", key="prospect_editor",
        )
        grade_of = {v: k for k, v in prospects.GRADE_LABELS.items()}
        for k, r in pdf.iterrows():
            grade = grade_of[pedit["Grade"].iloc[k]]
            pv = prospects.prospect_value(grade, int(pedit["ETA (yrs)"].iloc[k]),
                                          pedit["Type"].iloc[k] == "Pitcher",
                                          None if pd.isna(r["Age"]) else float(r["Age"]), season, growth)
            from_id = team_a if r.pid in pros_a else team_b
            prospect_values.append({
                "name": f"{r['Prospect']} (prospect, FV {grade})",
                "to_id": team_b if from_id == team_a else team_a, "from_id": from_id,
                "surplus": pv.surplus, "production": pv.production, "salary": pv.salary,
                "war_total": pv.war_total,
            })

# ------------------------------------------------------------------ the deal

value_players: list[dict] = []
moves_spec: list[dict] = []          # what goes into the depth-chart re-run
deal_rows: list[dict] = []           # one row per player for the tables below
for pid in send_a + send_b:
    frm, to = (team_a, team_b) if pid in send_a else (team_b, team_a)
    r = ctx.rates.loc[pid]
    moves_spec.append({"player_id": int(pid), "from_id": int(frm), "to_id": int(to)})
    deal_rows.append({**r[["name", "pos", "age"] + RATE_COLS + ["sp_share"]].to_dict(),
                      "player_id": int(pid), "from_id": frm, "to_id": to, "kind": "player"})
for pid in fa_a + fa_b:
    to = team_a if pid in fa_a else team_b
    r = fat.loc[pid]
    spec = {"player_id": int(pid), "from_id": None, "to_id": int(to), "pos": r.pos,
            **{c: float(r[c]) for c in RATE_COLS + ["sp_share"]}}
    moves_spec.append(spec)
    deal_rows.append({**r[["name", "pos", "age"] + RATE_COLS + ["sp_share"]].to_dict(),
                      "player_id": int(pid), "from_id": None, "to_id": to, "kind": "free agent",
                      "fa_salary": float(r.fa_salary), "fa_years": int(r.fa_years)})
for spec, to_id, from_id in custom_in:
    row = players.custom_player_rates(spec, ctx.pv.ctx)
    sp_share = 1.0 if spec.get("role") == "SP" else 0.0
    moves_spec.append({"player_id": int(spec["id"]), "from_id": None if from_id is None else int(from_id),
                       "to_id": int(to_id), "pos": row["pos"], "sp_share": sp_share,
                       **{c: float(row[c]) for c in RATE_COLS}})
    deal_rows.append({**{k: row[k] for k in ["name", "pos", "age", "war_pace"] + RATE_COLS},
                      "sp_share": sp_share, "player_id": int(spec["id"]), "from_id": from_id, "to_id": to_id,
                      "kind": "custom"})
# a custom player "from the other team" isn't on that roster; it's only removed there in spirit
for m in moves_spec:
    if m["player_id"] < 0:
        m["from_id"] = None

moves_json = None
if deal_rows:
    rates = pd.DataFrame(deal_rows)
    _, allocs_auto = trade.trade_effect(ctx, moves_spec)
    rates["season_pa"] = [float(allocs_auto[t].loc[p, "pa"]) if p in allocs_auto[t].index else 0.0
                          for p, t in zip(rates.player_id, rates.to_id)]
    rates["season_ip"] = [float(allocs_auto[t].loc[p, "ip"]) if p in allocs_auto[t].index else 0.0
                          for p, t in zip(rates.player_id, rates.to_id)]
    rates["exp_pa"] = (rates.season_pa * rates.to_id.map(rem_frac)).round(0)
    rates["exp_ip"] = (rates.season_ip * rates.to_id.map(rem_frac)).round(1)
    rates["from"] = [teams.loc[f, "abbr"] if pd.notna(f) else ("Free agent" if k == "free agent" else "Outside")
                     for f, k in zip(rates.from_id, rates.kind)]
    rates["to"] = [teams.loc[t, "abbr"] for t in rates.to_id]
    tbl = ctx.pv.table
    rates["recent_war"] = [tbl.loc[p, "war_todate"] if p in tbl.index else np.nan for p in rates.player_id]

    st.subheader("Players in the deal")
    span = f"the rest of the {season} season" if not offseason else f"the {season} season"
    st.caption(
        f"Expected PA and IP over {span} come from each player's spot on his new team's automatic depth chart. "
        "The depth chart is re-run after the move, so a star pushes a bench player out of the lineup. "
        "Edit them for injuries or a different role."
    )
    edit = st.data_editor(
        rates[["name", "pos", "from", "to", "recent_war", "exp_pa", "exp_ip"]],
        column_config={
            "name": "Player", "pos": "Pos", "from": "From", "to": "To",
            "recent_war": st.column_config.NumberColumn(f"WAR ({ctx.pv.recent_label})", disabled=True,
                                                        format="%.1f"),
            "exp_pa": st.column_config.NumberColumn("Expected PA", min_value=0, max_value=750, step=10),
            "exp_ip": st.column_config.NumberColumn("Expected IP", min_value=0.0, max_value=230.0, step=5.0,
                                                    format="%.1f"),
        },
        disabled=["name", "pos", "from", "to"], hide_index=True, width="stretch", key="deal_editor",
    )
    for k, m in enumerate(moves_spec):
        f = float(rem_frac[m["to_id"]]) or 1.0
        if abs(float(edit.exp_pa.iloc[k]) - float(rates.exp_pa.iloc[k])) > 0.5:
            m["fixed_pa"] = float(edit.exp_pa.iloc[k]) / f
        if abs(float(edit.exp_ip.iloc[k]) - float(rates.exp_ip.iloc[k])) > 0.05:
            m["fixed_ip"] = float(edit.exp_ip.iloc[k]) / f
    deltas_now, allocs_after = trade.trade_effect(ctx, moves_spec)
    rates["season_pa"] = [float(allocs_after[t].loc[p, "pa"]) if p in allocs_after[t].index else 0.0
                          for p, t in zip(rates.player_id, rates.to_id)]
    rates["season_ip"] = [float(allocs_after[t].loc[p, "ip"]) if p in allocs_after[t].index else 0.0
                          for p, t in zip(rates.player_id, rates.to_id)]
    moves_json = json.dumps(moves_spec)

    rpw = ctx.pv.ctx.runs_per_win
    with st.expander("Projected wins added, by model", expanded=False):
        wins = pd.DataFrame({
            "Team": [teams.loc[t, "abbr"] for t in deltas_now.index],
            "WAR (recent)": deltas_now.war_runs / rpw,
            "Projection": deltas_now.proj_runs / rpw,
            "Run diff: runs scored": deltas_now.rd_rs,
            "Run diff: runs allowed": deltas_now.rd_ra,
        })
        st.dataframe(wins.style.format({c: "{:+.1f}" for c in wins.columns[1:]}), hide_index=True, width="stretch")
        st.caption("Full-season effect of the move on each team after re-running its depth chart. WAR columns "
                   f"are wins, and run columns are runs. During the season only the remaining {season} games are "
                   "affected.")

    # ---------------------------------------------------------- trade value inputs
    st.subheader("Trade value")
    st.caption(
        "Surplus value = projected WAR × market $/WAR, minus salary, over the years of team control "
        f"({'the remaining ' + format(float(rem_frac[team_a]), '.0%') + ' of ' if not offseason else 'all of '}"
        f"the {season} season plus future seasons), discounted 8% a year. Free agents are valued on the "
        "contract you give them. "
        "**Salaries are predicted from historical comparable players**: the 30 most similar seasons "
        "by service class, age, service time and recent WAR (2000–2016), scaled to the league-average salary "
        f"of each season (growing {growth:.1%}/yr after 2025). Years of control come from service time. "
        "Edit any of them to match the real contract. "
        f"Market rate this season: ${value.dollars_per_war(season):.1f}M per WAR."
    )
    custom_by_id = {x["id"]: x for x in library}
    rates = rates.reset_index(drop=True)
    defaults = trade.player_value_defaults(ctx, rates, growth, custom_by_id)
    for k, d in enumerate(defaults):          # free agents: years and salary of the new contract
        if rates.kind.iloc[k] == "free agent":
            d["control"] = int(rates.fa_years.iloc[k]) - 1
            d["salary"] = float(rates.fa_salary.iloc[k])
    ages = [d["age"] for d in defaults]
    services = [d["service"] for d in defaults]
    priors = [d["priors"] for d in defaults]
    pitch_flags = [d["pitcher"] for d in defaults]
    paths = [d["path"] for d in defaults]
    rates["war_season"] = [d["war_season"] for d in defaults]
    rates["service"] = np.round(services, 1)
    rates["control"] = [d["control"] for d in defaults]
    rates["salary"] = [d["salary"] for d in defaults]
    rates["cls"] = [pth.first.cls for pth in paths]

    def predicted_path(k, war_s, control):
        wars = value.war_path(war_s, ages[k], control)
        return contracts.salary_path(ages[k] if ages[k] is not None else 27.0, services[k], priors[k],
                                     wars, pitch_flags[k], season, growth)

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
                help=f"Seasons the acquiring team controls the player beyond {season}. For a free agent, "
                     "the length of his new contract minus one."),
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
        if r.kind == "free agent" and abs(typed - float(r.salary)) < 0.05:
            typed = float(r.salary)
            salary_used = typed
        else:
            salary_used = pth.salaries if abs(typed - float(r.salary)) < 0.05 else typed
        rem_frac = float(ctx.rem_games[r.to_id]) / 162
        sv = value.surplus_value(war_s, ages[k], control, salary_used, rem_frac, season, growth)
        value_players.append({"name": r["name"], "to_id": int(r.to_id),
                              "from_id": None if pd.isna(r.from_id) else int(r.from_id),
                              "surplus": sv.total, "production": sv.production, "salary": sv.salary,
                              "war_total": sv.war_total})
        contract_rows.append((r["name"], pth, salary_used))

    with st.expander("Contract predictions and comparable players"):
        for k, (name, pth, used) in enumerate(contract_rows):
            if rates.kind.iloc[k] == "free agent":
                fac = trade.free_agent_contract(rates.iloc[k], season, growth)
                st.markdown(f"**{name}**: free agent, predicted **${fac.aav:.1f}M/yr × {fac.years} years** "
                            f"({fac.multiple:.1f}× the league-average salary)")
                c = fac.comps.head(8).copy()
                c["mult"] = np.exp(c.log_mult)
                c[f"In {season} $"] = c.mult * contracts.avg_salary(season, growth)
                st.dataframe(
                    c[["name", "year", "source", "age", "war_recent", "mult", f"In {season} $", "years"]].rename(
                        columns={"name": "Comparable signing", "year": "Year", "source": "Source", "age": "Age",
                                 "war_recent": "Recent WAR", "mult": "× avg", "years": "Years"}),
                    hide_index=True, width="stretch",
                    column_config={"Recent WAR": st.column_config.NumberColumn(format="%.1f"),
                                   "Age": st.column_config.NumberColumn(format="%.0f"),
                                   "× avg": st.column_config.NumberColumn(format="%.2f×"),
                                   f"In {season} $": st.column_config.NumberColumn(format="$%.1fM")})
                continue
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

res_raw, deltas = simulate_cached(as_of, moves_json, tuple(model_keys), sims, int(seed))
moves = moves_json
res = {k: pd.Series(v) for k, v in res_raw.items()}


def stat(model, label, key, team_id):
    return res[(model, label)][key][idx[team_id]]


def blend(label, key, team_id):
    return float(np.mean([stat(m, label, key, team_id) for m in model_keys]))


st.divider()

# ------------------------------------------------------------------ trade verdict

all_values = value_players + prospect_values
if all_values:
    pdelta = {t: blend("after", "playoff", t) - blend("before", "playoff", t) for t in (team_a, team_b)}
    verdicts = value.evaluate(all_values, pdelta, (team_a, team_b), spot_value, tol_pct)
    st.subheader("Deal verdict")
    no = [v for v in verdicts.values() if not v.accepts]
    if not no:
        st.success("✅ **Deal accepted.** Both teams come out within their tolerance.")
    else:
        msgs = []
        for v in no:
            short = teams.loc[v.team_id, "short"]
            gap = -v.net - v.tolerance
            msgs.append(f"the **{short}** say no: they'd lose **${-v.net:.1f}M** of value. Their limit is "
                        f"${v.tolerance:.1f}M, so add about **${gap:.1f}M** more for them.")
        st.error("❌ **Deal rejected.** " + " Also, ".join(m[0].upper() + m[1:] for m in msgs)
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
        } for p in all_values]).style.format({
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
st.caption("A move also shifts the odds of the teams competing with the two clubs for playoff spots.")
base_wp = trade.scenario_wp(ctx, None)
lg_rows = []
for tid in ctx.teams.team_id:
    b, a, d = shown(blend("before", "playoff", tid), blend("after", "playoff", tid))
    lg_rows.append({
        "Team": teams.loc[tid, "name"], "League": teams.loc[tid, "league"], "Division": teams.loc[tid, "division"],
        **({"Proj. W": round(float(np.mean([base_wp[m][idx[tid]] for m in model_keys]) * 162))}
           if offseason else {"W": int(ctx.base.loc[tid, "W"]), "L": int(ctx.base.loc[tid, "L"])}),
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
**Which playoffs.** Odds are always for the next playoffs after the chosen date: that season's through
the last day of its regular season, then the following season's. Games through the date count as played,
and every later game is simulated {sims:,} times.

**Simulation.** Each game's home-win probability comes from the two teams' talent win% (log5) plus
home-field advantage (.540). Each simulated season also draws its own "true" talent for every team. The
spread is about ±.041 (±6.6 wins) before a season, matching how far preseason estimates actually missed
in 2023–26, and it narrows as games are played. Playoff spots follow the 12-team format: 3 division
winners and 3 wild cards per league, with the top two division winners getting byes.

**Tiebreakers.** These follow MLB's rules (no tiebreaker games): head-to-head record (combined among
all tied clubs when 3 or more are tied), then intradivision record, then intraleague record, then record in
the last half of intraleague games, extended back one game at a time. Run on the full 2022–2025
seasons, the simulator reproduces every actual playoff field and seed.

**Rosters and depth charts.** Each club's roster on the date comes from every player's dated roster
history. An automatic depth chart ranks the roster by projection and hands out a season's playing time:
a starting and backup catcher, 8 regulars, 4 bench bats, a 5-man rotation plus 2 depth starters, 7 relievers
and 3 mop-up arms. The remainder (injuries, call-ups) is filled at replacement level, and players on the
injured list get less time during the season. Team strength is the sum of each player's rate × playing time.

**Team talent.** Before a season, each team's talent blends last season's Pythagorean record with its
roster projection, using weights fit on opening days 2023–26. Roster projections alone missed by more
(9.9–10.3 wins RMSE) than last season's record (9.3). The blend did slightly better (9.1–9.4), which is
about as well as preseason forecasts can do given 162-game luck. During the season, results so far
(half record, half Pythagorean) are blended with that preseason estimate at 70 games of weight
(runs scored and allowed separately at 40 games for the run-differential model).

**Player value.** Hitters get wOBA-based batting runs, plus baserunning (SB/CS), plus a positional
adjustment. Pitchers get FIP-based runs saved, with replacement level set separately for starters and
relievers. Replacement level is calibrated to 1,000 WAR per season (57% hitters, 43% pitchers).
Runs per win: **{c.runs_per_win:.2f}**.
- *WAR (recent)*: runs above replacement per PA or IP over {ctx.pv.recent_label}. Before about a month
  of games, this is last season plus the new season so far.
- *Projection (Marcel)*: weights this season and the three prior seasons 6/5/4/3 for hitters and
  4/3/2/1 for pitchers, regresses to league average, and adjusts for age.
- *Run differential*: recent offensive runs move the team's runs scored. Pitching and defensive runs move
  runs allowed. Win% is then recomputed with Pythagenpat, so a great player helps a bit less on an
  already-great team.

**Trades and signings.** A move re-runs the depth charts of both clubs. The arriving player takes the
role his projection earns, the players behind him lose playing time, and the departing player's time
goes to whoever is next on the depth chart. The full-season change in team strength applies to whatever
games remain. Free agents are anyone with no club on the date who played in the majors this season or
last. Their default contract comes from their 20 most similar actual free-agent signings: major 2022–25
deals, plus 2000–16 signings scaled up for today's market, where 3+ WAR players now cost about 1.5× more
relative to the league average. Contract length comes from the nearest recent signings.

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

**Prospects.** Each prospect's default grade is estimated from his 60 most similar minor-league seasons
from 2015–2019 (level, age vs level, performance vs level, playing time). His expected MLB WAR over six years
sets his rank, and the rank sets the grade. You can override the grade and the ETA. The grade sets the WAR he'd produce if he
pans out (50 ≈ 2, 60 ≈ 4, 70 ≈ 6). A bust-risk factor (higher for pitchers and for prospects further
away) scales that down. Six years of control at predicted pre-arb and arbitration salaries give the surplus
value, discounted from his debut. Prospects affect trade value only, not this season's odds.

**Not modeled.** Park factors, fielding metrics beyond the positional adjustment, reliever leverage,
platoons, minor leaguers outside the 40-man (except as prospects in trades), a new labor agreement's rules,
injuries after the as-of date (edit playing time instead), and postseason series odds.
""")
