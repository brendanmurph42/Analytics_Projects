"""Build playoff_odds/resources/contract_comps.csv from the Lahman database.

One row per player-season with a known salary (2000-2016, the last year
Lahman tracked salaries), with the features used to find comparable players:
age, MLB service (approximate), recent WAR, pitcher flag, and the salary as a
multiple of that season's league-average salary (so it can be carried forward
to any year).

Usage (one-time; the CSV is committed so the app doesn't need this):

    .venv/bin/pip install pyreadr
    curl -L -o /tmp/lahman.tar.gz https://cran.r-project.org/src/contrib/Lahman_14.0-0.tar.gz
    mkdir -p /tmp/lahman && tar -xzf /tmp/lahman.tar.gz -C /tmp/lahman
    .venv/bin/python scripts/build_contract_comps.py /tmp/lahman/Lahman/data

Also writes resources/fa_signings.csv: veterans who changed teams with a new
contract, with the average salary over their first seasons with the new club
(a proxy for the contract's annual value) and how long they stayed.

Data: Lahman Baseball Database (SABR), CC BY-SA 3.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playoff_odds import contracts, players  # noqa: E402

FIRST_SALARY_YEAR = 2000


def load(d: Path, name: str) -> pd.DataFrame:
    return list(pyreadr.read_r(str(d / f"{name}.RData")).values())[0]


def main(data_dir: str) -> None:
    d = Path(data_dir)
    bat = load(d, "Batting")
    pit = load(d, "Pitching")
    app = load(d, "Appearances")
    ppl = load(d, "People")
    sal = load(d, "Salaries")

    bat = bat[bat.yearID >= 1990].fillna(0)
    pit = pit[pit.yearID >= 1990].fillna(0)
    app = app[app.yearID >= 1990].fillna(0)

    # --- per player-season counting stats in this app's column names
    b = bat.rename(columns={"X2B": "2B", "X3B": "3B"})
    b["PA"] = b.AB + b.BB + b.HBP + b.SH + b.SF
    hcols = ["PA", "AB", "H", "2B", "3B", "HR", "BB", "IBB", "HBP", "SF", "SB", "CS", "G"]
    b = b.groupby(["playerID", "yearID"], as_index=False)[hcols].sum()

    p = pit.rename(columns={"IPouts": "OUTS", "BFP": "BF", "SO": "SO", "G": "GP"})
    pcols = ["BF", "OUTS", "HR", "BB", "IBB", "HBP", "SO", "R", "GS", "GP"]
    p = p.groupby(["playerID", "yearID"], as_index=False)[pcols].sum()

    # primary position from appearances
    poscols = {"G_c": "C", "G_1b": "1B", "G_2b": "2B", "G_3b": "3B", "G_ss": "SS",
               "G_lf": "LF", "G_cf": "CF", "G_rf": "RF", "G_dh": "DH", "G_p": "P"}
    a = app.groupby(["playerID", "yearID"], as_index=False)[list(poscols) + ["G_all"]].sum()
    a["pos"] = a[list(poscols)].idxmax(axis=1).map(poscols)
    b = b.merge(a[["playerID", "yearID", "pos"]], on=["playerID", "yearID"], how="left")
    b["pos"] = b.pos.fillna("DH")

    # --- WAR per player-season with the same framework the app uses
    wars = []
    for yr in sorted(b.yearID.unique()):
        hb = b[b.yearID == yr].assign(player_id=lambda x: x.playerID)
        pb = p[p.yearID == yr].assign(player_id=lambda x: x.playerID)
        ctx = players.build_context(int(yr), hb, pb)
        hv = players._hitter_values(hb, ctx)
        pv = players._pitcher_values(pb, ctx)
        hw = pd.DataFrame({"playerID": hb.playerID.values,
                           "w": ((hv.off_rar + hv.def_runs) / ctx.runs_per_win).values})
        pw = pd.DataFrame({"playerID": pb.playerID.values, "w": (pv.pit_rar / ctx.runs_per_win).values})
        w = pd.concat([hw, pw]).groupby("playerID", as_index=False).w.sum()
        w["yearID"] = yr
        wars.append(w)
    war = pd.concat(wars, ignore_index=True)

    # --- service-time proxy (debut date + seasons with MLB appearances)
    appeared = app[app.G_all > 0].groupby("playerID").yearID.apply(lambda x: set(int(v) for v in x))
    debut = load(d, "People").set_index("playerID").debut

    # --- recent WAR (3/2/1 weights on the three prior seasons)
    war_idx = war.set_index(["playerID", "yearID"]).w
    def recent(pid, yr):
        vals = [war_idx.get((pid, yr - k), 0.0) for k in (1, 2, 3)]
        return contracts.recent_war(vals)

    pitcher = p.set_index(["playerID", "yearID"]).OUTS / 3
    pa = b.set_index(["playerID", "yearID"]).PA

    # --- salaries
    s = sal[sal.yearID >= FIRST_SALARY_YEAR].groupby(["playerID", "yearID"], as_index=False).salary.sum()
    lg_avg = sal.groupby("yearID").salary.mean()
    s["salary_index"] = s.salary / s.yearID.map(lg_avg)

    ppl = ppl.set_index("playerID")
    rows = []
    for r in s.itertuples():
        pid, yr = r.playerID, int(r.yearID)
        if pid not in ppl.index or pd.isna(ppl.at[pid, "birthYear"]):
            continue
        by, bm = ppl.at[pid, "birthYear"], ppl.at[pid, "birthMonth"]
        age = yr - by - (1 if (not pd.isna(bm) and bm >= 7) else 0)  # baseball age (July 1)
        ip_prev = sum(pitcher.get((pid, yr - k), 0.0) for k in (1, 2, 3))
        pa_prev = sum(pa.get((pid, yr - k), 0.0) for k in (1, 2, 3))
        rows.append({
            "name": f"{ppl.at[pid, 'nameFirst']} {ppl.at[pid, 'nameLast']}",
            "year": yr, "age": int(age),
            "service": round(contracts.service_years(debut.get(pid), appeared.get(pid, set()), yr), 2),
            "war_recent": round(recent(pid, yr), 2),
            "pitcher": int(ip_prev * 4 > pa_prev),
            "salary_m": round(r.salary / 1e6, 3),
            "salary_index": round(r.salary_index, 4),
        })
    out = pd.DataFrame(rows)
    dest = ROOT / "playoff_odds" / "resources" / "contract_comps.csv"
    dest.parent.mkdir(exist_ok=True)
    out.to_csv(dest, index=False)

    # --- free-agent signings: veterans who changed teams with a real salary change.
    # Lahman has no transaction data. A team change with the salary moving more
    # than ~15% separates new deals from players traded mid-contract, and the
    # average salary over his first (up to five) seasons with the new club
    # approximates the contract's annual value (first-year salaries are often
    # backloaded or deferred).
    st = sal[sal.yearID >= FIRST_SALARY_YEAR - 1].sort_values("salary")
    teams_n = st.groupby(["playerID", "yearID"]).teamID.nunique()
    main_team = st.groupby(["playerID", "yearID"]).teamID.last()          # highest-paid stint
    tot = st.groupby(["playerID", "yearID"]).salary.sum()
    idx_of = out.set_index(["name", "year"])                               # features by name/year
    key = {(r.playerID, int(r.yearID)): r for r in s.itertuples()}
    last_year = int(sal.yearID.max())
    fa_rows = []
    for pid, grp in main_team.groupby(level=0):
        yrs = sorted(int(y) for y in grp.index.get_level_values(1))
        team = {int(y): grp.loc[(pid, y)] for y in yrs}
        for y in yrs:
            if y < FIRST_SALARY_YEAR or (y - 1) not in team or teams_n[(pid, y)] > 1:
                continue
            if team[y] == team[y - 1]:
                continue
            change = abs(np.log(tot[(pid, y)] / max(tot[(pid, y - 1)], 1)))
            if change < 0.15 or (pid, y) not in key or pid not in ppl.index:
                continue
            nm = f"{ppl.at[pid, 'nameFirst']} {ppl.at[pid, 'nameLast']}"
            if (nm, y) not in idx_of.index:
                continue
            feat = idx_of.loc[(nm, y)]
            if isinstance(feat, pd.DataFrame):
                feat = feat.iloc[0]
            if feat.service < 5.5:
                continue
            tenure = []
            for k in range(5):
                yy = y + k
                if yy in team and team[yy] == team[y] and teams_n[(pid, yy)] == 1 and (pid, yy) in key:
                    tenure.append(key[(pid, yy)].salary_index)
                else:
                    break
            fa_rows.append({
                "name": nm, "year": y, "age": int(feat.age), "war_recent": feat.war_recent,
                "pitcher": int(feat.pitcher), "service": feat.service,
                "aav_index": round(float(np.mean(tenure)), 4), "years": len(tenure),
                "censored": int(y + len(tenure) - 1 >= last_year),
                "first_salary_m": round(tot[(pid, y)] / 1e6, 3),
            })
    fa = pd.DataFrame(fa_rows)
    fa.to_csv(dest.parent / "fa_signings.csv", index=False)
    print(f"wrote {len(fa):,} free-agent signings")

    avg = lg_avg[lg_avg.index >= 1985].round(4)
    avg.rename("avg_salary").div(1e6).round(3).to_csv(dest.parent / "lahman_avg_salary.csv")
    print(f"wrote {len(out):,} rows to {dest}")


if __name__ == "__main__":
    main(sys.argv[1])
