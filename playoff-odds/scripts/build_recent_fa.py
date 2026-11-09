"""Add features to the hand-entered recent free-agent contracts (2022-2025).

Reads playoff_odds/resources/recent_fa_contracts.csv (name, first season,
years, total $M, entered from public reporting) and writes
playoff_odds/resources/recent_fa_signings.csv with the same features as the
Lahman-era signings: age, approximate service, recent WAR (3/2/1 over the three
prior seasons, with 2020 scaled to a 162-game pace), pitcher flag, and the
contract's annual value as a multiple of that season's league-average salary.

    .venv/bin/python scripts/build_recent_fa.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playoff_odds import contracts, data, players  # noqa: E402

RES = ROOT / "playoff_odds" / "resources"
src = pd.read_csv(RES / "recent_fa_contracts.csv")
rows = []
for r in src.itertuples():
    d = data._get("people/search", {"names": r.name, "sportIds": 1})
    season = int(r.first_season)
    people = [p for p in d.get("people", []) if p.get("mlbDebutDate")]
    if not people:
        print("not found:", r.name)
        continue
    # among namesakes, the one who actually played in the seasons before the deal
    played = lambda p: sum(p["id"] in players.season_war(season - j).index for j in (1, 2, 3))
    p = max(people, key=lambda p: (played(p), p["mlbDebutDate"]))
    pid = p["id"]
    prior = []
    for j in (1, 2, 3):
        w = float(players.season_war(season - j).get(pid, 0.0))
        prior.append(w * 162 / 60 if season - j == 2020 else w)
    age = (pd.Timestamp(f"{season}-07-01") - pd.Timestamp(p["birthDate"])).days / 365.25
    svc, _ = contracts.player_history(pid, season)
    aav = r.total_m / r.years
    rows.append({
        "name": r.name, "player_id": pid, "year": season, "age": round(age, 1),
        "war_recent": round(contracts.recent_war(prior), 2),
        "pitcher": int(p.get("primaryPosition", {}).get("abbreviation") == "P"),
        "service": round(svc, 2), "aav_m": round(aav, 2), "years": int(r.years),
        "aav_index": round(aav / contracts.avg_salary(season), 4),
    })
out = pd.DataFrame(rows)
out.to_csv(RES / "recent_fa_signings.csv", index=False)
print(out.sort_values("aav_index", ascending=False).to_string(index=False))
