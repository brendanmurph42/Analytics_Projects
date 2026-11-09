"""Build playoff_odds/resources/prospect_comps.csv: historical prospects and what they became.

For each minor leaguer in 2015-2019 who hadn't debuted in the majors yet:
level, age vs level, performance vs level, playing time (the same features
the app computes for current prospects), and the MLB WAR he produced over the
following six seasons.

    .venv/bin/python scripts/build_prospect_comps.py
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from playoff_odds import data, players, prospects  # noqa: E402

rows = []
for y in range(2015, 2020):
    milb = data.milb_stats_to_date(y, date(y, 11, 30))
    feat = prospects._prospect_pool(prospects.prospect_features(milb, y), y)
    feat = feat[feat.vol >= 30].copy()
    fut = pd.Series(0.0, index=feat.index)
    for k in range(1, 7):
        w = players.season_war(y + k)
        fut += feat.index.to_series().map(w).fillna(0.0).clip(lower=-1.0)
    feat["future_war"] = fut.round(2)
    feat["season"] = y
    rows.append(feat.reset_index().rename(columns={"index": "player_id"}))
    print(y, len(feat), "players;", "mean future WAR", round(feat.future_war.mean(), 2))

out = pd.concat(rows, ignore_index=True)
cols = ["player_id", "name", "season", "pitcher", "vol", "level_score", "age_rel", "perf", "age", "future_war"]
out[cols].round(3).to_csv(ROOT / "playoff_odds" / "resources" / "prospect_comps.csv", index=False)
print("wrote", len(out), "rows")
