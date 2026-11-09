"""Season targeting and automatic depth charts."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from playoff_odds import data, depth


def test_target_season(monkeypatch):
    monkeypatch.setattr(data, "season_dates", lambda y: (date(y, 3, 26), date(y, 9, 28)))
    assert data.target_season(date(2026, 7, 31)) == 2026
    assert data.target_season(date(2026, 9, 28)) == 2026      # last regular-season day
    assert data.target_season(date(2026, 9, 29)) == 2027      # postseason -> next year
    assert data.target_season(date(2026, 11, 9)) == 2027
    assert data.target_season(date(2027, 2, 15)) == 2027      # spring training


def _roster():
    rows = []
    for i in range(16):   # hitters, descending quality; two catchers
        rows.append({"player_id": i, "pos": "C" if i in (3, 12) else "SS", "status": "",
                     "proj_off_per_pa": 0.06 - i * 0.004, "def_per_pa": 0.0, "proj_pit_per_ip": 0.0,
                     "todate_off_per_pa": 0.05, "todate_pit_per_ip": 0.0, "sp_share": 0.0})
    for i in range(16, 34):   # pitchers: first 8 are starters
        rows.append({"player_id": i, "pos": "P", "status": "", "proj_off_per_pa": 0.0, "def_per_pa": 0.0,
                     "proj_pit_per_ip": 0.4 - (i - 16) * 0.02, "todate_off_per_pa": 0.0,
                     "todate_pit_per_ip": 0.3, "sp_share": 1.0 if i < 24 else 0.0})
    return pd.DataFrame(rows).set_index("player_id", drop=False)


def test_allocate_fills_slots_sensibly():
    a = depth.allocate(_roster())
    # best catcher starts, second is the backup; catchers don't take other lineup spots
    assert a.loc[3, "pa"] == 480 and a.loc[12, "pa"] == 180
    assert a.loc[0, "pa"] == 550 and a.loc[1, "pa"] == 550
    assert (a.pa > 0).sum() == len(depth.HIT_SLOTS)
    assert a.pa.sum() == sum(pa for _, pa in depth.HIT_SLOTS)
    # five best starters make the rotation
    assert all(a.loc[i, "ip"] == 150 for i in range(16, 21))
    assert a.ip.sum() == pytest.approx(sum(ip for _, ip in depth.PIT_SLOTS))


def test_overrides_displace_lowest_ranked():
    r = _roster()
    r["fixed_pa"] = np.nan
    r.loc[15, "fixed_pa"] = 600          # a scrub forced into an everyday role
    a = depth.allocate(r)
    assert a.loc[15, "pa"] == 600
    total = sum(pa for _, pa in depth.HIT_SLOTS) + 0   # totals stay the same: someone else lost 600 PA
    assert a.pa.sum() == pytest.approx(total)
    assert a.loc[0, "pa"] == 550          # the best hitters keep their jobs


def test_injured_players_get_less_time_in_season():
    r = _roster()
    r.loc[0, "status"] = "Injured 60-Day"
    assert depth.allocate(r, in_season=True).loc[0, "pa"] < depth.allocate(r, in_season=False).loc[0, "pa"]


def test_league_win_pct_is_centered():
    runs = {t: depth.TeamRuns({"war": 100 + 10 * t, "proj": 90 + 8 * t, "rd": 100 + 10 * t}, 5.0,
                              {"war": 80, "proj": 70, "rd": 80}) for t in range(30)}
    wp = depth.league_win_pct(runs, 10.0, 4.5)
    for m in ("war", "proj"):
        assert wp[m].mean() == pytest.approx(0.5, abs=1e-9)
        assert wp[m].is_monotonic_increasing
    # run differential goes through Pythagenpat, so it's only approximately centered
    assert wp["rd"].mean() == pytest.approx(0.5, abs=0.01)
    assert wp["rd"].is_monotonic_increasing
