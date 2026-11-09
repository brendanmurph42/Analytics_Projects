"""Prospect valuation from FV grades."""
import pandas as pd
import pytest

from playoff_odds import prospects as P


def test_values_increase_with_grade_and_land_near_targets():
    vals = [P.prospect_value(g, 1, False, 22, 2025).surplus for g in P.GRADES]
    assert vals == sorted(vals)
    v = dict(zip(P.GRADES, vals))
    assert v[70] == pytest.approx(110, rel=0.1)
    assert v[60] == pytest.approx(50, rel=0.1)
    assert v[50] == pytest.approx(15, rel=0.15)


def test_pitchers_and_distance_add_risk():
    h = P.prospect_value(60, 1, False, 22, 2025).surplus
    assert P.prospect_value(60, 1, True, 22, 2025).surplus < h
    assert P.prospect_value(60, 4, False, 19, 2025).surplus < h


def test_value_never_negative():
    assert P.prospect_value(40, 4, True, 19, 2025).surplus >= 0


def test_stat_summary_picks_highest_level():
    milb = pd.DataFrame([
        {"player_id": 1, "level": "A+", "sport": 13, "group": "hitting", "PA": 200, "AB": 180, "H": 60,
         "2B": 10, "3B": 1, "HR": 8, "BB": 15, "HBP": 3, "SF": 2, "SO": 40, "SB": 10, "OUTS": 0, "BF": 0, "ER": 0},
        {"player_id": 1, "level": "AA", "sport": 12, "group": "hitting", "PA": 150, "AB": 135, "H": 38,
         "2B": 8, "3B": 0, "HR": 5, "BB": 12, "HBP": 1, "SF": 2, "SO": 35, "SB": 6, "OUTS": 0, "BF": 0, "ER": 0},
    ])
    s = P.summarize_stats(milb, 1, 21.0)
    assert s["level"] == "AA" and s["line"].startswith("AA: ") and "(also A+)" in s["line"]
    assert s["age_vs_level"] == pytest.approx(21.0 - P.TYPICAL_AGE["AA"])
    assert P.summarize_stats(milb, 2, 20.0)["level"] is None
