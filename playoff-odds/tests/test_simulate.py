"""Integration checks against real seasons (needs network on first run; cached after)."""
from datetime import date

import numpy as np
import pytest

from playoff_odds import data, simulate

# Actual playoff seeds 1-6 per league.
ACTUAL = {
    2022: {"AL": ["HOU", "NYY", "CLE", "TOR", "SEA", "TB"], "NL": ["LAD", "ATL", "STL", "NYM", "SD", "PHI"]},
    2023: {"AL": ["BAL", "HOU", "MIN", "TB", "TEX", "TOR"], "NL": ["ATL", "LAD", "MIL", "PHI", "MIA", "AZ"]},
    2024: {"AL": ["NYY", "CLE", "HOU", "BAL", "KC", "DET"], "NL": ["LAD", "PHI", "MIL", "SD", "ATL", "NYM"]},
    2025: {"AL": ["TOR", "SEA", "CLE", "NYY", "BOS", "DET"], "NL": ["MIL", "PHI", "LAD", "CHC", "SD", "CIN"]},
}


@pytest.mark.parametrize("season", sorted(ACTUAL))
def test_end_of_season_reproduces_real_seeds(season):
    """With every game played, the tiebreak logic must match the real bracket."""
    teams, sched = data.get_teams(season), data.get_schedule(season)
    st = simulate.build_setup(teams, sched, date(season, 11, 30))
    assert len(st.rem_home) == 0
    res = simulate.simulate(st, np.full(30, 0.5), simulate.draw_uniforms(0, 20, 1))
    for lg_i, lg in enumerate(("AL", "NL")):
        seeds = {}
        for t in range(30):
            if st.league[t] == lg_i and res.playoff[t] == 1:
                seeds[int(np.argmax(res.seed_dist[t]))] = st.abbr[t]
        assert [seeds[k] for k in range(1, 7)] == ACTUAL[season][lg]


def test_probabilities_are_consistent_midseason():
    teams, sched = data.get_teams(2024), data.get_schedule(2024)
    st = simulate.build_setup(teams, sched, date(2024, 7, 31))
    res = simulate.simulate(st, np.full(30, 0.5), simulate.draw_uniforms(len(st.rem_home), 2000, 3))
    assert np.isclose(res.playoff.sum(), 12)       # 12 playoff spots each sim
    assert np.isclose(res.division.sum(), 6)
    assert np.isclose(res.bye.sum(), 4)
    assert (res.division <= res.playoff + 1e-12).all()
