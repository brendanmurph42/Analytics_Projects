"""Custom player conversions round-trip and behave sensibly."""
import pytest

from playoff_odds.players import LeagueContext, custom_player_rates

CTX = LeagueContext(season=2024, lg_woba=0.310, lg_wsb_rate=0.0075, lg_ra9=4.45, fip_const=3.6,
                    runs_per_win=9.7, hit_repl_per_pa=0.033, pit_repl_scale=0.95, lg_r_per_pa=0.117)


@pytest.mark.parametrize("spec", [
    {"name": "h", "kind": "Hitter", "pos": "SS", "mode": "WAR pace", "value": 5.0},
    {"name": "s", "kind": "Pitcher", "role": "SP", "mode": "WAR pace", "value": 4.0},
    {"name": "r", "kind": "Pitcher", "role": "RP", "mode": "WAR pace", "value": 1.0},
])
def test_war_pace_round_trips(spec):
    assert custom_player_rates(spec, CTX)["war_pace"] == pytest.approx(spec["value"])


def test_rate_stats_are_monotonic_and_sane():
    hit = lambda w: custom_player_rates({"name": "x", "kind": "Hitter", "pos": "1B", "mode": "Rate stat",
                                         "value": w}, CTX)["war_pace"]
    pit = lambda f: custom_player_rates({"name": "x", "kind": "Pitcher", "role": "SP", "mode": "Rate stat",
                                         "value": f}, CTX)["war_pace"]
    assert hit(80) < hit(100) < hit(150)
    assert 0 < hit(100) < 2.5          # league-average bat at 1B is a modest positive
    assert pit(2.5) > pit(4.0) > pit(5.5)
    assert 4 < pit(2.5) < 8            # ace-level FIP lands in ace-level WAR


def test_custom_player_same_in_both_models():
    r = custom_player_rates({"name": "x", "kind": "Hitter", "pos": "CF", "mode": "Rate stat", "value": 120}, CTX)
    assert r["todate_off_per_pa"] == r["proj_off_per_pa"]
    assert r["rem_pa_per_g"] > 0 and r["rem_ip_per_g"] == 0
