"""Trade value: surplus math and accept/reject logic."""
import pytest

from playoff_odds import value as V


def test_surplus_basics():
    young = V.surplus_value(5.0, 23, 5, 3.0, 0.3, 2025)
    rental = V.surplus_value(5.0, 31, 0, 25.0, 0.3, 2025)
    bad = V.surplus_value(1.0, 35, 3, 30.0, 0.3, 2025)
    assert young.total > 150          # cheap young star is worth a lot
    assert young.total > rental.total
    assert bad.total < 0              # overpaid veteran has negative value
    # rental: 30% of a season of 5 WAR at $9M/WAR minus 30% of salary
    assert rental.total == pytest.approx(0.3 * (5 * 9.0 - 25.0))


def test_negative_projection_counts_as_zero_production():
    r = V.surplus_value(-1.0, 30, 2, 1.0, 0.5, 2024)
    assert r.production == 0 and r.total < 0


def _deal(a_gets, b_gets, premium_a=0.0, premium_b=0.0, spot=40.0, tol=0.15):
    players = [{"surplus": a_gets, "to_id": 1, "from_id": 2}, {"surplus": b_gets, "to_id": 2, "from_id": 1}]
    return V.evaluate(players, {1: premium_a / spot, 2: premium_b / spot}, (1, 2), spot, tol)


def test_balanced_trade_accepted():
    v = _deal(30.0, 28.0)
    assert v[1].accepts and v[2].accepts


def test_lopsided_trade_rejected_by_loser_only():
    v = _deal(40.0, 5.0)
    assert v[1].accepts and not v[2].accepts
    assert v[2].net == pytest.approx(-35.0)


def test_contender_premium_can_flip_verdict():
    assert not _deal(10.0, 30.0)[1].accepts
    assert _deal(10.0, 30.0, premium_a=18.0)[1].accepts


def test_free_agent_from_outside_only_affects_receiver():
    players = [{"surplus": 12.0, "to_id": 1, "from_id": None}]
    v = V.evaluate(players, {}, (1, 2), 40.0, 0.15)
    assert v[1].net == 12.0 and v[2].net == 0.0
