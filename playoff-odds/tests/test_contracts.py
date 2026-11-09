"""Contract predictions from historical comps."""
import numpy as np
import pytest

from playoff_odds import contracts as C


def test_salary_series_and_growth():
    assert C.avg_salary(2025) == pytest.approx(5.16)
    g = C.trend_growth()
    assert 0.01 < g < 0.08
    assert C.avg_salary(2027) == pytest.approx(5.16 * (1 + g) ** 2)
    assert C.avg_salary(2027, growth=0.0) == pytest.approx(5.16)


def test_service_proxy():
    # debut in mid-August 2020 (60-game season) then full seasons 2021-2024
    s = C.service_years("2020-08-18", {2020, 2021, 2022, 2023, 2024, 2025}, 2025)
    assert 4.4 < s < 4.8
    assert C.service_years("2024-05-11", {2024, 2025}, 2025) == pytest.approx(C.debut_credit("2024-05-11"))
    assert C.service_years(None, set(), 2025) == 0.0
    assert C.default_control_years(0.8, 23) == 5
    assert C.default_control_years(4.6, 28) == 1


def test_predictions_are_ordered_sensibly():
    pre = C.predict_salary(24, 1.0, 4.0, False, 2025)
    arb = C.predict_salary(27, 4.5, 4.0, False, 2025)
    fa_good = C.predict_salary(30, 9.0, 5.0, False, 2025)
    fa_bad = C.predict_salary(30, 9.0, 0.5, False, 2025)
    assert pre.cls == "pre-arb" and arb.cls == "arb" and fa_good.cls == "free agent"
    assert pre.salary < arb.salary < fa_good.salary
    assert fa_bad.salary < fa_good.salary
    assert pre.salary < 1.5  # near the league minimum
    assert len(pre.comps) == C.K_NEIGHBORS


def test_growth_scales_future_salaries():
    lo = C.predict_salary(30, 9.0, 4.0, False, 2030, growth=0.0)
    hi = C.predict_salary(30, 9.0, 4.0, False, 2030, growth=0.05)
    assert hi.multiple == pytest.approx(lo.multiple)
    assert hi.salary / lo.salary == pytest.approx(1.05 ** 5)


def test_path_gives_arb_raises_and_flat_free_agent_salary():
    young = C.salary_path(24, 1.0, [4.0, 1.0, 0.0], [5.0] * 6, False, 2025)
    assert young.salaries[0] < young.salaries[-1]           # raises through arbitration
    vet = C.salary_path(31, 8.0, [4.0, 4.0, 4.0], [4.0, 3.5, 3.0], False, 2025)
    assert len(set(np.round(vet.salaries, 6))) == 1          # free-agent deal is flat
