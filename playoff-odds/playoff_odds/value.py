"""Trade value: surplus value of players plus a contender premium.

Surplus value is how front offices usually frame trade value: the market
price of a player's projected production over the years his team controls him,
minus what they'll pay him.

    surplus = sum over control years of (WAR_y x $/WAR_y - salary_y) x discount_y

* The current season counts only for the share of games remaining.
* WAR ages: young players improve, players past 30 decline. A projected WAR
  below zero counts as zero, since a team would bench or release him.
* $/WAR is the free-agent market rate for the season, inflated each year at
  the same growth rate used for salaries (default: the long-run salary trend).
  Future dollars are discounted 8% a year.

Salaries come from historical comparable players (see ``contracts.py``),
and the user can override them.

The contender premium turns a team's change in playoff probability into
dollars, so a contender values present-day help above its market price.
A team accepts a trade if its net value (players received - players sent +
contender premium) isn't worse than a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

DOLLARS_PER_WAR = {2022: 8.0, 2023: 8.3, 2024: 8.6, 2025: 9.0, 2026: 9.3}  # $M per WAR
WAR_INFLATION = 0.04
DISCOUNT_RATE = 0.08
LEAGUE_MIN_SALARY = 0.76  # $M


def dollars_per_war(season: int) -> float:
    if season in DOLLARS_PER_WAR:
        return DOLLARS_PER_WAR[season]
    last = max(DOLLARS_PER_WAR)
    return DOLLARS_PER_WAR[last] * (1 + WAR_INFLATION) ** (season - last)


def aging_delta(age: float) -> float:
    """Expected change in WAR per season going from ``age`` to ``age + 1``."""
    if age < 26:
        return 0.4
    if age < 29:
        return 0.0
    if age < 33:
        return -0.5
    return -0.7


def default_control_years(age: float | None) -> int:
    """Seasons of team control after this one, guessed from age."""
    if age is None:
        return 2
    if age < 25:
        return 5
    if age < 26:
        return 4
    if age < 27:
        return 3
    if age < 28:
        return 2
    return 1


def default_salary(age: float | None, war_per_season: float, season: int) -> float:
    """Average annual salary ($M) over the control years, guessed from age.

    Pre-arbitration players earn near the minimum, arbitration players a
    growing share of market value, and veterans roughly market value.
    """
    market = max(war_per_season, 0.0) * dollars_per_war(season)
    if age is None:
        share = 0.6
    elif age < 25:
        share = 0.12
    elif age < 27:
        share = 0.3
    elif age < 29:
        share = 0.55
    else:
        share = 0.9
    return round(max(LEAGUE_MIN_SALARY, share * market), 1)


@dataclass
class SurplusResult:
    total: float          # $M, discounted
    production: float     # $M of WAR value
    salary: float         # $M of salary owed
    war_total: float      # projected WAR over the control period


def war_path(war_per_season: float, age: float | None, control_years: int) -> list[float]:
    """Projected full-season WAR for this season and each control year, with aging."""
    age = 27.0 if age is None else float(age)
    out, war = [], war_per_season
    for y in range(int(control_years) + 1):
        if y > 0:
            war += aging_delta(age + y - 1)
        out.append(war)
    return out


def surplus_value(war_per_season: float, age: float | None, control_years: int,
                  salary: float | list[float], remaining_frac: float, season: int,
                  growth: float = WAR_INFLATION) -> SurplusResult:
    """``salary`` is a flat $M/yr or a per-season list (index 0 = this season).

    ``growth`` inflates $/WAR for future seasons; it should match the salary
    growth used to predict future contracts so both sides move together.
    """
    dpw = dollars_per_war(season)
    total = prod = paid = war_total = 0.0
    wars = war_path(war_per_season, age, control_years)
    for y, war in enumerate(wars):
        frac = remaining_frac if y == 0 else 1.0
        disc = 1.0 / (1 + DISCOUNT_RATE) ** y
        w = max(war, 0.0) * frac
        p = w * dpw * (1 + growth) ** y
        sal_y = salary[min(y, len(salary) - 1)] if isinstance(salary, (list, tuple)) else salary
        s = sal_y * frac
        prod += p * disc
        paid += s * disc
        war_total += w
    total = prod - paid
    return SurplusResult(total, prod, paid, war_total)


@dataclass
class TeamVerdict:
    team_id: int
    received: float
    sent: float
    premium: float
    net: float
    tolerance: float

    @property
    def accepts(self) -> bool:
        return self.net >= -self.tolerance


def evaluate(players: list[dict], playoff_delta: dict[int, float], team_ids: tuple[int, int],
             playoff_spot_value: float, tolerance_pct: float, tolerance_floor: float = 3.0
             ) -> dict[int, TeamVerdict]:
    """Each team's view of the deal.

    ``players``: dicts with ``surplus``, ``to_id`` and ``from_id`` (None for
    players from outside the deal). ``playoff_delta``: change in playoff
    probability (0-1) by team. ``playoff_spot_value``: $M per full playoff spot.
    """
    out = {}
    recv = {t: sum(p["surplus"] for p in players if p["to_id"] == t) for t in team_ids}
    sent = {t: sum(p["surplus"] for p in players if p["from_id"] == t) for t in team_ids}
    deal_size = max([abs(v) for v in recv.values()] + [abs(v) for v in sent.values()] + [0.0])
    tol = max(tolerance_floor, tolerance_pct * deal_size)
    for t in team_ids:
        prem = playoff_delta.get(t, 0.0) * playoff_spot_value
        out[t] = TeamVerdict(t, recv[t], sent[t], prem, recv[t] - sent[t] + prem, tol)
    return out
