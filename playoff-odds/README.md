# Trade Playoff Odds

A Streamlit app that estimates every MLB team's playoff probability **as of any date from 2022 to 2026**. It then re-runs the season with a **hypothetical trade** applied, so you can see how much a deal moves the needle for both teams and for everyone they're competing with.

## Run it

```bash
cd playoff-odds
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Data comes from the free MLB Stats API and is cached in `.cache/`. The first load of a season takes a few seconds, and later loads are instant.

## How to use

1. In the sidebar, pick a season and an as-of date. Games on or before that date count as played, and every later game is simulated.
2. Pick Team A and Team B, then the players each side sends. Rosters reflect who was in each organization on that date, including injured-list players.
3. Optionally add **custom players**: free agents, prospects, or hypothetical players that you describe by talent level instead of looking up. Choose which team each one joins, and whether they come from outside the deal or from the other team.
4. Adjust **Remaining PA / IP** for each player if you expect a different role or an injury.
5. Read the before/after odds: make playoffs, win division, first-round bye, projected wins, the seed distribution, and the league-wide ripple effects.

## The three models

| Model | Player value | Team effect |
|---|---|---|
| **WAR (to date)** | Season-to-date runs above replacement (fWAR-style: wOBA batting, baserunning, positional adjustment, FIP pitching) | Runs → wins, added to the team's talent win% |
| **Projection (Marcel)** | Weighted current + 3 prior seasons, regressed to average, age-adjusted | Same as WAR |
| **Run differential** | Season-to-date runs, split into offense and run prevention | Moves team runs scored / allowed, then win% via Pythagenpat (diminishing returns) |

The headline number is the average of whichever models you select.

## Custom players

You don't need a player to exist in MLB's data. Open **Custom players** and describe one by either:

- **WAR pace**: WAR per 650 PA for hitters, or per 180 IP (starters) / 65 IP (relievers) for pitchers. 2 is average, 5 is all-star, and 8+ is MVP-level.
- **Rate stat**: wRC+ for hitters (100 = league average, plus a position), or FIP / ERA for pitchers.

The app converts that to runs per PA or IP using the selected season's league constants. Custom players default to everyday-hitter, rotation-starter, or regular-reliever playing time, which you can edit. The list lasts for your browser session. Use **Download list** to save it as JSON, and upload that file next time.

Real players don't need manual updates either. Stats and rosters come live from the MLB Stats API for whatever date you pick.

## Simulation details

- Monte Carlo over every remaining game. Win probability uses log5 with home-field advantage, and each simulated season draws its own true-talent levels to reflect uncertainty.
- Uses the 12-team format: 3 division winners and 3 wild cards per league, with byes for the top two division winners.
- Uses MLB's full tiebreaker procedure: head-to-head (combined for 3+ teams), then intradivision record, then intraleague record, then the last half of intraleague games, extended one game at a time.
- Before and after runs use common random numbers, so the difference isolates the trade.
- The test suite checks that the simulator reproduces the real 2022–2025 playoff fields and seeds from final standings: `.venv/bin/python -m pytest -q`.

## Layout

```
app.py                    Streamlit UI
playoff_odds/data.py      MLB Stats API client + disk cache, rosters by date
playoff_odds/players.py   Player valuation (WAR-style rates, Marcel projections)
playoff_odds/strength.py  Team talent estimates, log5 game probabilities
playoff_odds/simulate.py  Season simulator + tiebreakers
playoff_odds/trade.py     Trade deltas per model, before/after runs
tests/                    Integration tests against real seasons
```

## Limitations

The model does not include park factors, fielding metrics beyond the positional adjustment, reliever leverage, injuries after the as-of date, or postseason series odds. Player playing time defaults to the season-to-date rate.
