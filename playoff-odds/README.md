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

## Trade value and verdicts

Every player in the deal gets a **surplus value** in dollars:

> projected WAR × market $/WAR − salary, over this season's remaining games plus each year of team control

WAR ages (young players improve, players past 30 decline), $/WAR starts at the season's market rate ($8.0M in 2022 to $9.3M in 2026) and grows with salaries, and future dollars are discounted 8% a year.

### Predicted contracts from historical comps

MLB's public data has no contracts, so salaries are **predicted from comparable players**:

1. Each player's MLB service time is estimated from his debut date, with partial credit for the debut season and a year for every later season in the majors. That puts him in a class: pre-arbitration (<3 years), arbitration (3–6), or free agent (6+). It also sets the years of control.
2. The app finds the **30 most similar player seasons from 2000–2016** in the same class, by age, service and recent WAR (3/2/1 over the prior three seasons). These ~14,000 seasons come from the Lahman database, the last free source with salaries.
3. Each comp's salary is stored as a **multiple of that season's league-average salary**, which strips out inflation. The median multiple is applied to the league average for each target season. Known averages (AP opening-day study) run through 2025, and later seasons **grow at the long-run trend**: about 3.4% a year over the last 20 years, adjustable in the sidebar.
4. Pre-arb and arbitration players are re-predicted every season, so raises show up. A veteran's salary is predicted once and held flat, like a signed contract.

On held-out 2014–2016 salaries (predicted only from 2000–2013 comps), the typical error is about 18%, and 83% of predictions land within 2x of the actual salary. Errors are largest for stars on early extensions, so type the real figure in the salary column when you know it. The **Contract predictions** expander shows each player's salary path and closest comps.

Each team's net = value received − value sent + a **contender premium**: the team's change in playoff odds × the value of a playoff spot (default $40M, set in the sidebar). This lets a contender overpay for help now, and it costs a contending seller something to weaken itself.

A team **rejects** the trade if its net loss is bigger than the tolerance (default 15% of the deal's size, minimum $3M). The app names who says no and how much value would fix it, and still shows the playoff odds.

## Prospects

Turn on **Include prospects** to add each organization's minor leaguers: everyone off the 40-man roster who belonged to the club on the chosen date. They're sorted by highest level, then youngest for that level, and each shows his stat line at his highest level this season.

Each prospect gets an **estimated Future Value grade** (FanGraphs-style 20–80 scale), which you can override:

- He's matched with the **60 most similar minor-league seasons from 2015–2019** (about 34,000 seasons). Similarity is based on level, age relative to the level's real average age, performance relative to the level (OPS for hitters, K−BB% for pitchers), and playing time. Every measure is weighted across levels by playing time, so a short cameo at a higher level doesn't dominate.
- His **expected MLB WAR** is what those comps produced over the next six seasons, busts included.
- Grades follow his rank among all eligible minor leaguers (no MLB debut before this season), in roughly the proportions scouting lists use: #1–4 → 70, through #15 → 65, #45 → 60, #120 → 55, #330 → 50, #900 → 45, and 40 after that.
- **Holdout check:** using only 2015–2017 comps, the top picks for 2018 were Vlad Jr., Eloy Jiménez, Kyle Tucker, Tatis, Yordan Alvarez, Soto, and Bichette. For 2019 they were Wander Franco, Luis Robert, Julio Rodríguez, and Gavin Lux. The top 50 averaged 7.0 MLB WAR over six years, versus 0.24 for all minor leaguers. It's stats only, with no scouting, injuries, or makeup, so override it when you know better.

The grade then becomes surplus value:

- **WAR if he reaches his grade:** 50 ≈ 2 WAR (average regular), 60 ≈ 4 (all-star level), 70 ≈ 6 (superstar).
- **Bust risk:** each grade gets a realization factor. Pitchers carry 25% more risk, and each year of ETA beyond one adds 8%.
- **Control and salary:** six seasons of team control from his debut, with salaries predicted by the contract comps model, discounted from his ETA.

The risk factors are tuned so a hitter about a year away lands near rough public prospect valuations: about $110M for a 70, $50M for a 60, $15M for a 50, and $5M for a 45. Those targets approximate published research rather than exact figures, so adjust `REALIZATION` in `playoff_odds/prospects.py` if you prefer different ones. Prospects affect trade value only, not this season's playoff odds.

## Data sources

- Stats, schedules, rosters: MLB Stats API (personal, non-commercial use).
- Historical salaries for contract comps: [Lahman Baseball Database](https://sabr.org/lahman-database/) (SABR, CC BY-SA 3.0). The derived table in `playoff_odds/resources/` is shared under the same license. To rebuild it, see `scripts/build_contract_comps.py`, which needs `pip install pyreadr`.
- League-average salaries for 2017–2025: AP opening-day salary study.

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
playoff_odds/value.py     Surplus value and accept/reject verdicts
playoff_odds/contracts.py Contract predictions from historical comps
playoff_odds/prospects.py Prospect grades (historical comps) and grade-to-value model
playoff_odds/resources/   Comps table + league-average salary series
scripts/                  One-time build scripts for the contract and prospect comps tables
tests/                    Integration tests against real seasons
```

## Limitations

The model does not include park factors, fielding metrics beyond the positional adjustment, reliever leverage, injuries after the as-of date, or postseason series odds. Player playing time defaults to the season-to-date rate.
