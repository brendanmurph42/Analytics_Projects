# Playoff Odds

A Streamlit app that estimates every MLB team's chance of making **the next playoffs** as of any date since 2022. It opens on today. Through the last day of a regular season, that means that season's playoffs. After it, during the postseason and the winter, it means next season's. As of November 2026 it shows the **2027 playoff odds**, projected from each club's current roster.

Then try a move, such as a **trade, a free-agent signing, or a prospect deal**. The app re-runs the depth charts and the season to show how the odds change for both clubs and for everyone they're competing with. It also judges whether each side would accept the deal.

## Run it

```bash
cd playoff-odds
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Data comes from the free MLB Stats API and is cached in `.cache/`. The first load for a new date takes 10–20 seconds (rosters for all 30 clubs), and later loads are instant.

## How to use

1. In the sidebar, pick the as-of date. It defaults to today. The title shows which playoffs the odds are for.
2. Pick Team A and Team B, then the players each side sends. Rosters show who was with each club on that date, including injured-list players.
3. Optionally **sign free agents**: unsigned players who played in the majors this season or last, with projected WAR and a predicted contract.
4. Optionally add **custom players** or turn on **prospects**.
5. Adjust **Expected PA / IP** if you expect a different role or an injury.
6. Read the verdict and the before/after odds: make playoffs, win division, first-round bye, projected wins, the seed distribution, and the league-wide ripple effects.

## How teams are projected

- **Rosters on the date** come from every player's dated roster history. That's more reliable than the API's roster lists, which return only today's players for the current year and drop anyone who later left a club.
- **Automatic depth charts** rank each roster by projection and assign a season's playing time: starting and backup catcher, 8 regulars, 4 bench bats, a 5-man rotation plus 2 depth starters, 7 relievers, and 3 mop-up arms. The rest is filled at replacement level. Team strength is the sum of rate × playing time.
- **Before a season**, talent blends last season's Pythagorean record with the roster projection, using weights fit on opening days 2023–26. In leave-one-season-out tests, the blend missed actual records by 9.1–9.4 wins (RMSE). That compares with 9.3 for last season's record alone, 9.9–10.3 for roster projections alone, and an SD of 12.0 wins in actual records. Preseason uncertainty in the simulation (±6.6 wins) is set from those misses.
- **During a season**, results so far are blended with the preseason estimate, at 70 games of weight.
- **Trades and signings** re-run both clubs' depth charts. The newcomer takes the role his projection earns, and everyone behind him loses playing time.

## The three models

| Model | Player rates | Team effect |
|---|---|---|
| **WAR (recent)** | Runs above replacement this season so far (last season plus this one before about a month of games): wOBA batting, baserunning, positional adjustment, FIP pitching | Runs → wins |
| **Projection (Marcel)** | Weighted current + 3 prior seasons, regressed to average, age-adjusted | Runs → wins |
| **Run differential** | Recent runs, split into offense and run prevention | Moves runs scored / allowed, then win% via Pythagenpat (diminishing returns) |

All three use the same depth charts, and the headline number is the average of whichever models you select. MLB's labor agreement expired December 1, 2026, so 2027 odds assume the current schedule and 12-team format.

## Custom players

You don't need a player to exist in MLB's data. Open **Custom players** and describe one by either:

- **WAR pace**: WAR per 650 PA for hitters, or per 180 IP (starters) / 65 IP (relievers) for pitchers. 2 is average, 5 is all-star, and 8+ is MVP-level.
- **Rate stat**: wRC+ for hitters (100 = league average, plus a position), or FIP / ERA for pitchers.

The app converts that to runs per PA or IP using the season's league constants. Custom players get whatever role their talent earns on the new club's depth chart, which you can edit. The list lasts for your browser session. Use **Download list** to save it as JSON, and upload that file next time.

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

### Free-agent contracts

Free agents are priced from **actual free-agent signings**, not from every veteran's salary. Veterans on pre-free-agency extensions or backloaded deals make general comps run well below the open market. There are two sources:

- **2022–25:** 55 major free-agent contracts, hand-entered from public reporting in `playoff_odds/resources/recent_fa_contracts.csv` (please double-check them). Their pre-signing WAR, age and service are computed by `scripts/build_recent_fa.py`.
- **2000–16:** about 1,600 signings inferred from Lahman: veterans who changed teams with a new salary. Annual value is the average salary over their first seasons with the new club.

Since 2016, the league-average salary barely grew ($4.4M → $5.2M) while good free agents' prices kept rising. Recent 3+ WAR signings ran about 1.5× what the 2000–16 market would have paid relative to the average, so older signings are scaled up by that measured gap (fully at 3+ WAR, not at all at 1 WAR or below). A free agent's annual value comes from his 20 nearest signings by age, recent WAR, and pitcher/hitter, with recent deals counted triple. Contract length comes from his nearest recent signings (pitchers match pitchers), blended toward 2000–16 tenure for lower-tier players.

**Leave-one-out on the recent contracts:** for 4+ WAR free agents the typical error is about 15% with no bias, versus 42% too low for the general comps. For example, Judge is predicted at $37.5M (actual $40M), Correa at $29M ($33M), and Freeman at $29.5M ($27M). Contract length is typically within a year. Soto's record $51M is underpredicted at about $40M. Lower-tier predictions run about 25% below the recent list, but that list only includes notable deals.

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
playoff_odds/data.py      MLB Stats API client + disk cache, rosters and free agents by date
playoff_odds/players.py   Player valuation (WAR-style rates, Marcel projections)
playoff_odds/strength.py  Team talent estimates, log5 game probabilities
playoff_odds/simulate.py  Season simulator + tiebreakers
playoff_odds/trade.py     Context for a date, trade/signing effects, before/after runs
playoff_odds/depth.py     Automatic depth charts and roster-based team projections
playoff_odds/value.py     Surplus value and accept/reject verdicts
playoff_odds/contracts.py Contract predictions from historical comps
playoff_odds/prospects.py Prospect grades (historical comps) and grade-to-value model
playoff_odds/resources/   Comps table + league-average salary series
scripts/                  One-time build scripts for the contract and prospect comps tables
tests/                    Integration tests against real seasons
```

## Limitations

The model does not include park factors, fielding metrics beyond the positional adjustment, reliever leverage, injuries after the as-of date, or postseason series odds. Player playing time defaults to the season-to-date rate.
