"""Data access for the MLB Stats API, with an on-disk JSON cache.

Everything the model needs comes from statsapi.mlb.com (free, no key):
  * teams / leagues / divisions for a season
  * the full regular-season schedule with final scores
  * player hitting & pitching stats over any date range (season-to-date)
  * full-season player stats for prior seasons (for projections)
  * team rosters as of a given date
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE = "https://statsapi.mlb.com/api/v1"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
SUPPORTED_SEASONS = (2022, 2023, 2024, 2025, 2026)

_session = requests.Session()
_session.headers["User-Agent"] = "playoff-odds-tool/1.0"


def _get(path: str, params: dict, *, cache: bool = True, ttl_days: float | None = None) -> dict:
    """GET a Stats API endpoint, caching the JSON response on disk."""
    key = hashlib.sha1(json.dumps([path, sorted(params.items())]).encode()).hexdigest()
    fp = CACHE_DIR / f"{key}.json"
    if cache and fp.exists():
        fresh = ttl_days is None or (time.time() - fp.stat().st_mtime) < ttl_days * 86400
        if fresh:
            return json.loads(fp.read_text())
    last_err = None
    for attempt in range(3):
        try:
            r = _session.get(f"{BASE}/{path}", params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            break
        except requests.RequestException as e:  # pragma: no cover - network
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    else:  # pragma: no cover - network
        raise RuntimeError(f"Stats API request failed: {path} {params}: {last_err}")
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------- teams


def get_teams(season: int) -> pd.DataFrame:
    d = _get("teams", {"sportId": 1, "season": season})
    rows = []
    for t in d["teams"]:
        rows.append(
            {
                "team_id": t["id"],
                "abbr": t["abbreviation"],
                "name": t["name"],
                "short": t.get("teamName", t["name"]),
                "league_id": t["league"]["id"],
                "league": "AL" if t["league"]["id"] == 103 else "NL",
                "division_id": t["division"]["id"],
                "division": t["division"]["name"]
                .replace("American League", "AL")
                .replace("National League", "NL"),
            }
        )
    return pd.DataFrame(rows).sort_values(["league", "division", "abbr"]).reset_index(drop=True)


# ---------------------------------------------------------------- schedule


def get_schedule(season: int) -> pd.DataFrame:
    """All regular-season games that were completed, one row per game.

    Postponed / cancelled entries are dropped; a suspended game that was
    resumed later is kept once (its final completion). Because these are
    historical seasons, "remaining games" from an as-of date are exactly the
    completed games dated after it.
    """
    is_current = season >= date.today().year
    d = _get(
        "schedule",
        {
            "sportId": 1,
            "season": season,
            "gameType": "R",
            "startDate": f"{season}-02-01",
            "endDate": f"{season}-11-30",
        },
        ttl_days=1 if is_current else None,
    )
    rows = {}
    for dd in d["dates"]:
        for g in dd["games"]:
            st = g["status"]
            if st.get("abstractGameState") != "Final":
                continue
            if st.get("detailedState") in ("Postponed", "Cancelled"):
                continue
            home, away = g["teams"]["home"], g["teams"]["away"]
            if "score" not in home or "score" not in away:
                continue
            rows[g["gamePk"]] = {
                "game_pk": g["gamePk"],
                "date": pd.Timestamp(g.get("officialDate", dd["date"])),
                "game_dt": pd.Timestamp(g["gameDate"]),
                "home_id": home["team"]["id"],
                "away_id": away["team"]["id"],
                "home_score": home["score"],
                "away_score": away["score"],
            }
    df = pd.DataFrame(rows.values()).sort_values(["date", "game_dt", "game_pk"]).reset_index(drop=True)
    df["home_win"] = df["home_score"] > df["away_score"]
    return df


# ---------------------------------------------------------------- player stats

HIT_COLS = {
    "plateAppearances": "PA",
    "atBats": "AB",
    "hits": "H",
    "doubles": "2B",
    "triples": "3B",
    "homeRuns": "HR",
    "baseOnBalls": "BB",
    "intentionalWalks": "IBB",
    "hitByPitch": "HBP",
    "sacFlies": "SF",
    "stolenBases": "SB",
    "caughtStealing": "CS",
    "gamesPlayed": "G",
}
PIT_COLS = {
    "battersFaced": "BF",
    "outs": "OUTS",
    "homeRuns": "HR",
    "baseOnBalls": "BB",
    "intentionalWalks": "IBB",
    "hitBatsmen": "HBP",
    "strikeOuts": "SO",
    "runs": "R",
    "gamesStarted": "GS",
    "gamesPitched": "GP",
}


def _stats_frame(splits: list, cols: dict) -> pd.DataFrame:
    rows = []
    for s in splits:
        st = s["stat"]
        row = {"player_id": s["player"]["id"], "name": s["player"]["fullName"]}
        pos = s.get("position", {}).get("abbreviation")
        row["pos"] = pos
        for k, v in cols.items():
            row[v] = int(st.get(k, 0) or 0)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["player_id", "name", "pos", *cols.values()])
    df = pd.DataFrame(rows)
    agg = {v: "sum" for v in cols.values()}
    agg.update({"name": "first", "pos": "first"})
    return df.groupby("player_id", as_index=False).agg(agg)


def _fetch_stats(group: str, params: dict, ttl_days=None) -> list:
    out, offset = [], 0
    while True:
        d = _get(
            "stats",
            {**params, "group": group, "sportId": 1, "playerPool": "ALL", "limit": 2000, "offset": offset},
            ttl_days=ttl_days,
        )
        s = d["stats"][0] if d.get("stats") else {"splits": []}
        out.extend(s["splits"])
        total = s.get("totalSplits", len(out))
        offset += 2000
        if offset >= total or not s["splits"]:
            return out


def stats_to_date(season: int, as_of: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Season-to-date hitting and pitching through `as_of` (inclusive)."""
    params = {
        "stats": "byDateRange",
        "season": season,
        "startDate": f"{season}-02-01",
        "endDate": as_of.isoformat(),
    }
    ttl = 1 if as_of >= date.today() else None
    hit = _stats_frame(_fetch_stats("hitting", params, ttl), HIT_COLS)
    pit = _stats_frame(_fetch_stats("pitching", params, ttl), PIT_COLS)
    return hit, pit


def season_stats(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-season hitting and pitching totals for a (completed) season."""
    params = {"stats": "season", "season": season}
    hit = _stats_frame(_fetch_stats("hitting", params), HIT_COLS)
    pit = _stats_frame(_fetch_stats("pitching", params), PIT_COLS)
    return hit, pit


# ---------------------------------------------------------------- rosters


@dataclass
class RosterPlayer:
    player_id: int
    name: str
    pos: str
    birth_date: str | None
    status: str


def get_roster(team_id: int, as_of: date) -> list[RosterPlayer]:
    """Players in the team's organization (40-man incl. injured list) on a date.

    The API's dated roster endpoint silently drops players who later left
    the club, so we instead take everyone who was on the 40-man at any point
    that season (plus the dated roster) and keep those whose roster-entry
    history places them with this team on ``as_of``.
    """
    season = as_of.year
    season_r = _get(f"teams/{team_id}/roster", {"rosterType": "40Man", "season": season, "hydrate": "person"})
    dated_r = _get(f"teams/{team_id}/roster",
                   {"rosterType": "40Man", "date": as_of.isoformat(), "hydrate": "person"})
    status = {r["person"]["id"]: r.get("status", {}).get("description", "") for r in dated_r.get("roster", [])}
    persons = {}
    for r in season_r.get("roster", []) + dated_r.get("roster", []):
        persons.setdefault(r["person"]["id"], r)

    ids = sorted(persons)
    on_team: set[int] = set()
    day = as_of.isoformat()
    for i in range(0, len(ids), 100):
        chunk = ids[i : i + 100]
        d = _get("people", {"personIds": ",".join(map(str, chunk)), "hydrate": "rosterEntries"}, ttl_days=7)
        for p in d.get("people", []):
            for e in p.get("rosterEntries", []):
                if e.get("team", {}).get("id") != team_id:
                    continue
                start, end = e.get("startDate") or "0000", e.get("endDate") or "9999"
                if start <= day <= end:
                    on_team.add(p["id"])
                    break
    on_team |= set(status)  # anything the dated roster lists is definitely there

    out = []
    for pid in ids:
        if pid not in on_team:
            continue
        r = persons[pid]
        p = r["person"]
        out.append(
            RosterPlayer(
                player_id=pid,
                name=p["fullName"],
                pos=r.get("position", {}).get("abbreviation") or p.get("primaryPosition", {}).get("abbreviation", ""),
                birth_date=p.get("birthDate"),
                status=status.get(pid, ""),
            )
        )
    return out


def get_people(player_ids: list[int]) -> dict[int, dict]:
    """Birth date and primary position for a list of players."""
    out: dict[int, dict] = {}
    ids = sorted(set(int(i) for i in player_ids))
    for i in range(0, len(ids), 150):
        chunk = ids[i : i + 150]
        d = _get("people", {"personIds": ",".join(map(str, chunk))})
        for p in d.get("people", []):
            out[p["id"]] = {
                "birth_date": p.get("birthDate"),
                "pos": p.get("primaryPosition", {}).get("abbreviation", ""),
                "name": p.get("fullName"),
            }
    return out
