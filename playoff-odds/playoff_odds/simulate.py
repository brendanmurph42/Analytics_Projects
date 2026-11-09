"""Monte Carlo simulation of the rest of a season under the 12-team format.

Format (2022-present): per league, three division winners plus three wild
cards. Seeds 1-3 are division winners ordered by record (1 and 2 get byes);
seeds 4-6 are wild cards ordered by record. There are no tiebreaker games.

Tiebreakers (MLB's procedure since 2022), applied to any tie that affects a
division title, a playoff spot, or seeding:
  1. Head-to-head winning percentage (for 3+ clubs, combined record in
     games among all tied clubs).
  2. Winning percentage in intradivision games.
  3. Winning percentage in intraleague games.
  4. Winning percentage in the last half of intraleague games, then
     extending back one intraleague game at a time until the tie breaks.
  5. (Practically unreachable) random draw.
When a step singles out one club as best, it is placed and the procedure
restarts from step 1 for the remaining clubs. When a step narrows the
leaders to a smaller group, that group is ranked by restarting the procedure,
followed by the others.

The same uniform random draws are reused across scenarios (common random
numbers), so before/after differences reflect the trade, not noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strength import game_probs


@dataclass
class SeasonSetup:
    team_ids: list[int]
    abbr: list[str]
    league: np.ndarray        # (T,) 0=AL 1=NL
    division: np.ndarray      # (T,) division id
    base_w: np.ndarray        # (T,) wins to date
    base_l: np.ndarray
    rem_home: np.ndarray      # (N,) team idx
    rem_away: np.ndarray
    # tiebreak bookkeeping
    base_h2h: np.ndarray      # (T,T) wins of i over j to date
    h2h_games: dict = field(default_factory=dict)  # (i,j) -> (idx_i_home, idx_j_home) remaining
    base_div_w: np.ndarray = None
    base_div_g: np.ndarray = None
    base_lg_w: np.ndarray = None
    base_lg_g: np.ndarray = None
    rem_div_mask: np.ndarray = None   # (N,) bool
    rem_lg_mask: np.ndarray = None
    # chronological intraleague sequence per team: list of (kind, value)
    # kind 0 -> known result value (1 win / 0 loss); kind 1 -> remaining game idx, value=+1 if home else -1
    lg_seq: list = field(default_factory=list)


def build_setup(teams: pd.DataFrame, sched: pd.DataFrame, as_of) -> SeasonSetup:
    ids = teams.team_id.tolist()
    idx = {t: i for i, t in enumerate(ids)}
    T = len(ids)
    league = (teams.league == "NL").astype(int).to_numpy()
    division = teams.division_id.to_numpy()

    s = sched.copy()
    s["h"] = s.home_id.map(idx)
    s["a"] = s.away_id.map(idx)
    s = s.dropna(subset=["h", "a"])
    s["h"], s["a"] = s.h.astype(int), s.a.astype(int)
    s["is_div"] = division[s.h] == division[s.a]
    s["is_lg"] = league[s.h] == league[s.a]
    s = s.sort_values(["date", "game_dt", "game_pk"]).reset_index(drop=True)
    cutoff = pd.Timestamp(as_of)
    is_played = (s.date <= cutoff) & s.final
    played = s[is_played]
    rem = s[~is_played].reset_index(drop=True)

    base_w = np.zeros(T, int)
    base_l = np.zeros(T, int)
    h2h = np.zeros((T, T), int)
    div_w = np.zeros(T, int); div_g = np.zeros(T, int)
    lg_w = np.zeros(T, int); lg_g = np.zeros(T, int)
    for h, a, hw, isd, isl in zip(played.h, played.a, played.home_win, played.is_div, played.is_lg):
        w, l = (h, a) if hw else (a, h)
        base_w[w] += 1; base_l[l] += 1; h2h[w, l] += 1
        if isd:
            div_w[w] += 1; div_g[h] += 1; div_g[a] += 1
        if isl:
            lg_w[w] += 1; lg_g[h] += 1; lg_g[a] += 1

    h2h_games = {}
    rh, ra = rem.h.to_numpy(), rem.a.to_numpy()
    for i in range(T):
        for j in range(T):
            if i < j and league[i] == league[j]:
                ih = np.where((rh == i) & (ra == j))[0]
                jh = np.where((rh == j) & (ra == i))[0]
                h2h_games[(i, j)] = (ih, jh)

    # Chronological intraleague sequences (needed for the "last half" step)
    lg_seq = [[] for _ in range(T)]
    for h, a, hw, isl in zip(played.h, played.a, played.home_win, played.is_lg):
        if isl:
            lg_seq[h].append((0, 1 if hw else 0))
            lg_seq[a].append((0, 0 if hw else 1))
    for k, (h, a, isl) in enumerate(zip(rem.h, rem.a, rem.is_lg)):
        if isl:
            lg_seq[h].append((1, k + 1))      # +k+1 -> home side of rem game k
            lg_seq[a].append((1, -(k + 1)))   # -(k+1) -> away side

    return SeasonSetup(
        team_ids=ids, abbr=teams.abbr.tolist(), league=league, division=division,
        base_w=base_w, base_l=base_l, rem_home=rh, rem_away=ra, base_h2h=h2h,
        h2h_games=h2h_games, base_div_w=div_w, base_div_g=div_g, base_lg_w=lg_w, base_lg_g=lg_g,
        rem_div_mask=rem.is_div.to_numpy(), rem_lg_mask=rem.is_lg.to_numpy(), lg_seq=lg_seq,
    )


class _SimTiebreak:
    """Tiebreak helpers bound to one simulated season (row of results)."""

    def __init__(self, st: SeasonSetup, res: np.ndarray, div_w, div_g, lg_w, lg_g, rng):
        self.st, self.res, self.rng = st, res, rng
        self.div_w, self.div_g, self.lg_w, self.lg_g = div_w, div_g, lg_w, lg_g

    def h2h(self, i, j):
        """(wins of i vs j, games) for this sim."""
        st = self.st
        a, b = (i, j) if i < j else (j, i)
        ih, jh = st.h2h_games[(a, b)]
        # wins for 'a': a home & home win, plus b home & home loss
        wa = self.res[ih].sum() + (len(jh) - self.res[jh].sum())
        ga = len(ih) + len(jh)
        wins_a = st.base_h2h[a, b] + wa
        wins_b = st.base_h2h[b, a] + (ga - wa)
        games = wins_a + wins_b
        return (wins_a, games) if i == a else (wins_b, games)

    def _crit_h2h(self, group):
        vals = {}
        for i in group:
            w = g = 0
            for j in group:
                if i != j:
                    wi, gi = self.h2h(i, j)
                    w += wi; g += gi
            vals[i] = w / g if g else None
        if any(v is None for v in vals.values()):
            return None  # no head-to-head games -> step does not apply
        return vals

    def _crit_pct(self, group, w, g):
        return {i: (w[i] / g[i] if g[i] else 0.0) for i in group}

    def _seq_results(self, t):
        out = []
        for kind, v in self.st.lg_seq[t]:
            if kind == 0:
                out.append(v)
            else:
                k = abs(v) - 1
                hw = self.res[k]
                out.append(int(hw) if v > 0 else int(not hw))
        return out

    def _last_half_steps(self, group):
        seqs = {i: self._seq_results(i) for i in group}
        n = min(len(s) for s in seqs.values())
        if n == 0:
            return
        k = n // 2
        while k <= n:
            yield {i: (sum(seqs[i][-k:]) / k if k else 0.0) for i in group}
            k += 1

    def rank(self, group: list[int]) -> list[int]:
        group = list(group)
        if len(group) <= 1:
            return group

        def steps():
            yield self._crit_h2h(group)
            yield self._crit_pct(group, self.div_w, self.div_g)
            yield self._crit_pct(group, self.lg_w, self.lg_g)
            yield from self._last_half_steps(group)

        for vals in steps():
            if vals is None:
                continue
            best = max(vals.values())
            top = [i for i in group if abs(vals[i] - best) < 1e-12]
            if len(top) < len(group):
                rest = [i for i in group if i not in top]
                if len(top) == 1:
                    return top + self.rank(rest)
                return self.rank(top) + self.rank(rest)
        g = list(group)
        self.rng.shuffle(g)
        return g


def _order_by_wins(teams, wins, tb_factory):
    """Order teams by wins desc, breaking ties with the tiebreak procedure."""
    teams = sorted(teams, key=lambda t: -wins[t])
    out, i = [], 0
    while i < len(teams):
        j = i
        while j + 1 < len(teams) and wins[teams[j + 1]] == wins[teams[i]]:
            j += 1
        tier = teams[i : j + 1]
        out.extend(tier if len(tier) == 1 else tb_factory().rank(tier))
        i = j + 1
    return out


@dataclass
class SimResult:
    playoff: np.ndarray   # (T,) probability
    division: np.ndarray
    bye: np.ndarray
    mean_wins: np.ndarray
    seed_dist: np.ndarray  # (T, 7) col 0 = missed, 1..6 = seed


def simulate(st: SeasonSetup, wp: np.ndarray, U: np.ndarray, seed: int = 0,
             talent_noise: np.ndarray | None = None) -> SimResult:
    """Simulate remaining games.

    ``U`` is (sims, N) uniforms shared across scenarios. ``talent_noise`` is an
    optional (sims, T) array added to ``wp`` so each simulated season draws its
    own "true" team strengths (uncertainty in talent, not just in outcomes).
    """
    T = len(st.team_ids)
    S = U.shape[0]
    wp_s = wp[None, :] + talent_noise if talent_noise is not None else wp
    p = game_probs(st.rem_home, st.rem_away, wp_s)
    R = U < p  # home wins
    N = U.shape[1]

    H = np.zeros((N, T), np.float32); H[np.arange(N), st.rem_home] = 1
    A = np.zeros((N, T), np.float32); A[np.arange(N), st.rem_away] = 1
    Rf = R.astype(np.float32)
    wins = st.base_w + (Rf @ H + (1 - Rf) @ A).round().astype(int)

    dm = st.rem_div_mask.astype(np.float32)
    lm = st.rem_lg_mask.astype(np.float32)
    div_w = st.base_div_w + ((Rf * dm) @ H + ((1 - Rf) * dm) @ A).round().astype(int)
    div_g = st.base_div_g + (dm @ (H + A)).round().astype(int)
    lg_w = st.base_lg_w + ((Rf * lm) @ H + ((1 - Rf) * lm) @ A).round().astype(int)
    lg_g = st.base_lg_g + (lm @ (H + A)).round().astype(int)

    rng = np.random.default_rng(seed + 7919)
    seed_counts = np.zeros((T, 7), int)
    div_count = np.zeros(T, int)
    divisions = {lg: sorted(set(st.division[st.league == lg])) for lg in (0, 1)}
    league_teams = {lg: np.where(st.league == lg)[0].tolist() for lg in (0, 1)}
    div_teams = {d: np.where(st.division == d)[0].tolist() for d in set(st.division)}

    for s in range(S):
        w = wins[s]
        cache = {}

        def tb():
            if "tb" not in cache:
                cache["tb"] = _SimTiebreak(st, R[s], div_w[s], div_g, lg_w[s], lg_g, rng)
            return cache["tb"]

        for lg in (0, 1):
            winners = []
            for d in divisions[lg]:
                members = div_teams[d]
                top = max(w[t] for t in members)
                tied = [t for t in members if w[t] == top]
                winners.append(tied[0] if len(tied) == 1 else tb().rank(tied)[0])
            for t in winners:
                div_count[t] += 1
            ordered_winners = _order_by_wins(winners, w, tb)
            others = [t for t in league_teams[lg] if t not in winners]
            # Only rank as deep as needed: teams with wins >= 3rd-best non-winner.
            sorted_w = sorted((w[t] for t in others), reverse=True)
            cutoff = sorted_w[2]
            contenders = [t for t in others if w[t] >= cutoff]
            wc = _order_by_wins(contenders, w, tb)[:3]
            for k, t in enumerate(ordered_winners + wc):
                seed_counts[t, k + 1] += 1

    seed_counts[:, 0] = S - seed_counts[:, 1:].sum(axis=1)
    return SimResult(
        playoff=1 - seed_counts[:, 0] / S,
        division=div_count / S,
        bye=(seed_counts[:, 1] + seed_counts[:, 2]) / S,
        mean_wins=wins.mean(axis=0),
        seed_dist=seed_counts / S,
    )


def draw_uniforms(n_games: int, sims: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((sims, n_games), dtype=np.float32)


def draw_talent_noise(sd: np.ndarray, sims: int, seed: int) -> np.ndarray:
    """(sims, T) normal draws scaled by each team's talent SD."""
    z = np.random.default_rng(seed + 104729).standard_normal((sims, len(sd)))
    return (z * np.asarray(sd)[None, :]).astype(np.float32)
