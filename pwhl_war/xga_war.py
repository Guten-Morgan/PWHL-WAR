"""
xga_war.py
----------
xG-based WAR for PWHL skaters using shot-quality-adjusted xGA as the
defensive component instead of residual plus/minus.

Data source: pwhl.hockey-statistics.com API (play-by-play + game summaries)

Method
------
1. Fetch play-by-play and game summaries for each completed game.

2. Offensive value: individual xG per 60 min
     ixG         = sum of xG for all shots taken by the player
     ixG60       = ixG / toi_min × 60

   xG values (empirically derived from 2024-25 full season, 102 games):
     Quality shot      → 0.128  (incl. "Quality goal")
     Non-quality shot  → 0.051  (incl. "Non quality goal")

3. Defensive value: team xGA attributed by TOI share

   a. For each game, compute team-level Fenwick xGA:
        team_xGA_game = Σ xG(shot) for all shots against team T in that game
        (uses shot events only — blocked shots excluded, matching Fenwick)

   b. Attribute to each skater by their TOI share within the game:
        player_xGA_game  = team_xGA_game × (player_toi_game / team_skater_toi_game)

   c. Aggregate across all games the player appeared in:
        xGA60 = Σ(player_xGA_game) / toi_min × 60

   d. Players with higher offensive output tend to spend more time in the
      offensive zone, facing fewer shots against.  Remove this correlation:
        β           = OLS slope of xGA60 ~ ixG60 (qualified players)
        xGA60_resid = xGA60 − (α + β × ixG60)

   e. Team-quality adjustment (same as box_war):
        team_xGA_resid = TOI-weighted mean xGA60_resid per team
        xGA60_resid    = xGA60_resid − team_xGA_resid

   f. League-mean-adjust and scale:
        league_resid = TOI-weighted mean of team-adjusted xGA60_resid
        d_adj_xGA60  = xGA60_resid − league_resid
        d_value60    = −d_adj_xGA60 × defense_weight
        (negated so that fewer xGA than expected → positive d_value → good dWAR)

4. Blocked shots (optional, same framework as box_war):
        d_value60 += blocks60_adj × block_weight

5. Replacement levels, GAR, WAR: identical to box_war.

Limitation
----------
Without on-ice player IDs per shot event (not published by PWHL), xGA is
attributed proportionally by TOI — not by actual defensive presence.
This makes it a team-quality proxy, but using xG rather than goals
provides more stable signal (less luck) than residual +/-.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression

log = logging.getLogger(__name__)

API_BASE   = "https://pwhl.hockey-statistics.com/api"
CACHE_DIR  = Path(__file__).parent / "data" / "raw" / "pbp_cache"

# Empirical xG rates from 2024-25 full season (102 games)
XG_MAP = {
    "Quality on net":    0.128,
    "Quality goal":      0.128,
    "Non quality on net":0.051,
    "Non quality goal":  0.051,
}

TEAM_MAP = {
    "Boston Fleet":         "BOS",
    "Minnesota Frost":      "MIN",
    "Montreal Victoire":    "MTL",
    "Montréal Victoire":    "MTL",   # accented form from schedule API
    "New York Sirens":      "NY",
    "Ottawa Charge":        "OTT",
    "Toronto Sceptres":     "TOR",
    "Seattle Torrent":      "SEA",
    "Vancouver Goldeneyes": "VAN",
}

SEASON_YEARS = {
    "2023-24": "2023/2024",
    "2024-25": "2024/2025",
    "2025-26": "2025/2026",
}

DEFAULT_MIN_TOI     = 50.0
DEFAULT_REPLACEMENT = 25.0


# ---------------------------------------------------------------------------
# Data fetching & caching
# ---------------------------------------------------------------------------

class PWHLApiLoader:
    """Fetch and cache PBP + summary data from the PWHL API."""

    def __init__(self, cache_dir: Path = CACHE_DIR, delay: float = 0.15):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "Mozilla/5.0 (compatible; PWHL-WAR/1.0)"

    def get_game_ids(self, season: str) -> tuple[list[int], dict]:
        """
        Return (game_ids, schedule_map) for a season.
        schedule_map: {game_id: {home_abbr, away_abbr, home_id, away_id}}
        """
        season_year = SEASON_YEARS.get(season, season)
        r = self._session.get(f"{API_BASE}/schedule", timeout=20)
        r.raise_for_status()
        game_ids    = []
        sched_map   = {}
        for g in r.json().get("games", []):
            if g.get("season_year") != season_year:
                continue
            if not g.get("status", "").startswith("Final"):
                continue
            gid = int(g["game_id"])
            game_ids.append(gid)
            sched_map[gid] = {
                "home_abbr": TEAM_MAP.get(g.get("home_team", ""), g.get("home_team", "UNK")[:3].upper()),
                "away_abbr": TEAM_MAP.get(g.get("away_team", ""), g.get("away_team", "UNK")[:3].upper()),
                "home_id":   str(g.get("home_team_id", "")),
                "away_id":   str(g.get("away_team_id", "")),
            }
        log.info("Season %s: %d completed games", season, len(game_ids))
        return game_ids, sched_map

    def get_pbp(self, game_id: int) -> list[dict]:
        return self._cached(f"pbp_{game_id}.json",
                            f"{API_BASE}/game/playbyplay/{game_id}")

    def get_summary(self, game_id: int) -> dict:
        return self._cached(f"summ_{game_id}.json",
                            f"{API_BASE}/game/summary/{game_id}")

    def _cached(self, fname: str, url: str):
        path = self.cache_dir / fname
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(self.delay)
        r = self._session.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        path.write_text(json.dumps(data), encoding="utf-8")
        return data


# ---------------------------------------------------------------------------
# Helper parsers
# ---------------------------------------------------------------------------

def _parse_toi(s) -> float:
    """'MM:SS' or numeric seconds → float minutes."""
    if not s:
        return 0.0
    s = str(s).strip()
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) + int(parts[1]) / 60
        except ValueError:
            return 0.0
    try:
        return float(s) / 60
    except ValueError:
        return 0.0


def _pid_from_url(url: str) -> int | None:
    """Extract numeric player ID from headshot URL."""
    try:
        stem = Path(url).stem
        return int(stem)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class XGAWar:
    """
    xG-based WAR using Fenwick shot quality as the defensive input.

    Parameters
    ----------
    min_toi_min     : Minimum TOI (minutes) to qualify (default 50).
    replacement_pct : Percentile defining replacement level (default 25).
    defense_weight  : Scale applied to the d_adj_xGA60 defensive component.
                      Default 0.36 (same tuning as box_war).
    block_weight    : xG value per position+team-adjusted block per 60.
                      Default 0.04.  Only used when blocks_df is supplied.
    goals_per_win   : Override Pythagorean estimate if desired.
    """

    def __init__(
        self,
        min_toi_min:     float = DEFAULT_MIN_TOI,
        replacement_pct: float = DEFAULT_REPLACEMENT,
        defense_weight:  float = 0.36,
        block_weight:    float = 0.04,
        goals_per_win:   float | None = None,
    ):
        self.min_toi_min     = min_toi_min
        self.replacement_pct = replacement_pct
        self.defense_weight  = defense_weight
        self.block_weight    = block_weight
        self._gpw_override   = goals_per_win

        self.results_:              pd.DataFrame | None = None
        self.goals_per_win_:        float | None        = None
        self.o_replacement_val60_:  float | None        = None
        self.d_replacement_val60_:  float | None        = None

    # ------------------------------------------------------------------

    def fit(
        self,
        game_ids:   list[int],
        loader:     PWHLApiLoader,
        sched_map:  dict | None = None,
        blocks_df:  pd.DataFrame | None = None,
    ) -> "XGAWar":
        """
        Parameters
        ----------
        game_ids  : list of completed game IDs for the season
        loader    : PWHLApiLoader instance (handles caching)
        blocks_df : DataFrame with [PlayerID, blocks] — optional
        """
        records = self._aggregate_games(game_ids, loader, sched_map or {})
        if not records:
            raise ValueError("No player data aggregated — check game IDs.")

        df = pd.DataFrame(records)
        df["toi_min"] = df["toi_min"].clip(lower=0)

        # Pythagorean goals-per-win
        gpw = self._gpw_override or self._compute_gpw(df)
        self.goals_per_win_ = gpw

        # --- Offensive xG60 ---
        df["ixG60"] = df["ixG"] / df["toi_min"].clip(lower=0.1) * 60

        # --- Defensive xGA60 ---
        df["xGA60"] = df["xGA"] / df["toi_min"].clip(lower=0.1) * 60

        qual_mask = df["toi_min"] >= self.min_toi_min
        qual_fit  = df[qual_mask]

        # Regress out correlation with offensive zone time
        reg = LinearRegression().fit(
            qual_fit[["ixG60"]].values,
            qual_fit["xGA60"].values,
        )
        df["xGA60_resid"] = df["xGA60"] - (
            reg.intercept_ + reg.coef_[0] * df["ixG60"]
        )
        log.info("xGA60 ~ ixG60: intercept=%.3f slope=%.3f",
                 reg.intercept_, reg.coef_[0])

        # Team-quality adjustment
        team_resid = (
            df[qual_mask]
            .groupby("team")["xGA60_resid"]
            .apply(lambda g: np.average(
                g, weights=df.loc[g.index, "toi_min"].clip(lower=0.1)
            ))
        )
        df["team_xGA_resid"] = df["team"].map(team_resid).fillna(0)
        df["xGA60_resid"]    = df["xGA60_resid"] - df["team_xGA_resid"]
        log.info("Team xGA60_resid adjustments: %s",
                 team_resid.round(3).to_dict())

        # League-mean-adjust; negate so fewer xGA → positive d_value
        league_resid = np.average(
            df.loc[qual_mask, "xGA60_resid"],
            weights=df.loc[qual_mask, "toi_min"].clip(lower=0.1),
        )
        df["d_adj_xGA60"] = df["xGA60_resid"] - league_resid
        df["d_value60"]   = -df["d_adj_xGA60"] * self.defense_weight

        # --- Blocked shots (optional) ---
        if blocks_df is not None and not blocks_df.empty:
            blk = blocks_df.copy()
            if "PlayerID" in blk.columns:
                blk = blk.rename(columns={"PlayerID": "player_id"})
            df = df.merge(
                blk[["player_id", "blocks"]].rename(columns={"blocks": "_blocks"}),
                on="player_id", how="left",
            )
            df["_blocks"] = df["_blocks"].fillna(0)
            df["blocks60"] = df["_blocks"] / df["toi_min"].clip(lower=0.1) * 60

            df["_pos"] = df["pos"].str.upper().map(
                lambda p: "D" if p in {"LD", "RD", "D"} else "F"
            )
            pos_means = df[qual_mask].groupby("_pos")["blocks60"].mean()
            df["blocks60_pos_adj"] = (
                df["blocks60"] - df["_pos"].map(pos_means).fillna(0)
            )
            blk_team = (
                df[qual_mask]
                .groupby("team")["blocks60_pos_adj"]
                .apply(lambda g: np.average(
                    g, weights=df.loc[g.index, "toi_min"].clip(lower=0.1)
                ))
            )
            df["blocks60_adj"] = (
                df["blocks60_pos_adj"] - df["team"].map(blk_team).fillna(0)
            )
            df["block_val60"] = df["blocks60_adj"] * self.block_weight
            df["d_value60"]  += df["block_val60"]
        else:
            df["blocks60"] = df["blocks60_adj"] = df["block_val60"] = 0.0

        # --- Combined value ---
        df["value60"] = df["ixG60"] + df["d_value60"]

        # --- Replacement levels ---
        qual = df[df["toi_min"] >= self.min_toi_min]
        toi_sum = qual["toi_min"].sum()
        if toi_sum == 0:
            raise ValueError("No qualified players.")

        o_repl = float(np.percentile(qual["ixG60"],   self.replacement_pct))
        d_repl = float(np.percentile(qual["d_value60"], self.replacement_pct))
        self.o_replacement_val60_ = o_repl
        self.d_replacement_val60_ = d_repl

        # --- GAR → WAR ---
        df["oGAR"] = ((df["ixG60"]   - o_repl) * df["toi_min"] / 60).round(3)
        df["dGAR"] = ((df["d_value60"] - d_repl) * df["toi_min"] / 60).round(3)
        df["GAR"]  = (df["oGAR"] + df["dGAR"]).round(3)

        df["oWAR"]  = (df["oGAR"] / gpw).round(3)
        df["dWAR"]  = (df["dGAR"] / gpw).round(3)
        df["WAR"]   = (df["oWAR"] + df["dWAR"]).round(3)
        df["war60"] = (df["WAR"] / df["toi_min"].clip(lower=0.1) * 60).round(3)

        self.results_ = df.sort_values("WAR", ascending=False).reset_index(drop=True)
        return self

    def get_war(self, min_toi: float | None = None) -> pd.DataFrame:
        if self.results_ is None:
            raise RuntimeError("Call fit() first.")
        df  = self.results_.copy()
        cut = min_toi if min_toi is not None else self.min_toi_min
        df  = df[df["toi_min"] >= cut].reset_index(drop=True)
        cols = [c for c in [
            "name", "player_id", "team", "pos", "gp", "toi_min",
            "ixG", "xGA", "ixG60", "xGA60", "xGA60_resid", "d_adj_xGA60",
            "blocks60", "blocks60_adj", "block_val60",
            "d_value60", "value60",
            "oGAR", "dGAR", "GAR", "oWAR", "dWAR", "WAR", "war60",
            "o_replacement_val60", "d_replacement_val60", "goals_per_win",
        ] if c in df.columns]
        df["o_replacement_val60"] = round(self.o_replacement_val60_, 4)
        df["d_replacement_val60"] = round(self.d_replacement_val60_, 4)
        df["goals_per_win"]       = round(self.goals_per_win_, 4)
        return df[cols]

    def summary(self, top_n: int = 20) -> None:
        df = self.get_war()
        print(f"\n{'='*68}")
        print(f"  PWHL xGA-WAR Leaderboard  (top {top_n})")
        print(f"  Off. replacement (p{self.replacement_pct:.0f}): {self.o_replacement_val60_:+.3f} ixG60")
        print(f"  Def. replacement (p{self.replacement_pct:.0f}): {self.d_replacement_val60_:+.3f} d_val60")
        print(f"  Goals per win: {self.goals_per_win_:.3f}")
        print(f"{'='*68}")
        show = [c for c in ["name", "team", "pos", "gp", "toi_min",
                             "ixG60", "xGA60", "d_value60", "oWAR", "dWAR", "WAR"]
                if c in df.columns]
        print(df[show].head(top_n).to_string(index=False))
        print(f"{'='*68}\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _aggregate_games(
        self,
        game_ids:  list[int],
        loader:    PWHLApiLoader,
        sched_map: dict,
    ) -> list[dict]:
        """Process each game's PBP + summary into player-level season totals."""

        players: dict[str, dict] = {}
        total_goals  = 0
        n_team_games = 0

        for i, gid in enumerate(game_ids):
            if i % 20 == 0:
                log.info("  Processing game %d/%d (id=%d)...", i+1, len(game_ids), gid)

            try:
                pbp  = loader.get_pbp(gid)
                summ = loader.get_summary(gid)
            except Exception as exc:
                log.warning("Game %d failed: %s — skipping", gid, exc)
                continue

            # Team abbreviations from pre-built schedule map
            sm        = sched_map.get(gid, {})
            home_abbr = sm.get("home_abbr", "HOM")
            away_abbr = sm.get("away_abbr", "VIS")

            # Build teamId → abbr from goal events (has id + abbreviation)
            tid_to_abbr: dict[str, str] = {}
            for e in pbp:
                if e.get("event") == "goal":
                    t = e["details"].get("team", {})
                    if t.get("id") and t.get("abbreviation"):
                        tid_to_abbr[str(t["id"])] = t["abbreviation"]

            # Collect all shooter team IDs from shot events
            all_shot_tids: set[str] = set()
            for e in pbp:
                if e.get("event") == "shot":
                    tid = str(e["details"].get("shooterTeamId", ""))
                    if tid:
                        all_shot_tids.add(tid)

            # Resolve home_id / away_id using goal-event tid→abbr mapping.
            # Goal events are the authoritative abbr source; schedule names
            # are a fallback.  We match by checking which goal-event abbr
            # most closely corresponds to each side's schedule name.
            home_tid, away_tid = "", ""
            sched_abbrs = {home_abbr, away_abbr}
            for tid, abbr in tid_to_abbr.items():
                # Direct abbr match against schedule-derived abbrs
                if abbr == home_abbr:
                    home_tid = tid
                elif abbr == away_abbr:
                    away_tid = tid
                # If schedule abbr was wrong (e.g. "MON" vs "MTL"), check
                # whether the goal-event abbr belongs to neither side — if
                # there are exactly 2 unique goal abbrs, assign by order.
            # Fallback: infer from the two shot team IDs
            if len(all_shot_tids) == 2 and (not home_tid or not away_tid):
                known   = {home_tid, away_tid} - {""}
                unknown = all_shot_tids - known
                if len(unknown) == 1:
                    inferred = unknown.pop()
                    if not home_tid:
                        home_tid = inferred
                    else:
                        away_tid = inferred
                elif len(unknown) == 2:
                    # Neither side resolved — assign arbitrarily but consistently
                    sorted_tids = sorted(unknown)
                    home_tid, away_tid = sorted_tids[0], sorted_tids[1]
            # Use goal-event abbrs as canonical team labels where available
            if home_tid and home_tid in tid_to_abbr:
                home_abbr = tid_to_abbr[home_tid]
            if away_tid and away_tid in tid_to_abbr:
                away_abbr = tid_to_abbr[away_tid]

            # --- Team xG from shot events ---
            team_xG_for: dict[str, float] = {}
            player_ixG:  dict[int, float] = {}

            for e in pbp:
                if e.get("event") != "shot":
                    continue
                d   = e["details"]
                q   = d.get("shotQuality", "")
                xg  = XG_MAP.get(q, 0.0)
                if xg == 0.0:
                    continue
                tid = str(d.get("shooterTeamId", ""))
                pid = d.get("shooter", {}).get("id")
                team_xG_for[tid] = team_xG_for.get(tid, 0.0) + xg
                if pid:
                    player_ixG[pid] = player_ixG.get(pid, 0.0) + xg
                if "goal" in q:
                    total_goals += 1

            home_xGA = team_xG_for.get(away_tid, 0.0)   # away scored on home
            away_xGA = team_xG_for.get(home_tid, 0.0)   # home scored on away

            # --- Parse player TOI from summary and attribute xGA ---
            for side, team_abbr, team_xGA in [
                ("homeTeam",     home_abbr, home_xGA),
                ("visitingTeam", away_abbr, away_xGA),
            ]:
                skaters = summ.get(side, {}).get("skaters", [])
                skater_tois: list[tuple[int, float, str, str]] = []

                for sk in skaters:
                    pid = _pid_from_url(sk.get("playerImageURL", ""))
                    toi = _parse_toi(sk.get("stats", {}).get("toi", ""))
                    if pid and toi > 0:
                        skater_tois.append((
                            pid, toi,
                            sk.get("name", str(pid)),
                            sk.get("position", "F"),
                        ))

                team_total_toi = sum(t for _, t, _, _ in skater_tois)
                if team_total_toi <= 0:
                    continue

                n_team_games += 1

                for pid, toi, name, pos in skater_tois:
                    key = f"{pid}|{team_abbr}"
                    if key not in players:
                        players[key] = {
                            "player_id": pid,
                            "name": name,
                            "pos":  pos,
                            "team": team_abbr,
                            "gp": 0, "toi_min": 0.0,
                            "ixG": 0.0, "xGA": 0.0,
                        }
                    p = players[key]
                    p["gp"]      += 1
                    p["toi_min"] += toi
                    p["ixG"]     += player_ixG.get(pid, 0.0)
                    p["xGA"]     += team_xGA * (toi / team_total_toi)

        self._total_goals  = total_goals
        self._n_team_games = n_team_games
        log.info("Aggregated %d player-game entries → %d unique players",
                 sum(p["gp"] for p in players.values()), len(players))
        return list(players.values())

    def _compute_gpw(self, df: pd.DataFrame) -> float:
        """Estimate goals_per_win from total goals and team-games seen."""
        ng = getattr(self, "_n_team_games", 0)
        tg = getattr(self, "_total_goals", 0)
        if ng > 0 and tg > 0:
            avg = tg / ng
            gpw = 2.0 * avg
            log.info("Avg goals/team/game=%.3f → goals_per_win=%.3f", avg, gpw)
            return gpw
        return 6.0
