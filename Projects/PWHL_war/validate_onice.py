"""
validate_onice.py
-----------------
Comparison harness for the on-ice defensive metric experiment
(rpi/on-ice-defensive-metric-for-pwhl-war). Scores each candidate by how
repeatable its player-level defence score (d_value60) is:

  * split-half: the same season's regular-season games split by alternating
    game order (odd vs even positions); correlate half A with half B
  * season-to-season: 2023-24 > 2024-25 and 2024-25 > 2025-26
  * cohort: skaters with >= 100 five-on-five minutes in the full season(s), the
    same players for every candidate

Candidates (each supplies an fa_df [player_id, xGA, toi_min] to the unchanged
XGWar defence math; blocks and OZS covariates are off for all of them):
  baseline         current TOI-share attribution, API xG
  toi_share_ownxg  same TOI-share attribution but with the own on-ice xG (control)
  onice_xga        true on-ice attribution, own xG (Option 1)
  naive            team-average rate x player minutes, own xG (no individual signal)
  rapm             shrunk ridge (Option 2; added in Phase 6)

    python validate_onice.py --candidate baseline      -> out/cand_baseline.csv
    python validate_onice.py --compare                 -> out/model_comparison_table.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pwhl_war.onice import (
    _regulation_shots, attach_onice, build_fa_frame, credit_xga, five_on_five_minutes,
    fit_onice_xg, load_schedule, load_season, score_shots,
)

log = logging.getLogger("validate_onice")

SEASONS = ("2023-24", "2024-25", "2025-26")
TRANSITIONS = (("2023-24", "2024-25"), ("2024-25", "2025-26"))
MIN_5V5 = 100.0
OUT_DIR = Path("rpi/on-ice-defensive-metric-for-pwhl-war/out")
N_BOOT = 2000
SEED = 42
CANDIDATES = ("baseline", "toi_share_ownxg", "onice_xga", "naive", "rapm")


@dataclass
class SeasonData:
    """Everything one season's candidates need, restricted to regular-season games."""
    season: str
    game_data: pd.DataFrame            # player-game rows (GameID, Team, PlayerID, TOI, ...)
    shots: pd.DataFrame                # shots on goal with rebuilt sets and own xg
    shifts: pd.DataFrame
    sched_map: dict                    # game_id -> {home_abbr, away_abbr, home_id, away_id}
    coord_df: pd.DataFrame | None      # API shots (baseline xG, offensive xG)
    skater_ids: set = field(default_factory=set)
    train_coord_df: pd.DataFrame | None = None   # API shots from the other seasons


# ---------------------------------------------------------------- splitting
def split_half_games(game_ids) -> tuple[list[int], list[int]]:
    """Alternate whole games in sorted order: positions 0,2,4... vs 1,3,5..."""
    ids = sorted(int(g) for g in game_ids)
    return ids[0::2], ids[1::2]


def spearman_brown(r: float) -> float:
    """Full-length reliability from a half-length correlation."""
    return 2 * r / (1 + r)


# ---------------------------------------------------------- fa_df candidates
def _is_5v5(shots: pd.DataFrame) -> pd.Series:
    return shots["rebuild_ok"] & (shots["n_home"] == 5) & (shots["n_away"] == 5) & shots["xg"].notna()


def _team_xga(data: SeasonData, ids) -> pd.DataFrame:
    """Own-xG 5v5 xG against each team in each game: [game_id, abbr, xg]."""
    sh = data.shots[data.shots["game_id"].isin(ids) & _is_5v5(data.shots)]
    rows = []
    for gid, g in sh.groupby("game_id"):
        sm = data.sched_map[gid]
        by_home = g["team_id"] == g["home_team_id"]
        rows.append((gid, sm["away_abbr"], float(g.loc[by_home, "xg"].sum())))      # home shots -> away defends
        rows.append((gid, sm["home_abbr"], float(g.loc[~by_home, "xg"].sum())))
    return pd.DataFrame(rows, columns=["game_id", "abbr", "xg"])


def _toi_all(data: SeasonData, ids) -> pd.Series:
    gd = data.game_data[data.game_data["GameID"].isin(ids)]
    return gd.groupby("PlayerID")["TOI"].sum().rename("toi_min")


def fa_onice_xga(data: SeasonData, ids) -> pd.DataFrame:
    sh = data.shots[data.shots["game_id"].isin(ids)]
    sf = data.shifts[data.shifts["game_id"].isin(ids)]
    return build_fa_frame(credit_xga(sh), five_on_five_minutes(sf, data.skater_ids))


def fa_toi_share_ownxg(data: SeasonData, ids) -> pd.DataFrame:
    gd = data.game_data[data.game_data["GameID"].isin(ids)]
    totals = _team_xga(data, ids)
    credit: dict[int, float] = {}
    for _, t in totals.iterrows():
        team = gd[(gd["GameID"] == t["game_id"]) & (gd["Team"] == t["abbr"])]
        tot = team["TOI"].sum()
        if tot <= 0:
            continue
        for pid, toi in zip(team["PlayerID"], team["TOI"]):
            credit[int(pid)] = credit.get(int(pid), 0.0) + t["xg"] * toi / tot
    return build_fa_frame(pd.Series(credit, dtype=float), _toi_all(data, ids))


def fa_naive(data: SeasonData, ids) -> pd.DataFrame:
    gd = data.game_data[data.game_data["GameID"].isin(ids)]
    team_xga = _team_xga(data, ids).groupby("abbr")["xg"].sum()
    team_toi = gd.groupby("Team")["TOI"].sum()
    rate = (team_xga / team_toi).fillna(0.0)
    pt = gd.groupby(["PlayerID", "Team"])["TOI"].sum().reset_index()
    pt["xGA"] = pt["TOI"] * pt["Team"].map(rate).fillna(0.0)
    return build_fa_frame(pt.groupby("PlayerID")["xGA"].sum(), _toi_all(data, ids))


class _SubsetLoader:
    """PWHLApiLoader limited to given games, with no schedule request."""

    def __new__(cls, ids, sched_map):
        from pwhl_war.xga_war import PWHLApiLoader

        class Loader(PWHLApiLoader):
            def get_game_ids(self, season):
                return list(ids), {g: sched_map[g] for g in ids}
        return Loader()


def fa_baseline(data: SeasonData, ids) -> pd.DataFrame:
    from pwhl_war.xga_war import XGAWar
    coord = data.coord_df[data.coord_df["game_id"].astype(int).isin(ids)]
    # XGAWar matches shots with `coord_df["game_id"] == str(game_id)`: keep ids as strings
    coord = coord.assign(game_id=coord["game_id"].astype(int).astype(str))
    loader = _SubsetLoader(sorted(ids), data.sched_map)
    return XGAWar.build_fa_season(data.season, loader, coord_df=coord)


def fa_rapm(data: SeasonData, ids) -> pd.DataFrame:
    """Option 2: ridge on five-on-five stints, shrunk toward the Option 1 on-ice xGA."""
    from pwhl_war.onice_rapm import build_stints, fit_rapm
    prior = fa_onice_xga(data, ids)
    shots = data.shots[data.shots["game_id"].isin(ids)]
    shifts = data.shifts[data.shifts["game_id"].isin(ids)]
    return fit_rapm(build_stints(shots, shifts, data.skater_ids), prior)


_FA = {"baseline": fa_baseline, "toi_share_ownxg": fa_toi_share_ownxg,
       "onice_xga": fa_onice_xga, "naive": fa_naive, "rapm": fa_rapm}


def fa_candidate(candidate: str, data: SeasonData, game_ids) -> pd.DataFrame:
    """The candidate's [player_id, xGA, toi_min] over only the given games."""
    if candidate not in _FA:
        raise ValueError(f"unknown candidate {candidate!r}; choose from {sorted(_FA)}")
    return _FA[candidate](data, set(int(g) for g in game_ids))


# ------------------------------------------------------------ d_value60 run
def candidate_d_value60(candidate: str, data: SeasonData, game_ids, min_toi_min: float = 50.0) -> pd.DataFrame:
    """Run the unchanged XGWar defence math with the candidate's fa_df.

    Returns [player_id, toi_5v5, d_value60], one row per player (traded players
    collapsed by TOI-weighted mean). Blocks and OZS are off for every candidate.
    """
    from pwhl_war.box_war import XGWar
    ids = set(int(g) for g in game_ids)
    fa = fa_candidate(candidate, data, ids)
    gd = data.game_data[data.game_data["GameID"].isin(ids)]
    coord = data.coord_df[data.coord_df["game_id"].astype(int).isin(ids)]
    model = XGWar(min_toi_min=min_toi_min, replacement_pct=25.0, defense_weight=-0.10,
                  block_weight=0.04, goals_per_win=4.6667, team_adjust=False)
    model.fit(gd, pbp_df=coord, train_pbp_df=data.train_coord_df, fa_df=fa, blocks_df=None, ozs_df=None)
    war = model.get_war(min_toi=0)
    w = war.assign(_w=war["toi_min"].clip(lower=0.1))
    collapsed = (w.assign(_dw=w["d_value60"] * w["_w"]).groupby("player_id")
                 .agg(_dw=("_dw", "sum"), _w=("_w", "sum")))
    d = (collapsed["_dw"] / collapsed["_w"]).rename("d_value60").reset_index()
    mins = five_on_five_minutes(data.shifts[data.shifts["game_id"].isin(ids)], data.skater_ids)
    d["toi_5v5"] = d["player_id"].map(mins).fillna(0.0)
    return d[["player_id", "toi_5v5", "d_value60"]]


# ------------------------------------------------------------------- metrics
def season_to_season_rows(full: dict[str, pd.DataFrame], min_toi: float = MIN_5V5) -> pd.DataFrame:
    """Pairs [player_id, transition, key, x, y] for players above the 5v5 cut in both seasons."""
    out = []
    for a, b in TRANSITIONS:
        if a not in full or b not in full:
            continue
        m = full[a].merge(full[b], on="player_id", suffixes=("_a", "_b"))
        m = m[(m["toi_5v5_a"] >= min_toi) & (m["toi_5v5_b"] >= min_toi)]
        out.append(pd.DataFrame({
            "player_id": m["player_id"], "transition": f"{a}>{b}",
            "key": m["player_id"].astype(str) + f"|{a}>{b}",
            "x": m["d_full_a"], "y": m["d_full_b"],
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["player_id", "transition", "key", "x", "y"])


def _r(x, y) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else float("nan")


def bootstrap_diff(rows_a: pd.DataFrame, rows_b: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Bootstrap (over players) of r(a) - r(b) for two candidates on the same rows.

    Each frame has columns key, player_id, x, y. A player's rows travel together.
    """
    m = rows_a.merge(rows_b, on="key", suffixes=("_a", "_b"))
    pid = m["player_id_a"].to_numpy()
    xa, ya, xb, yb = (m[c].to_numpy(float) for c in ("x_a", "y_a", "x_b", "y_b"))
    pos: dict = {}
    for i, p in enumerate(pid):
        pos.setdefault(p, []).append(i)
    players = list(pos)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(players), len(players))
        idx = np.concatenate([pos[players[k]] for k in pick])
        diffs.append(_r(xa[idx], ya[idx]) - _r(xb[idx], yb[idx]))
    diffs = np.array(diffs)
    return {"r_a": _r(xa, ya), "r_b": _r(xb, yb), "diff": _r(xa, ya) - _r(xb, yb),
            "lo": float(np.nanpercentile(diffs, 2.5)), "hi": float(np.nanpercentile(diffs, 97.5)),
            "n": int(len(m))}


# ------------------------------------------------------------- data assembly
def build_all_data() -> dict[str, SeasonData]:
    """Load, rebuild on-ice sets and score own xG for all three seasons.

    Own xG and the offensive-xG training data use leave-one-season-out: each
    season is scored by a model fit on the other two.
    """
    from pwhl_war.coord_loader import CoordLoader
    from pwhl_war.scope import regular_season_game_ids
    from pwhl_war.xga_war import PWHLApiLoader, _parse_toi, _pid_from_url

    sched = load_schedule()
    raw, reg_ids, api = {}, {}, {}
    for s in SEASONS:
        raw[s] = load_season(s)
        reg_ids[s] = regular_season_game_ids(sched, s)
        api[s] = CoordLoader().fetch_pbp([s])
        api[s]["game_id"] = api[s]["game_id"].astype(int)

    train_shots = {s: _regulation_shots(raw[s]["pbp"], reg_ids[s]) for s in SEASONS}
    out = {}
    for s in SEASONS:
        others = [t for t in SEASONS if t != s]
        model = fit_onice_xg(pd.concat([train_shots[t] for t in others], ignore_index=True))
        skaters = set(raw[s]["box"]["player_id"].astype(int))
        shifts = raw[s]["shifts"][raw[s]["shifts"]["game_id"].isin(set(reg_ids[s]))]
        shots = attach_onice(train_shots[s], shifts, skaters)
        shots["xg"] = score_shots(model, shots)

        loader = PWHLApiLoader()
        game_ids, sched_map = loader.get_game_ids(s)
        keep = [g for g in game_ids if g in set(reg_ids[s])]
        sched_map = {g: sched_map[g] for g in keep}
        records = []
        for gid in keep:
            summ = loader.get_summary(gid)
            sm = sched_map[gid]
            for side, abbr in (("homeTeam", sm["home_abbr"]), ("visitingTeam", sm["away_abbr"])):
                for sk in summ.get(side, {}).get("skaters", []):
                    pid = _pid_from_url(sk.get("playerImageURL", ""))
                    toi = _parse_toi(sk.get("stats", {}).get("toi", ""))
                    if not pid or toi <= 0:
                        continue
                    st = sk.get("stats", {})
                    pos = "D" if str(sk.get("position", "F")).upper() in {"LD", "RD", "D"} else "F"
                    records.append({
                        "GameID": gid, "Season": s.replace("-", ""), "SeasonStage": "Regular", "Team": abbr,
                        "PlayerID": pid, "Name": sk.get("name", str(pid)), "position": pos, "TOI": toi,
                        "EV_ixG": 0.0, "PP_ixG": 0.0, "SH_ixG": 0.0,
                        "EV_G": st.get("goals", 0), "PP_G": 0, "SH_G": 0, "EN_G": 0,
                        "EV_A1": st.get("assists", 0), "EV_A2": 0, "PP_A1": 0, "PP_A2": 0,
                        "SH_A1": 0, "SH_A2": 0,
                        "plusMinus": st.get("plusMinus", 0), "PIM": st.get("penaltyMinutes", 0),
                        "hits": st.get("hits", 0),
                    })
        coord = api[s][api[s]["game_id"].isin(set(keep))]
        train_coord = pd.concat([api[t] for t in others], ignore_index=True)
        out[s] = SeasonData(season=s, game_data=pd.DataFrame(records), shots=shots, shifts=shifts,
                            sched_map=sched_map, coord_df=coord, skater_ids=skaters,
                            train_coord_df=train_coord)
    return out


def run_candidate(candidate: str, data: dict[str, SeasonData]) -> pd.DataFrame:
    """One row per player-season: full-season and half-season d_value60, plus 5v5 minutes."""
    rows = []
    for s in SEASONS:
        d = data[s]
        ids = sorted(d.sched_map)
        a_ids, b_ids = split_half_games(ids)
        full = candidate_d_value60(candidate, d, ids, 50.0).rename(columns={"d_value60": "d_full"})
        ha = candidate_d_value60(candidate, d, a_ids, 25.0).rename(columns={"d_value60": "d_half_a"})
        hb = candidate_d_value60(candidate, d, b_ids, 25.0).rename(columns={"d_value60": "d_half_b"})
        m = full.merge(ha[["player_id", "d_half_a"]], on="player_id", how="left") \
                .merge(hb[["player_id", "d_half_b"]], on="player_id", how="left")
        m.insert(1, "season", s)
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def _split_half_rows(res: pd.DataFrame, min_toi: float = MIN_5V5) -> pd.DataFrame:
    m = res[(res["toi_5v5"] >= min_toi)].dropna(subset=["d_half_a", "d_half_b"])
    return pd.DataFrame({"player_id": m["player_id"], "season": m["season"],
                         "key": m["player_id"].astype(str) + "|" + m["season"],
                         "x": m["d_half_a"], "y": m["d_half_b"]})


def write_comparison(candidates=CANDIDATES) -> Path:
    """Repeatability table for every candidate that has a cand_*.csv, plus bootstrap differences."""
    res = {c: pd.read_csv(OUT_DIR / f"cand_{c}.csv") for c in candidates if (OUT_DIR / f"cand_{c}.csv").exists()}
    cohort_note = {}
    sh, s2s = {}, {}
    for c, df in res.items():
        sh[c] = _split_half_rows(df)
        full = {s: df[df["season"] == s][["player_id", "toi_5v5", "d_full"]] for s in SEASONS}
        s2s[c] = season_to_season_rows(full)
        cohort_note[c] = (len(sh[c]), len(s2s[c]))

    lines = ["# Repeatability comparison (generated by `validate_onice.py --compare`)", "",
             f"Cohort: skaters with at least {MIN_5V5:.0f} five-on-five minutes in the full season(s). "
             "Metric: d_value60 from the unchanged XGWar defence math; blocks and OZS off for every candidate.", "",
             "| candidate | split-half r (pooled) | Spearman-Brown | split-half by season | n | season-to-season r (pooled) | by transition | n |",
             "|---|---|---|---|---|---|---|---|"]
    for c in res:
        r = _r(sh[c]["x"], sh[c]["y"])
        by_s = ", ".join(f"{s[-5:]}: {_r(g['x'], g['y']):.2f}" for s, g in sh[c].groupby("season"))
        r2 = _r(s2s[c]["x"], s2s[c]["y"]) if len(s2s[c]) else float("nan")
        by_t = ", ".join(f"{t}: {_r(g['x'], g['y']):.2f}" for t, g in s2s[c].groupby("transition"))
        lines.append(f"| {c} | {r:.3f} | {spearman_brown(r):.3f} | {by_s} | {len(sh[c])} | {r2:.3f} | {by_t} | {len(s2s[c])} |")

    pairs = [("onice_xga", "baseline"), ("onice_xga", "toi_share_ownxg"), ("toi_share_ownxg", "baseline"),
             ("onice_xga", "naive"), ("rapm", "baseline"), ("rapm", "onice_xga"), ("rapm", "naive")]
    lines += ["", "Bootstrap differences in r (over players, 2000 resamples, seed 42); interval excludes 0 = a clear difference.", "",
              "| comparison | measure | r(first) | r(second) | difference | 95% interval | n |", "|---|---|---|---|---|---|---|"]
    for a, b in pairs:
        if a in res and b in res:
            for name, tab in (("split-half", sh), ("season-to-season", s2s)):
                o = bootstrap_diff(tab[a], tab[b])
                lines.append(f"| {a} - {b} | {name} | {o['r_a']:.3f} | {o['r_b']:.3f} | {o['diff']:+.3f} | "
                             f"[{o['lo']:+.3f}, {o['hi']:+.3f}] | {o['n']} |")
    path = OUT_DIR / "comparison_table.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=CANDIDATES)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.candidate:
        res = run_candidate(args.candidate, build_all_data())
        path = OUT_DIR / f"cand_{args.candidate}.csv"
        res.to_csv(path, index=False)
        log.info("wrote %s (%d rows)", path, len(res))
    if args.compare:
        write_comparison()
    return 0


if __name__ == "__main__":
    sys.exit(main())
