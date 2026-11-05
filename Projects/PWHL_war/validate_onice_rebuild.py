"""
validate_onice_rebuild.py
-------------------------
Validate the on-ice sets rebuilt from shifts (pwhl_war.onice.attach_onice)
against two independent checks, per season, regular-season games, periods 1-3:

  1. Goal credits: the goal rows' plus_player_* (scoring team) and
     minus_player_* (conceding team) columns name the skaters on the ice at
     each goal. Share of those slots found in the rebuilt set (gate >= 89%),
     next to the share found in the shipped on_ice_home / on_ice_away lists.
  2. Shooter-in-own-set: the shooter of each shot is on the ice for their team.

Writes rpi/on-ice-defensive-metric-for-pwhl-war/out/onice_validation.md.
Exit status 1 if any season misses the goal-credit gate.

Run from Projects/PWHL_war/:
    /c/Users/morga/AppData/Local/Python/bin/python.exe validate_onice_rebuild.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from pwhl_war.onice import attach_onice, dedupe_goals, load_season
from pwhl_war.scope import regular_season_game_ids

SEASONS = ("2023-24", "2024-25", "2025-26")
GATE = 0.89
OUT = Path("rpi/on-ice-defensive-metric-for-pwhl-war/out/onice_validation.md")
PLUS = [f"plus_player_{n}_id" for n in ("one", "two", "three", "four", "five")]
MINUS = [f"minus_player_{n}_id" for n in ("one", "two", "three", "four", "five")]


def _parse_list(value) -> set[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return set()
    return {int(x) for x in str(value).split(",") if x.strip()}


def _scorer_is_home(df: pd.DataFrame) -> pd.Series:
    return df["team_id"] == df["home_team_id"]


def goal_credit_recovery(goals: pd.DataFrame) -> dict:
    """Slots found in the rebuilt sets and in the shipped lists."""
    tot = {"plus": 0, "minus": 0}
    found_new = {"plus": 0, "minus": 0}
    found_old = {"plus": 0, "minus": 0}
    scorer_home = _scorer_is_home(goals)
    for (_, r), home in zip(goals.iterrows(), scorer_home):
        sets = {
            "new": (set(r["home_skaters"]), set(r["away_skaters"])),
            "old": (_parse_list(r["on_ice_home"]), _parse_list(r["on_ice_away"])),
        }
        for kind, cols, same_side in (("plus", PLUS, True), ("minus", MINUS, False)):
            ids = [int(r[c]) for c in cols if pd.notna(r[c])]
            tot[kind] += len(ids)
            for which, found in (("new", found_new), ("old", found_old)):
                h, a = sets[which]
                side = (h if home else a) if same_side else (a if home else h)
                found[kind] += sum(1 for i in ids if i in side)
    allv = sum(tot.values())
    return {
        "goals": len(goals), "slots": allv,
        "new_plus": found_new["plus"] / tot["plus"], "new_minus": found_new["minus"] / tot["minus"],
        "new_all": sum(found_new.values()) / allv,
        "old_plus": found_old["plus"] / tot["plus"], "old_minus": found_old["minus"] / tot["minus"],
        "old_all": sum(found_old.values()) / allv,
    }


def shooter_in_own_set(shots: pd.DataFrame) -> dict:
    home = _scorer_is_home(shots)
    new_hit = old_hit = 0
    for (_, r), h in zip(shots.iterrows(), home):
        pid = int(r["player_id"])
        new_hit += pid in (set(r["home_skaters"]) if h else set(r["away_skaters"]))
        old_hit += pid in (_parse_list(r["on_ice_home"]) if h else _parse_list(r["on_ice_away"]))
    n = len(shots)
    return {"shots": n, "new": new_hit / n, "old": old_hit / n}


def main() -> int:
    sched = pd.read_parquet("pwhl_war/data/raw/sdv/schedule_master.parquet")
    lines = ["# On-ice rebuild validation (Phase 3)", "",
             "Regular-season games, periods 1-3. Rebuilt = shifts with `end_s <= clock < start_s` "
             "(clock = seconds remaining), skaters only. Shipped = the dataset's `on_ice_*` lists.", "",
             "| season | goals | credit slots | rebuilt: scorer side | rebuilt: conceding side | **rebuilt: all** | shipped: all | "
             "shooter in own set, rebuilt | shooter, shipped | gate |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    failed = False
    for season in SEASONS:
        d = load_season(season)
        reg = set(regular_season_game_ids(sched, season))
        skaters = set(d["box"]["player_id"].astype(int))
        pbp = d["pbp"]
        pbp = pbp[pbp["game_id"].isin(reg)]

        raw_goals = pbp[pbp["event"] == "goal"]
        goals = attach_onice(raw_goals, d["shifts"][d["shifts"]["game_id"].isin(reg)], skaters)
        goals = goals[goals["period_of_game"].isin(["1", "2", "3"])]
        rec = goal_credit_recovery(goals)

        shots = dedupe_goals(pbp)
        shots = shots[(shots["event"] == "shot") & shots["player_id"].notna()
                      & shots["period_of_game"].isin(["1", "2", "3"])]
        shots = attach_onice(shots, d["shifts"][d["shifts"]["game_id"].isin(reg)], skaters)
        sh = shooter_in_own_set(shots)

        ok = rec["new_all"] >= GATE
        failed |= not ok
        lines.append(
            f"| {season} | {rec['goals']} | {rec['slots']} | {rec['new_plus']:.1%} | {rec['new_minus']:.1%} | "
            f"**{rec['new_all']:.1%}** | {rec['old_all']:.1%} | {sh['new']:.1%} | {sh['old']:.1%} | "
            f"{'PASS' if ok else 'FAIL'} |"
        )
        print(lines[-1])

    lines += ["", f"Gate: rebuilt sets recover at least {GATE:.0%} of credited goal players in every season.",
              f"Result: {'FAIL' if failed else 'PASS'}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
