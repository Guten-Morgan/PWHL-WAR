"""
compare_fa_xga.py
-----------------
Compare YtY stability of xGA-dWAR vs FA-dWAR (raw shot count against).
Uses the cached PBP files — no new API calls for game data.
"""
import sys, json, logging
if hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression

sys.path.insert(0, ".")
from pwhl_war.xga_war    import PWHLApiLoader, XG_MAP, _parse_toi, _pid_from_url
from pwhl_war.constants  import SEASON_YEARS, TEAM_MAP, DEFAULT_MIN_TOI, DEFAULT_REPLACEMENT_PCT

logging.basicConfig(level=logging.WARNING)

CACHE    = Path("pwhl_war/data/raw/pbp_cache")
LOADER   = PWHLApiLoader(cache_dir=CACHE)
MIN_TOI  = DEFAULT_MIN_TOI
REPL_PCT = DEFAULT_REPLACEMENT_PCT

API = "https://pwhl.hockey-statistics.com/api"


def get_season_games(season):
    r = requests.get(f"{API}/schedule", timeout=20,
                     headers={"User-Agent":"Mozilla/5.0"})
    games, sched = [], {}
    for g in r.json().get("games", []):
        if g.get("season_year") != SEASON_YEARS[season]:
            continue
        if not g.get("status","").startswith("Final"):
            continue
        gid = int(g["game_id"])
        games.append(gid)
        ht = g.get("home_team","")
        at = g.get("away_team","")
        sched[gid] = {
            "home_abbr": TEAM_MAP.get(ht, ht[:3].upper()),
            "away_abbr": TEAM_MAP.get(at, at[:3].upper()),
            "home_id":   str(g.get("home_team_id","")),
            "away_id":   str(g.get("away_team_id","")),
        }
    return games, sched


def aggregate_games(game_ids, sched_map, use_fa=False):
    players = {}
    total_goals = n_team_games = 0

    for gid in game_ids:
        try:
            pbp  = LOADER.get_pbp(gid)
            summ = LOADER.get_summary(gid)
        except Exception:
            continue

        sm        = sched_map.get(gid, {})
        home_abbr = sm.get("home_abbr", "HOM")
        away_abbr = sm.get("away_abbr", "VIS")

        tid_to_abbr = {}
        for e in pbp:
            if e.get("event") == "goal":
                t = e["details"].get("team", {})
                if t.get("id") and t.get("abbreviation"):
                    tid_to_abbr[str(t["id"])] = t["abbreviation"]

        all_shot_tids = set()
        for e in pbp:
            if e.get("event") == "shot":
                tid = str(e["details"].get("shooterTeamId",""))
                if tid: all_shot_tids.add(tid)

        home_tid = away_tid = ""
        for tid, abbr in tid_to_abbr.items():
            if abbr == home_abbr: home_tid = tid
            elif abbr == away_abbr: away_tid = tid
        if len(all_shot_tids) == 2 and (not home_tid or not away_tid):
            known   = {home_tid, away_tid} - {""}
            unknown = all_shot_tids - known
            if len(unknown) == 1:
                inf = unknown.pop()
                if not home_tid: home_tid = inf
                else: away_tid = inf
            elif len(unknown) == 2:
                st = sorted(unknown)
                home_tid, away_tid = st[0], st[1]
        if home_tid and home_tid in tid_to_abbr: home_abbr = tid_to_abbr[home_tid]
        if away_tid and away_tid in tid_to_abbr: away_abbr = tid_to_abbr[away_tid]

        team_def   = {}
        player_ixG = {}

        for e in pbp:
            if e.get("event") != "shot": continue
            d   = e["details"]
            q   = d.get("shotQuality","")
            xg  = XG_MAP.get(q, 0.0)
            tid = str(d.get("shooterTeamId",""))
            pid = d.get("shooter",{}).get("id")

            if use_fa:
                if not q: continue          # skip events with no quality label
                team_def[tid] = team_def.get(tid, 0.0) + 1.0
                if pid: player_ixG[pid] = player_ixG.get(pid, 0.0) + xg
            else:
                if xg == 0.0: continue
                team_def[tid] = team_def.get(tid, 0.0) + xg
                if pid: player_ixG[pid] = player_ixG.get(pid, 0.0) + xg

            if "goal" in q: total_goals += 1

        home_def = team_def.get(away_tid, 0.0)
        away_def = team_def.get(home_tid, 0.0)

        for side, team_abbr, team_d in [
            ("homeTeam",     home_abbr, home_def),
            ("visitingTeam", away_abbr, away_def),
        ]:
            skaters  = summ.get(side, {}).get("skaters", [])
            toi_list = []
            for sk in skaters:
                pid = _pid_from_url(sk.get("playerImageURL",""))
                toi = _parse_toi(sk.get("stats",{}).get("toi",""))
                if pid and toi > 0:
                    toi_list.append((pid, toi, sk.get("name", str(pid)),
                                     sk.get("position","F")))
            team_toi = sum(t for _,t,_,_ in toi_list)
            if team_toi <= 0: continue
            n_team_games += 1
            for pid, toi, name, pos in toi_list:
                key = f"{pid}|{team_abbr}"
                if key not in players:
                    players[key] = {"player_id":pid,"name":name,"pos":pos,
                                    "team":team_abbr,"gp":0,"toi_min":0.0,
                                    "ixG":0.0,"def_metric":0.0}
                p = players[key]
                p["gp"]         += 1
                p["toi_min"]    += toi
                p["ixG"]        += player_ixG.get(pid, 0.0)
                p["def_metric"] += team_d * (toi / team_toi)

    return list(players.values()), total_goals, n_team_games


def compute_dwar(records, gpw, label):
    df = pd.DataFrame(records)
    df["toi_min"] = df["toi_min"].clip(lower=0)
    df["ixG60"]   = df["ixG"]         / df["toi_min"].clip(lower=0.1) * 60
    df["def60"]   = df["def_metric"]  / df["toi_min"].clip(lower=0.1) * 60

    qual = df["toi_min"] >= MIN_TOI

    reg = LinearRegression().fit(df[qual][["ixG60"]], df[qual]["def60"])
    df["def60_resid"] = df["def60"] - (reg.intercept_ + reg.coef_[0] * df["ixG60"])

    team_resid = (
        df[qual].groupby("team")["def60_resid"]
        .apply(lambda g: np.average(g, weights=df.loc[g.index,"toi_min"].clip(0.1)))
    )
    df["def60_resid"] -= df["team"].map(team_resid).fillna(0)

    league_resid = np.average(df.loc[qual,"def60_resid"],
                              weights=df.loc[qual,"toi_min"].clip(0.1))
    df["d_adj"] = df["def60_resid"] - league_resid

    d_repl  = float(np.percentile(df[qual]["d_adj"], REPL_PCT))
    # negate: fewer shots/xGA against = positive defensive contribution
    df["dGAR"] = (-df["d_adj"] - (-d_repl)) * df["toi_min"] / 60
    df["dWAR"]  = (df["dGAR"] / gpw).round(4)

    col = f"dWAR_{label}"
    return df[["player_id","name","team","toi_min","ixG60","dWAR"]].rename(
        columns={"dWAR": col})


if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # Run
    # ---------------------------------------------------------------------------
    seasons = ["2023-24","2024-25","2025-26"]
    xga_frames, fa_frames = [], []

    for s in seasons:
        print(f"Processing {s} ...", end=" ", flush=True)
        game_ids, sched = get_season_games(s)
        print(f"{len(game_ids)} games")

        rec_xga, tg, ng = aggregate_games(game_ids, sched, use_fa=False)
        rec_fa,  _,  _  = aggregate_games(game_ids, sched, use_fa=True)
        gpw = 2 * (tg / ng) if ng > 0 else 6.0

        df_xga = compute_dwar(rec_xga, gpw, "xGA"); df_xga["Season"] = s
        df_fa  = compute_dwar(rec_fa,  gpw, "FA");  df_fa["Season"]  = s
        xga_frames.append(df_xga)
        fa_frames.append(df_fa)

    xga_all = pd.concat(xga_frames, ignore_index=True)
    fa_all  = pd.concat(fa_frames,  ignore_index=True)

    # ---------------------------------------------------------------------------
    # YtY stability
    # ---------------------------------------------------------------------------
    pairs = [("2023-24","2024-25"), ("2024-25","2025-26")]

    print("\n" + "="*62)
    print("YtY STABILITY: xGA-dWAR vs FA-dWAR")
    print("(Spearman r of dWAR in season N vs season N+1)")
    print("="*62)

    for s1, s2 in pairs:
        for tag, all_df, col in [("xGA-dWAR", xga_all, "dWAR_xGA"),
                                  ("FA-dWAR",  fa_all,  "dWAR_FA")]:
            y1 = all_df[all_df["Season"]==s1][["player_id","team",col]].rename(columns={col:"d1"})
            y2 = all_df[all_df["Season"]==s2][["player_id","team",col]].rename(columns={col:"d2"})
            m  = y1.merge(y2, on=["player_id","team"])
            if len(m) < 5:
                m = y1.merge(y2, on="player_id")
            r, p = stats.spearmanr(m["d1"], m["d2"])
            sig = "*" if p < 0.05 else " "
            print(f"  {s1}->{s2}  {tag:<12}  n={len(m):>3}  r={r:+.3f}  p={p:.4f}  {sig}")
        print()
