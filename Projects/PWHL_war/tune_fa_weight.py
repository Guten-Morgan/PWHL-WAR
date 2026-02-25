"""
tune_fa_weight.py
-----------------
Sweep defense_weight for FA-based dWAR and find the value that maximises
team-level Spearman correlation with goal differential.

Outputs the optimal weight, then compares team-GD Spearman for:
  - xGA-dWAR (current model)
  - FA-dWAR  (tuned weight)
"""
import sys, logging
if hasattr(sys.stdout,"reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr,"reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression

sys.path.insert(0, ".")
from pwhl_war.xga_war import PWHLApiLoader, XG_MAP, _parse_toi, _pid_from_url

logging.basicConfig(level=logging.WARNING)

CACHE    = Path("pwhl_war/data/raw/pbp_cache")
LOADER   = PWHLApiLoader(cache_dir=CACHE)
MIN_TOI  = 50.0
REPL_PCT = 25.0

API          = "https://pwhl.hockey-statistics.com/api"
SEASON_YEARS = {"2023-24":"2023/2024","2024-25":"2024/2025","2025-26":"2025/2026"}
TEAM_MAP = {
    "Boston Fleet":"BOS","Minnesota Frost":"MIN","Montreal Victoire":"MTL",
    "Montréal Victoire":"MTL","New York Sirens":"NY","Ottawa Charge":"OTT",
    "Toronto Sceptres":"TOR","Seattle Torrent":"SEA","Vancouver Goldeneyes":"VAN",
}

# HockeyTech standings
HT_BASE    = "https://lscluster.hockeytech.com/feed/index.php"
HT_KEY     = "446521baf8c38984"
HT_CLI     = "pwhl"
SEASON_IDS = {"2023-24":"1","2024-25":"5","2025-26":"8"}


def get_season_games(season):
    r = requests.get(f"{API}/schedule", timeout=20,
                     headers={"User-Agent":"Mozilla/5.0"})
    games, sched = [], {}
    for g in r.json().get("games", []):
        if g.get("season_year") != SEASON_YEARS[season]: continue
        if not g.get("status","").startswith("Final"): continue
        gid = int(g["game_id"])
        games.append(gid)
        ht = g.get("home_team",""); at = g.get("away_team","")
        sched[gid] = {
            "home_abbr": TEAM_MAP.get(ht, ht[:3].upper()),
            "away_abbr": TEAM_MAP.get(at, at[:3].upper()),
            "home_id": str(g.get("home_team_id","")),
            "away_id": str(g.get("away_team_id","")),
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
        sm = sched_map.get(gid, {})
        home_abbr = sm.get("home_abbr","HOM")
        away_abbr = sm.get("away_abbr","VIS")

        tid_to_abbr = {}
        for e in pbp:
            if e.get("event") == "goal":
                t = e["details"].get("team",{})
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
            known = {home_tid, away_tid} - {""}
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

        team_def = {}; player_ixG = {}
        for e in pbp:
            if e.get("event") != "shot": continue
            d = e["details"]
            q = d.get("shotQuality","")
            xg = XG_MAP.get(q, 0.0)
            tid = str(d.get("shooterTeamId",""))
            pid = d.get("shooter",{}).get("id")
            if use_fa:
                if not q: continue
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
            skaters = summ.get(side,{}).get("skaters",[])
            toi_list = []
            for sk in skaters:
                pid = _pid_from_url(sk.get("playerImageURL",""))
                toi = _parse_toi(sk.get("stats",{}).get("toi",""))
                if pid and toi > 0:
                    toi_list.append((pid, toi, sk.get("name",str(pid)), sk.get("position","F")))
            team_toi = sum(t for _,t,_,_ in toi_list)
            if team_toi <= 0: continue
            n_team_games += 1
            for pid, toi, name, pos in toi_list:
                key = f"{pid}|{team_abbr}"
                if key not in players:
                    players[key] = {"player_id":pid,"name":name,"pos":pos,"team":team_abbr,
                                    "gp":0,"toi_min":0.0,"ixG":0.0,"def_metric":0.0}
                p = players[key]
                p["gp"] += 1; p["toi_min"] += toi
                p["ixG"] += player_ixG.get(pid, 0.0)
                p["def_metric"] += team_d * (toi / team_toi)
    return list(players.values()), total_goals, n_team_games


def build_player_df(records, gpw):
    """Return df with ixG60, d_adj (pre-weight), o_repl, gpw."""
    df = pd.DataFrame(records)
    df["toi_min"] = df["toi_min"].clip(lower=0)
    df["ixG60"]   = df["ixG"] / df["toi_min"].clip(lower=0.1) * 60
    df["def60"]   = df["def_metric"] / df["toi_min"].clip(lower=0.1) * 60
    qual = df["toi_min"] >= MIN_TOI
    reg = LinearRegression().fit(df[qual][["ixG60"]], df[qual]["def60"])
    df["def60_resid"] = df["def60"] - (reg.intercept_ + reg.coef_[0] * df["ixG60"])
    team_resid = (df[qual].groupby("team")["def60_resid"]
                  .apply(lambda g: np.average(g, weights=df.loc[g.index,"toi_min"].clip(0.1))))
    df["def60_resid"] -= df["team"].map(team_resid).fillna(0)
    league_resid = np.average(df.loc[qual,"def60_resid"],
                              weights=df.loc[qual,"toi_min"].clip(0.1))
    df["d_adj"] = df["def60_resid"] - league_resid
    # Offensive side (same for both models)
    o_repl = float(np.percentile(df[qual]["ixG60"], REPL_PCT))
    df["oGAR"] = (df["ixG60"] - o_repl) * df["toi_min"] / 60
    df["oWAR"]  = df["oGAR"] / gpw
    df["gpw"]   = gpw
    df["o_repl"] = o_repl
    return df


def apply_weight(df, weight):
    """Compute dWAR + WAR for a given defense_weight."""
    df = df.copy()
    qual = df["toi_min"] >= MIN_TOI
    df["d_value60"] = -df["d_adj"] * weight
    d_repl = float(np.percentile(df[qual]["d_value60"], REPL_PCT))
    df["dGAR"] = (df["d_value60"] - d_repl) * df["toi_min"] / 60
    df["dWAR"]  = df["dGAR"] / df["gpw"]
    df["WAR"]   = df["oWAR"] + df["dWAR"]
    return df


def fetch_standings(season_label, season_id):
    sess = requests.Session()
    r = sess.get(HT_BASE, params={
        "feed":"modulekit","view":"schedule","season_id":season_id,
        "key":HT_KEY,"client_code":HT_CLI,"fmt":"json",
    }, timeout=20)
    games = r.json().get("SiteKit",{}).get("Schedule",[])
    rows = []
    for g in games:
        status = (g.get("game_status") or "").strip().lower()
        if status != "final": continue
        try:
            hgf = int(g.get("home_goal_count",0) or 0)
            agf = int(g.get("visiting_goal_count",0) or 0)
        except (TypeError, ValueError): continue
        ha = (g.get("home_team_code") or "").strip()
        aa = (g.get("visiting_team_code") or "").strip()
        if not ha or not aa: continue
        rows.append({"team":ha,"GF":hgf,"GA":agf,"W":int(hgf>agf)})
        rows.append({"team":aa,"GF":agf,"GA":hgf,"W":int(agf>hgf)})
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    agg = df.groupby("team",as_index=False).agg(
        GP=("GF","count"),GF=("GF","sum"),GA=("GA","sum"),W=("W","sum"))
    agg["GD"] = agg["GF"] - agg["GA"]
    agg["Season"] = season_label
    return agg


# ── Run ──────────────────────────────────────────────────────────────────
seasons = ["2023-24","2024-25","2025-26"]
xga_player, fa_player = {}, {}

for s in seasons:
    print(f"Processing {s} ...", end=" ", flush=True)
    game_ids, sched = get_season_games(s)
    print(f"{len(game_ids)} games")
    rec_xga, tg, ng = aggregate_games(game_ids, sched, use_fa=False)
    rec_fa,  _,  _  = aggregate_games(game_ids, sched, use_fa=True)
    gpw = 2 * (tg / ng) if ng > 0 else 6.0
    xga_player[s] = build_player_df(rec_xga, gpw)
    fa_player[s]  = build_player_df(rec_fa,  gpw)

print("\nFetching standings...", end=" ", flush=True)
standings_frames = []
for s, sid in SEASON_IDS.items():
    df = fetch_standings(s, sid)
    if not df.empty: standings_frames.append(df)
standings = pd.concat(standings_frames, ignore_index=True)
print(f"done ({len(standings)} team-seasons)")

def team_spearman(player_dfs, weight, metric="GD"):
    """Aggregate player WAR to team level and correlate with GD."""
    rows = []
    for s, df in player_dfs.items():
        w = apply_weight(df, weight)
        t = w.groupby("team",as_index=False)["WAR"].sum().rename(columns={"WAR":"team_WAR"})
        t["Season"] = s
        rows.append(t)
    team_war = pd.concat(rows, ignore_index=True)
    merged = standings.merge(team_war, on=["Season","team"], how="inner")
    if len(merged) < 3: return np.nan, np.nan
    r, p = stats.spearmanr(merged["team_WAR"], merged[metric])
    return r, p

# ── Sweep defense_weight ─────────────────────────────────────────────────
weights = np.round(np.arange(0.005, 0.121, 0.005), 4)
print("\nSweeping defense_weight (FA-dWAR vs team GD):")
print(f"  {'weight':>8}  {'r_GD':>8}  {'p':>8}")

best_w, best_r = 0.03, -999
for w in weights:
    r, p = team_spearman(fa_player, w)
    sig = "*" if p < 0.05 else " "
    print(f"  {w:8.3f}  {r:8.3f}  {p:8.4f}  {sig}")
    if r > best_r:
        best_r, best_w = r, w

print(f"\nOptimal FA defense_weight: {best_w:.3f}  (r_GD={best_r:.3f})")

# ── Head-to-head: optimal FA vs current xGA (weight=0.36) ───────────────
r_xga, p_xga = team_spearman(xga_player, 0.36)
r_fa,  p_fa  = team_spearman(fa_player,  best_w)

print("\n" + "="*52)
print("TEAM-LEVEL GD VALIDATION (pooled, all seasons)")
print("="*52)
print(f"  xGA-dWAR (w=0.360)  r={r_xga:+.3f}  p={p_xga:.4f}")
print(f"  FA-dWAR  (w={best_w:.3f})  r={r_fa:+.3f}  p={p_fa:.4f}")
print(f"\nOptimal FA weight to use in xga_war.py: {best_w}")
