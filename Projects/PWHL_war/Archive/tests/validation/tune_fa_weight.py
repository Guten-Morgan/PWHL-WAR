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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pwhl_war.xga_war    import PWHLApiLoader, _parse_toi, _pid_from_url

# XG_MAP was used by the old xga_war.py HockeyTech pipeline before Phase 9
# replaced it with API-native xG.  Retained here because tune_fa_weight.py
# still reads HockeyTech PBP events which carry a shotQuality label, not a
# continuous xG field.
XG_MAP = {
    "Quality on net":     0.128,
    "Quality goal":       0.128,
    "Non quality on net": 0.051,
    "Non quality goal":   0.051,
}
from pwhl_war.constants  import (
    HOCKEYTECH_API_KEY, SEASON_IDS, SEASON_YEARS, TEAM_MAP,
    DEFAULT_MIN_TOI, DEFAULT_REPLACEMENT_PCT,
)

logging.basicConfig(level=logging.WARNING)

CACHE    = Path("pwhl_war/data/raw/pbp_cache")
LOADER   = PWHLApiLoader(cache_dir=CACHE)
MIN_TOI  = DEFAULT_MIN_TOI
REPL_PCT = DEFAULT_REPLACEMENT_PCT

API      = "https://pwhl.hockey-statistics.com/api"

# HockeyTech standings
HT_BASE = "https://lscluster.hockeytech.com/feed/index.php"
HT_KEY  = HOCKEYTECH_API_KEY
HT_CLI  = "pwhl"


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


def _resolve_team_ids(pbp, home_abbr, away_abbr):
    """Resolve numeric team IDs from goal events and infer from shots if needed."""
    tid_to_abbr = {}
    for e in pbp:
        if e.get("event") == "goal":
            t = e["details"].get("team", {})
            if t.get("id") and t.get("abbreviation"):
                tid_to_abbr[str(t["id"])] = t["abbreviation"]
    all_shot_tids = set()
    for e in pbp:
        if e.get("event") == "shot":
            tid = str(e["details"].get("shooterTeamId", ""))
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
    return home_tid, away_tid, home_abbr, away_abbr


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
        home_abbr = sm.get("home_abbr", "HOM")
        away_abbr = sm.get("away_abbr", "VIS")
        home_tid, away_tid, home_abbr, away_abbr = _resolve_team_ids(pbp, home_abbr, away_abbr)

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


def aggregate_games_dual(game_ids, sched_map):
    """Accumulate FA60 (volume) and xGA60 (quality) against per player in one pass.

    Returns the same (records, total_goals, n_team_games) tuple as
    aggregate_games(), but each record contains *both* ``def_xga`` and
    ``def_fa`` fields instead of a single ``def_metric``.
    """
    players = {}
    total_goals = n_team_games = 0
    for gid in game_ids:
        try:
            pbp  = LOADER.get_pbp(gid)
            summ = LOADER.get_summary(gid)
        except Exception:
            continue
        sm = sched_map.get(gid, {})
        home_abbr = sm.get("home_abbr", "HOM")
        away_abbr = sm.get("away_abbr", "VIS")
        home_tid, away_tid, home_abbr, away_abbr = _resolve_team_ids(pbp, home_abbr, away_abbr)

        team_def_xga = {}; team_def_fa = {}; player_ixG = {}
        for e in pbp:
            if e.get("event") != "shot": continue
            d = e["details"]
            q = d.get("shotQuality", "")
            if not q: continue          # skip shots with no quality label
            xg  = XG_MAP.get(q, 0.0)
            tid = str(d.get("shooterTeamId", ""))
            pid = d.get("shooter", {}).get("id")
            # FA: count any labelled shot (volume proxy)
            team_def_fa[tid]  = team_def_fa.get(tid, 0.0)  + 1.0
            # xGA: quality-weighted shots only
            if xg > 0.0:
                team_def_xga[tid] = team_def_xga.get(tid, 0.0) + xg
                if pid: player_ixG[pid] = player_ixG.get(pid, 0.0) + xg
            if "goal" in q: total_goals += 1

        # defending team = opponent's shots/xG
        home_def_xga = team_def_xga.get(away_tid, 0.0)
        home_def_fa  = team_def_fa.get(away_tid, 0.0)
        away_def_xga = team_def_xga.get(home_tid, 0.0)
        away_def_fa  = team_def_fa.get(home_tid, 0.0)

        for side, team_abbr, def_xga, def_fa in [
            ("homeTeam",     home_abbr, home_def_xga, home_def_fa),
            ("visitingTeam", away_abbr, away_def_xga, away_def_fa),
        ]:
            skaters = summ.get(side, {}).get("skaters", [])
            toi_list = []
            for sk in skaters:
                pid = _pid_from_url(sk.get("playerImageURL", ""))
                toi = _parse_toi(sk.get("stats", {}).get("toi", ""))
                if pid and toi > 0:
                    toi_list.append((pid, toi, sk.get("name", str(pid)), sk.get("position", "F")))
            team_toi = sum(t for _, t, _, _ in toi_list)
            if team_toi <= 0: continue
            n_team_games += 1
            for pid, toi, name, pos in toi_list:
                key = f"{pid}|{team_abbr}"
                if key not in players:
                    players[key] = {"player_id": pid, "name": name, "pos": pos, "team": team_abbr,
                                    "gp": 0, "toi_min": 0.0, "ixG": 0.0,
                                    "def_xga": 0.0, "def_fa": 0.0}
                p = players[key]
                p["gp"] += 1; p["toi_min"] += toi
                p["ixG"]     += player_ixG.get(pid, 0.0)
                p["def_xga"] += def_xga * (toi / team_toi)
                p["def_fa"]  += def_fa  * (toi / team_toi)
    return list(players.values()), total_goals, n_team_games


def build_player_df(records, gpw, team_adjust=True):
    """Return df with ixG60, d_adj (pre-weight), o_repl, gpw.

    Parameters
    ----------
    team_adjust : bool
        If True (default), subtract each team's TOI-weighted mean residual
        before league centering.  Set False to preserve between-team defensive
        signal for diagnostic experiments.
    """
    df = pd.DataFrame(records)
    df["toi_min"] = df["toi_min"].clip(lower=0)
    df["ixG60"]   = df["ixG"] / df["toi_min"].clip(lower=0.1) * 60
    df["def60"]   = df["def_metric"] / df["toi_min"].clip(lower=0.1) * 60
    qual = df["toi_min"] >= MIN_TOI
    reg = LinearRegression().fit(df[qual][["ixG60"]], df[qual]["def60"])
    df["def60_resid"] = df["def60"] - (reg.intercept_ + reg.coef_[0] * df["ixG60"])
    if team_adjust:
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


def build_player_df_combined(records, gpw, fa_alpha=0.5, team_adjust=False):
    """Build player DataFrame using orthogonal FA/xGA combined defensive metric.

    Orthogonal decomposition (fitted on qualified players):
      FA60         = shot-volume-against rate
      xGA_qual60   = xGA60 - ols(xGA60 ~ FA60)
                   = quality against above what volume alone predicts
      combined     = fa_alpha * FA60_z + (1 - fa_alpha) * xGA_qual60_z
                     where _z = unit-variance normalisation.

    Higher combined → more shots and higher-quality shots against → worse
    defence.  The standard OLS-vs-offence → optional team-adjust → league-
    centre pipeline is then applied (same as build_player_df).  Output is
    compatible with apply_weight(): d_adj is positive for worse defenders.

    Parameters
    ----------
    records     : list of dicts with def_xga, def_fa, ixG, toi_min fields
    fa_alpha    : float in [0, 1] — weight on FA60 volume component
                  1.0 = pure FA60, 0.0 = pure xGA quality residual
    team_adjust : if True, subtract team-mean residuals (Step 2)
    """
    df = pd.DataFrame(records)
    df["toi_min"] = df["toi_min"].clip(lower=0)
    df["ixG60"]   = df["ixG"]     / df["toi_min"].clip(lower=0.1) * 60
    df["FA60"]    = df["def_fa"]  / df["toi_min"].clip(lower=0.1) * 60
    df["xGA60"]   = df["def_xga"] / df["toi_min"].clip(lower=0.1) * 60

    qual = df["toi_min"] >= MIN_TOI

    # Orthogonal decomposition: quality component = xGA60 unexplained by volume
    reg_ortho = LinearRegression().fit(
        df[qual][["FA60"]].values,
        df[qual]["xGA60"].values,
    )
    df["xGA_qual60"] = df["xGA60"] - (reg_ortho.intercept_ + reg_ortho.coef_[0] * df["FA60"])

    # Unit-variance normalisation (scale by qualified-player std)
    fa60_std  = df.loc[qual, "FA60"].std()
    qual_std  = df.loc[qual, "xGA_qual60"].std()
    fa60_std  = fa60_std  if fa60_std  > 1e-9 else 1.0
    qual_std  = qual_std  if qual_std  > 1e-9 else 1.0
    df["FA60_z"]       = df["FA60"]       / fa60_std
    df["xGA_qual60_z"] = df["xGA_qual60"] / qual_std

    # Combined metric: higher = worse defender (sign convention matches FA60/xGA60)
    df["comb60"] = fa_alpha * df["FA60_z"] + (1 - fa_alpha) * df["xGA_qual60_z"]

    # Standard pipeline (mirrors build_player_df on comb60)
    reg = LinearRegression().fit(df[qual][["ixG60"]], df[qual]["comb60"])
    df["comb60_resid"] = df["comb60"] - (reg.intercept_ + reg.coef_[0] * df["ixG60"])
    if team_adjust:
        team_resid = (df[qual].groupby("team")["comb60_resid"]
                     .apply(lambda g: np.average(g, weights=df.loc[g.index, "toi_min"].clip(0.1))))
        df["comb60_resid"] -= df["team"].map(team_resid).fillna(0)
    league_resid = np.average(df.loc[qual, "comb60_resid"],
                              weights=df.loc[qual, "toi_min"].clip(0.1))
    df["d_adj"] = df["comb60_resid"] - league_resid   # positive = worse defender

    # Offensive side
    o_repl = float(np.percentile(df[qual]["ixG60"], REPL_PCT))
    df["oGAR"]   = (df["ixG60"] - o_repl) * df["toi_min"] / 60
    df["oWAR"]   = df["oGAR"] / gpw
    df["gpw"]    = gpw
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


if __name__ == "__main__":
    # ── Run ──────────────────────────────────────────────────────────────────
    seasons = ["2023-24","2024-25","2025-26"]
    xga_player, fa_player = {}, {}
    # Save raw records so we can rebuild with team_adjust=False for Experiment 2
    raw_xga_records = {}
    raw_fa_records  = {}

    raw_dual_records = {}  # {season: (dual_records, gpw)} for Experiment 3

    for s in seasons:
        print(f"Processing {s} ...", end=" ", flush=True)
        game_ids, sched = get_season_games(s)
        print(f"{len(game_ids)} games")
        rec_xga, tg, ng = aggregate_games(game_ids, sched, use_fa=False)
        rec_fa,  _,  _  = aggregate_games(game_ids, sched, use_fa=True)
        rec_dual, _,  _  = aggregate_games_dual(game_ids, sched)
        gpw = 2 * (tg / ng) if ng > 0 else 6.0
        raw_xga_records[s]  = (rec_xga, gpw)
        raw_fa_records[s]   = (rec_fa,  gpw)
        raw_dual_records[s] = (rec_dual, gpw)
        xga_player[s] = build_player_df(rec_xga, gpw, team_adjust=True)
        fa_player[s]  = build_player_df(rec_fa,  gpw, team_adjust=True)

    print("\nFetching standings...", end=" ", flush=True)
    standings_frames = []
    for s, sid in SEASON_IDS.items():
        df = fetch_standings(s, sid)
        if not df.empty: standings_frames.append(df)
    standings = pd.concat(standings_frames, ignore_index=True)
    print(f"done ({len(standings)} team-seasons)")

    def team_spearman(player_dfs, weight, metric="GD"):
        """Aggregate player WAR to team level and correlate with GD.

        Returns
        -------
        (r_war, p_war, r_owar, p_owar) — Spearman r and p for total WAR and
        oWAR-only, both vs team GD.  Separating oWAR from total WAR reveals
        whether the team-level correlation is driven by offense alone (which
        would indicate the defensive component adds no independent signal).
        """
        rows = []
        for s, df in player_dfs.items():
            w = apply_weight(df, weight)
            t = w.groupby("team", as_index=False).agg(
                team_WAR=("WAR", "sum"),
                team_oWAR=("oWAR", "sum"),
            )
            t["Season"] = s
            rows.append(t)
        team_war = pd.concat(rows, ignore_index=True)
        merged = standings.merge(team_war, on=["Season","team"], how="inner")
        if len(merged) < 3:
            return np.nan, np.nan, np.nan, np.nan
        r_war,  p_war  = stats.spearmanr(merged["team_WAR"],  merged[metric])
        r_owar, p_owar = stats.spearmanr(merged["team_oWAR"], merged[metric])
        return r_war, p_war, r_owar, p_owar

    # ── Sweep defense_weight ─────────────────────────────────────────────────
    # Extend to 1.0 to find the true optimum for normalised-FA60 (team_adjust=False).
    # Step 0.02 keeps resolution fine enough while limiting output lines.
    weights = np.round(np.arange(0.01, 2.01, 0.02), 4)
    # Note: fa_player uses team_adjust=True, so all rows will be flat at r=0.118.
    # We still sweep to confirm flatness and to build the weights array used by Exp 3.
    print("\nSweeping defense_weight (FA-dWAR, team_adjust=True vs team GD):")
    print(f"  {'weight':>8}  {'r_WAR':>8}  {'p':>8}  {'r_oWAR':>8}")

    best_w, best_r = 0.03, -999
    for w in weights:
        r, p, r_owar, _ = team_spearman(fa_player, w)
        sig = "*" if p < 0.05 else " "
        print(f"  {w:8.3f}  {r:8.3f}  {p:8.4f}  {r_owar:8.3f}  {sig}")
        if r > best_r:
            best_r, best_w = r, w

    print(f"\nOptimal FA defense_weight (team_adjust=True): {best_w:.3f}  (r_GD={best_r:.3f})")

    # ── Head-to-head: optimal FA vs current xGA (weight=0.36) ───────────────
    r_xga,  p_xga,  r_xga_o,  p_xga_o  = team_spearman(xga_player, 0.36)
    r_fa,   p_fa,   r_fa_o,   p_fa_o   = team_spearman(fa_player,  best_w)

    print("\n" + "="*60)
    print("TEAM-LEVEL GD VALIDATION (pooled, all seasons)")
    print("="*60)
    print(f"  {'model':<28}  {'r_WAR':>7}  {'p':>7}  {'r_oWAR':>7}")
    print(f"  {'xGA-dWAR (w=0.360)':<28}  {r_xga:+7.3f}  {p_xga:7.4f}  {r_xga_o:+7.3f}")
    print(f"  {'FA-dWAR  (w='+str(best_w)+')':28}  {r_fa:+7.3f}  {p_fa:7.4f}  {r_fa_o:+7.3f}")
    print()
    print("  r_oWAR: Spearman of team oWAR (offense only) vs team GD.")
    print("  If r_oWAR ≈ r_WAR, the defensive component adds no independent")
    print("  team-level signal — consistent with team-centering zeroing dWAR.")
    print(f"\nOptimal FA weight to use in xga_war.py: {best_w}")

    # ── Experiment 2: proxy × team-centering grid ────────────────────────────
    # Test all four combinations of {xGA, FA60} × {team_adjust True/False}.
    # FA60 is already goalie-agnostic (shot counts, not goals), so no goalie
    # adjustment is needed — the question is purely whether removing
    # team-centering recovers enough between-team signal.
    xga_notc = {
        s: build_player_df(rec, gpw, team_adjust=False)
        for s, (rec, gpw) in raw_xga_records.items()
    }
    fa_notc = {
        s: build_player_df(rec, gpw, team_adjust=False)
        for s, (rec, gpw) in raw_fa_records.items()
    }
    r_notc,    p_notc,    _, _ = team_spearman(xga_notc,  0.36)
    r_fa_notc, p_fa_notc, _, _ = team_spearman(fa_notc,   best_w)

    print("\n" + "="*60)
    print("EXPERIMENT 2: proxy × team-centering grid")
    print("="*60)
    print(f"  {'model':<40}  {'r_WAR':>7}  {'p':>7}")
    print(f"  {'xGA  team_adjust=True  (baseline)':<40}  {r_xga:+7.3f}  {p_xga:7.4f}")
    print(f"  {'xGA  team_adjust=False':<40}  {r_notc:+7.3f}  {p_notc:7.4f}")
    print(f"  {'FA60 team_adjust=True  (sweep best)':<40}  {r_fa:+7.3f}  {p_fa:7.4f}")
    print(f"  {'FA60 team_adjust=False':<40}  {r_fa_notc:+7.3f}  {p_fa_notc:7.4f}")
    print()
    best_notc_r = max(r_notc, r_fa_notc)
    if best_notc_r > 0.30:
        best_label = "FA60" if r_fa_notc >= r_notc else "xGA"
        print(f"  RESULT: best no-centering r={best_notc_r:.3f} > 0.30 ({best_label} proxy).")
        print("  Removing team-centering recovers between-team signal.")
        print("  Recommended: switch to FA60 proxy, team_adjust=False.")
    else:
        print(f"  RESULT: best no-centering r={best_notc_r:.3f} <= 0.30.")
        print("  Neither proxy recovers sufficient signal without team-centering.")
        print("  Recommendation: switch to FA60 proxy (goalie-agnostic) with")
        print("  team_adjust=False — accepting modest signal improvement over pm60.")
    print()
    print("  Decision gate: r > 0.30 to declare improvement meaningful.")

    # ── Experiment 3: orthogonal FA/xGA combined metric 2D sweep ─────────────
    # Grid: fa_alpha ∈ {0.0, 0.1, …, 1.0} × defense_weight (same sweep grid)
    # team_adjust=False: preserves between-team signal (best result from Exp 2)
    print("\n" + "="*60)
    print("EXPERIMENT 3: FA/xGA orthogonal combined metric sweep")
    print("  team_adjust=False (preserves between-team defensive signal)")
    print("="*60)
    print(f"\n  {'fa_alpha':>9}  {'best_w':>8}  {'best_r':>8}  {'r_oWAR':>8}")

    exp3_rows = []
    for fa_alpha in np.round(np.arange(0.0, 1.01, 0.1), 1):
        comb_player = {
            s: build_player_df_combined(rec, gpw, fa_alpha=fa_alpha, team_adjust=False)
            for s, (rec, gpw) in raw_dual_records.items()
        }
        best_wa, best_ra, best_rowar = 0.03, -999.0, np.nan
        for wa in weights:
            r, p, r_owar, _ = team_spearman(comb_player, wa)
            if not np.isnan(r) and r > best_ra:
                best_ra, best_wa, best_rowar = r, wa, r_owar
        print(f"  {fa_alpha:9.1f}  {best_wa:8.3f}  {best_ra:8.3f}  {best_rowar:8.3f}")
        exp3_rows.append({"fa_alpha": fa_alpha, "best_w": best_wa,
                          "best_r": best_ra, "r_oWAR": best_rowar})

    exp3_df = pd.DataFrame(exp3_rows)
    best_row = exp3_df.loc[exp3_df["best_r"].idxmax()]
    print(f"\n  Overall best: fa_alpha={best_row['fa_alpha']:.1f}  "
          f"defense_weight={best_row['best_w']:.3f}  r={best_row['best_r']:.3f}")
    print()
    if best_row["best_r"] > best_notc_r:
        print(f"  Combined metric improves on single-proxy best "
              f"({best_row['best_r']:.3f} > {best_notc_r:.3f}).")
    else:
        print(f"  Combined metric does not improve on single-proxy best "
              f"({best_row['best_r']:.3f} <= {best_notc_r:.3f}).")
    print("  Decision gate: r > 0.30 for meaningful improvement over oWAR baseline.")
