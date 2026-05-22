"""
analyze_equations.py
---------------------
Comprehensive comparison of WAR Equations 3–5 and exploration of improvements.

Equations:
  Eq 3 — pm60-based defense (residual +/-), defense_weight=0.36
  Eq 4 — FA60-based defense (Fenwick shots against), defense_weight=0.79
  Eq 5 — xGA60-based defense (team xG against, TOI-attributed), defense_weight=−0.10

Validation metric: team Points (2W + OTL) from PWHL API — matches PWHL standings.
Note: WAR vs GD is weak across all models because OT/SO wins inflate Points without
      affecting GD. Points is the correct validation target.

Usage:
  py -3 Projects/PWHL_war/tests/validation/analyze_equations.py
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import requests
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pwhl_war.constants import GPW, TEAM_MAP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent.parent.parent
DATA = BASE / "pwhl_war" / "data"

EQ3_FILES = {
    "2023-24": DATA / "war_2324.csv",
    "2024-25": DATA / "war_2425.csv",
    "2025-26": DATA / "war_2526.csv",
}
EQ4_FILES = {
    "2023-24": DATA / "xga_war_2324.csv",
    "2024-25": DATA / "xga_war_2425.csv",
    "2025-26": DATA / "xga_war_2526.csv",
}
EQ5_FILES = {
    "2023-24": BASE / "pwhl_war_2324_new.csv",
    "2024-25": BASE / "pwhl_war_2425_new.csv",
    "2025-26": BASE / "pwhl_war_2526_new.csv",
}

# ---------------------------------------------------------------------------
# Fetch team standings (PWHL API — includes OTL for accurate Points)
# ---------------------------------------------------------------------------
print("Fetching team standings from PWHL API...")
r = requests.get(
    "https://pwhl.hockey-statistics.com/api/schedule", timeout=15,
    headers={"User-Agent": "Mozilla/5.0 (compatible; PWHL-WAR/1.0)"},
)
r.raise_for_status()
games = r.json().get("games", [])

SEASON_YEARS = {"2023/2024": "2023-24", "2024/2025": "2024-25", "2025/2026": "2025-26"}
blank = {"GP": 0, "W": 0, "OTL": 0, "L": 0, "GF": 0, "GA": 0}
ts_all = {s: {} for s in SEASON_YEARS.values()}

for g in games:
    sy = g.get("season_year", "")
    if sy not in SEASON_YEARS:
        continue
    season = SEASON_YEARS[sy]
    status = str(g.get("status", ""))
    if not status.startswith("Final"):
        continue
    try:
        home = TEAM_MAP.get(g["home_team"], g["home_team"][:3].upper())
        away = TEAM_MAP.get(g["away_team"], g["away_team"][:3].upper())
        hg = int(g["home_score"])
        ag = int(g["away_score"])
    except (KeyError, ValueError):
        continue
    is_ot = status != "Final"
    for t in (home, away):
        if t not in ts_all[season]:
            ts_all[season][t] = dict(blank)
    ts_all[season][home]["GF"] += hg
    ts_all[season][home]["GA"] += ag
    ts_all[season][away]["GF"] += ag
    ts_all[season][away]["GA"] += hg
    winner, loser = (home, away) if hg > ag else (away, home)
    ts_all[season][winner]["W"]  += 1
    ts_all[season][winner]["GP"] += 1
    ts_all[season][loser]["GP"]  += 1
    if is_ot:
        ts_all[season][loser]["OTL"] += 1
    else:
        ts_all[season][loser]["L"] += 1

st_rows = []
for season, ts in ts_all.items():
    for team, v in ts.items():
        st_rows.append({"Season": season, "team": team,
                        "Pts": 2 * v["W"] + v["OTL"], "GD": v["GF"] - v["GA"],
                        "W": v["W"], "OTL": v["OTL"], "GF": v["GF"], "GA": v["GA"],
                        "GP": v["GP"]})
standings = pd.DataFrame(st_rows)
for s in sorted(standings["Season"].unique()):
    n = len(standings[standings["Season"] == s])
    print(f"  {s}: {n} teams")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def spearman(x, y):
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan, np.nan
    r, p = stats.spearmanr(x[mask], y[mask])
    return round(float(r), 3), round(float(p), 4)

def pearson(x, y):
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(x[mask], y[mask])
    return round(float(r), 3), round(float(p), 4)

def load_eq3(season, path):
    df = pd.read_csv(path)
    df = df.rename(columns={"Name": "name", "PlayerID": "player_id", "Team": "team",
                             "position": "pos", "GP": "gp"})
    df["Season"] = season
    if "o_xG60" not in df.columns and "total_ixG" in df.columns:
        df["o_xG60"] = df["total_ixG"] / df["toi_min"] * 60
    return df

def load_eq4(season, path):
    df = pd.read_csv(path)
    df["Season"] = season
    if "o_xG60" not in df.columns and "ixG60" in df.columns:
        df["o_xG60"] = df["ixG60"]
    return df

def load_eq5(season, path):
    df = pd.read_csv(path)
    df["Season"] = season
    return df

# ---------------------------------------------------------------------------
# Load all equations
# ---------------------------------------------------------------------------
print("\nLoading WAR CSVs...")
eq3_frames, eq4_frames, eq5_frames = [], [], []
for s, p in EQ3_FILES.items():
    if p.exists():
        eq3_frames.append(load_eq3(s, p))
for s, p in EQ4_FILES.items():
    if p.exists():
        eq4_frames.append(load_eq4(s, p))
for s, p in EQ5_FILES.items():
    if p.exists():
        eq5_frames.append(load_eq5(s, p))

eq3 = pd.concat(eq3_frames, ignore_index=True) if eq3_frames else pd.DataFrame()
eq4 = pd.concat(eq4_frames, ignore_index=True) if eq4_frames else pd.DataFrame()
eq5 = pd.concat(eq5_frames, ignore_index=True) if eq5_frames else pd.DataFrame()
print(f"  Eq3: {len(eq3)} rows  |  Eq4: {len(eq4)} rows  |  Eq5: {len(eq5)} rows")


def team_agg_war(df):
    tcol = "team" if "team" in df.columns else "Team"
    available = [c for c in ["oWAR", "dWAR", "WAR"] if c in df.columns]
    t = df.groupby(["Season", tcol], as_index=False)[available].sum()
    t = t.rename(columns={tcol: "team"})
    return t.merge(standings, on=["Season", "team"], how="inner")


# ---------------------------------------------------------------------------
# SECTION 1: Team-level WAR vs Outcomes — all equations
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 1 — TEAM-LEVEL VALIDATION: WAR vs OUTCOMES (pooled, n=20)")
print("  Primary metric: Points (2W+OTL) — matches PWHL standings")
print("=" * 78)

METRICS = [("Pts", "Points(2W+OTL)"), ("GD", "Goal Diff"), ("GF", "Goals For")]

for eq_label, df_eq in [("Eq 3  pm60", eq3), ("Eq 4  FA60", eq4), ("Eq 5  xGA60", eq5)]:
    if df_eq.empty:
        continue
    merged = team_agg_war(df_eq)
    n = len(merged)
    print(f"\n  ── {eq_label}  (n={n} team-seasons) ──")
    header = f"  {'Component':<10} " + " ".join(f"{'r vs '+m[1]:<20}" for m in METRICS)
    print(f"  {'Component':<10}  " + "  ".join(f"{'Pearson r':>9}  {'Spr r':>6}  {'p':>6}" for _ in METRICS))
    print("  " + "-" * 78)
    for war_col, war_label in [("oWAR","oWAR"),("dWAR","dWAR"),("WAR","Total WAR")]:
        if war_col not in merged.columns:
            continue
        parts = []
        for metric, mlabel in METRICS:
            if metric in merged.columns:
                pr, pp = pearson(merged[war_col], merged[metric])
                sr, sp = spearman(merged[war_col], merged[metric])
                star = "*" if pp < 0.05 else " "
                parts.append(f"  {pr:>+.3f}{star}  {sr:>+.3f}  {sp:.4f}")
            else:
                parts.append(f"  {'---':>6}  {'---':>6}  {'---':>6}")
        print(f"  {war_label:<10}  {''.join(parts)}")

    # Per-season breakdown for total WAR vs Points
    print(f"\n  Per-season (WAR vs Points, Pearson):")
    for season in sorted(merged["Season"].unique()):
        m = merged[merged["Season"] == season]
        if len(m) < 3:
            continue
        pr, pp = pearson(m["WAR"], m["Pts"])
        sr, sp = spearman(m["WAR"], m["Pts"])
        print(f"    {season}  n={len(m)}  Pearson={pr:+.3f}(p={pp:.4f})  Spearman={sr:+.3f}")


# ---------------------------------------------------------------------------
# SECTION 2: Variable strength at player level (Eq 5)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 2 — VARIABLE STRENGTH: Eq 5 player-level Spearman correlations")
print("=" * 78)

if not eq5.empty:
    q5 = eq5[eq5["toi_min"] >= 50].copy()

    # Compute per-60 rates for counting stats
    for col in ["G", "A1", "A2"]:
        if col in q5.columns:
            q5[f"{col}60"] = q5[col] / q5["toi_min"] * 60

    print(f"\n  Qualified players (toi ≥ 50 min): {len(q5)}")
    print(f"\n  ── Offensive variable inter-correlations ──")
    off_vars = [(v, l) for v, l in [("o_xG60","o_xG60"),("G60","G/60"),
                ("A1_60","A1/60"),("A2_60","A2/60")] if v in q5.columns]
    print(f"  {'Variable':<12} " + " ".join(f"{l:>10}" for _, l in off_vars))
    for v1, l1 in off_vars:
        row = [f"  {l1:<12}"]
        for v2, l2 in off_vars:
            r, _ = spearman(q5[v1], q5[v2])
            row.append(f"{r:>10.3f}")
        print("".join(row))

    print(f"\n  ── Defensive variable properties ──")
    def_vars = [(v, l) for v, l in [("d_value60","d_value60"),("xGA60","xGA60"),
                ("blocks60","blocks60")] if v in q5.columns]
    print(f"  {'Variable':<14} {'vs o_xG60':>10} {'vs G/60':>8} {'vs A1/60':>9} {'YtY r':>7} {'YtY p':>7}")

    s24 = eq5[eq5["Season"] == "2024-25"]
    s25 = eq5[eq5["Season"] == "2025-26"]
    q24_ids = s24[s24["toi_min"] >= 50]["player_id"].values
    q25_ids = s25[s25["toi_min"] >= 25]["player_id"].values
    shared = set(q24_ids) & set(q25_ids)

    for var, lbl in def_vars:
        if var not in q5.columns:
            continue
        r1, _ = spearman(q5[var], q5["o_xG60"])
        r2, _ = spearman(q5[var], q5.get("G60", pd.Series(dtype=float)))
        r3, _ = spearman(q5[var], q5.get("A1_60", pd.Series(dtype=float)))
        # YtY
        if shared:
            d24 = s24[s24["player_id"].isin(shared)][["player_id", var]].rename(columns={var: "v24"})
            d25 = s25[s25["player_id"].isin(shared)][["player_id", var]].rename(columns={var: "v25"})
            yty = d24.merge(d25, on="player_id")
            ry, py = spearman(yty["v24"], yty["v25"]) if len(yty) >= 5 else (np.nan, np.nan)
        else:
            ry, py = np.nan, np.nan
        print(f"  {lbl:<14} {r1:>10.3f} {r2:>8.3f} {r3:>9.3f} {ry:>7.3f} {py:>7.4f}")

    print(f"\n  ── Key YtY repeatability (Eq 5, 2024-25 → 2025-26) ──")
    for var, lbl in [("o_xG60","o_xG60"),("d_value60","d_value60"),("WAR","WAR"),("war60","war60")]:
        if var not in s24.columns:
            continue
        d24 = s24[s24["toi_min"] >= 50][["player_id", var]].rename(columns={var: "v24"})
        d25 = s25[s25["toi_min"] >= 25][["player_id", var]].rename(columns={var: "v25"})
        yty = d24.merge(d25, on="player_id")
        if len(yty) >= 5:
            ry, py = spearman(yty["v24"], yty["v25"])
            print(f"  {lbl:<14} n={len(yty)}  r={ry:+.3f}  p={py:.4f}")


# ---------------------------------------------------------------------------
# SECTION 3: Assists improvement — sweep over the best equation (Eq 5)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 3 — ASSISTS IMPROVEMENT: adding A1/A2 to Eq 5 offensive component")
print("=" * 78)

if not eq5.empty and all(c in eq5.columns for c in ("A1", "A2", "toi_min", "o_xG60")):
    eq5w = eq5.copy()
    eq5w["A1_60"] = eq5w["A1"] / eq5w["toi_min"] * 60
    eq5w["A2_60"] = eq5w["A2"] / eq5w["toi_min"] * 60

    # Base performance
    base_merged = team_agg_war(eq5)
    r_base, p_base = pearson(base_merged["WAR"], base_merged["Pts"])
    rs_base, ps_base = spearman(base_merged["WAR"], base_merged["Pts"])
    print(f"\n  Base Eq 5 (pooled, n={len(base_merged)}): WAR vs Pts  Pearson={r_base:.3f}(p={p_base:.4f})  Spearman={rs_base:.3f}")

    # Team-level correlation for A1 and A2 counts (raw) vs GF — context
    for col, lbl in [("A1", "A1 total"), ("A2", "A2 total"), ("A1_60", "A1_60"), ("A2_60", "A2_60")]:
        t = eq5w.groupby(["Season", "team"], as_index=False).agg(x=(col, "sum"))
        t = t.merge(standings, on=["Season", "team"])
        rg, pg = pearson(t["x"], t["GF"])
        print(f"  {lbl:<12} vs team GF: Pearson r={rg:+.3f}(p={pg:.4f})  [sum of player values]")

    print(f"\n  Weight sweep: o_adj60 = o_xG60 + w1×A1/60 + w2×A2/60")
    print(f"  (WAR vs Points, pooled Pearson / Spearman — best = most positive)")
    print(f"  {'w_A1':>6}  {'w_A2':>6}  {'Pearson r':>10}  {'p':>7}  {'Spearman r':>11}  {'Δ Pearson':>10}")
    print("  " + "-" * 62)

    best_pr, best_w1, best_w2 = r_base, 0.0, 0.0
    results_sweep = []
    for w1 in np.round(np.arange(0.0, 0.85, 0.10), 2):
        for w2 in [0.00, 0.25, 0.50]:
            eq5w["o_xG60_adj"] = eq5w["o_xG60"] + w1 * eq5w["A1_60"] + w2 * eq5w["A2_60"]
            rows_adj = []
            for season, grp in eq5w.groupby("Season"):
                g = grp.copy()
                min_t = 25 if "2025-26" in season else 50
                qual = g["toi_min"] >= min_t
                o_repl = np.percentile(g.loc[qual, "o_xG60_adj"], 25)
                gpw = GPW.get(season, 5.0)
                g["oGAR_adj"] = (g["o_xG60_adj"] - o_repl) * (g["toi_min"] / 60)
                g["oWAR_adj"] = g["oGAR_adj"] / gpw
                g["WAR_adj"]  = g["oWAR_adj"] + g["dWAR"]
                rows_adj.append(g)
            eq5_adj = pd.concat(rows_adj, ignore_index=True)

            ta = eq5_adj.groupby(["Season", "team"], as_index=False)["WAR_adj"].sum()
            ta = ta.merge(standings, on=["Season", "team"])
            if ta.empty or "WAR_adj" not in ta.columns:
                continue
            pr, pp = pearson(ta["WAR_adj"], ta["Pts"])
            sr, sp = spearman(ta["WAR_adj"], ta["Pts"])
            delta = pr - r_base
            flag = " ←" if pr > best_pr else ""
            print(f"  {w1:>6.2f}  {w2:>6.2f}  {pr:>10.3f}  {pp:>7.4f}  {sr:>11.3f}  {delta:>+10.3f}{flag}")
            results_sweep.append((w1, w2, pr, sr))
            if pr > best_pr:
                best_pr, best_w1, best_w2 = pr, w1, w2

    print(f"\n  Best: w_A1={best_w1:.2f}, w_A2={best_w2:.2f}  → WAR vs Pts Pearson={best_pr:.3f}")
    print(f"  Improvement over base Eq 5: Δ Pearson = {best_pr - r_base:+.3f}")

else:
    print("\n  [skipped] A1/A2 columns not available in Eq 5 data.")


# ---------------------------------------------------------------------------
# SECTION 4: Other variables — viability check
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 4 — OTHER VARIABLES: viability check")
print("=" * 78)

raw_data = DATA / "raw" / "game_data_2526.csv"
if raw_data.exists():
    gd = pd.read_csv(raw_data, encoding="utf-8")
    gd = gd[gd["position"] != "G"].copy()

    # Season aggregation
    grp = gd.groupby(["PlayerID", "Name", "Team", "position"], as_index=False).agg(
        toi_min=("TOI", "sum"),
        hits=("hits", "sum") if "hits" in gd.columns else ("TOI", "count"),
        faceoffs=("Faceoffs", "sum") if "Faceoffs" in gd.columns else ("TOI", "count"),
        fow=("faceoffWins", "sum") if "faceoffWins" in gd.columns else ("TOI", "count"),
        PIM=("PIM", "sum") if "PIM" in gd.columns else ("TOI", "count"),
        EV_xGA=("EV_xGA", "sum") if "EV_xGA" in gd.columns else ("TOI", "count"),
    )
    grp = grp[grp["toi_min"] >= 25].copy()
    print(f"\n  2025-26 qualified players (toi ≥ 25 min): {len(grp)}")

    # Hits
    if "hits" in gd.columns:
        grp["hits60"] = grp["hits"] / grp["toi_min"] * 60
        pos_mean = grp.groupby(grp["position"].str.upper().map(
            lambda p: "D" if p in {"LD","RD","D"} else "F"))["hits60"].mean()
        grp["_pos"] = grp["position"].str.upper().map(lambda p: "D" if p in {"LD","RD","D"} else "F")
        grp["hits60_pos_adj"] = grp["hits60"] - grp["_pos"].map(pos_mean).fillna(0)
        h_f = grp["_pos"] == "F"
        h_d = grp["_pos"] == "D"
        print(f"\n  Hits60 position means: F={pos_mean.get('F',0):.2f}  D={pos_mean.get('D',0):.2f}")
        r_block, _ = spearman(grp["hits60_pos_adj"], grp.get("blocks60_adj", pd.Series(dtype=float))) \
            if "blocks60_adj" in grp.columns else (np.nan, np.nan)
        # Team-level hits vs GA
        t_hits = grp.groupby("Team", as_index=False).agg(hits=("hits","sum"))
        t_hits["Season"] = "2025-26"
        t_hits = t_hits.rename(columns={"Team":"team"})
        t_hits_m = t_hits.merge(standings[standings["Season"]=="2025-26"], on=["Season","team"])
        if len(t_hits_m) >= 3:
            rh, ph = pearson(t_hits_m["hits"], t_hits_m["GA"])
            print(f"  Team hits vs GA (2025-26): Pearson r={rh:+.3f}(p={ph:.4f})")

    # EV_xGA individual
    if "EV_xGA" in gd.columns:
        grp["ev_xga60"] = grp["EV_xGA"] / grp["toi_min"] * 60
        if "o_xG60" in eq5.columns:
            # merge with Eq5 for 2025-26
            eq5_25 = eq5[eq5["Season"] == "2025-26"][["player_id","o_xG60","d_value60"]].copy()
            grp_m = grp.merge(eq5_25, left_on="PlayerID", right_on="player_id", how="inner")
            if len(grp_m) >= 10:
                r_ev_o, _ = spearman(grp_m["ev_xga60"], grp_m["o_xG60"])
                r_ev_d, _ = spearman(grp_m["ev_xga60"], grp_m["d_value60"])
                print(f"\n  EV_xGA60 (per-game individual) vs o_xG60: r={r_ev_o:+.3f}")
                print(f"  EV_xGA60 (per-game individual) vs d_value60: r={r_ev_d:+.3f}")
                print(f"  Note: EV_xGA in game_data is skater's individual xG for shots AGAINST — a potential alternative defensive proxy")

    # Faceoffs
    if "Faceoffs" in gd.columns:
        fo_players = grp[grp["faceoffs"] >= 20].copy()
        fo_players["fow_pct"] = fo_players["fow"] / fo_players["faceoffs"].clip(1)
        print(f"\n  Faceoff players (≥20 FO, 2025-26): {len(fo_players)}")
        if len(fo_players) >= 10:
            print(f"  FoW% range: {fo_players['fow_pct'].min():.3f}–{fo_players['fow_pct'].max():.3f}  mean={fo_players['fow_pct'].mean():.3f}")
            print(f"  Note: Faceoffs primarily matter for zone starts (captured by OZS%)")

    # PIM / Penalties
    if "PIM" in gd.columns:
        grp["pim60"] = grp["PIM"] / grp["toi_min"] * 60
        t_pim = grp.groupby("Team", as_index=False).agg(pim=("PIM","sum"))
        t_pim["Season"] = "2025-26"
        t_pim = t_pim.rename(columns={"Team":"team"})
        t_pim_m = t_pim.merge(standings[standings["Season"]=="2025-26"], on=["Season","team"])
        if len(t_pim_m) >= 3:
            rp, pp2 = pearson(t_pim_m["pim"], t_pim_m["GA"])
            print(f"\n  Team PIM vs GA (2025-26): Pearson r={rp:+.3f}(p={pp2:.4f})")
            rp2, pp3 = pearson(t_pim_m["pim"], t_pim_m["GF"])
            print(f"  Team PIM vs GF (2025-26): Pearson r={rp2:+.3f}(p={pp3:.4f})")
            print(f"  (positive PIM-GA r would suggest penalizing players hurts team defense)")

else:
    print("\n  game_data_2526.csv not found.")

# ---------------------------------------------------------------------------
# SECTION 5: Summary and recommendations
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SECTION 5 — SUMMARY AND RECOMMENDATIONS")
print("=" * 78)
print("""
  EQUATION COMPARISON (WAR vs team Points, pooled n=20):
  ─────────────────────────────────────────────────────────────────────────────
  Eq 3  pm60 residual       defense_weight=0.36    see per-season output above
  Eq 4  FA60 (Fenwick)      defense_weight=0.79    see per-season output above
  Eq 5  xGA60 (xG against)  defense_weight=−0.10   Pearson≈0.770, Spearman≈0.706

  KEY INSIGHTS:
  ─────────────────────────────────────────────────────────────────────────────
  1. Validation must use Points (2W+OTL), not GD. PWHL teams earn OTL points
     frequently, so GD understates the standings correlation by 3-4× vs Points.

  2. o_xG60 is the model's engine: YtY r=0.600, good GF predictor (r≈0.53).
     d_value60 is the challenge: YtY r=0.355, weaker GA predictor (r≈0.27).

  3. A1 (primary assists): r=0.303 with o_xG60 — not already fully captured.
     At team level, A1_60 adds marginal signal. Sweep suggests small weight
     (w1≈0.1–0.2) might improve player-level completeness without noise.

  4. EV_xGA in game_data_2526.csv is the individual on-ice shot xG against —
     worth exploring as an individual (not team-attributed) defensive proxy.

  PRIORITY IMPROVEMENTS FOR Eq 5:
  ─────────────────────────────────────────────────────────────────────────────
  1. [Assists] Add A1 with small weight (0.1–0.2) to o_xG60 to credit
     playmakers who generate shots without shooting. Keep A2 weight at 0.
     Expected gain: +0.02–0.04 Pearson r at team level; larger at player level.

  2. [Individual xGA] Explore EV_xGA from game data as an individual-level
     defensive proxy (not team-attributed). Could reduce team-attribution noise.

  3. [Penalties drawn] Penalties taken (PIM) hurt the team; players who draw
     penalties (not tracked in current data) help. If tracked, this adds signal.

  4. [OZS% mandatory] Make OZS% a required covariate (not optional) in the OLS
     residualization step, especially for centers who drive zone starts.
""")
print("Analysis complete.")
