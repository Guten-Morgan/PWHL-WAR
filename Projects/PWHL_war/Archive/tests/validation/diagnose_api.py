"""
diagnose_api.py
---------------
Probe the PWHL HockeyTech API to find which endpoints actually work
and what data structure they return.  Run this BEFORE run_war.py.

  python diagnose_api.py
"""

import json
import time
import requests

BASE   = "https://lscluster.hockeytech.com/feed/index.php"
KEY    = "446521baf8c38984"
CLIENT = "pwhl"
GAME   = "210"          # first game of 2025-26 season
SEASON = "8"

session = requests.Session()


def get(params: dict, label: str) -> dict | None:
    p = {"key": KEY, "client_code": CLIENT, "fmt": "json", **params}
    try:
        r = session.get(BASE, params=p, timeout=15)
        r.raise_for_status()
        data = r.json()
        print(f"\n✓  {label}")
        print(f"   Top-level keys: {list(data.keys())[:8]}")
        sk = data.get("SiteKit", {})
        print(f"   SiteKit keys  : {list(sk.keys())[:10]}")
        return data
    except Exception as exc:
        print(f"\n✗  {label}  →  {exc}")
        return None


# -----------------------------------------------------------------------
# 1. Endpoints we know work
# -----------------------------------------------------------------------
print("=" * 60)
print("KNOWN-GOOD ENDPOINTS")
print("=" * 60)
get({"feed": "modulekit",   "view": "seasons"},                        "seasons")
get({"feed": "modulekit",   "view": "schedule", "season_id": SEASON},  "schedule")
get({"feed": "statviewfeed","view": "players",  "season": SEASON,
     "team": "all", "position": "skaters"},                            "player stats (skaters)")
get({"feed": "statviewfeed","view": "players",  "season": SEASON,
     "team": "all", "position": "goalies"},                            "player stats (goalies)")
get({"feed": "statviewfeed","view": "teams",    "season": SEASON},     "team stats")

time.sleep(0.5)

# -----------------------------------------------------------------------
# 2. PbP / game-level candidates
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("GAME-LEVEL ENDPOINT CANDIDATES")
print("=" * 60)

candidates = [
    # modulekit variations
    ({"feed": "modulekit",    "view": "gameSummary",        "game_id": GAME}, "modulekit gameSummary"),
    ({"feed": "modulekit",    "view": "gameCenter",         "game_id": GAME}, "modulekit gameCenter"),
    ({"feed": "modulekit",    "view": "gamecenter",         "game_id": GAME}, "modulekit gamecenter"),
    ({"feed": "modulekit",    "view": "recap",              "game_id": GAME}, "modulekit recap"),
    ({"feed": "modulekit",    "view": "boxscore",           "game_id": GAME}, "modulekit boxscore"),

    # statviewfeed variations
    ({"feed": "statviewfeed", "view": "gameSummary",        "game_id": GAME}, "statviewfeed gameSummary"),
    ({"feed": "statviewfeed", "view": "gamecentersummary",  "game_id": GAME}, "statviewfeed gamecentersummary"),
    ({"feed": "statviewfeed", "view": "gameCenterSummary",  "game_id": GAME}, "statviewfeed gameCenterSummary"),
    ({"feed": "statviewfeed", "view": "playbyplay",         "game_id": GAME}, "statviewfeed playbyplay"),
    ({"feed": "statviewfeed", "view": "playbyplayV2",       "game_id": GAME}, "statviewfeed playbyplayV2"),
    ({"feed": "statviewfeed", "view": "gameDetails",        "game_id": GAME}, "statviewfeed gameDetails"),
    ({"feed": "statviewfeed", "view": "scoring",            "game_id": GAME}, "statviewfeed scoring"),
]

working = {}
for params, label in candidates:
    time.sleep(0.3)
    data = get(params, label)
    if data and "error" not in data:
        working[label] = data

# -----------------------------------------------------------------------
# 3. Deep-dive the first working endpoint
# -----------------------------------------------------------------------
if working:
    print("\n" + "=" * 60)
    print(f"DEEP DIVE: first working endpoint")
    print("=" * 60)
    label, data = next(iter(working.items()))
    print(f"Endpoint: {label}")
    print(json.dumps(data, indent=2)[:4000])   # first 4000 chars
else:
    print("\n✗  No game-level endpoints worked.")
    print("   Falling back to player-stat-based WAR (no PbP needed).")

# -----------------------------------------------------------------------
# 4. Check player stats structure
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("PLAYER STATS SAMPLE (first player)")
print("=" * 60)
pdata = get({"feed": "statviewfeed", "view": "players",
             "season": SEASON, "team": "all", "position": "skaters"},
            "player stats detail")
if pdata:
    players = (pdata.get("SiteKit", {}).get("Players") or
               pdata.get("SiteKit", {}).get("skaters") or [])
    if players:
        print("Available fields:", list(players[0].keys()))
        print("\nSample player:")
        print(json.dumps(players[0], indent=2))
