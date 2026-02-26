"""
run_war.py
----------
PWHL xG-based Wins Above Replacement.

Data: hockey-statistics.com  (auto-downloaded on first run)
Method: individual xG (offense) + team-adjusted on-ice xGA (defense)

Usage
-----
  python run_war.py                            # all seasons combined
  python run_war.py --season 2024-25           # single season
  python run_war.py --season 2025-26           # current season
  python run_war.py --min-toi 20               # lower TOI threshold
  python run_war.py --refresh                  # re-download data
  python run_war.py --inspect                  # show column names and exit
  python run_war.py --output my_results.csv    # custom output path
"""

import argparse
import io
import logging
import sys
from pathlib import Path

# Windows terminals default to cp1252; force UTF-8 so accented player names print correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import requests
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).parent))

from pwhl_war.csv_loader  import PWHLCsvLoader
from pwhl_war.box_war     import XGWar
from pwhl_war.constants   import DEFAULT_MIN_TOI, DEFAULT_DEFENSE_WEIGHT, DEFAULT_BLOCK_WEIGHT
from pwhl_war.io_utils    import load_blocks

HEADSHOT_DIR = Path("pwhl_war/data/raw/headshots")
HEADSHOT_URL = "https://assets.leaguestat.com/pwhl/120x160/{player_id}.jpg"
HEADSHOT_SIZE = (60, 60)   # pixels — resize before plotting

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt= "%H:%M:%S",
)
log = logging.getLogger("run_war")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PWHL xG-WAR")
    p.add_argument("--season", choices=["2023-24", "2024-25", "2025-26"],
                   default=None,
                   help="Season to analyse (default: all seasons combined).")
    p.add_argument("--min-toi",         type=float, default=50.0,
                   help="Minimum TOI in minutes to qualify (default 50).")
    p.add_argument("--replacement-pct", type=float, default=25.0,
                   help="Percentile defining replacement level (default 25).")
    p.add_argument("--defense-weight",  type=float, default=0.36,
                   help="Weight on defensive component 0-1 (default 0.36).")
    p.add_argument("--block-weight",   type=float, default=0.04,
                   help="xG value per position+team-adjusted block/60 (default 0.04).")
    p.add_argument("--no-blocks",      action="store_true",
                   help="Disable the blocked-shots component.")
    p.add_argument("--refresh",  action="store_true",
                   help="Force re-download of data files.")
    p.add_argument("--inspect",  action="store_true",
                   help="Show column names from each table then exit.")
    p.add_argument("--output",   type=str, default="pwhl_war_results.csv",
                   help="Output CSV file path.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_leaderboard(war_df: pd.DataFrame, season_label: str, top_n: int = 25) -> None:
    name_col = "Name" if "Name" in war_df.columns else "PlayerID"
    fig, axes = plt.subplots(1, 3, figsize=(18, 9))
    fig.suptitle(f"PWHL xG-WAR — {season_label}", fontsize=14, fontweight="bold")

    for ax, (metric, colour, title) in zip(
        axes,
        [("WAR",  "steelblue",  "Total WAR"),
         ("oWAR", "darkorange", "Offensive WAR"),
         ("dWAR", "seagreen",   "Defensive WAR")],
    ):
        top = war_df.nlargest(top_n, metric).copy()
        sns.barplot(data=top, x=metric, y=name_col, color=colour, ax=ax)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(metric)
        ax.set_ylabel("")

    plt.tight_layout()
    plt.savefig("pwhl_war_leaderboard.png", dpi=150, bbox_inches="tight")
    log.info("Saved: pwhl_war_leaderboard.png")
    plt.close()


# ---------------------------------------------------------------------------
# Headshot helpers
# ---------------------------------------------------------------------------

def _fetch_headshot(player_id, session: requests.Session) -> Image.Image | None:
    """
    Download (and cache) a player headshot.  Returns a PIL Image or None.
    The image is cropped to a circle and resized to HEADSHOT_SIZE.
    """
    HEADSHOT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = HEADSHOT_DIR / f"{player_id}.png"

    if cache_path.exists():
        try:
            return Image.open(cache_path).convert("RGBA")
        except Exception:
            cache_path.unlink(missing_ok=True)

    url = HEADSHOT_URL.format(player_id=player_id)
    try:
        resp = session.get(url, timeout=8)
        if resp.status_code != 200 or not resp.content:
            return None
        img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception:
        return None

    # Resize and crop to circle
    img = img.resize(HEADSHOT_SIZE, Image.LANCZOS)
    img = _circle_crop(img)
    img.save(cache_path)
    return img


def _circle_crop(img: Image.Image) -> Image.Image:
    """Crop a PIL image to a circle with a thin white border."""
    size = img.size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((2, 2, size[0] - 2, size[1] - 2), fill=255)

    result = Image.new("RGBA", size, (0, 0, 0, 0))
    result.paste(img, mask=mask)

    # Thin white ring border
    border_mask = Image.new("L", size, 0)
    bdraw = ImageDraw.Draw(border_mask)
    bdraw.ellipse((0, 0, size[0], size[1]), fill=255)
    bdraw.ellipse((2, 2, size[0] - 2, size[1] - 2), fill=0)
    white_ring = Image.new("RGBA", size, (255, 255, 255, 200))
    result.paste(white_ring, mask=border_mask)
    return result


def _silhouette(war_value: float) -> Image.Image:
    """Plain coloured circle fallback when no headshot is available."""
    colour = (70, 130, 180) if war_value >= 0 else (180, 70, 70)
    img = Image.new("RGBA", HEADSHOT_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((2, 2, HEADSHOT_SIZE[0] - 2, HEADSHOT_SIZE[1] - 2),
                 fill=(*colour, 200))
    return img


# ---------------------------------------------------------------------------
# WAR vs TOI headshot scatter
# ---------------------------------------------------------------------------

def plot_war_vs_toi_headshots(
    war_df: pd.DataFrame,
    season_label: str,
    zoom: float = 0.55,
) -> None:
    """
    Scatter plot of WAR (Y) vs minutes played (X) using player headshots
    as the data points.  A horizontal dashed line marks WAR = 0.
    Players above the line contributed positive value; below is negative.
    """
    name_col = "Name" if "Name" in war_df.columns else "PlayerID"

    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Reference lines
    ax.axhline(0, color="#555555", linewidth=1.2, linestyle="--", zorder=1)

    # Light WAR grid lines
    war_ticks = np.arange(
        np.floor(war_df["WAR"].min()) - 0.5,
        np.ceil(war_df["WAR"].max())  + 0.5,
        0.5,
    )
    for w in war_ticks:
        ax.axhline(w, color="#2a2a2a", linewidth=0.5, linestyle="-", zorder=0)

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible; PWHL-WAR/1.0)"

    log.info("Fetching %d headshots...", len(war_df))
    for _, row in war_df.iterrows():
        pid  = int(row["PlayerID"]) if "PlayerID" in row.index else None
        x    = row["toi_min"]
        y    = row["WAR"]

        # Get headshot (or fallback silhouette)
        img = (_fetch_headshot(pid, session) if pid is not None else None
               ) or _silhouette(y)

        arr = np.asarray(img)
        oi  = OffsetImage(arr, zoom=zoom)
        oi.image.axes = ax
        ab  = AnnotationBbox(
            oi, (x, y),
            frameon      = False,
            pad          = 0,
            box_alignment= (0.5, 0.5),
            zorder       = 3,
        )
        ax.add_artist(ab)

    # Invisible scatter to set axis limits properly
    ax.scatter(war_df["toi_min"], war_df["WAR"], alpha=0, s=HEADSHOT_SIZE[0]**2 * zoom**2)

    # Axis styling
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.tick_params(colors="white", labelsize=10)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.set_xlabel("Time on Ice (minutes)", fontsize=12, labelpad=10)
    ax.set_ylabel("WAR  (Wins Above Replacement)", fontsize=12, labelpad=10)
    ax.set_title(
        f"PWHL xG-WAR  ·  {season_label}",
        fontsize=15, fontweight="bold", color="white", pad=14,
    )

    # Annotation: top 5 by WAR name labels
    for _, row in war_df.nlargest(5, "WAR").iterrows():
        ax.annotate(
            row[name_col],
            xy       = (row["toi_min"], row["WAR"]),
            xytext   = (6, 14),
            textcoords="offset points",
            fontsize = 7.5,
            color    = "white",
            fontweight="bold",
            zorder   = 5,
        )

    plt.tight_layout()
    plt.savefig("pwhl_war_vs_toi.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    log.info("Saved: pwhl_war_vs_toi.png")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    loader = PWHLCsvLoader()

    if args.refresh:
        loader.refresh()
        log.info("Cache cleared.")

    if args.inspect:
        for t in ("game_data", "schedule", "players"):
            try:
                loader.inspect(t)
            except Exception as exc:
                print(f"\n✗ {t}: {exc}")
        return

    # ------------------------------------------------------------------
    # Step 1: Load
    # ------------------------------------------------------------------
    log.info("=== Step 1: Load data (season=%s) ===", args.season or "all")

    try:
        game_data = loader.get_game_data(season=args.season)
        schedule  = loader.get_schedule(season=args.season)
    except Exception as exc:
        log.error("Data load failed: %s", exc)
        sys.exit(1)

    log.info("game_data rows: %d  |  schedule rows: %d", len(game_data), len(schedule))

    season_label = args.season or "All Seasons"

    # Sanity: confirm we have xG columns
    required = ["EV_ixG", "PP_ixG", "EV_xGA", "TOI"]
    missing  = [c for c in required if c not in game_data.columns]
    if missing:
        log.error("Missing expected columns: %s", missing)
        log.error("Run --inspect to see actual column names.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Compute WAR
    # ------------------------------------------------------------------
    log.info("=== Step 2: Compute xG-WAR ===")

    model = XGWar(
        min_toi_min    = args.min_toi,
        replacement_pct= args.replacement_pct,
        defense_weight = args.defense_weight,
        block_weight   = args.block_weight,
    )

    # Load blocked-shots data (raw observed counts — no extrapolation).
    blocks = None
    if not args.no_blocks:
        blocks = load_blocks(args.season)

    try:
        model.fit(game_data, schedule_df=schedule, blocks_df=blocks)
    except Exception as exc:
        log.error("WAR failed: %s", exc)
        log.error("Try --min-toi 10 or --inspect to debug.")
        raise

    war_df = model.get_war()
    model.summary(top_n=25)

    print(f"\nSeason summary ({season_label}):")
    print(f"  Qualified players : {len(war_df)}")
    print(f"  Total WAR in pool : {war_df['WAR'].sum():.2f}")
    print(f"  Max WAR           : {war_df['WAR'].max():.3f}  ({_top_name(war_df)})")
    print(f"  Goals per win        : {model.goals_per_win_:.3f}")
    print(f"  Off. replacement/60  : {model.o_replacement_val60_:+.4f} xG/60")
    print(f"  Def. replacement/60  : {model.d_replacement_val60_:+.4f} d_val/60")

    # ------------------------------------------------------------------
    # Step 3: Output
    # ------------------------------------------------------------------
    log.info("=== Step 3: Output ===")
    try:
        war_df.to_csv(args.output, index=False, encoding="utf-8")
        log.info("Results saved: %s", args.output)
    except PermissionError:
        alt = args.output.replace(".csv", "_new.csv")
        war_df.to_csv(alt, index=False, encoding="utf-8")
        log.warning("Permission denied on %s — saved to %s instead", args.output, alt)

    try:
        plot_leaderboard(war_df, season_label)
        plot_war_vs_toi_headshots(war_df, season_label)
    except Exception as exc:
        log.warning("Plotting skipped: %s", exc)

    log.info("Done.")


def _top_name(df: pd.DataFrame) -> str:
    col = next((c for c in ["name", "Name", "player_id", "PlayerID"] if c in df.columns), None)
    return str(df.iloc[0][col]) if (len(df) and col) else "N/A"


if __name__ == "__main__":
    main()
