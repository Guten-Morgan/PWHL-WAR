"""
onice_rapm.py
-------------
Option 2: a ridge regression (RAPM) of five-on-five xG on the skaters on the
ice, shrunk toward the Option 1 on-ice xGA instead of toward zero.

Stints are stretches with a constant set of five skaters per side. Each
stint gives two rows (home attacking, away attacking). The design has an
offence and a defence column per skater:

    xG per 60 = c + sum(offence of attackers) + sum(defence of defenders)

Defence coefficient = prior + shrunk deviation, where the prior for a
skater is (his or her Option 1 on-ice xGA/60 - league mean) / 5, because five
defenders share each shot. With a huge penalty the fit returns Option 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .onice import PERIOD_SECONDS, _elapsed_seconds, _usable

DEFAULT_ALPHAS = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)


def build_stints(events: pd.DataFrame, shifts: pd.DataFrame, skater_ids=None) -> pd.DataFrame:
    """Five-on-five stints with the xG each side generated.

    events: shots with game_id, period_of_game, time_of_period, team_id,
    home_team_id, xg. A shot belongs to the stint with start < elapsed <= end,
    the same half-open convention as attach_onice and five_on_five_minutes.
    Seconds with other than five skaters per side are not in any stint.
    Returns [game_id, period, start, end, duration, home_skaters, away_skaters,
    xg_home, xg_away].
    """
    s = _usable(shifts, skater_ids)
    s = s[s["period"].isin([1, 2, 3])]

    ev = events.copy()
    ev["_per"] = pd.to_numeric(ev["period_of_game"], errors="coerce")
    ev["_e"] = _elapsed_seconds(ev["time_of_period"])
    ev = ev.dropna(subset=["_per", "_e"])
    ev_by = {k: g for k, g in ev.groupby(["game_id", "_per"])}

    rows = []
    for (gid, per), g in s.groupby(["game_id", "period"]):
        lo = (PERIOD_SECONDS - g["start_s"]).to_numpy(float)
        hi = (PERIOD_SECONDS - g["end_s"]).to_numpy(float)
        home = (g["home"] == 1).to_numpy()
        pid = g["player_id"].to_numpy(int)
        pts = np.unique(np.concatenate([lo, hi]))

        local: list[list] = []
        for a, b in zip(pts[:-1], pts[1:]):
            on = (lo <= a) & (hi >= b)
            if on[home].sum() == 5 and on[~home].sum() == 5:
                h = tuple(sorted(int(p) for p in pid[on & home]))
                aw = tuple(sorted(int(p) for p in pid[on & ~home]))
                if local and local[-1][1] == a and local[-1][2] == h and local[-1][3] == aw:
                    local[-1][1] = b                       # same skaters continue
                else:
                    local.append([a, b, h, aw])

        eg = ev_by.get((gid, per))
        for a, b, h, aw in local:
            xh = xa = 0.0
            if eg is not None:
                inside = eg[(eg["_e"] > a) & (eg["_e"] <= b)]
                by_home = inside["team_id"] == inside["home_team_id"]
                xh = float(inside.loc[by_home, "xg"].sum())
                xa = float(inside.loc[~by_home, "xg"].sum())
            rows.append({"game_id": gid, "period": per, "start": float(a), "end": float(b),
                         "duration": float(b - a), "home_skaters": h, "away_skaters": aw,
                         "xg_home": xh, "xg_away": xa})
    return pd.DataFrame(rows, columns=["game_id", "period", "start", "end", "duration",
                                       "home_skaters", "away_skaters", "xg_home", "xg_away"])


def _design(stints: pd.DataFrame, index: dict[int, int], n: int):
    """Sparse design [offence | defence], target (xG per 60), weight (minutes), game group."""
    m = len(stints)
    r, c = [], []
    for i, (h, a) in enumerate(zip(stints["home_skaters"], stints["away_skaters"])):
        for p in h:
            r += [i, m + i]; c += [index[p], n + index[p]]        # home: offence in row i, defence in row m+i
        for p in a:
            r += [m + i, i]; c += [index[p], n + index[p]]        # away: offence in row m+i, defence in row i
    X = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(2 * m, 2 * n))
    dur = stints["duration"].to_numpy(float)
    y = np.concatenate([stints["xg_home"].to_numpy(float), stints["xg_away"].to_numpy(float)]) * 3600.0 / np.tile(dur, 2)
    w = np.tile(dur / 60.0, 2)
    groups = np.tile(stints["game_id"].to_numpy(), 2)
    return X, y, w, groups


def _best_alpha(X, y, w, groups, alphas, n_splits=5, seed: int = 0) -> float:
    """Penalty with the lowest cross-validated weighted error; folds are whole games."""
    folds = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    best, best_err = alphas[0], np.inf
    for alpha in alphas:
        errs = []
        for tr, va in folds.split(X, y, groups):
            m = Ridge(alpha=alpha, solver="sparse_cg").fit(X[tr], y[tr], sample_weight=w[tr])
            errs.append(np.average((y[va] - m.predict(X[va])) ** 2, weights=w[va]))
        err = float(np.mean(errs))
        if err < best_err:
            best, best_err = alpha, err
    return best


def fit_rapm(stints: pd.DataFrame, prior: pd.DataFrame, alphas=DEFAULT_ALPHAS) -> pd.DataFrame:
    """Shrunk ridge xGA per player, as [player_id, xGA, toi_min].

    prior: the Option 1 table [player_id, xGA, toi_min]. A skater's rating is
    league mean on-ice xGA/60 + 5 x (prior coefficient + fitted deviation),
    converted back to a total with the prior's minutes. The penalty is chosen
    by cross-validation over whole games (skipped when one value is given).
    """
    pr = prior.set_index("player_id")
    league = pr["xGA"].sum() / pr["toi_min"].sum() * 60.0
    rate = pr["xGA"] / pr["toi_min"].clip(lower=1e-9) * 60.0

    players = sorted({p for t in stints["home_skaters"] for p in t}
                     | {p for t in stints["away_skaters"] for p in t} | set(pr.index))
    index = {p: i for i, p in enumerate(players)}
    n = len(players)
    p_coef = ((rate - league) / 5.0).reindex(players).fillna(0.0).to_numpy()

    X, y, w, groups = _design(stints, index, n)
    # subtract the prior defence of the five defenders from every row
    y_adj = y - X[:, n:] @ p_coef

    alphas = list(alphas)
    alpha = alphas[0] if len(alphas) == 1 else _best_alpha(X, y_adj, w, groups, alphas)
    fit = Ridge(alpha=alpha, solver="sparse_cg").fit(X, y_adj, sample_weight=w)
    delta = fit.coef_[n:]

    rating60 = league + 5.0 * (p_coef + delta)
    stint_min = pd.Series(0.0, index=players)
    for col in ("home_skaters", "away_skaters"):
        for t, d in zip(stints[col], stints["duration"]):
            for p in t:
                stint_min[p] += d / 60.0
    toi = pr["toi_min"].reindex(players)
    toi = toi.fillna(stint_min)
    out = pd.DataFrame({"player_id": players, "xGA": rating60 * toi.to_numpy() / 60.0, "toi_min": toi.to_numpy()})
    out.attrs["alpha"] = alpha
    return out
