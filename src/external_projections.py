from __future__ import annotations
import config
"""
external_projections.py — third-party projection feeds, aligned to the board.
=============================================================================
The project already benchmarks against Solio (`solio_ensemble`), which publishes a
single gameweek. Fantasy Football Scout members get a SIX-gameweek grid, which is a
materially better comparator: it covers the whole early-season window the model is
tuned for, so agreement can be measured per gameweek rather than at one point.

This module only LOADS and ALIGNS. It does not blend, and it does not touch the board.
Whether an external feed should move a projection is a modelling decision with a
decision rule attached (see `solio_ensemble.benchmark_*`); silently averaging two
numbers because both exist is exactly the move this project avoids.

MATCHING IS THE RISK, AS ALWAYS
-------------------------------
FFS ships display names and its own club spellings ("Man Utd", "Spurs", "Joao Pedro"
without the diacritic). Matching is therefore:

  1. normalise club to frame spelling, and position to GK/DEF/MID/FWD
  2. fold accents and punctuation on the name
  3. match WITHIN CLUB, never across it — a bare surname is ambiguous league-wide but
     usually unique inside a squad, and a cross-club fallback would silently attach
     one Palmer's projection to the other (Chelsea and Ipswich both have one)
  4. anything still unmatched is REPORTED, not guessed

Run:  python src/external_projections.py --selftest
      python src/external_projections.py --check     (needs the real workbook)
"""
import os
import unicodedata
import numpy as np
import pandas as pd

GW_COLS = [f"GW{i}" for i in range(1, 7)]

# FFS club spellings -> the short names used across the model
TEAM_NORM = {
    "Man Utd": "Man United", "Man United": "Man United", "Spurs": "Tottenham",
    "Tottenham Hotspur": "Tottenham", "Nott'm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest", "Coventry City": "Coventry",
    "Hull City": "Hull", "Ipswich Town": "Ipswich", "Brighton & Hove Albion": "Brighton",
    "Wolves": "Wolves", "Newcastle United": "Newcastle", "West Ham United": "West Ham",
    "Leeds United": "Leeds", "Sunderland AFC": "Sunderland",
}
POS_NORM = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD",
            "GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}

_FOLD = {"ø": "o", "æ": "ae", "ß": "ss", "đ": "d", "ð": "d", "ł": "l",
         "þ": "th", "œ": "oe", "ı": "i"}


def _key(s):
    """Casefold, fold non-decomposing letters, strip accents, drop punctuation.
    'João Pedro' and 'Joao Pedro' must land on the same key, as must 'B.Fernandes'
    and 'B Fernandes'."""
    s = str(s).lower().replace(".", " ").replace("-", " ").replace("'", "")
    s = "".join(_FOLD.get(c, c) for c in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join("".join(c for c in s if c.isalnum() or c == " ").split())


def _last(k):
    toks = [t for t in k.split() if t not in ("jr", "sr", "ii", "iii")]
    return toks[-1] if toks else ""


def load_ffs(path=None) -> pd.DataFrame:
    """Read the FFS six-gameweek projection workbook.

    Returns [player_ffs, team, pos, gw1..gw6, ffs_total, price, ffs_value, _key, _last].
    Prices arrive as strings ('15.5m', '12m'); they are parsed to float millions so the
    workbook can be sanity-checked against FPL's own prices rather than trusted blind.
    """
    p = path or config.FFS_PROJECTION
    d = pd.read_excel(p)
    missing = [c for c in ["Player", "Team", "Pos"] + GW_COLS if c not in d.columns]
    if missing:
        raise RuntimeError(f"FFS workbook missing columns: {missing}")

    out = pd.DataFrame({
        "player_ffs": d["Player"].astype(str).str.strip(),
        "team": d["Team"].map(lambda t: TEAM_NORM.get(str(t).strip(), str(t).strip())),
        "pos": d["Pos"].map(lambda p_: POS_NORM.get(str(p_).strip(), str(p_).strip())),
    })
    for i, c in enumerate(GW_COLS, start=1):
        out[f"ffs_gw{i}"] = pd.to_numeric(d[c], errors="coerce")
    out["ffs_total"] = out[[f"ffs_gw{i}" for i in range(1, 7)]].sum(axis=1)
    if "Price" in d.columns:
        out["ffs_price"] = pd.to_numeric(
            d["Price"].astype(str).str.replace("m", "", regex=False).str.strip(),
            errors="coerce")
    out["_key"] = out["player_ffs"].map(_key)
    out["_last"] = out["_key"].map(_last)
    return out


def align_to_board(board_players: pd.DataFrame, ffs: pd.DataFrame = None,
                   verbose=True) -> pd.DataFrame:
    """Attach FFS gw1..gw6 to a board frame keyed on (player, team).

    `board_players` needs [player, team] and ideally [pos, player_code]. Matching runs
    WITHIN club only, in two passes: exact normalised name, then unique surname inside
    that club. Returns the board frame with ffs_* columns added (NaN where unmatched).
    """
    ffs = load_ffs() if ffs is None else ffs
    b = board_players.copy()
    b["_key"] = b["player"].map(_key)
    b["_last"] = b["_key"].map(_last)

    ffs_cols = [c for c in ffs.columns if c.startswith("ffs_")]
    taken = set()
    assign = {}

    # pass 1 — exact normalised name within club
    fidx = {}
    for i, r in ffs.iterrows():
        fidx.setdefault((r["_key"], r["team"]), []).append(i)
    for bi, r in b.iterrows():
        cands = [i for i in fidx.get((r["_key"], r["team"]), []) if i not in taken]
        if len(cands) == 1:
            assign[bi] = cands[0]; taken.add(cands[0])

    # pass 2 — unique surname within club (FFS shortens some names, FPL shortens others)
    lidx = {}
    for i, r in ffs.iterrows():
        lidx.setdefault((r["_last"], r["team"]), []).append(i)
    for bi, r in b.iterrows():
        if bi in assign:
            continue
        cands = [i for i in lidx.get((r["_last"], r["team"]), []) if i not in taken]
        if len(cands) == 1:
            assign[bi] = cands[0]; taken.add(cands[0])

    for c in ffs_cols:
        b[c] = np.nan
    for bi, fi in assign.items():
        for c in ffs_cols:
            b.at[bi, c] = ffs.at[fi, c]
    b["ffs_matched"] = b.index.isin(assign)

    if verbose:
        n = len(assign)
        print(f"[ffs] matched {n}/{len(b)} board players ({n/max(len(b),1):.1%}); "
              f"{len(ffs) - len(taken)} FFS rows unused")
        un = b[~b["ffs_matched"]]
        if len(un):
            top = un.assign(_o=pd.to_numeric(un.get("own", 0), errors="coerce")) \
                    .nlargest(8, "_o")[["player", "team", "pos"]]
            print("[ffs] highest-ownership board players with NO FFS row:")
            for _, r in top.iterrows():
                print(f"       {r['player']:18s} {r['team']:16s} {r.get('pos','')}")
    return b.drop(columns=["_key", "_last"])


def agreement(df: pd.DataFrame, ours="mean", theirs="ffs_gw1"):
    """Pearson / Spearman / MAE / signed bias on the rows where both exist."""
    d = df[[ours, theirs]].dropna()
    if len(d) < 3:
        return {"n": len(d)}
    return {"n": int(len(d)),
            "pearson": float(d[ours].corr(d[theirs])),
            "spearman": float(d[ours].corr(d[theirs], method="spearman")),
            "mae": float((d[ours] - d[theirs]).abs().mean()),
            "bias_ours_minus_theirs": float((d[ours] - d[theirs]).mean())}


def selftest():
    ffs = pd.DataFrame({
        "player_ffs": ["Joao Pedro", "B.Fernandes", "Palmer", "Palmer"],
        "team": ["Chelsea", "Man United", "Chelsea", "Ipswich"],
        "pos": ["FWD", "MID", "MID", "GK"],
        "ffs_gw1": [4.14, 5.97, 4.97, 2.10],
        "ffs_total": [26.47, 34.38, 32.56, 12.0],
    })
    ffs["_key"] = ffs["player_ffs"].map(_key)
    ffs["_last"] = ffs["_key"].map(_last)

    board = pd.DataFrame({
        "player": ["João Pedro", "B.Fernandes", "Palmer", "Palmer", "Nobody"],
        "team": ["Chelsea", "Man United", "Chelsea", "Ipswich", "Hull"],
        "pos": ["FWD", "MID", "MID", "GK", "DEF"],
        "own": [61.2, 48.1, 10.5, 5.7, 0.1],
    })
    out = align_to_board(board, ffs, verbose=False)

    assert abs(out.loc[0, "ffs_gw1"] - 4.14) < 1e-9, "accent folding failed"
    # the two Palmers must NOT cross clubs
    assert abs(out.loc[2, "ffs_gw1"] - 4.97) < 1e-9, "Chelsea Palmer mismatched"
    assert abs(out.loc[3, "ffs_gw1"] - 2.10) < 1e-9, "Ipswich Palmer mismatched"
    assert pd.isna(out.loc[4, "ffs_gw1"]), "unmatched player should stay NaN"
    assert out["ffs_matched"].sum() == 4

    a = agreement(pd.DataFrame({"mean": [1.0, 2.0, 3.0], "ffs_gw1": [1.1, 2.1, 2.9]}))
    assert a["n"] == 3 and a["pearson"] > 0.99
    print("SELFTEST OK: accents folded, same-name players kept inside their own clubs, "
          "unmatched left NaN, agreement computed.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    if "--check" in sys.argv:
        import warnings; warnings.filterwarnings("ignore")
        f = load_ffs()
        print(f"loaded {len(f)} FFS rows")
        print(f.head(10).to_string(index=False))
        print("\nclubs:", sorted(f["team"].unique()))
        print("positions:", sorted(f["pos"].unique()))
        sys.exit(0)
    print(__doc__)
