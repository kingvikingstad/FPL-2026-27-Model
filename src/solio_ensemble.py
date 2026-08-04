"""
solio_ensemble.py — blend our projections with Solio Analytics' public feed
===========================================================================
Solio publishes a free, no-auth feed (refreshed every 4h) documented for AI
agents: https://fpl.solioanalytics.com/api/data/latest.md. It is a sharp,
market-driven model that is *methodologically independent* of ours (efficient
market odds vs our two-season component priors), which is exactly what makes it
a good ensemble partner: independent errors average down.

This module does four things:
  1. fetch   — GET the live feed (falls back to a local cache when offline)
  2. parse   — the documented markdown tables -> a consolidated player frame
  3. align   — join to our single-GW projection by normalised name
  4. blend + benchmark — accuracy-weighted ensemble, agreement stats, and a
     disagreement table that flags where we and a sharp model diverge (a genuine
     differential edge, or a bug in one of the two — worth a look either way).

IMPORTANT basis rule: Solio publishes SINGLE-gameweek projections. Align against
OUR single-GW projection (project(gw,gw)), never the GW1-6 cumulative board.
"""
from __future__ import annotations
import re, io, os, unicodedata
import numpy as np, pandas as pd

SOLIO_MD_URL = "https://fpl.solioanalytics.com/api/data/latest.md"
CACHE = "/home/claude/fpl/solio_cache.md"

# Solio short codes -> our schedule short names (extend as needed)
TEAM_MAP = {"MCI": "Man City", "MUN": "Man United", "MUN.": "Man United",
            "ARS": "Arsenal", "CHE": "Chelsea", "LIV": "Liverpool",
            "TOT": "Tottenham", "NEW": "Newcastle", "AVL": "Aston Villa",
            "BOU": "Bournemouth", "BRE": "Brentford", "BHA": "Brighton",
            "CRY": "Crystal Palace", "EVE": "Everton", "FUL": "Fulham",
            "NFO": "Nott'm Forest", "WHU": "West Ham", "WOL": "Wolves",
            "BUR": "Burnley", "LEE": "Leeds", "SUN": "Sunderland",
            "COV": "Coventry", "HUL": "Hull", "IPS": "Ipswich"}


# ---------------------------------------------------------------- fetch
def fetch_solio(url: str = SOLIO_MD_URL, cache: str = CACHE, timeout: int = 20,
                write_cache: bool = True) -> str:
    """Return the feed markdown. Tries the network; on any failure (e.g. the
    sandbox has no egress) falls back to the local cache."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "fpl-model/1.0"})
        text = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8")
        if write_cache:
            try: open(cache, "w").write(text)
            except Exception: pass
        return text
    except Exception as e:
        if os.path.exists(cache):
            print(f"[solio] network unavailable ({type(e).__name__}); using cache {cache}")
            return open(cache).read()
        raise


# ---------------------------------------------------------------- parse
def _num(x):
    if x is None: return np.nan
    s = str(x).replace("£", "").replace("m", "").replace("%", "").replace(",", "").strip()
    try: return float(s)
    except ValueError: return np.nan


def _tables(md: str) -> dict:
    """Split the markdown into {section_title: DataFrame} for every pipe table."""
    out = {}
    for block in re.split(r"\n##+ ", md):
        lines = block.strip().split("\n")
        title = lines[0].strip()
        rows = [ln for ln in lines if ln.strip().startswith("|")]
        if len(rows) < 2:
            continue
        header = [c.strip() for c in rows[0].strip("|").split("|")]
        data = []
        for ln in rows[2:]:                       # skip header + separator
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) == len(header):
                data.append(cells)
        if data:
            out[title] = pd.DataFrame(data, columns=header)
    return out


def parse_solio(md: str) -> dict:
    """Consolidated per-player frame + the raw section frames + metadata."""
    gw = None
    m = re.search(r"Gameweek (\d+)", md)
    if m: gw = int(m.group(1))
    tabs = _tables(md)

    def pick(title_contains, cols):
        for t, df in tabs.items():
            if title_contains.lower() in t.lower():
                d = df.copy()
                keep = {c: cols[c] for c in cols if c in d.columns}
                d = d[list(keep)].rename(columns=keep)
                return d
        return pd.DataFrame()

    top = pick("Top-projected", {"Player": "player", "Team": "team_code", "Pos": "pos",
                                 "Price": "price", "Proj. Points": "solio_proj", "Own%": "own"})
    goals = pick("projected goals", {"Player": "player", "prG": "prG"})
    assists = pick("projected assists", {"Player": "player", "prA": "prA"})
    bonus = pick("projected bonus", {"Player": "player", "Proj. P (Bonus)": "prBonus"})
    defcon = pick("projected DefCon", {"Player": "player", "DefCon %": "defcon_pct"})

    base = top.copy()
    for extra in (goals, assists, bonus, defcon):
        if len(extra):
            base = base.merge(extra, on="player", how="left")
    for c in ["price", "solio_proj", "own", "prG", "prA", "prBonus", "defcon_pct"]:
        if c in base.columns:
            base[c] = base[c].map(_num)
    if "team_code" in base.columns:
        base["team"] = base["team_code"].map(lambda t: TEAM_MAP.get(str(t).strip(), str(t).strip()))
    base["key"] = base["player"].map(norm_name)
    return {"gameweek": gw, "players": base, "sections": tabs,
            "cs": pick("clean sheet", {"Team": "team", "CS %": "cs_pct"}),
            "att_fix": pick("attacking fixtures", {"Team": "team",
                             "Proj. G For": "gF", "Proj. G Against": "gA"})}


# ---------------------------------------------------------------- align
def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().strip()
    s = re.sub(r"[.\-'`]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def align_single_gw(ours: pd.DataFrame, solio_players: pd.DataFrame,
                    our_proj_col: str = "mean", our_name_col: str = "player",
                    require_team: bool = True) -> pd.DataFrame:
    """Inner-join our SINGLE-GW projection to Solio by normalised name.
    `ours` must be a single-GW projection (project(gw,gw)), not the cumulative board.

    Matches on (name, team) by default to avoid name collisions (e.g. a Palmer GK
    at one club vs a Palmer MID at another). Set require_team=False only when the
    two feeds are known to be on different seasons and you accept collision risk.
    """
    o = ours.copy(); o["key"] = o[our_name_col].map(norm_name)
    o = o.rename(columns={our_proj_col: "our_proj"})
    keep = [c for c in ["key", "player", "pos", "team", "cost", "own", "our_proj"] if c in o.columns]
    o = o[keep]
    s = solio_players[["key", "team", "solio_proj", "prG", "prA", "defcon_pct", "own"]].rename(
        columns={"own": "solio_own", "team": "team_s"})
    if require_team and "team" in o.columns:
        a = o.merge(s, on="key", how="inner")
        a = a[a["team"] == a["team_s"]]                        # same club only
        dropped = o.merge(s, on="key", how="inner")
        n_collide = len(dropped) - len(a)
        if n_collide:
            print(f"[align] dropped {n_collide} name matches with mismatched club "
                  f"(collisions / transfers avoided)")
    else:
        a = o.merge(s, on="key", how="inner")
    return a.drop(columns=[c for c in ["team_s"] if c in a.columns]).dropna(
        subset=["our_proj", "solio_proj"])


# ---------------------------------------------------------------- blend + benchmark
def accuracy_weights(mae_ours: float, mae_solio: float) -> float:
    """Inverse-MAE weight on OUR model (needs realised GW results to estimate).
    Returns w_ours in [0,1]; equal weight when unknown."""
    if not (mae_ours and mae_solio):
        return 0.5
    inv_o, inv_s = 1.0 / mae_ours, 1.0 / mae_solio
    return inv_o / (inv_o + inv_s)


def blend(aligned: pd.DataFrame, w_ours: float = 0.5) -> pd.DataFrame:
    a = aligned.copy()
    a["ensemble"] = w_ours * a["our_proj"] + (1 - w_ours) * a["solio_proj"]
    a["disagreement"] = a["our_proj"] - a["solio_proj"]
    return a.sort_values("ensemble", ascending=False).reset_index(drop=True)


def benchmark(aligned: pd.DataFrame) -> dict:
    a = aligned.dropna(subset=["our_proj", "solio_proj"])
    return {"n_matched": len(a),
            "pearson": a["our_proj"].corr(a["solio_proj"]),
            "spearman": a["our_proj"].corr(a["solio_proj"], method="spearman"),
            "mae": (a["our_proj"] - a["solio_proj"]).abs().mean(),
            "bias_ours_minus_solio": (a["our_proj"] - a["solio_proj"]).mean()}


def disagreements(aligned: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    a = blend(aligned)
    cols = [c for c in ["player", "pos", "team", "cost", "our_proj", "solio_proj",
                        "disagreement", "ensemble"] if c in a.columns]
    hi = a.nlargest(n, "disagreement")[cols]      # we rate ABOVE Solio
    lo = a.nsmallest(n, "disagreement")[cols]     # we rate BELOW Solio
    return hi, lo


def leverage(df: pd.DataFrame, proj="solio_proj", own="own") -> pd.Series:
    """Solio's own differential metric: proj * (1 - ownership)."""
    return df[proj] * (1 - df[own].astype(float) / 100.0)
