from __future__ import annotations
import config
"""
team_news.py — capture team news so it can be scored later (studies/omit_doubt.py INSTRUMENT)
=============================================================================================
`studies/omit_doubt.py` asks whether a player a preview OMITS and also FLAGS as a doubt should
be pulled down harder than an unexplained omission. It could not be answered from what the
repo stored on 28 Sep 2026, for three reasons this module removes:

  1. fpl.page's doubts line was parsed and thrown away. `archive()` keeps it, per capture,
     beside the raw article and every line-up graphic — the graphic is the only copy of the
     XI, and it is hosted on a third party that may purge it.
  2. No board ever logged a player's start prior BEFORE team news next to the prior after
     it. `ledger_rows()` records, for every player at a covered club: the pre-team-news
     Beta, the weighted source agreement, the resolved doubt flag, the graphic's band,
     FPL's chance of playing, and the start probability after consensus and after the
     injury ceiling. With the pre-team-news prior logged, every arm the study defines can
     be recomputed offline whichever one is live.
  3. The locked boards carry no team news, so nothing downstream could be joined back.

This is DATA CAPTURE. Nothing in `predicted_xi` or `bayes_model` reads it; a board built
with it is bit-identical on `mean` to one built without (the acceptance test for shipping
ahead of the study's result). Every capture failure is reported and swallowed: losing a
log line must never cost a deadline board.

STORES (committed, NOT regenerable — a capture not taken is evidence lost for good):
  config.TEAM_NEWS/gw{N}/{stamp}/   article.html, img_{club}.{ext}, doubts.csv, meta.json
  config.TEAM_NEWS_LEDGER           one row per (board run, player at a covered club)

Doubt names are resolved to `player_code` WITHIN CLUB with the same matcher as the XI
(`predicted_xi.resolve`), so the study's cell is defined by the names the board saw. An
unresolved doubt name is kept in the ledger's club-level count, because the study excludes
any club-gw with one — it would otherwise mislabel an omitted-and-flagged player as
unflagged.

Run:  python src/team_news.py --selftest
"""
import datetime as _dt
import glob
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

LEDGER_COLS = ["gw", "run_ts", "deadline", "env_fp", "player_code", "team", "start_a_pre",
               "start_b_pre", "p_pre", "n_start", "n_sources", "named", "doubt_flag",
               "club_doubts_unresolved", "band", "p_band", "chance_play", "p_cons",
               "p_ceil", "doubts_capture"]
# Env knobs that change the start prior itself. A ledger row built under any of them set is
# not the canonical board's prior; the fingerprint lets the study keep only default runs.
_FP_PREFIXES = ("INSEASON_", "PRED_XI", "FPL_MINUTES", "REGIME", "INJURY_IMPACT", "LIVE_FPL")
# Selects WHICH gameweek's news applies, not how the prior is built — and lock_board pins it
# on every lock, so fingerprinting it would mark every canonical row as non-default.
_FP_EXEMPT = {"PRED_XI_GW"}


def _now():
    return _dt.datetime.now(_dt.timezone.utc)


def _stamp(ts):
    return ts.strftime("%Y%m%dT%H%M%SZ")


def env_fingerprint(environ=None):
    """'' for a default run; otherwise a short hash plus the sorted non-default knobs."""
    env = os.environ if environ is None else environ
    kv = sorted(f"{k}={v}" for k, v in env.items()
                if k.startswith(_FP_PREFIXES) and k not in _FP_EXEMPT)
    if not kv:
        return ""
    return hashlib.sha1(";".join(kv).encode()).hexdigest()[:8] + ":" + ";".join(kv)


# ----------------------------------------------------------------------------- archive
def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_") or "club"


def archive(gw, html, blocks, images, as_of, source="fplpage", captured_at=None,
            root=None):
    """Write one capture: the raw article, each graphic, the doubts line, and metadata.

    `blocks` is `fplpage.parse_blocks(html)[1]`; `images` maps club -> raw bytes. Returns
    the capture directory. Never overwrites: the stamp is to the second, and a same-second
    collision gets a suffix.
    """
    ts = captured_at or _now()
    base = os.path.join(root or config.TEAM_NEWS, f"gw{int(gw)}", f"{_stamp(ts)}_{source}")
    d, k = base, 1
    while os.path.exists(d):
        d, k = f"{base}_{k}", k + 1
    os.makedirs(d)
    with open(os.path.join(d, "article.html"), "w", encoding="utf-8") as fh:
        fh.write(html or "")
    for club, raw in (images or {}).items():
        if raw:
            ext = "png" if raw[:4] == b"\x89PNG" else "jpg"
            with open(os.path.join(d, f"img_{_slug(club)}.{ext}"), "wb") as fh:
                fh.write(raw)
    rows = [{"team": b["club"], "raw_club": b.get("raw_club"), "doubt_name": n,
             "as_of": as_of, "captured_at": ts.isoformat()}
            for b in blocks for n in b.get("doubts", [])]
    pd.DataFrame(rows, columns=["team", "raw_club", "doubt_name", "as_of",
                                "captured_at"]).to_csv(os.path.join(d, "doubts.csv"),
                                                       index=False)
    meta = {"gw": int(gw), "source": source, "captured_at": ts.isoformat(), "as_of": as_of,
            "clubs": [b["club"] for b in blocks],
            "clubs_with_doubts_line": [b["club"] for b in blocks if b.get("doubts_line")],
            "doubts_line_unparsed": [b["club"] for b in blocks
                                     if b.get("doubts_line") and not b.get("doubts")]}
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return d


def latest_capture(gw, before=None, source="fplpage", root=None):
    """The newest capture directory for `gw` taken strictly before `before` (UTC)."""
    caps = sorted(glob.glob(os.path.join(root or config.TEAM_NEWS, f"gw{int(gw)}",
                                         f"*_{source}*")))
    if before is not None:
        cut = _stamp(pd.Timestamp(before).tz_convert("UTC")
                     if pd.Timestamp(before).tzinfo else pd.Timestamp(before, tz="UTC"))
        caps = [c for c in caps if os.path.basename(c)[:16] < cut]
    return caps[-1] if caps else None


def deadline(gw, season="2026-2027", base=None):
    """The gameweek's deadline (UTC) from gameweek_summaries, selected by id — the feed is
    not sorted, so nothing positional. None if it cannot be read."""
    try:
        g = pd.read_csv(os.path.join(base or config.repo(season), "gameweek_summaries.csv"))
        hit = g.loc[g["id"] == int(gw), "deadline_time"]
        return pd.Timestamp(hit.iloc[0]).tz_convert("UTC") if len(hit) == 1 else None
    except Exception:
        return None


# ----------------------------------------------------------------------------- ledger
def _p(frame):
    a, b = frame["start_a"].astype(float), frame["start_b"].astype(float)
    return (a / (a + b)).where((a + b) > 0)


def ledger_rows(gw, deadline, pre, cons_out, ceil_out, cons, squad, sig=None,
                xi_files=(), capture=None, run_ts=None, environ=None):
    """One row per player at a club covered by the consensus.

    pre       players BEFORE any team news (start_a, start_b, player_code, team)
    cons_out  after apply_consensus / apply_soft
    ceil_out  after apply_injury_ceiling
    cons      consensus() output (team, player_code, n_start, n_sources)
    xi_files  predicted-XI CSVs for this gw; the fpl.page one supplies band / p_band
    capture   an archive() directory whose doubts.csv supplies the doubt flag
    """
    import predicted_xi as pxi
    covered = set(cons["team"]) if len(cons) else set()
    base = pre[pre["team"].isin(covered)][["player_code", "team", "start_a", "start_b"]]
    base = base.dropna(subset=["player_code"]).drop_duplicates("player_code")
    if base.empty:
        return pd.DataFrame(columns=LEDGER_COLS)
    out = base.rename(columns={"start_a": "start_a_pre", "start_b": "start_b_pre"})
    out["p_pre"] = _p(base).to_numpy()
    for name, fr in (("p_cons", cons_out), ("p_ceil", ceil_out)):
        m = fr.dropna(subset=["player_code"]).drop_duplicates("player_code")
        out[name] = out["player_code"].map(dict(zip(m["player_code"], _p(m))))
    c = cons.drop_duplicates(["team", "player_code"])
    ns = {(r.team, r.player_code): r.n_start for r in c.itertuples()}
    tot = c.drop_duplicates("team").set_index("team")["n_sources"].to_dict()
    out["n_start"] = [float(ns.get((t, pc), 0.0)) for t, pc in zip(out["team"],
                                                                     out["player_code"])]
    out["n_sources"] = out["team"].map(tot).astype(float)
    out["named"] = out["n_start"] > 0

    # graphic band, from the fpl.page file resolved within club
    out["band"], out["p_band"] = None, np.nan
    for f in xi_files:
        if not os.path.basename(str(f)).endswith("_fplpage.csv"):
            continue
        r = pxi.resolve(pd.read_csv(f), squad, verbose=False)
        r = r[r["player_code"].notna()].drop_duplicates("player_code")
        if "band" in r:
            out["band"] = out["player_code"].map(dict(zip(r["player_code"], r["band"])))
            out["p_band"] = out["player_code"].map(dict(zip(r["player_code"], r["p_band"])))

    # doubt flag, resolved with the XI's own matcher
    out["doubt_flag"] = False
    out["club_doubts_unresolved"] = 0
    out["doubts_capture"] = ""
    if capture:
        dpath = os.path.join(capture, "doubts.csv")
        dd = pd.read_csv(dpath) if os.path.exists(dpath) else pd.DataFrame()
        out["doubts_capture"] = os.path.basename(capture)
        if len(dd):
            r = pxi.resolve(dd.rename(columns={"doubt_name": "player"}).assign(role="bench"),
                            squad, verbose=False)
            flagged = set(r.loc[r["player_code"].notna(), "player_code"])
            unres = r[r["player_code"].isna()].groupby("team").size().to_dict()
            out["doubt_flag"] = out["player_code"].isin(flagged)
            out["club_doubts_unresolved"] = out["team"].map(unres).fillna(0).astype(int)

    out["chance_play"] = np.nan
    if sig is not None and "chance_play" in getattr(sig, "columns", []):
        s = sig.dropna(subset=["player_code"]).drop_duplicates("player_code")
        out["chance_play"] = out["player_code"].map(dict(zip(s["player_code"],
                                                             s["chance_play"])))
    out["gw"] = int(gw)
    out["run_ts"] = (run_ts or _now()).isoformat()
    out["deadline"] = str(deadline) if deadline is not None else ""
    out["env_fp"] = env_fingerprint(environ)
    return out[LEDGER_COLS]


def append_ledger(rows, path=None):
    """Append, never rewrite. Header only when the file is new."""
    path = path or config.TEAM_NEWS_LEDGER
    if rows is None or not len(rows):
        return 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.exists(path)
    rows[LEDGER_COLS].to_csv(path, mode="a", header=new, index=False)
    return len(rows)


# ----------------------------------------------------------------------------- selftest
def selftest():
    import tempfile
    tmp = tempfile.mkdtemp()
    root = os.path.join(tmp, "team_news")
    t0 = pd.Timestamp("2026-10-09T18:00:00Z").to_pydatetime()
    blocks = [{"club": "Chelsea", "raw_club": "CHELSEA", "doubts": ["Joao Pedro", "Zzzqx"],
               "doubts_line": True},
              {"club": "Arsenal", "raw_club": "ARSENAL", "doubts": [], "doubts_line": True}]
    d1 = archive(6, "<html>x</html>", blocks, {"Chelsea": b"\xff\xd8jpg"}, "2026-10-09",
                 captured_at=t0, root=root)
    d2 = archive(6, "<html>y</html>", blocks, {}, "2026-10-09", captured_at=t0, root=root)
    assert d1 != d2, "same-second captures must not overwrite"
    assert os.path.exists(os.path.join(d1, "img_Chelsea.jpg"))
    meta = json.load(open(os.path.join(d1, "meta.json"), encoding="utf-8"))
    assert meta["doubts_line_unparsed"] == ["Arsenal"], meta
    assert latest_capture(6, before="2026-10-10T10:00:00Z", root=root) is not None
    assert latest_capture(6, before="2026-10-09T17:00:00Z", root=root) is None

    squad = pd.DataFrame({"web_name": ["João Pedro", "Palmer", "Saka", "Rice"],
                          "team": ["Chelsea", "Chelsea", "Arsenal", "Arsenal"],
                          "player_code": [1, 2, 3, 4],
                          "first_name": ["João Pedro", "Cole", "Bukayo", "Declan"],
                          "second_name": ["Junqueira de Jesus", "Palmer", "Saka", "Rice"]})
    pre = squad.assign(start_a=[6.3, 9.0, 9.0, 9.0], start_b=[3.7, 1.0, 1.0, 1.0])
    cons = pd.DataFrame({"team": ["Chelsea", "Arsenal", "Arsenal"],
                         "player_code": [2, 3, 4], "n_start": [1.0, 1.0, 1.0],
                         "n_sources": [1.0, 1.0, 1.0]})
    cons_out = pre.copy()
    cons_out.loc[0, ["start_a", "start_b"]] = [5.15, 4.85]
    ceil_out = cons_out.copy()
    sig = pd.DataFrame({"player_code": [1, 2, 3, 4], "chance_play": [0.75, 1, 1, 1]})
    L = ledger_rows(6, "2026-10-10T10:00:00Z", pre, cons_out, ceil_out, cons, squad, sig,
                    capture=d1, environ={})
    assert list(L.columns) == LEDGER_COLS
    jp = L.set_index("player_code").loc[1]
    assert bool(jp["doubt_flag"]) and not bool(jp["named"]), jp
    assert abs(jp["p_pre"] - 0.63) < 1e-9 and abs(jp["p_cons"] - 0.515) < 1e-9
    assert jp["club_doubts_unresolved"] == 1          # "Zzzqx" did not resolve
    assert not L.set_index("player_code").loc[3, "doubt_flag"]
    assert (L["env_fp"] == "").all()
    assert env_fingerprint({"INSEASON_KAPPA": "8", "PATH": "x"}).endswith("INSEASON_KAPPA=8")
    assert env_fingerprint({"PRED_XI_GW": "6"}) == ""       # lock_board pins it every lock
    lp = os.path.join(root, "ledger.csv")
    assert append_ledger(L, lp) == 4 and append_ledger(L, lp) == 4
    back = pd.read_csv(lp)
    assert len(back) == 8 and list(back.columns) == LEDGER_COLS   # one header, appended
    print("team_news selftest OK")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(__doc__)
