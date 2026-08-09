from __future__ import annotations
import config
"""
crosswalk.py — Understat player id -> FPL `player_code`.
========================================================
G5: downstream code joins on `player_code`. Never `player_id` (FPL reassigns it
between seasons), never name (ambiguous, and the ambiguity is concentrated exactly
among the squad-player transfers that matter for minutes).

The cascade is deliberately conservative:

  1. exact   — normalised name + club
  2. exact+pos — position breaks a stage-1 tie (two normalised-identical names at
                 the same club); it cannot create a match stage 1 missed, it can
                 only resolve one stage 1 refused
  3. fuzzy   — ratio >= 0.92 within club, 1:1 only, written with verified=False

**Never auto-resolve a collision.** Any many-to-one in either direction, and anything
below the fuzzy threshold, is left UNMATCHED and written to `crosswalk_review.csv`
with its candidates. Hand-verify that file once, set `verified=True`, and commit the
result — an unverified fuzzy match silently attributing one player's shots to another
is worse than a missing row, because a missing row is visible.

Fuzzy matching uses stdlib `difflib`; no new dependency.

Provenance: every row is [CHECK] until `verified` is True. `load_crosswalk(strict=True)`
refuses to hand back unverified fuzzy rows, so the modelling path cannot consume them
by accident.
"""
import argparse, os, sys, unicodedata, re
from difflib import SequenceMatcher
import numpy as np, pandas as pd

FUZZY_MIN = 0.92
OUT_COLS = ["player_code", "understat_player_id", "name_fpl", "name_understat",
            "team", "match_method", "confidence", "verified"]


# ------------------------------------------------------------- normalisation
def _norm(s):
    """Casefold, strip accents, drop punctuation, collapse whitespace.
    'Ødegaard' -> 'odegaard'; 'N'Golo Kanté' -> 'ngolo kante'."""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("ø", "o").replace("Ø", "o").replace("ß", "ss")
    s = re.sub(r"[^\w\s]", "", s.casefold())
    return re.sub(r"\s+", " ", s).strip()


def _aliases(row):
    """Every plausible spelling of one FPL player. Understat uses full names, FPL's
    `web_name` is usually a surname — so both must be candidates, or every player
    known by a single name fails stage 1."""
    out = set()
    for c in ("full_name", "web_name", "name"):
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            out.add(_norm(v))
    first, second = row.get("first_name"), row.get("second_name")
    if isinstance(first, str) and isinstance(second, str):
        out.add(_norm(f"{first} {second}"))
        out.add(_norm(second))
    return {a for a in out if a}


# ------------------------------------------------------------------ matching
def _pairs_to_frame(pairs, method):
    return pd.DataFrame(pairs, columns=["player_code", "understat_player_id",
                                        "name_fpl", "name_understat", "team",
                                        "confidence"]).assign(match_method=method)


def _one_to_one(df):
    """Split a candidate frame into unambiguous 1:1 matches and collisions.
    A code claimed by two Understat ids, or an id claimed by two codes, is a
    collision — both sides go to review untouched."""
    if df.empty:
        return df, df
    dup_code = df["player_code"].duplicated(keep=False)
    dup_uid = df["understat_player_id"].duplicated(keep=False)
    bad = dup_code | dup_uid
    return df[~bad].copy(), df[bad].copy()


def build_understat_crosswalk(understat_players: pd.DataFrame,
                              fpl_players: pd.DataFrame,
                              fuzzy_min: float = FUZZY_MIN,
                              review_path: str = None) -> pd.DataFrame:
    """Returns [player_code, understat_player_id, name_fpl, name_understat, team,
    match_method, confidence, verified].

    `understat_players`: [understat_player_id, player_name, team]
    `fpl_players`:       [player_code, team, pos] + at least one of
                         (full_name | web_name | first_name+second_name)

    Both `team` columns must already be in frame spelling — pass Understat names
    through `sd_ingest.normalise_team` first.

    Writes everything it refused to `review_path` (default config.CROSSWALK_REVIEW).
    """
    u = understat_players.copy()
    u["_name"] = u["player_name"].map(_norm)
    f = fpl_players.copy()
    f["_aliases"] = [_aliases(r) for _, r in f.iterrows()]

    matched, collisions = [], []
    remaining_u = set(u["understat_player_id"])
    taken_codes = set()

    # ---- stage 1: exact normalised name + club (positions break ties in stage 2)
    idx = {}
    for _, r in f.iterrows():
        for a in r["_aliases"]:
            idx.setdefault((a, r["team"]), []).append(r)

    stage1, ambiguous = [], []
    for _, r in u.iterrows():
        cands = idx.get((r["_name"], r["team"]), [])
        if len(cands) == 1:
            stage1.append((cands[0]["player_code"], r["understat_player_id"],
                           cands[0].get("web_name") or cands[0].get("full_name"),
                           r["player_name"], r["team"], 1.0))
        elif len(cands) > 1:
            ambiguous.append((r, cands))

    ok, bad = _one_to_one(_pairs_to_frame(stage1, "exact_name_team"))
    matched.append(ok); collisions.append(bad)
    remaining_u -= set(ok["understat_player_id"]); taken_codes |= set(ok["player_code"])

    # ---- stage 2: position breaks a stage-1 tie
    stage2 = []
    for r, cands in ambiguous:
        pos = r.get("position") or r.get("pos")
        narrowed = [c for c in cands if pos and c.get("pos") == pos
                    and c["player_code"] not in taken_codes]
        if len(narrowed) == 1:
            stage2.append((narrowed[0]["player_code"], r["understat_player_id"],
                           narrowed[0].get("web_name") or narrowed[0].get("full_name"),
                           r["player_name"], r["team"], 0.98))
        else:
            collisions.append(_pairs_to_frame(
                [(c["player_code"], r["understat_player_id"],
                  c.get("web_name") or c.get("full_name"), r["player_name"],
                  r["team"], np.nan) for c in cands], "collision_name_team"))

    ok2, bad2 = _one_to_one(_pairs_to_frame(stage2, "exact_name_team_pos"))
    matched.append(ok2); collisions.append(bad2)
    remaining_u -= set(ok2["understat_player_id"]); taken_codes |= set(ok2["player_code"])

    # ---- stage 3: fuzzy within club, flagged for review
    stage3, weak = [], []
    for _, r in u[u["understat_player_id"].isin(remaining_u)].iterrows():
        pool = f[(f["team"] == r["team"]) & (~f["player_code"].isin(taken_codes))]
        best, best_score = None, 0.0
        for _, c in pool.iterrows():
            score = max((SequenceMatcher(None, r["_name"], a).ratio()
                         for a in c["_aliases"]), default=0.0)
            if score > best_score:
                best, best_score = c, score
        row = (None if best is None else best["player_code"],
               r["understat_player_id"],
               None if best is None else (best.get("web_name") or best.get("full_name")),
               r["player_name"], r["team"], round(best_score, 4))
        (stage3 if best_score >= fuzzy_min else weak).append(row)

    ok3, bad3 = _one_to_one(_pairs_to_frame(stage3, "fuzzy"))
    matched.append(ok3); collisions.append(bad3)
    remaining_u -= set(ok3["understat_player_id"])

    # ---- assemble
    xw = pd.concat([m for m in matched if not m.empty], ignore_index=True) \
        if any(not m.empty for m in matched) else _pairs_to_frame([], "none")
    xw["verified"] = xw["match_method"] != "fuzzy"     # exact matches need no review
    xw = xw.reindex(columns=OUT_COLS)

    review = pd.concat(
        [c for c in collisions if not c.empty] +
        ([_pairs_to_frame(weak, "below_threshold")] if weak else []),
        ignore_index=True) if (any(not c.empty for c in collisions) or weak) \
        else _pairs_to_frame([], "none")
    if not review.empty:
        review = review.reindex(columns=OUT_COLS[:-1]).assign(verified=False)
    # fuzzy matches are ACCEPTED but unverified — they belong in review too
    if not ok3.empty:
        review = pd.concat([review, ok3.assign(verified=False)
                            .reindex(columns=OUT_COLS)], ignore_index=True)

    path = review_path or config.CROSSWALK_REVIEW
    review.to_csv(path, index=False)
    n_fuzzy = int((xw["match_method"] == "fuzzy").sum())
    print(f"[crosswalk] {len(xw)} matched ({len(xw) - n_fuzzy} exact, {n_fuzzy} fuzzy "
          f"UNVERIFIED), {len(u) - len(xw)} unmatched; {len(review)} rows -> {path}")
    return xw


# ------------------------------------------------------------------- reading
def save_crosswalk(xw, path=None):
    p = path or config.CROSSWALK_UNDERSTAT
    xw.to_csv(p, index=False)
    print(f"[crosswalk] wrote {len(xw)} rows -> {p}")
    return p


def load_crosswalk(path=None, strict=True) -> pd.DataFrame:
    """Read the committed crosswalk. `strict` drops unverified fuzzy rows, so the
    modelling path cannot consume a hand-review that never happened."""
    p = path or config.CROSSWALK_UNDERSTAT
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"no crosswalk at {p}. Build it with:\n"
            f"  python crosswalk.py --build\n"
            f"then hand-verify {config.CROSSWALK_REVIEW} and commit the result.")
    xw = pd.read_csv(p)
    xw["verified"] = xw["verified"].astype(str).str.lower().isin(("true", "1", "yes"))
    if strict:
        n = len(xw)
        xw = xw[xw["verified"]]
        if len(xw) < n:
            print(f"[crosswalk] dropped {n - len(xw)} unverified rows (strict). "
                  f"Hand-verify {config.CROSSWALK_REVIEW} to promote them.")
    for col in ("player_code", "understat_player_id"):
        dup = xw.duplicated(col).sum()
        if dup:
            raise RuntimeError(f"crosswalk is not 1:1 — {dup} duplicate {col} rows in {p}")
    return xw


def attach_player_code(df: pd.DataFrame, xw: pd.DataFrame = None,
                       id_col="understat_player_id") -> pd.DataFrame:
    """Map an Understat-keyed frame onto `player_code` (G5). Rows with no crosswalk
    entry are DROPPED and counted — never silently carried with a null key."""
    xw = load_crosswalk() if xw is None else xw
    m = xw.set_index(id_col)["player_code"].to_dict()
    out = df.copy()
    out["player_code"] = out[id_col].map(m)
    lost = int(out["player_code"].isna().sum())
    if lost:
        print(f"[crosswalk] dropped {lost}/{len(out)} rows with no player_code")
    return out[out["player_code"].notna()].copy()


# ------------------------------------------------------------------ selftest
def selftest():
    import tempfile
    fpl = pd.DataFrame([
        {"player_code": 101, "first_name": "Martin", "second_name": "Ødegaard",
         "web_name": "Ødegaard", "team": "Arsenal", "pos": "MID"},
        {"player_code": 102, "first_name": "Bukayo", "second_name": "Saka",
         "web_name": "Saka", "team": "Arsenal", "pos": "MID"},
        {"player_code": 103, "first_name": "Danny", "second_name": "Welbeck",
         "web_name": "Welbeck", "team": "Brighton", "pos": "FWD"},
        {"player_code": 106, "first_name": "Gabriel", "second_name": "Martinelli",
         "web_name": "Martinelli", "team": "Arsenal", "pos": "MID"},
        # a genuine same-name collision at one club, split only by position
        {"player_code": 104, "first_name": "Joe", "second_name": "Gomez",
         "web_name": "Gomez", "team": "Liverpool", "pos": "DEF"},
        {"player_code": 105, "first_name": "Joe", "second_name": "Gomez",
         "web_name": "Gomez", "team": "Liverpool", "pos": "FWD"},
    ])
    und = pd.DataFrame([
        {"understat_player_id": 1, "player_name": "Martin Odegaard",
         "team": "Arsenal", "position": "MID"},
        {"understat_player_id": 2, "player_name": "Bukayo Saka",
         "team": "Arsenal", "position": "MID"},
        {"understat_player_id": 3, "player_name": "Daniel Welbeck",
         "team": "Brighton", "position": "FWD"},
        {"understat_player_id": 4, "player_name": "Joe Gomez",
         "team": "Liverpool", "position": "DEF"},
        {"understat_player_id": 5, "player_name": "Totally Different Person",
         "team": "Brighton", "position": "MID"},
        # a scraper typo — close enough to accept, not close enough to trust
        {"understat_player_id": 6, "player_name": "Gabriel Martineli",
         "team": "Arsenal", "position": "MID"},
    ])

    tmp = tempfile.mkdtemp(prefix="crosswalk_selftest_")
    review = os.path.join(tmp, "review.csv")
    xw = build_understat_crosswalk(und, fpl, review_path=review)
    by_uid = xw.set_index("understat_player_id")

    # accents must not break an exact match
    assert by_uid.loc[1, "player_code"] == 101
    assert by_uid.loc[1, "match_method"] == "exact_name_team"
    assert by_uid.loc[2, "player_code"] == 102

    # a near-identical typo clears the bar, and is accepted UNVERIFIED
    assert by_uid.loc[6, "player_code"] == 106
    assert by_uid.loc[6, "match_method"] == "fuzzy"
    assert not bool(by_uid.loc[6, "verified"]), "fuzzy must never self-certify"

    # the same-name collision resolves on position, and only on position
    assert by_uid.loc[4, "player_code"] == 104

    # 'Daniel' vs 'Danny' Welbeck scores 0.81 — plausible to a human, below the bar.
    # It must stay UNMATCHED rather than be auto-resolved.
    assert 3 not in by_uid.index, "a sub-threshold name must not be auto-resolved"
    assert 5 not in by_uid.index, "a name with no counterpart must stay UNMATCHED"
    rev = pd.read_csv(review)
    for uid in (3, 5):
        assert (rev["understat_player_id"] == uid).any(), \
            f"unmatched id {uid} must reach review"

    # strict load hides the unverified fuzzy row
    path = os.path.join(tmp, "xw.csv")
    save_crosswalk(xw, path)
    strict = load_crosswalk(path, strict=True)
    assert 106 not in set(strict["player_code"]), "strict load must hide unverified rows"
    assert 101 in set(strict["player_code"])

    # attach_player_code drops what it cannot key (G5), and says so
    shots = pd.DataFrame({"understat_player_id": [1, 2, 999], "xg": [0.1, 0.2, 0.3]})
    keyed = attach_player_code(shots, strict)
    assert len(keyed) == 2 and set(keyed["player_code"]) == {101, 102}

    print(f"SELFTEST OK: {len(xw)} matched, collision split on position, "
          f"fuzzy left unverified, unmatched routed to review, strict load enforced.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seasons", nargs="+", default=["2526"])
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    if a.build:
        import sd_ingest, core_insights as ci
        shots = sd_ingest.understat_shots(a.seasons)
        und = (shots.assign(team=shots["team"].map(sd_ingest.normalise_team))
               .groupby(["understat_player_id", "player_name", "team"], as_index=False)
               .size().drop(columns="size"))
        d, _, _ = ci.load(base=config.repo("2026-2027"))
        fpl = d[["player_code", "web_name", "team", "pos"]]
        xw = build_understat_crosswalk(und, fpl)
        save_crosswalk(xw)
        print(f"\nNow hand-verify {config.CROSSWALK_REVIEW}, set verified=True on the "
              f"rows you confirm, merge them into {config.CROSSWALK_UNDERSTAT}, and commit.")
        sys.exit(0)
    ap.print_help()
