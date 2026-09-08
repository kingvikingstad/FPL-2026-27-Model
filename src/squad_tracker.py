from __future__ import annotations
import config
"""
squad_tracker.py — record the squad you actually picked, compare it to the model's.
====================================================================================
Two jobs, kept separate because they answer different questions:

  COMPARE (before the deadline)  what did you pick that the model would not, and what
                                 does that difference cost in projected points?
  SCORE   (after the results)    what did each squad ACTUALLY score? This is the only
                                 question that eventually matters, and it is the one a
                                 projection model cannot answer about itself.

WHY THE SECOND ONE IS THE POINT
--------------------------------
Everything in this repo so far is projection against projection. The model has been
validated against other models (Solio, FFS) and against measured base rates, but never
against a scored gameweek — the docs are explicit that "the decisive validation is
post-GW1 scored data". This ledger is what turns that from an intention into a record:
once actual points are entered, every week adds a paired observation of model vs manager
on the same fixtures, and after enough weeks the difference is measurable rather than
arguable.

Keep entering results even in weeks you follow the model exactly. A week where both
squads are identical still scores the model against reality, which is the harder test.

FILE FORMAT  (data/my_squad.csv — edit this by hand)
-----------------------------------------------------
    gw,player,in_xi,is_captain,is_vice
    1,Raya,yes,no,no
    1,B.Fernandes,yes,yes,no
    ...
`player` is matched exactly, then on a whole word, against display_name / web_name.
Accents are optional, not forbidden: 'Petrovic' and 'Petrović' both find Petrović.
An ambiguous name is REPORTED, never guessed — see `resolve_names`.

Actual points go in data/my_results.csv:
    gw,my_points,model_points,note

Run:  python src/squad_tracker.py --selftest
"""
import os
import re
import unicodedata
import numpy as np
import pandas as pd

SQUAD_FILE = os.path.join(config.DATA, "my_squad.csv")
RESULTS_FILE = os.path.join(config.DATA, "my_results.csv")

SQUAD_RULES = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
BUDGET = 100.0
MAX_PER_CLUB = 3
XI_SIZE = 11


def _truthy(v):
    return str(v).strip().lower() in ("1", "y", "yes", "true", "t", "x")


def template(gw=1):
    """A blank sheet to fill in."""
    return pd.DataFrame({"gw": [gw] * 15, "player": [""] * 15,
                         "in_xi": ["yes"] * 11 + ["no"] * 4,
                         "is_captain": ["no"] * 15, "is_vice": ["no"] * 15})


def load_squad(path=None, gw=None):
    p = path or SQUAD_FILE
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} does not exist — run scripts/track.py --init to create a template")
    d = pd.read_csv(p)
    d.columns = [c.strip().lower() for c in d.columns]
    for c in ("in_xi", "is_captain", "is_vice"):
        d[c] = d[c].map(_truthy) if c in d.columns else False
    d["player"] = d["player"].astype(str).str.strip()
    d = d[d["player"] != ""]
    if gw is not None:
        d = d[d["gw"] == gw]
    return d


# Letters NFKD does NOT decompose — they are distinct characters, not letter+diacritic.
# Without these, 'Odegaard' and 'Petrovic' never reach 'Ødegaard' and 'Đ. Petrović'.
_FOLD = {"ø": "o", "æ": "ae", "ß": "ss", "đ": "d", "ð": "d", "ł": "l",
         "þ": "th", "œ": "oe", "ı": "i"}


def _fold(s):
    """Lowercase and strip diacritics, leaving punctuation and spacing untouched.

    Applied to BOTH sides of every comparison, which is what makes the accent optional
    rather than merely relocated: 'Petrovic' and 'Petrović' meet at the same key, and
    'Petrović' still finds itself. Punctuation deliberately survives — the whole-word
    pass in `resolve_names` uses '.', '-' and an apostrophe as word boundaries, so
    folding those away would turn 'Amad' back into a substring match on 'Al-Hamadi'.
    """
    s = str(s).lower()
    s = "".join(_FOLD.get(c, c) for c in s)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def resolve_names(entered, board):
    """Map typed names onto player_code. Exact, then whole-word; ambiguity is reported.

    Same rule as the ban matcher, and for the same reason: 'Amad' is a substring of
    'Kamada' and 'Al-Hamadi', so a contains() match silently picks the wrong player.

    Both the typed name and the board columns are diacritic-folded (see `_fold`), so a
    keyboard without accents is not a reason to be told the player does not exist. That
    widens what can match, never what is guessed: a name folding onto two players is
    still reported as ambiguous rather than resolved to whichever sorted first.
    """
    u = board.drop_duplicates("player_code")
    # Folded once per call rather than once per entered name; na_action keeps a missing
    # name missing instead of turning it into the literal string 'nan'.
    disp = u["display_name"].map(_fold, na_action="ignore")
    plyr = u["player"].map(_fold, na_action="ignore")
    label = (u.get("unique_label", pd.Series("", index=u.index))
              .astype(str).map(_fold))
    out, problems = [], []
    for nm in entered:
        key = _fold(str(nm).strip())
        hit = u[disp.eq(key) | plyr.eq(key) | label.eq(key)]
        if len(hit) == 0:
            pat = rf"(?:^|[\s.\-']){re.escape(key)}(?:$|[\s.\-'])"
            hit = u[disp.str.contains(pat, regex=True, na=False)
                    | plyr.str.contains(pat, regex=True, na=False)]
        if len(hit) == 0:
            problems.append(f"'{nm}' matched no player")
            out.append(None)
        elif len(hit) > 1:
            who = ", ".join(f"{r.display_name} ({r.team})" for r in hit.itertuples())
            problems.append(f"'{nm}' is ambiguous: {who} — use the fuller name")
            out.append(None)
        else:
            out.append(int(hit.iloc[0]["player_code"]))
    return out, problems


def validate(sq):
    """Check the entered squad against the actual FPL rules.

    Returns a list of problems. An entry error that silently passes here would show up
    later as a mysterious points gap, so this is deliberately strict and complains about
    everything at once rather than stopping at the first fault.
    """
    problems = []
    n = len(sq)
    if n != 15:
        problems.append(f"squad has {n} players, needs 15")
    for pos, k in SQUAD_RULES.items():
        got = int((sq["pos"] == pos).sum())
        if got != k:
            problems.append(f"{got} {pos}, needs {k}")
    cost = float(sq["price"].sum())
    if cost > BUDGET + 1e-6:
        problems.append(f"squad costs £{cost:.1f}m, over the £{BUDGET:.0f}m budget")
    clubs = sq.groupby("team").size()
    for club, c in clubs[clubs > MAX_PER_CLUB].items():
        problems.append(f"{c} players from {club}, max is {MAX_PER_CLUB}")
    xi = sq[sq["in_xi"]]
    if len(xi) != XI_SIZE:
        problems.append(f"{len(xi)} in the XI, needs {XI_SIZE}")
    else:
        if int((xi["pos"] == "GK").sum()) != 1:
            problems.append("XI needs exactly 1 GK")
        if int((xi["pos"] == "DEF").sum()) < 3:
            problems.append("XI needs at least 3 DEF")
        if int((xi["pos"] == "FWD").sum()) < 1:
            problems.append("XI needs at least 1 FWD")
    caps = int(sq["is_captain"].sum())
    if caps != 1:
        problems.append(f"{caps} captains, needs exactly 1")
    elif not bool(sq.loc[sq["is_captain"], "in_xi"].iloc[0]):
        problems.append("captain is not in the XI")
    if int(sq["is_vice"].sum()) > 1:
        problems.append("more than one vice-captain")
    if sq["player_code"].duplicated().any():
        problems.append("the same player appears twice")
    return problems


def score_squad(sq, ep_col="ep", bench_boost=False, triple_captain=False):
    """Projected points for a squad as entered: XI + captain copy (+ bench if boosted)."""
    xi = sq[sq["in_xi"]]
    cap = sq[sq["is_captain"]]
    xi_ep = float(xi[ep_col].sum())
    cap_ep = float(cap[ep_col].sum()) if len(cap) else 0.0
    bench_ep = float(sq.loc[~sq["in_xi"], ep_col].sum())
    total = xi_ep + cap_ep * (2.0 if triple_captain else 1.0)
    if bench_boost:
        total += bench_ep
    return {"xi_ep": xi_ep, "captain_ep": cap_ep, "bench_ep": bench_ep,
            "total_ep": total}


def compare(mine, proposed, ep_col="ep"):
    """Set difference plus the projected cost of every disagreement."""
    a = set(mine["player_code"])
    b = set(proposed["player_code"])
    only_mine = mine[mine.player_code.isin(a - b)]
    only_prop = proposed[proposed.player_code.isin(b - a)]
    return {"shared": len(a & b),
            "only_mine": only_mine.sort_values(ep_col, ascending=False),
            "only_proposed": only_prop.sort_values(ep_col, ascending=False)}


# The three squads tracked. `hybrid` is the model optimising under the manager's stated
# constraints; `model` is the same chip and transfer plan with those constraints removed.
# Keeping both separates two different questions — is the model any good, and do my
# preferences cost me — which a two-way comparison silently merges.
TRACKS = ("mine", "hybrid", "model")


def append_result(gw, my_points, hybrid_points=None, model_points=None, note="",
                  path=None):
    """Add a scored gameweek, replacing any existing row for that gameweek.

    A missing track is stored as NaN rather than zero: "not recorded" and "scored
    nothing" are different, and averaging a zero into a mean would quietly libel it.
    """
    p = path or RESULTS_FILE
    row = {"gw": int(gw), "my_points": float(my_points),
           "hybrid_points": np.nan if hybrid_points is None else float(hybrid_points),
           "model_points": np.nan if model_points is None else float(model_points),
           "note": note}
    row = pd.DataFrame([row])
    if os.path.exists(p):
        d = pd.read_csv(p)
        for c in ("hybrid_points", "model_points"):
            if c not in d.columns:
                d[c] = np.nan
        d = d[d["gw"] != int(gw)]
        row = pd.concat([d, row], ignore_index=True)
    row = row.sort_values("gw")
    row.to_csv(p, index=False)
    return row


def ledger(path=None):
    """Running record across all three tracks, per gameweek and cumulative."""
    p = path or RESULTS_FILE
    cols = ["gw", "my_points", "hybrid_points", "model_points"]
    if not os.path.exists(p):
        return pd.DataFrame(columns=cols)
    d = pd.read_csv(p).sort_values("gw")
    for c in ("hybrid_points", "model_points"):
        if c not in d.columns:
            d[c] = np.nan
    d["vs_hybrid"] = d["my_points"] - d["hybrid_points"]
    d["vs_model"] = d["my_points"] - d["model_points"]
    d["hybrid_vs_model"] = d["hybrid_points"] - d["model_points"]
    for a, b in (("my_points", "cum_mine"), ("hybrid_points", "cum_hybrid"),
                 ("model_points", "cum_model")):
        d[b] = d[a].cumsum()
    return d


def selftest():
    import tempfile
    board = pd.DataFrame({
        "player_code": [1, 2, 3, 4, 5, 6, 7, 8],
        "display_name": ["Raya", "C. Palmer (CHE)", "A. Palmer (IPS)", "Haaland",
                         "Đ. Petrović (BOU)", "D. Núñez (LIV)", "M. Nunes (MCI)",
                         "João Pedro (CHE)"],
        "player": ["Raya", "Palmer", "Palmer", "Haaland",
                   "Đ. Petrović", "Núñez", "Nunes", "João Pedro"],
        "team": ["Arsenal", "Chelsea", "Ipswich", "Man City",
                 "Bournemouth", "Liverpool", "Man City", "Chelsea"],
        "pos": ["GK", "MID", "GK", "FWD", "GK", "FWD", "MID", "FWD"],
        "price": [6.0, 9.5, 4.0, 15.5, 5.0, 7.5, 6.5, 7.5],
        "ep": [4.2, 5.5, 2.0, 7.6, 3.4, 4.1, 3.9, 4.6]})

    codes, probs = resolve_names(["Raya", "Haaland"], board)
    assert codes == [1, 4] and not probs

    # ambiguity must be reported, not guessed
    codes, probs = resolve_names(["Palmer"], board)
    assert codes == [None] and "ambiguous" in probs[0]
    # the disambiguated name resolves cleanly
    codes, probs = resolve_names(["C. Palmer (CHE)"], board)
    assert codes == [2] and not probs
    codes, probs = resolve_names(["Nobody"], board)
    assert codes == [None] and "no player" in probs[0]

    # an accent on the board must not be required on the keyboard. 'Petrovic' reaches
    # 'Đ. Petrović' only if BOTH sides are folded — and only through the whole-word
    # pass, since neither board column equals the typed name outright.
    codes, probs = resolve_names(["Petrovic"], board)
    assert codes == [5] and not probs, (codes, probs)
    # the accent is optional, not forbidden: typing it still resolves the same player
    codes, probs = resolve_names(["Petrović"], board)
    assert codes == [5] and not probs, (codes, probs)
    # 'João Pedro' matched before this change because the user typed the accent; both
    # spellings must land on the same code now, or the fix has moved the problem
    codes, probs = resolve_names(["João Pedro", "Joao Pedro"], board)
    assert codes == [8, 8] and not probs, (codes, probs)

    # folding must not collide two distinct players into a false ambiguity: 'Núñez' and
    # 'Nunes' differ by one letter and both survive the fold, so each still resolves to
    # exactly one code with nothing reported. If folding ever widened into a fuzzy
    # match, these two would be the first pair to be wrongly merged.
    codes, probs = resolve_names(["Nunez", "Nunes"], board)
    assert codes == [6, 7] and not probs, (codes, probs)

    # validation catches a squad that breaks the rules
    bad = pd.DataFrame({"player_code": [1, 2], "pos": ["GK", "MID"],
                        "team": ["Arsenal", "Chelsea"], "price": [6.0, 9.5],
                        "in_xi": [True, True], "is_captain": [True, True],
                        "is_vice": [False, False]})
    pr = validate(bad)
    assert any("15" in x for x in pr), "should flag squad size"
    assert any("captains" in x for x in pr), "should flag two captains"

    # scoring: captain adds a second copy; bench only counts when boosted
    sq = pd.DataFrame({"ep": [5.0, 3.0, 2.0], "in_xi": [True, True, False],
                       "is_captain": [True, False, False]})
    s = score_squad(sq)
    assert abs(s["total_ep"] - 13.0) < 1e-9, s          # 5+3 XI, +5 captain
    sb = score_squad(sq, bench_boost=True)
    assert abs(sb["total_ep"] - 15.0) < 1e-9, sb        # +2 bench
    st = score_squad(sq, triple_captain=True)
    assert abs(st["total_ep"] - 18.0) < 1e-9, st        # captain counted twice more

    # ledger accumulates across three tracks and replaces rather than duplicating
    tmp = os.path.join(tempfile.mkdtemp(), "res.csv")
    append_result(1, 70, 69, 68, path=tmp)
    append_result(2, 55, 57, 60, path=tmp)
    append_result(2, 58, 57, 60, note="corrected", path=tmp)
    L = ledger(tmp)
    assert len(L) == 2, "a re-entered gameweek must replace, not duplicate"
    assert float(L.loc[L.gw == 2, "my_points"].iloc[0]) == 58.0
    assert abs(float(L["cum_mine"].iloc[-1]) - 128.0) < 1e-9
    assert abs(float(L["cum_hybrid"].iloc[-1]) - 126.0) < 1e-9
    assert abs(float(L["cum_model"].iloc[-1]) - 128.0) < 1e-9
    assert abs(float(L["vs_model"].iloc[0]) - 2.0) < 1e-9
    assert abs(float(L["hybrid_vs_model"].iloc[0]) - 1.0) < 1e-9

    # a missing track stays NaN — "not recorded" must not be read as "scored nothing"
    append_result(3, 60, path=tmp)
    L2 = ledger(tmp)
    assert pd.isna(L2.loc[L2.gw == 3, "model_points"].iloc[0]), \
        "an unrecorded track must be NaN, not 0"
    assert abs(float(L2["my_points"].mean()) - 62.6667) < 0.01

    print("SELFTEST OK: exact and whole-word name matching, accents optional on both "
          "sides without merging distinct players, ambiguity reported not guessed, "
          "squad rules validated, captain doubled and bench counted only when boosted, "
          "ledger replaces a re-entered gameweek.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    print(__doc__)
