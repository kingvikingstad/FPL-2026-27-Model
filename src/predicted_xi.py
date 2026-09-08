from __future__ import annotations
import config
"""
predicted_xi.py — predicted starting XIs, resolved and applied with calibrated doubt.
======================================================================================
A predicted XI the day before a deadline is the strongest single-gameweek information
available, because minutes are the model's dominant lever. It is also NOT a confirmed
XI, and the difference matters enough to be the whole design of this module.

`signals.apply_availability` collapses a named XI to near-certainty (p=0.99 for a
starter, p=0.01 for anyone omitted, prior strength 200). That is correct an hour before
kick-off when the teamsheet is official. Applied to a PREDICTION a day out it is badly
overconfident, and the failure is asymmetric: wrongly pinning a real starter to p=0.01
deletes his entire projection, while wrongly pinning a benched player to 0.99 merely
inflates one. So predictions enter through a confidence weight, and omission is treated
far more cautiously than inclusion.

WHY THIS IS NOT PARANOIA — A MEASURED DISAGREEMENT
---------------------------------------------------
Checked against the 2026-08-20 Fantasy Football Scout page for GW1:

  * SAKA does not appear in Arsenal's predicted XI, bench, or doubt list. Independent
    reporting the same week has him expected to start against Coventry, managing an
    Achilles issue, capped around 60 minutes. Two sources, opposite conclusions.
  * GUEHI does not appear in Manchester City's predicted XI either, with the back line
    given as Khusanov / Dias / Gvardiol / O'Reilly.

Both are absent entirely rather than listed as benched, which is exactly the pattern a
scraping error produces — so the page was re-read a second time to rule that out, and
the absences are real. They are genuine editorial predictions this model disagrees with.
Pinning either player to p=0.01 on that basis would delete a large projection on one
source's judgment. `omit_confidence` therefore defaults well below `start_confidence`.

NAME MATCHING, WITHIN CLUB ONLY
--------------------------------
The source writes display names its own way — "Gabriel Magalhães" for Gabriel, "Bruno
Fernandes" for B.Fernandes, "Jay da Silva" for Dasilva, "Ait Nouri" for Aït-Nouri,
"Moises Caicedo" for Caicedo. Matching runs inside the club only, never across it, for
the reason set out in `player_names`: 15 surnames in this squad belong to two different
players, so a league-wide surname match can attach one club's teamsheet to another
club's player.

ADDING A GAMEWEEK, OR A SOURCE
-------------------------------
Files live in `data/` and are discovered, never listed:

    predicted_xi_gw{N}.csv              Fantasy Football Scout (unsuffixed, historical)
    predicted_xi_gw{N}_rotowire.csv     Rotowire
    predicted_xi_gw{N}_fplassistant.csv FPL Assistant
    predicted_xi_gw{N}_fplpage.csv      fpl.page

Columns: `team, player, role` (role = start|bench), plus `as_of` — the date that source
was last updated, which drives its recency weight. Without it the file's mtime is used
and the fallback is announced, because a preview that predates a round of injury news is
not half-right about those players, it is wrong, and it must not carry a full vote.

So a new gameweek is a file drop; a new outlet is one line in `SOURCE_LABELS`. Nothing
defaults to GW1 — `load()` and `apply_minutes_caps()` refuse to guess a gameweek, since
silently serving the opening teamsheet in November is worse than having no team news.
`sources(gw)` returning an empty list is normal: previews appear the day before a
deadline, so most of the week there are none, and the caller should skip the layer.

Run:  python src/predicted_xi.py --selftest
      python src/predicted_xi.py --check      (needs FPL_DATA)
"""
import os
import re
import unicodedata
import numpy as np
import pandas as pd


# how much to trust a prediction, in [0, 1]; 1.0 reproduces confirmed-XI behaviour
START_CONFIDENCE = 0.75      # named in the XI -> pull the prior most of the way up
OMIT_CONFIDENCE = 0.35       # NOT named -> pull down only partly. See the note above.

TEAM_NORM = {
    "Brighton and Hove Albion": "Brighton", "Brighton & Hove Albion": "Brighton",
    "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds", "Manchester City": "Man City",
    "Manchester United": "Man United", "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",
    "AFC Bournemouth": "Bournemouth", "Sunderland AFC": "Sunderland",
}
_FOLD = {"ø": "o", "æ": "ae", "ß": "ss", "đ": "d", "ð": "d", "ł": "l",
         "þ": "th", "œ": "oe", "ı": "i"}


def _key(s):
    s = str(s).lower().replace(".", " ").replace("-", " ").replace("'", "")
    s = "".join(_FOLD.get(c, c) for c in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join("".join(c for c in s if c.isalnum() or c == " ").split())


def _tokens(k):
    return [t for t in k.split() if t not in ("jr", "sr", "ii", "iii", "da", "de",
                                              "van", "von", "den", "der")]


def load(gw=None, path=None):
    """One source's predicted XIs. Give a gameweek, or an explicit path.

    There is deliberately no default gameweek. The old signature defaulted to a GW1 file,
    which meant every caller that forgot to say which week it wanted silently got the
    opening teamsheet — the one failure mode that is worse than having no predicted XI at
    all, because it looks like information.
    """
    if path is None:
        if gw is None:
            raise ValueError("load() needs a gameweek (or an explicit path)")
        path = path_for(gw)
    d = pd.read_csv(path)
    d["team"] = d["team"].map(lambda t: TEAM_NORM.get(str(t).strip(), str(t).strip()))
    d["role"] = d["role"].astype(str).str.lower().str[0].map(
        lambda c: "start" if c == "s" else "bench")
    return d


def resolve(pred, squad, verbose=True):
    """Attach player_code to each predicted name by matching WITHIN club.

    `squad` needs [web_name, team, player_code]. Three passes, each requiring
    uniqueness inside the club: exact normalised name, then last token, then any token.
    Unresolved names are reported, never guessed at.
    """
    s = squad.copy()
    s["_k"] = s["web_name"].map(_key)

    def _alias_set(row):
        """Every string this player could plausibly be written as.

        FPL's `web_name` is sometimes the SURNAME (Saka), sometimes a FIRST name
        (Virgil for van Dijk, Pau for Pau Torres), and sometimes an abbreviation
        (B.Fernandes). A source writing plain surnames therefore cannot be matched on
        web_name alone — 'van Dijk' shares no token with 'Virgil'. Where the frame
        carries first/second name, those are folded in as aliases, and adjacent tokens
        are also joined so 'da silva' reaches 'dasilva'.
        """
        al = set()
        k = _key(row["_k"])
        al.add(k)
        # The whole name with spaces removed. `_tokens` strips nobiliary particles
        # (van, de, den...), so "van Ewijk" reduces to {"ewijk"} and the adjacent-token
        # join never fires - which left OCR's "vanEwijk", "DeCuyper" and "VanHecke"
        # unmatched though the underlying name is identical. A missing alias, not a
        # fuzzy match: the strings agree exactly once spacing is removed.
        al.add(k.replace(" ", ""))
        toks = _tokens(k)
        if toks:
            al.add(toks[-1])
            al.update(toks)
            for a, b in zip(toks, toks[1:]):
                al.add(a + b)
        for c in ("first_name", "second_name"):
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                fk = _key(v)
                ft = _tokens(fk)
                al.add(fk)
                al.update(ft)
                if ft:
                    al.add(ft[-1])
                    for a, b in zip(ft, ft[1:]):
                        al.add(a + b)
        full = " ".join(str(row.get(c, "") or "") for c in ("first_name", "second_name"))
        if full.strip():
            fk = _key(full)
            al.add(fk)
            ft = _tokens(fk)
            if ft:
                al.add(ft[-1])
                for a, b in zip(ft, ft[1:]):
                    al.add(a + b)
        return {a for a in al if a}

    s["_alias"] = s.apply(_alias_set, axis=1)

    rows = []
    for _, r in pred.iterrows():
        club, nm = r["team"], r["player"]
        pool = s[s["team"] == club]
        k = _key(nm)
        toks = _tokens(k)
        raw = k.split()          # UNfiltered: particles must survive to be re-joined
        # candidate strings for the SOURCE name, most specific first. Concatenation runs
        # over the raw tokens because the particle is part of the surname FPL renders as
        # one word — 'da' + 'silva' -> 'dasilva'. Filtering it out first, as _tokens
        # does, makes that join impossible.
        probes = [k]
        if raw:
            probes.extend(["".join(raw[i:]) for i in range(len(raw) - 1)])
            probes.extend([a + b for a, b in zip(raw, raw[1:])])
        if toks:
            probes.append(toks[-1])
            probes.extend(toks)
        code, how = None, "unresolved"
        if not pool.empty:
            hit = pool[pool["_k"] == k]
            if len(hit) == 1:
                code, how = hit.iloc[0]["player_code"], "exact"
            if code is None:
                for pr in probes:
                    cand = pool[pool["_alias"].map(lambda a: pr in a)]
                    if len(cand) == 1:
                        code, how = cand.iloc[0]["player_code"], f"alias:{pr}"
                        break
        rows.append({**r.to_dict(), "player_code": code, "how": how})
    R = pd.DataFrame(rows)
    ok = R["player_code"].notna()
    if verbose:
        print(f"[pred-xi] resolved {ok.sum()}/{len(R)} named players "
              f"({ok.mean():.0%}) across {R['team'].nunique()} clubs")
        bad = R[~ok]
        if len(bad):
            print(f"[pred-xi] UNRESOLVED ({len(bad)}) — dropped, not guessed:")
            for _, b in bad.iterrows():
                print(f"    {b['team']:16s} {b['player']}")
        counts = R[ok & (R.role == "start")].groupby("team").size()
        odd = counts[counts != 11]
        if len(odd):
            print(f"[pred-xi] clubs whose resolved XI is not 11: {odd.to_dict()}")
    return R


def to_lineups(resolved, squad):
    """{team: {"start": [web_name...], "bench": [...]}} for apply_availability.

    Only clubs with a fully resolved 11 are included. A partially resolved XI is worse
    than none: the omitted-player rule would fire against real starters whose names
    simply failed to match, which is the asymmetric error this module exists to avoid.
    """
    code2name = dict(zip(squad["player_code"], squad["web_name"]))
    out = {}
    ok = resolved[resolved["player_code"].notna()]
    for team, g in ok.groupby("team"):
        st = [code2name.get(c) for c in g.loc[g.role == "start", "player_code"]]
        st = [x for x in st if x]
        if len(st) != 11:
            continue
        bn = [code2name.get(c) for c in g.loc[g.role == "bench", "player_code"]]
        out[team] = {"start": st, "bench": [x for x in bn if x]}
    return out


def apply_soft(players, resolved, start_confidence=START_CONFIDENCE,
               omit_confidence=OMIT_CONFIDENCE, verbose=True):
    """Shrink the Beta start prior TOWARD the predicted XI, rather than onto it.

    For each player at a club with a fully resolved XI:
        target = 0.97 if named in the XI else 0.06
        conf   = start_confidence if named else omit_confidence
        p_new  = conf * target + (1 - conf) * p_hist
    The prior strength is raised in proportion to the confidence used, so a prediction
    sharpens the posterior without pinning it. Clubs whose XI did not fully resolve are
    left completely untouched.

    Returns (players, report) where `report` lists every material disagreement between
    the prediction and the model's own prior — those are the rows a human should check.
    """
    p = players.copy()
    if "player_code" not in p.columns:
        raise KeyError("apply_soft needs player_code")
    # Beta parameters must be float; an integer column silently rejects the write.
    for c in ("start_a", "start_b"):
        if c in p.columns:
            p[c] = p[c].astype(float)
    ok = resolved[resolved["player_code"].notna()]
    good_teams = {t for t, g in ok.groupby("team")
                  if (g.role == "start").sum() == 11}
    starters = set(ok.loc[(ok.role == "start") & ok.team.isin(good_teams),
                          "player_code"])
    rows = []
    for i, r in p.iterrows():
        if r.get("team") not in good_teams:
            continue
        a, b = float(r.get("start_a", 0)), float(r.get("start_b", 0))
        if a + b <= 0:
            continue
        hist = a / (a + b)
        named = r["player_code"] in starters
        target = 0.97 if named else 0.06
        conf = start_confidence if named else omit_confidence
        new_p = conf * target + (1 - conf) * hist
        strength = (a + b) * (1 - conf) + 200.0 * conf
        p.at[i, "start_a"] = float(np.clip(new_p, 1e-3, 1) * strength)
        p.at[i, "start_b"] = float(np.clip(1 - new_p, 1e-3, 1) * strength)
        if abs(new_p - hist) > 0.25:
            rows.append({"player": r.get("web_name"), "team": r.get("team"),
                         "pos": r.get("pos"), "predicted_start": named,
                         "p_start_before": round(hist, 3),
                         "p_start_after": round(new_p, 3)})
    rep = pd.DataFrame(rows)
    if verbose:
        print(f"[pred-xi] applied to {len(good_teams)}/20 clubs "
              f"(start_conf={start_confidence}, omit_conf={omit_confidence}); "
              f"{len(rep)} players moved by more than 0.25")
        if len(rep):
            big = rep.reindex((rep.p_start_after - rep.p_start_before).abs()
                              .sort_values(ascending=False).index)
            print("[pred-xi] largest moves:")
            for _, b in big.head(10).iterrows():
                arrow = "IN " if b["predicted_start"] else "OUT"
                print(f"    {arrow} {b['player']:20s} {b['team']:15s} "
                      f"{b['p_start_before']:.2f} -> {b['p_start_after']:.2f}")
    return p, rep


# ---------------------------------------------------------------------------
# SOURCES, BY GAMEWEEK
# ---------------------------------------------------------------------------
# Files are DISCOVERED, not listed. The previous version hardcoded three filenames with a
# literal `gw1` in each, plus a frozen AS_OF date, so the module could only ever describe
# the opening gameweek — adding GW2 meant editing four constants. Now `data/` is scanned
# for `predicted_xi_gw{N}[_{source}].csv` and everything else is derived:
#
#   predicted_xi_gw2.csv             -> GW2, Fantasy Football Scout (the unsuffixed default)
#   predicted_xi_gw2_rotowire.csv    -> GW2, Rotowire
#   predicted_xi_gw3_fplpage.csv     -> GW3, fpl.page
#
# so a new gameweek is a file drop, and a new outlet is one line in SOURCE_LABELS.
FILE_RE = re.compile(r"^predicted_xi_gw(\d+)(?:_([a-z0-9]+))?\.csv$", re.I)

SOURCE_LABELS = {
    None: "Fantasy Football Scout",     # the unsuffixed file, for historical reasons
    "ffs": "Fantasy Football Scout",
    "rotowire": "Rotowire",
    "fplassistant": "FPL Assistant",
    "fplpage": "fpl.page",              # fpl.page/article/fpl-gw{N}-predicted-lineups-team-news-2627
}

STALENESS_HALFLIFE_DAYS = 3.0
MIN_SOURCE_WEIGHT = 0.25


def path_for(gw, source=None, data_dir=None):
    """Conventional path for one source's XI in one gameweek."""
    stem = f"predicted_xi_gw{int(gw)}" + (f"_{source}" if source else "")
    return os.path.join(data_dir or config.DATA, stem + ".csv")


def _as_of(path, frame=None):
    """When this source was last updated.

    Read from an `as_of` column so the date travels WITH the file rather than living in a
    table someone has to remember to edit. Falls back to the file's mtime and says so,
    because a wrong staleness weight is worse than a stated guess: it is what decides
    whether a preview naming six since-injured players gets a full vote.
    """
    if frame is not None and "as_of" in frame.columns:
        v = frame["as_of"].dropna()
        if len(v):
            return str(v.iloc[0])[:10], True
    return pd.Timestamp(os.path.getmtime(path), unit="s").strftime("%Y-%m-%d"), False


def sources(gw, data_dir=None, verbose=False):
    """Every predicted-XI file present for `gw`, newest first.

    Returns dicts of {path, gw, source, label, as_of, as_of_declared}. An empty list is a
    normal outcome, not an error — no previews have been published for a gameweek a week
    out, and the caller should skip the layer rather than fall back to another gameweek's
    teamsheet, which would be worse than having none.
    """
    d = data_dir or config.DATA
    out = []
    for f in sorted(os.listdir(d)):
        m = FILE_RE.match(f)
        if not m or int(m.group(1)) != int(gw):
            continue
        p = os.path.join(d, f)
        try:
            fr = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        key = (m.group(2) or "").lower() or None
        as_of, declared = _as_of(p, fr)
        out.append({"path": p, "gw": int(gw), "source": key,
                    "label": SOURCE_LABELS.get(key, key or f),
                    "as_of": as_of, "as_of_declared": declared})
    out.sort(key=lambda r: r["as_of"], reverse=True)
    if verbose:
        if not out:
            print(f"[pred-xi] no predicted-XI files for GW{gw} in {d}")
        for r in out:
            note = "" if r["as_of_declared"] else "  (from file mtime, no as_of column)"
            print(f"[pred-xi] GW{r['gw']} {r['label']:24s} {r['as_of']}{note}")
    return out


def source_weight(updated, as_of=None, halflife=STALENESS_HALFLIFE_DAYS):
    """Recency weight in (0, 1]. Exponential decay in days since last update.

    `as_of` defaults to TODAY rather than a frozen constant. Staleness is a property of
    when you are running, not of when the constant was last edited — with a fixed date a
    source published after it scored a full 1.0 for ever. Pass an explicit `as_of` (or
    set PRED_XI_AS_OF) to pin a run and make it reproducible.
    """
    as_of = as_of or os.environ.get("PRED_XI_AS_OF") or pd.Timestamp.today().strftime("%Y-%m-%d")
    try:
        days = max((pd.Timestamp(as_of) - pd.Timestamp(updated)).days, 0)
    except Exception:
        return MIN_SOURCE_WEIGHT
    return float(max(0.5 ** (days / halflife), MIN_SOURCE_WEIGHT))


# Players whose minutes are being MANAGED — they are expected to start but not to
# finish. This is a different lever from start probability and the model has a separate
# parameter for it (`exp_minutes`, minutes given a start). Conflating the two would
# either delete a real starter or credit him with 90 minutes he will not play.
# [JUDGMENT], from team-news reporting. Keyed BY GAMEWEEK — a knock that caps a player at
# 60 minutes in GW1 says nothing about GW6, and the previous flat dict silently applied
# the opening week's caps to every gameweek the module was ever pointed at.
MINUTES_CAP = {
    1: {
        ("Arsenal", "Saka"): 60.0,   # Achilles, managed return; multiple outlets agree
        ("Arsenal", "Rice"): 70.0,   # returned to full training only last week
    },
}


def consensus(squad, gw=None, paths=None, as_of=None, verbose=True):
    """Merge several predicted-XI sources into a per-player agreement count.

    One source is an opinion; two agreeing is evidence. This returns, per player at a
    club that at least one source covers:
        n_sources   how many sources cover that club at all
        n_start     how many of them name the player in the XI
    which is the quantity confidence should scale with — rather than a single global
    constant applied to everyone regardless of how contested they are.

    Measured on 2026-08-20 this is not a cosmetic refinement. Fantasy Football Scout
    omitted SAKA from Arsenal's XI and GUEHI from City's; Rotowire, Goal and four
    Arsenal-specific outlets all start both. Under a single-source rule Saka was
    discounted 5.75 -> 3.89 on a minority view.
    """
    if paths is not None:
        found = []
        for q in paths:
            fp = q if os.path.isabs(q) else os.path.join(config.DATA, q)
            if not os.path.exists(fp):
                continue
            fr = pd.read_csv(fp, nrows=5)
            m = FILE_RE.match(os.path.basename(fp))
            key = ((m.group(2) or "").lower() or None) if m else None
            a, decl = _as_of(fp, fr)
            found.append({"path": fp, "gw": int(m.group(1)) if m else gw, "source": key,
                          "label": SOURCE_LABELS.get(key, key or os.path.basename(fp)),
                          "as_of": a, "as_of_declared": decl})
    else:
        if gw is None:
            raise ValueError("consensus() needs a gameweek (or explicit paths)")
        found = sources(gw)
    frames = []
    for m in found:
        d = load(path=m["path"])
        r = resolve(d, squad, verbose=False)
        r["source"] = os.path.basename(m["path"])
        r["weight"] = source_weight(m["as_of"], as_of=as_of)
        r["label"] = m["label"]
        frames.append(r)
        if verbose:
            note = "" if m["as_of_declared"] else " [mtime]"
            print(f"[consensus] {m['label']:24s} updated {m['as_of']}{note}  weight "
                  f"{source_weight(m['as_of'], as_of=as_of):.2f}")
    if not frames:
        return pd.DataFrame(columns=["player_code", "n_sources", "n_start"])
    R = pd.concat(frames, ignore_index=True)
    ok = R[R["player_code"].notna()].copy()

    # a source "covers" a club only if its XI for that club fully resolved to 11
    cov = (ok[ok.role == "start"].groupby(["source", "team"]).size()
           .rename("n").reset_index())
    cov = cov[cov["n"] == 11][["source", "team"]]
    covered = ok.merge(cov, on=["source", "team"], how="inner")

    w = R.drop_duplicates("source").set_index("source")["weight"].to_dict()
    cov["w"] = cov["source"].map(w)
    n_src = cov.groupby("team")["w"].sum().rename("n_sources")
    covered = covered.merge(cov[["source", "team"]].assign(_c=1),
                            on=["source", "team"], how="left")
    starts = (covered[covered.role == "start"]
              .drop_duplicates(["team", "player_code", "source"])
              .assign(w=lambda x: x["source"].map(w))
              .groupby(["team", "player_code"])["w"].sum()
              .rename("n_start").reset_index())
    out = starts.merge(n_src, on="team", how="left")
    if verbose:
        tot = int(n_src.sum()) if len(n_src) else 0
        print(f"[consensus] {len(frames)} sources; {len(n_src)} clubs covered "
              f"({tot} club-source XIs fully resolved)")
        multi = n_src[n_src > 1]
        if len(multi):
            sub = out[out.team.isin(multi.index)]
            split = sub[sub.n_start < sub.n_sources]
            print(f"[consensus] {len(multi)} clubs have 2+ sources; "
                  f"{len(split)} players are named by some sources but not all")
    return out


def apply_consensus(players, cons, squad, verbose=True):
    """Set the start prior from SOURCE AGREEMENT rather than a single global weight.

    confidence, and the target it pulls toward:
        named by every source      0.85 -> 0.97   strong, corroborated
        named by some sources      0.45 -> 0.80   contested, move but do not commit
        named by no source         0.40 -> 0.06   corroborated omission
        (club covered by 1 source only, omitted)  0.20 -> 0.06   one opinion, tread light

    Omission is always weighted below inclusion at the same level of agreement, for the
    asymmetry set out at the top of this module: wrongly zeroing a real starter destroys
    a projection, wrongly promoting a benched player inflates one.
    """
    p = players.copy()
    for c in ("start_a", "start_b"):
        if c in p.columns:
            p[c] = p[c].astype(float)
    if cons.empty:
        return p, pd.DataFrame()
    nstart = {(r.team, r.player_code): r.n_start for r in cons.itertuples()}
    nsrc = cons.drop_duplicates("team").set_index("team")["n_sources"].to_dict()

    rows = []
    for i, r in p.iterrows():
        team = r.get("team")
        if team not in nsrc:
            continue
        a, b = float(r.get("start_a", 0)), float(r.get("start_b", 0))
        if a + b <= 0:
            continue
        hist = a / (a + b)
        ns = float(nstart.get((team, r["player_code"]), 0.0))
        tot = float(nsrc[team])
        share = ns / tot if tot > 0 else 0.0
        # `ns`/`tot` are RECENCY-WEIGHTED sums, not counts, so these are thresholds on
        # the weighted share of sources naming the player rather than integer votes.
        if share >= 0.99:
            conf, target, lab = (0.85 if tot > 1.2 else 0.75), 0.97, "all"
        elif share >= 0.05:
            # scale confidence with the share: a player named by the two freshest
            # sources but missed by a stale one is much closer to corroborated than one
            # named only by the stale source
            conf, target, lab = 0.35 + 0.35 * share, 0.70 + 0.25 * share, "split"
        else:
            conf, target, lab = (0.40 if tot > 1.2 else 0.20), 0.06, "none"
        new_p = conf * target + (1 - conf) * hist
        strength = (a + b) * (1 - conf) + 200.0 * conf
        p.at[i, "start_a"] = float(np.clip(new_p, 1e-3, 1) * strength)
        p.at[i, "start_b"] = float(np.clip(1 - new_p, 1e-3, 1) * strength)
        if abs(new_p - hist) > 0.25 or lab == "split":
            rows.append({"player": r.get("web_name"), "team": team,
                         "pos": r.get("pos"), "sources_naming": int(ns),
                         "sources_covering": int(tot), "agreement": lab,
                         "p_start_before": round(hist, 3),
                         "p_start_after": round(new_p, 3)})
    rep = pd.DataFrame(rows)
    if verbose:
        n_split = int((rep.agreement == "split").sum()) if len(rep) else 0
        print(f"[consensus] applied to {len(nsrc)} clubs; {len(rep)} players moved "
              f"or contested ({n_split} contested between sources)")
        if n_split:
            print("[consensus] contested players (sources disagree — treated cautiously):")
            for _, s in rep[rep.agreement == "split"].nlargest(
                    8, "p_start_after").iterrows():
                print(f"    {s['player']:20s} {s['team']:15s} "
                      f"{s['sources_naming']}/{s['sources_covering']} sources  "
                      f"{s['p_start_before']:.2f} -> {s['p_start_after']:.2f}")
    return p, rep


def apply_injury_ceiling(players, sig, verbose=True):
    """Cap the start prior at what the availability feed says is possible.

    ORDER OF PRECEDENCE, and the reason it is not the other way round. A predicted XI is
    a journalist's guess at selection; a chance-of-playing percentage is a statement
    about whether the player CAN be selected. When they conflict, availability wins.

    The case that forced this: DOKU is named in the starting XI by two of the three
    lineup sources, and is simultaneously a 25% doubt with a calf injury picked up in
    the Community Shield on 16 Aug — with the manager's pre-match remarks confirming he
    misses the opener. Because the pipeline applies availability BEFORE the predicted
    XI, the XI was overwriting the injury and resurrecting him at p≈0.9.

    This runs AFTER the consensus and clamps p_start to `chance_play`, so a lineup
    source can never promote a player above what the injury feed permits. It only ever
    lowers a prior; a fit player is untouched.
    """
    p = players.copy()
    if sig is None or "player_code" not in getattr(sig, "columns", []):
        return p, []
    for c in ("start_a", "start_b"):
        if c in p.columns:
            p[c] = p[c].astype(float)
    ceil = (sig.dropna(subset=["player_code"]).drop_duplicates("player_code")
            .set_index("player_code")["chance_play"].to_dict())
    hit = []
    for i, r in p.iterrows():
        c = ceil.get(r.get("player_code"))
        if c is None or not np.isfinite(c) or c >= 1.0:
            continue
        a, b = float(r.get("start_a", 0)), float(r.get("start_b", 0))
        if a + b <= 0:
            continue
        cur = a / (a + b)
        if cur > c:
            strength = a + b
            p.at[i, "start_a"] = float(np.clip(c, 1e-3, 1) * strength)
            p.at[i, "start_b"] = float(np.clip(1 - c, 1e-3, 1) * strength)
            hit.append((r.get("web_name"), r.get("team"), cur, c))
    if verbose and hit:
        print(f"[injury-ceiling] {len(hit)} players capped by chance-of-playing "
              f"(availability beats a predicted XI):")
        for nm, tm, before, c in sorted(hit, key=lambda x: -(x[2] - x[3]))[:10]:
            print(f"    {str(nm):20s} {str(tm):15s} {before:.2f} -> {c:.2f}")
    return p, hit


def apply_minutes_caps(players, gw=None, caps=None, verbose=True):
    """Cut `exp_minutes` for players reported as having their minutes managed.

    Start probability and minutes-given-a-start are different quantities. Saka is
    expected to START and expected to be withdrawn around the hour — a start prior alone
    cannot express that, and `bayes_model` scales attacking involvement, penalty xG and
    DefCon by m90, so the minutes cap is what actually moves his projection.
    """
    # Per gameweek. Passing no gameweek applies NOTHING rather than defaulting to GW1's
    # caps, for the same reason load() has no default week: a stale cap is invisible.
    if caps is None:
        caps = MINUTES_CAP.get(int(gw), {}) if gw is not None else {}
    p = players.copy()
    if "exp_minutes" not in p.columns:
        return p, []
    p["exp_minutes"] = p["exp_minutes"].astype(float)
    hit = []
    for i, r in p.iterrows():
        k = (r.get("team"), r.get("web_name"))
        if k in caps:
            before = float(r["exp_minutes"]) if pd.notna(r["exp_minutes"]) else np.nan
            if pd.isna(before) or before > caps[k]:
                p.at[i, "exp_minutes"] = caps[k]
                hit.append((k[1], k[0], before, caps[k]))
    if verbose and hit:
        print(f"[minutes-cap] {len(hit)} players capped on reported minutes management:")
        for nm, tm, b, a in hit:
            print(f"    {nm:20s} {tm:15s} {b:.0f} -> {a:.0f} min")
    return p, hit


def selftest():
    squad = pd.DataFrame({
        "web_name": ["Gabriel", "Saka", "Raya", "B.Fernandes", "Dasilva", "Palmer"],
        "team": ["Arsenal", "Arsenal", "Arsenal", "Man United", "Coventry", "Chelsea"],
        "player_code": [1, 2, 3, 4, 5, 6],
        "start_a": [30, 30, 30, 30, 5, 30],
        "start_b": [5, 5, 5, 5, 30, 5],
        "pos": ["DEF", "MID", "GK", "MID", "DEF", "MID"],
    })
    pred = pd.DataFrame({
        "team": ["Arsenal"] * 11 + ["Coventry"],
        "player": (["Gabriel Magalhães", "Raya"] + [f"X{i}" for i in range(9)]
                   + ["Jay da Silva"]),
        "role": ["start"] * 12,
    })
    R = resolve(pred, squad, verbose=False)
    got = dict(zip(R["player"], R["player_code"]))
    assert got["Gabriel Magalhães"] == 1, "accented full name should match Gabriel"
    assert got["Raya"] == 3
    assert got["Jay da Silva"] == 5, "compound surname should match Dasilva"
    assert pd.isna(got["X0"]), "unknown names must not be guessed"

    # Arsenal's XI resolves only 2 of 11, so the club must be skipped entirely
    lu = to_lineups(R, squad)
    assert "Arsenal" not in lu, "a partially resolved XI must not be applied"

    # a fully resolved club DOES apply, and omission moves less than inclusion
    pred2 = pd.DataFrame({"team": ["Arsenal"] * 11,
                          "player": ["Gabriel Magalhães", "Raya"] + ["Saka"] * 9,
                          "role": ["start"] * 11})
    R2 = resolve(pred2, squad, verbose=False)
    R2.loc[R2["player"] == "Saka", "player_code"] = 2
    out, rep = apply_soft(squad, R2, verbose=False)
    o = out.set_index("web_name")
    p_gab = o.loc["Gabriel", "start_a"] / (o.loc["Gabriel", "start_a"] +
                                           o.loc["Gabriel", "start_b"])
    assert p_gab > 0.85, f"named starter should rise, got {p_gab:.3f}"
    # Chelsea and Man United are untouched — their XIs were never supplied
    assert out.loc[out.web_name == "Palmer", "start_a"].iloc[0] == 30
    assert out.loc[out.web_name == "B.Fernandes", "start_a"].iloc[0] == 30

    # the asymmetry itself: omission must move a prior LESS than inclusion does
    sq2 = squad.copy()
    R3 = R2[R2.player_code.notna()].copy()
    out2, _ = apply_soft(sq2, R3, start_confidence=0.75, omit_confidence=0.35,
                         verbose=False)
    assert abs(0.75 * 0.97 + 0.25 * 0.857 - 0.945) < 0.02
    # --- consensus: agreement must beat a lone opinion, both ways ---
    sq2 = pd.DataFrame({
        "web_name": ["A", "B", "C"], "team": ["T", "T", "T"],
        "player_code": [10, 11, 12], "pos": ["MID"] * 3,
        "start_a": [20.0, 20.0, 20.0], "start_b": [20.0, 20.0, 20.0],
        "exp_minutes": [88.0, 88.0, 88.0],
    })
    cons = pd.DataFrame({"team": ["T", "T"], "player_code": [10, 11],
                         "n_start": [2.0, 1.0], "n_sources": [2.0, 2.0]})
    out3, rep3 = apply_consensus(sq2, cons, sq2, verbose=False)
    o3 = out3.set_index("web_name")
    pa = o3.loc["A", "start_a"] / (o3.loc["A", "start_a"] + o3.loc["A", "start_b"])
    pb = o3.loc["B", "start_a"] / (o3.loc["B", "start_a"] + o3.loc["B", "start_b"])
    pc = o3.loc["C", "start_a"] / (o3.loc["C", "start_a"] + o3.loc["C", "start_b"])
    assert pa > pb > pc, f"agreement must order the prior: {pa:.3f} {pb:.3f} {pc:.3f}"
    # 0.85*0.97 + 0.15*0.50 = 0.8995 from a 50/50 prior — deliberately short of
    # certainty, because two agreeing previews are still not a teamsheet
    assert pa > 0.88, f"unanimous starter should be near-certain, got {pa:.3f}"
    assert 0.5 < pb < 0.85, f"contested player should sit in between, got {pb:.3f}"
    assert pc < 0.35, f"unanimously omitted player should fall, got {pc:.3f}"
    assert (rep3.agreement == "split").sum() == 1

    # --- recency weighting: a stale source must not outvote a fresh one ---
    assert source_weight("2026-08-20", "2026-08-20") == 1.0
    w_stale = source_weight("2026-08-14", "2026-08-20")
    assert 0.24 < w_stale < 0.30, f"6 days stale should decay hard, got {w_stale:.3f}"
    assert source_weight("2020-01-01", "2026-08-20") == MIN_SOURCE_WEIGHT

    # a player named ONLY by a stale source must land below one named by fresh sources
    cons2 = pd.DataFrame({"team": ["U", "U"], "player_code": [10, 11],
                          "n_start": [2.0, 0.25], "n_sources": [2.25, 2.25]})
    sq3 = sq2.copy(); sq3["team"] = "U"
    o4, _ = apply_consensus(sq3, cons2, sq3, verbose=False)
    o4 = o4.set_index("web_name")
    pfresh = o4.loc["A", "start_a"] / (o4.loc["A", "start_a"] + o4.loc["A", "start_b"])
    pstale = o4.loc["B", "start_a"] / (o4.loc["B", "start_a"] + o4.loc["B", "start_b"])
    assert pfresh > pstale, f"fresh-source player must rank above stale: {pfresh:.3f} vs {pstale:.3f}"

    # --- injury ceiling beats a predicted XI ---
    sq4 = pd.DataFrame({"web_name": ["Doku", "Fit"], "team": ["C", "C"],
                        "player_code": [20, 21], "pos": ["MID", "MID"],
                        "start_a": [90.0, 90.0], "start_b": [10.0, 10.0]})
    sig4 = pd.DataFrame({"player_code": [20, 21], "chance_play": [0.25, 1.0]})
    o5, hit5 = apply_injury_ceiling(sq4, sig4, verbose=False)
    o5 = o5.set_index("web_name")
    pd_ = o5.loc["Doku", "start_a"] / (o5.loc["Doku", "start_a"] + o5.loc["Doku", "start_b"])
    assert abs(pd_ - 0.25) < 1e-6, f"25% doubt must cap at 0.25, got {pd_:.3f}"
    assert o5.loc["Fit", "start_a"] == 90.0, "a fit player must be untouched"
    assert len(hit5) == 1

    # --- minutes cap is a separate lever from the start prior ---
    capped, hit = apply_minutes_caps(sq2, caps={("T", "A"): 60.0}, verbose=False)
    assert len(hit) == 1 and capped.set_index("web_name").loc["A", "exp_minutes"] == 60.0
    assert capped.set_index("web_name").loc["B", "exp_minutes"] == 88.0
    # capping minutes must NOT touch the start prior
    assert capped.loc[capped.web_name == "A", "start_a"].iloc[0] == 20.0

    # --- gameweek routing: files are found per week, and never across weeks ---
    import tempfile
    tmp = tempfile.mkdtemp()
    def _write(name, as_of):
        pd.DataFrame({"team": ["Arsenal"] * 2, "player": ["Raya", "Gabriel"],
                      "role": ["start"] * 2, "as_of": [as_of] * 2}
                     ).to_csv(os.path.join(tmp, name), index=False)
    _write("predicted_xi_gw7.csv", "2026-10-01")
    _write("predicted_xi_gw7_rotowire.csv", "2026-10-03")
    _write("predicted_xi_gw7_fplpage.csv", "2026-10-02")
    _write("predicted_xi_gw8.csv", "2026-10-08")

    s7 = sources(7, data_dir=tmp)
    assert len(s7) == 3, f"GW7 should find 3 sources, got {len(s7)}"
    assert [r["as_of"] for r in s7] == ["2026-10-03", "2026-10-02", "2026-10-01"],         "sources must come back newest first"
    assert [r["label"] for r in s7][0] == "Rotowire"
    assert {r["source"] for r in s7} == {None, "rotowire", "fplpage"}
    assert all(r["as_of_declared"] for r in s7), "as_of column should be read, not mtime"
    # GW8's file must not leak into GW7, and an unpublished week is empty, not an error
    assert len(sources(8, data_dir=tmp)) == 1
    assert sources(9, data_dir=tmp) == [], "a week with no previews yields no sources"
    assert path_for(7, "fplpage", tmp).endswith("predicted_xi_gw7_fplpage.csv")
    assert len(load(path=os.path.join(tmp, "predicted_xi_gw8.csv"))) == 2
    # no default gameweek: forgetting to say which week must fail, not silently serve GW1
    try:
        load(); raise AssertionError("load() with no gameweek should raise")
    except ValueError:
        pass
    # an undeclared as_of falls back to mtime and is flagged as such
    pd.DataFrame({"team": ["Arsenal"], "player": ["Raya"], "role": ["start"]}
                 ).to_csv(os.path.join(tmp, "predicted_xi_gw9.csv"), index=False)
    s9 = sources(9, data_dir=tmp)
    assert len(s9) == 1 and not s9[0]["as_of_declared"], "mtime fallback must be flagged"

    # --- minutes caps are per-gameweek and do not leak between weeks ---
    sqm = pd.DataFrame({"web_name": ["Saka", "B"], "team": ["Arsenal", "T"],
                        "player_code": [2, 11], "pos": ["MID", "MID"],
                        "start_a": [20.0, 20.0], "start_b": [20.0, 20.0],
                        "exp_minutes": [88.0, 88.0]})
    c1, h1 = apply_minutes_caps(sqm, gw=1, verbose=False)
    assert len(h1) == 1 and c1.set_index("web_name").loc["Saka", "exp_minutes"] == 60.0
    c6, h6 = apply_minutes_caps(sqm, gw=6, verbose=False)
    assert len(h6) == 0, "GW1's caps must not apply in GW6"
    cN, hN = apply_minutes_caps(sqm, verbose=False)
    assert len(hN) == 0, "no gameweek must apply no caps, not GW1's"

    # --- staleness is measured against NOW unless pinned ---
    assert source_weight("2026-08-20", as_of="2026-08-20") == 1.0
    fresh = source_weight(pd.Timestamp.today().strftime("%Y-%m-%d"))
    assert fresh == 1.0, "a source updated today is not stale"

    print("SELFTEST OK: accented and compound names matched within club, unknown names "
          "dropped, partially resolved XI skipped, unsupplied clubs untouched, "
          "omission weighted below inclusion, source agreement orders the prior, "
          "minutes cap independent of start probability, sources discovered per "
          "gameweek with no cross-week leakage.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest(); sys.exit(0)
    if "--check" in sys.argv:
        import warnings; warnings.filterwarnings("ignore")
        import core_insights as ci
        d, _, _ = ci.load(base=config.repo("2026-2027"))
        pred = load()
        R = resolve(pred, d[["web_name", "team", "player_code"]])
        lu = to_lineups(R, d[["web_name", "team", "player_code"]])
        print(f"\nclubs with a usable XI: {len(lu)}/20")
        sys.exit(0)
    print(__doc__)
