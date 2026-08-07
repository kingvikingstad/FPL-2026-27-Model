"""
signals.py — leading-indicator signals that sharpen single-gameweek accuracy
=============================================================================
These convert pre-match information into TIGHTER PRIORS in the Bayesian model,
which is what actually helps when the sample is one gameweek. Sources are FREE
unless flagged.

  FREE (no key):
   - FPL API bootstrap-static : availability (chance_of_playing, status, news),
     ep_next, and set-piece order (penalties / free-kicks / corners).
     https://fantasy.premierleague.com/api/bootstrap-static/
   - FPL API team/set-piece-notes : confirmation notes per team.
   - ClubElo : dynamic team strength.   http://api.clubelo.com/<Club or date>
   - Understat : shot-level xG (shot volume stabilises far faster than goals).
     via `understatapi` / `underdata` / `soccerdata`.
   - vaastav/Fantasy-Premier-League : multi-season per-GW history for priors.

  NOT FREE / key-gated (wired as optional hooks, not defaults):
   - Player-prop odds (anytime goalscorer, shots o/u) — sharpest single-match
     attacking signal, but paid or free-tier-with-key.
   - Pre-match predicted/confirmed-lineup feeds — some free tiers need a key.
     (Confirmed XIs also appear ~1h pre-KO in several APIs.)

All loaders require network and run in the user's environment; the prior-
modification functions run anywhere and are what the model actually consumes.
"""
from __future__ import annotations
import numpy as np, pandas as pd

FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_SETPIECE  = "https://fantasy.premierleague.com/api/team/set-piece-notes/"
CLUBELO_API   = "http://api.clubelo.com/"          # + ClubName or YYYY-MM-DD

# league-average penalty supply: a team wins ~0.13 pens/game; primary taker
# converts ~0.79 -> ~0.10 pen goals/90 for the designated taker (tunable).
PEN_XG90_PRIMARY = 0.10
PEN_CONVERSION   = 0.79


# =============================================================================
# FREE LOADERS  (network; run in your environment)
# =============================================================================
def load_fpl_signals(bootstrap: dict | str) -> pd.DataFrame:
    """From the FPL bootstrap-static JSON: availability + set-piece order + ep_next.
    Accepts a parsed dict or a path to a saved JSON. FREE, no key."""
    if isinstance(bootstrap, str):
        import json
        with open(bootstrap) as f:
            bootstrap = json.load(f)
    rows = []
    for e in bootstrap["elements"]:
        cop = e.get("chance_of_playing_next_round")
        rows.append({
            "name": e["web_name"],
            "status": e.get("status", "a"),           # a/d/i/s/u/n
            "chance_play": (100 if cop is None else cop) / 100.0,
            "news": e.get("news", ""),
            "ep_next": float(e.get("ep_next", 0) or 0),
            "pen_order": e.get("penalties_order"),
            "fk_order": e.get("direct_freekicks_order"),
            "corner_order": e.get("corners_and_indirect_freekicks_order"),
        })
    return pd.DataFrame(rows)


CLUBELO_NORM = {
    "Man City": "Man City", "Man United": "Man United", "Tottenham": "Tottenham",
    "Nottingham": "Nott'm Forest", "Forest": "Nott'm Forest",
    "Bournemouth": "Bournemouth", "Brighton": "Brighton", "Newcastle": "Newcastle",
    "Leeds": "Leeds", "Coventry": "Coventry", "Hull": "Hull", "Ipswich": "Ipswich",
}
def _norm_elo(name):
    return CLUBELO_NORM.get(str(name).strip(), str(name).strip())


def _name_key(name):
    """Normalise a player name for cross-source joins: strip accents, drop
    punctuation, expand initial-dotted forms ('B.Fernandes' -> 'b fernandes'),
    collapse whitespace, lowercase. Understat, FPL and FM all spell the same
    player differently, and this is the join key that makes them agree."""
    import re, unicodedata
    s = str(name or "").strip()
    if not s:
        return ""
    # decompose accents: Ødegaard -> Odegaard, Martínez -> Martinez
    s = s.replace("ø", "o").replace("Ø", "O").replace("ð", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace(".", " ").replace("-", " ").replace("'", "")
    s = re.sub(r"[^A-Za-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def load_clubelo(when="today", club=None, timeout=45, retries=3):
    """ClubElo ratings. FREE. Snapshot for a date (all clubs) or one club's
    history. Returns a frame with normalised `team` and `Elo`. Requires network.

    ClubElo's endpoint is frequently slow to first byte (it renders the CSV on
    demand), so a short single-shot timeout drops the source unnecessarily. We
    allow a generous timeout and retry with backoff before giving up — losing
    ClubElo costs real information in the team prior, especially preseason when
    it is the only forward-looking team signal available."""
    import io, time, urllib.request
    from datetime import date, timedelta

    # Endpoint semantics (learned the hard way):
    #  * there is NO "today" endpoint — the literal string 404s;
    #  * a date only resolves if ClubElo has published that day's snapshot, so a
    #    machine running ahead of their publishing clock (or simply in a timezone
    #    east of theirs) asks for a date that does not exist yet and gets a 404;
    #  * https can return an empty body where http returns the CSV.
    # So: walk backwards a few days over both schemes and take the first real hit.
    if club:
        candidates = [club.replace(" ", "")]
    elif str(when).lower() in ("today", "now", "", "none"):
        today = date.today()
        candidates = [(today - timedelta(days=d)).isoformat() for d in range(0, 8)]
    else:
        candidates = [str(when)]

    bases = [CLUBELO_API]
    if CLUBELO_API.startswith("http://"):
        bases.append("https://" + CLUBELO_API[len("http://"):])

    last = None
    for endpoint in candidates:
        for base in bases:
            req = urllib.request.Request(base + endpoint,
                                         headers={"User-Agent": "Mozilla/5.0"})
            for attempt in range(retries):
                try:
                    with urllib.request.urlopen(req, timeout=timeout) as r:
                        payload = r.read()
                    if not payload.strip():
                        raise ValueError("empty body")
                    df = pd.read_csv(io.BytesIO(payload))
                    if not len(df) or "Club" not in df.columns:
                        raise ValueError("no Club column / no rows")
                    df["team"] = df["Club"].map(_norm_elo)
                    df.attrs["clubelo_date"] = endpoint
                    return df
                except Exception as e:               # noqa: BLE001 - try next candidate
                    last = e
                    # only worth retrying transport hiccups, not a 404/empty
                    if isinstance(e, ValueError) or "HTTP Error 4" in str(e):
                        break
                    if attempt < retries - 1:
                        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"ClubElo unreachable for {candidates[0]}..{candidates[-1]}: {last}")


def load_football_data(season_code):
    """Latest results+odds CSV from football-data.co.uk (FREE). `season_code`
    like '2526' for 2025/26. Returns the raw frame; save to a path for the model."""
    import io, urllib.request
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
    with urllib.request.urlopen(url, timeout=20) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def load_understat_shots(season="2025"):
    """Per-player shot volume & xG from Understat. FREE via `understatapi`.
    Shot volume/quality stabilises in ~5-6 games vs ~15+ for goals — the key
    small-sample win. Requires network + `pip install understatapi`."""
    from understatapi import UnderstatClient
    with UnderstatClient() as u:
        players = u.league(league="EPL").get_player_data(season=season)
    df = pd.DataFrame(players)
    for c in ["shots", "xG", "npg", "npxG", "key_passes", "time", "assists", "goals"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =============================================================================
# PRIOR-MODIFICATION LAYER  (runs anywhere; this is what the model consumes)
# =============================================================================
def apply_availability(players: pd.DataFrame, signals: pd.DataFrame,
                       lineups: dict | None = None, tighten=25.0) -> pd.DataFrame:
    """Rewrite the Beta start prior from current availability. This is the single
    biggest single-GW lever: it moves the minutes gate from a season-average
    guess to what we actually know this week.

    - status i/s/u (injured/suspended/unavailable) or chance_play==0 -> ~no start
    - status d (doubtful): expected start scaled by chance_play, prior tightened
    - status a: keep historical Beta (optionally nudged by chance_play)
    - `lineups` (optional): {team: {"start":[names], "bench":[names]}} collapses
      the prior to near-certainty once an XI is confirmed (~1h pre-KO).
    """
    p = players.copy()
    sig = signals.set_index(signals.name.str.lower().str.strip())
    for i, r in p.iterrows():
        key = str(r.web_name).lower().strip()
        # historical mean start prob from the existing Beta
        hist = r.start_a / (r.start_a + r.start_b)
        new_p, strength = hist, r.start_a + r.start_b
        if key in sig.index:
            s = sig.loc[key]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            if s.status in ("i", "s", "u", "n") or s.chance_play <= 0.0:
                new_p, strength = 0.01, 40.0                 # ruled out
            elif s.status == "d" or s.chance_play < 1.0:
                new_p = hist * s.chance_play                 # doubtful -> scale
                strength = tighten                           # we now have info
            # status 'a' with chance 1.0 -> leave historical prior
        # confirmed lineup overrides everything (near-degenerate)
        if lineups and r.team in lineups:
            lu = lineups[r.team]
            if r.web_name in lu.get("start", []):
                new_p, strength = 0.99, 200.0
            elif r.web_name in lu.get("bench", []):
                new_p, strength = max(r.sub_app_rate, 0.05), 60.0
            elif lu.get("start"):                            # named XI, not in it
                new_p, strength = 0.01, 200.0
        p.at[i, "start_a"] = float(np.clip(new_p, 1e-3, 1) * strength)
        p.at[i, "start_b"] = float(np.clip(1 - new_p, 1e-3, 1) * strength)
    return p


def apply_setpieces(players: pd.DataFrame, signals: pd.DataFrame,
                    pen_xg90=PEN_XG90_PRIMARY) -> pd.DataFrame:
    """Designated penalty takers get an additive penalty-goal rate (FPL's npxGI
    is non-penalty, so this is not double counting). Primary FK/corner takers get
    a modest assist-rate bump. Set-piece role is stable and materially shifts a
    player's goal/assist expectation beyond open-play xG."""
    p = players.copy()
    p["pen_xg90"] = 0.0
    sig = signals.set_index(signals.name.str.lower().str.strip())
    for i, r in p.iterrows():
        key = str(r.web_name).lower().strip()
        if key not in sig.index:
            continue
        s = sig.loc[key]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[0]
        if s.get("pen_order") == 1:
            p.at[i, "pen_xg90"] = pen_xg90                    # ~+0.10 goals/90
        # primary dead-ball taker -> small assist-rate prior bump (+10%)
        if s.get("fk_order") == 1 or s.get("corner_order") == 1:
            p.at[i, "xa_alpha"] = r.xa_alpha * 1.10
    return p


def apply_clubelo_prior(team_prior_means: pd.DataFrame, elo: pd.DataFrame,
                        weight=0.4) -> pd.DataFrame:
    """Blend ClubElo (dynamic, updates every match) into the team attack/defence
    PRIOR means. Helps most early season, when last year's averages are stale.
    `team_prior_means` indexed by team with columns [att, def]; `elo` has
    [team, Elo]. Returns blended means (centred log-Elo mapped to strength)."""
    e = elo.set_index("team")["Elo"]
    le = np.log(e / e.mean())
    out = team_prior_means.copy()
    for t in out.index:
        if t in le.index:
            out.loc[t, "att"] = (1 - weight) * out.loc[t, "att"] + weight * le[t]
            out.loc[t, "def"] = (1 - weight) * out.loc[t, "def"] + weight * le[t]
    return out


def apply_understat_prior(players: pd.DataFrame, ushots: pd.DataFrame,
                          blend=0.5) -> pd.DataFrame:
    """Faster-stabilising attacking prior: build involvement from SHOT VOLUME x
    mean xG/shot (repeatable) rather than realised goals (noisy), and blend with
    the FPL-derived npxGI prior. Matches on lowercased player name."""
    p = players.copy()
    if ushots is None or not len(ushots):
        return p
    u = ushots.copy()
    # A season that has not kicked off returns an EMPTY player list with no
    # columns at all, so guard on schema rather than assuming shape.
    need = ["time", "shots", "npxG", "key_passes", "player_name"]
    missing = [c for c in need if c not in u.columns]
    if missing:
        return p
    for c in ["time", "shots", "npxG", "key_passes"]:
        u[c] = pd.to_numeric(u[c], errors="coerce")
    u = u.dropna(subset=["time", "shots"])
    if not len(u):
        return p
    u["nnf"] = u["time"] / 90.0
    u["shotvol90"] = u["shots"] / u["nnf"].clip(lower=1e-6)
    u["xgpsh"] = (u["npxG"] / u["shots"].clip(lower=1)).clip(0.03, 0.3)
    u["inv90_understat"] = (u["shotvol90"] * u["xgpsh"] +
                            u["key_passes"] / u["nnf"].clip(lower=1e-6) * 0.08)

    # NAME MATCHING. Understat stores full names ("Mohamed Salah", "Martin
    # Odegaard"); FPL's web_name is usually just the surname, sometimes
    # initial-dotted ("B.Fernandes"), and either side may carry accents. Exact
    # lowercase matching therefore finds only a small fraction of the squad and
    # silently discards the sharpest attacking signal we have. Match on a
    # normalised full name first, then fall back to surname.
    full, sur = {}, {}
    for _, r in u.iterrows():
        v = r["inv90_understat"]
        if not np.isfinite(v):
            continue
        mins = float(r.get("time", 0) or 0)
        f = _name_key(r["player_name"])
        if f:
            # keep the higher-minutes player on collision (the real starter)
            if f not in full or mins > full[f][1]:
                full[f] = (float(v), mins)
        s = f.split()[-1] if f else ""
        if s:
            if s not in sur:
                sur[s] = [(float(v), mins)]
            else:
                sur[s].append((float(v), mins))

    n = 0
    for i, r in p.iterrows():
        nm = _name_key(r.web_name)
        if not nm:
            continue
        val = None
        if nm in full:
            val = full[nm][0]
        else:
            s = nm.split()[-1]
            cands = sur.get(s, [])
            # only trust a surname match when it is UNAMBIGUOUS — several PL
            # players share surnames (two Jameses, two Hendersons), and a wrong
            # join here quietly corrupts a player's attacking prior.
            if len(cands) == 1:
                val = cands[0][0]
        if val is None or not np.isfinite(val):
            continue
        hist_mean = r.npxgi_alpha / r.npxgi_beta
        blended = (1 - blend) * hist_mean + blend * val
        p.at[i, "npxgi_alpha"] = max(blended, 1e-3) * r.npxgi_beta
        n += 1
    p.attrs["understat_matched"] = n
    return p


# =============================================================================
# NOT-FREE HOOK — player-prop odds fusion (supply your own odds)
# =============================================================================
def fuse_prop_odds(players: pd.DataFrame, ags_odds: dict, devig=1.08) -> pd.DataFrame:
    """OPTIONAL (paid data). Fuse anytime-goalscorer odds into the per-player goal
    prior for a specific fixture. `ags_odds` = {player_name: decimal_odds}. We
    de-vig crudely (divide implied prob by an overround factor) and set the goal
    prior so E[goal] matches the market's P(scores) ~ 1 - exp(-lambda_goal).
    This is the sharpest single-match attacking signal when you can get it."""
    p = players.copy()
    if "pen_xg90" not in p:
        p["pen_xg90"] = 0.0
    for i, r in p.iterrows():
        o = ags_odds.get(r.web_name)
        if not o:
            continue
        p_score = np.clip((1.0 / o) / devig, 1e-3, 0.95)
        lam_goal = -np.log(1 - p_score)                  # P(>=1 goal) -> mean goals
        # map to a per-90 open-play rate (assume ~1 start); keep pens separate
        k0 = r.npxgi_beta
        p.at[i, "npxgi_alpha"] = max(lam_goal, 1e-3) * k0
    return p


if __name__ == "__main__":
    print("signals.py loaded. Free loaders: load_fpl_signals, load_clubelo, "
          "load_understat_shots. Prior mods: apply_availability, apply_setpieces, "
          "apply_clubelo_prior, apply_understat_prior. Paid hook: fuse_prop_odds.")
