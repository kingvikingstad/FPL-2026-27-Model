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


def load_clubelo(when="today", club=None):
    """ClubElo ratings. FREE. Snapshot for a date (all clubs) or one club's
    history. Returns a frame with normalised `team` and `Elo`. Requires network."""
    import io, urllib.request
    url = CLUBELO_API + (club.replace(" ", "") if club else str(when))
    with urllib.request.urlopen(url, timeout=20) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    if "Club" in df:
        df["team"] = df["Club"].map(_norm_elo)
    return df


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
FPL_BOOTSTRAP = "https://fantasy.premierleague.com/api/bootstrap-static/"


def fetch_live_signals(url=FPL_BOOTSTRAP, timeout=30):
    """Availability straight from the live FPL endpoint, keyed on `player_code`.

    WHY THIS EXISTS. The board is normally built from the data-repo snapshot, which is
    refreshed on a cron and can trail team news by days. Measured 2026-08-19 against a
    snapshot from 2026-08-14: 36 players the snapshot still treats as available were
    projected a combined 356 points over GW1-6 while the live endpoint had them out —
    and many were not injuries at all but completed transfers (Romero to Atletico,
    Spence to Internazionale, Vicario to Juventus, Unal to Getafe). A departed player
    cannot be priced by a rotation prior; he has to be removed.

    Returns a signals-shaped frame [name, player_code, status, chance_play, news,
    ep_next] usable directly by `apply_availability`. Raises on network failure so the
    caller decides whether to fall back — silently returning a stale frame is how the
    above happened in the first place.
    """
    import json, urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    chance = lambda v: 1.0 if v is None else max(0.0, min(1.0, float(v) / 100.0))
    return pd.DataFrame([{
        "name": e.get("web_name"),
        "player_code": e.get("code"),
        "status": e.get("status", "a") or "a",
        "chance_play": chance(e.get("chance_of_playing_next_round")),
        "news": (e.get("news") or "").strip(),
        "ep_next": pd.to_numeric(e.get("ep_next"), errors="coerce"),
    } for e in data["elements"]])


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

    Joins on `player_code` when BOTH frames carry it, falling back to lowercased
    web_name otherwise. The name fallback is genuinely ambiguous — 15 web_names in the
    26/27 squad belong to players at two different clubs (Palmer is at Chelsea and at
    Ipswich), so a name-keyed availability lookup can rule out the wrong player.
    """
    p = players.copy()
    use_code = ("player_code" in signals.columns and "player_code" in p.columns
                and signals["player_code"].notna().any())
    if use_code:
        sig = signals.dropna(subset=["player_code"]).drop_duplicates("player_code")
        sig = sig.set_index(sig["player_code"])
    else:
        sig = signals.set_index(signals.name.str.lower().str.strip())
    for i, r in p.iterrows():
        key = r.get("player_code") if use_code else str(r.web_name).lower().strip()
        # historical mean start prob from the existing Beta
        hist = r.start_a / (r.start_a + r.start_b)
        new_p, strength = hist, r.start_a + r.start_b
        if key is not None and key in sig.index:
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
    u = ushots.copy()
    u["nnf"] = u["time"] / 90.0
    u["shotvol90"] = u["shots"] / u["nnf"].clip(lower=1e-6)
    u["xgpsh"] = (u["npxG"] / u["shots"].clip(lower=1)).clip(0.03, 0.3)
    u["inv90_understat"] = (u["shotvol90"] * u["xgpsh"] +
                            u["key_passes"] / u["nnf"].clip(lower=1e-6) * 0.08)
    key = u.assign(k=u["player_name"].str.lower().str.strip()).set_index("k")["inv90_understat"]
    for i, r in p.iterrows():
        nm = str(r.web_name).lower().strip()
        if nm in key.index and np.isfinite(key[nm]):
            hist_mean = r.npxgi_alpha / r.npxgi_beta
            blended = (1 - blend) * hist_mean + blend * float(key[nm])
            k0 = r.npxgi_beta
            p.at[i, "npxgi_alpha"] = max(blended, 1e-3) * k0
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
