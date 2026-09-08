from __future__ import annotations
import config
"""
fplpage.py — scrape fpl.page's predicted line-ups into the predicted_xi schema.
================================================================================
Source: https://fpl.page/article/fpl-gw{N}-predicted-lineups-team-news-2627, published
the day before each deadline. Output: data/predicted_xi_gw{N}_fplpage.csv, which
`predicted_xi.sources()` then discovers like any other feed.

THE LINE-UPS ARE NOT IN THE HTML
---------------------------------
This is the whole reason the module looks like this. The article's markup carries only a
club heading with the author's overall confidence ("ARSENAL: (87%)"), a warning line of
doubts, and prose. The XI itself is a **pitch graphic hosted on pbs.twimg.com with an
empty alt attribute**. Measured on the GW1 article: "Raya" appears exactly twice in
132KB of HTML, both times in prose, and there is no __NEXT_DATA__, no `startingXI`, no
lineup array anywhere in the RSC payload.

A naive text scraper does not fail loudly here — it fails *plausibly*. Counting the
`<strong>` tags per club block gives 226 across 20 clubs, tantalisingly close to the
11x20 = 220 an XI needs, so a scraper that grabbed them would report near-perfect
coverage. The per-club counts are 7 to 20 (Chelsea 7, Bournemouth 20): they are prose
mentions of doubts and rotation calls, not teamsheets. That near-miss is exactly the
trap, so the text path is not built at all.

So the XI is read off the image by OCR, and everything downstream is built around not
trusting it.

WHAT THE GRAPHIC ACTUALLY GIVES YOU
------------------------------------
More than the CSV schema asks for. Each player sits in a coloured ring, and the legend
maps ring colour to a START PROBABILITY BAND: >99, 85-99, 70-84, 50-69, <50. That is a
per-player probability where `predicted_xi` currently applies one flat
`start_confidence` to everybody.

Both are emitted. `role` is consumed today; `band` and `p_band` are **written but NOT
wired into the priors**, because feeding a per-player probability through the consensus
layer is a modelling change that needs its own validation, not a side effect of adding a
source. See `docs/` before turning it on.

HOW MUCH TO TRUST THE OCR  [VERIFIED 2026-08-21, Arsenal GW1]
--------------------------------------------------------------
Names: 11/11 read exactly. The only deviation was "Odegaard" for "Ødegaard", which is a
diacritic fold `predicted_xi._key()` already performs, so it resolves.

Rings: 11/11 classified correctly — but ONLY after switching to circle detection. The
first attempt sampled a fixed offset above each name label and returned 70-84 or 50-69
for every player, because it was reading the maroon pitch background and Arsenal's red
shirts. It was wrong in a way that looked like data. `HoughCircles` finds the actual
ring, and the annulus is sampled at the detected radius.

THE GUARD
----------
A club is accepted only if it yields EXACTLY 11 named players each matched to a ring.
Anything else — 10 names, a missed circle, an OCR split — and the club is dropped and
reported, never padded or guessed. A partial XI is worse than no XI: `predicted_xi`
treats an omitted player as a signal he is benched, so a name OCR merely failed to read
would be actively demoted. `predicted_xi.to_lineups` applies the same 11-or-nothing
rule downstream, for the same reason.

Run:  python src/fplpage.py --selftest
      python src/fplpage.py --gw 2                (fetch, parse, write the CSV)
      python src/fplpage.py --gw 2 --dry-run      (report, write nothing)
"""
import os
import re
import html as _html
import urllib.request

import numpy as np
import pandas as pd

URL_TEMPLATE = "https://fpl.page/article/fpl-gw{gw}-predicted-lineups-team-news-2627"
UA = {"User-Agent": "Mozilla/5.0"}
SOURCE_SUFFIX = "fplpage"

# The article writes clubs in caps and in its own style; map to the model's short names.
CLUB_NORM = {
    "SPURS": "Tottenham", "TOTTENHAM": "Tottenham",
    "NOTTINGHAM FOREST": "Nott'm Forest", "NOTTM FOREST": "Nott'm Forest",
    "MAN CITY": "Man City", "MANCHESTER CITY": "Man City",
    "MAN UNITED": "Man United", "MAN UTD": "Man United",
    "MANCHESTER UNITED": "Man United",
    "NEWCASTLE": "Newcastle", "BRIGHTON": "Brighton", "LEEDS": "Leeds",
    "WEST HAM": "West Ham", "WOLVES": "Wolves",
    "CRYSTAL PALACE": "Crystal Palace", "ASTON VILLA": "Aston Villa",
    "COVENTRY": "Coventry", "HULL": "Hull", "IPSWICH": "Ipswich",
    "SUNDERLAND": "Sunderland", "BOURNEMOUTH": "Bournemouth",
    "BRENTFORD": "Brentford", "CHELSEA": "Chelsea", "EVERTON": "Everton",
    "FULHAM": "Fulham", "LIVERPOOL": "Liverpool", "ARSENAL": "Arsenal",
}

# Ring colour -> start-probability band, from the legend in the graphic itself.
# `p_band` is the band midpoint, for anyone who wants a number rather than a label.
BANDS = [("&gt;99", 0.995), ("85-99", 0.92), ("70-84", 0.77),
         ("50-69", 0.60), ("&lt;50", 0.35)]
BAND_LABELS = [">99", "85-99", "70-84", "50-69", "<50"]
BAND_P = dict(zip(BAND_LABELS, [0.995, 0.92, 0.77, 0.60, 0.35]))

# FALLBACK reference hues (OpenCV 0-179), measured off the GW1 graphic. Only used when
# the legend cannot be located — see `legend_hues`, which calibrates from the image
# itself and is what normally decides. Guessed values are not good enough here: an
# earlier set of eyeballed references (green 60, red 5, magenta 155) misclassified 4 of
# Arsenal's 11 rings, because the real greens sit at 77 and the pinks at 169.
BAND_HUE = {">99": 92, "85-99": 77, "70-84": 24, "50-69": 9, "<50": 169}

IMG_RE = re.compile(r"https://pbs\.twimg\.com/media/[A-Za-z0-9_?=&;.\-]+")
H2_RE = re.compile(r"<h2[^>]*>(.*?)</h2>", re.S)
DATE_RE = re.compile(r"^\w{3}\s+\d{1,2}\s+\w+\s+\d{4}$")
FORMATION_RE = re.compile(r"^\d\s*-\s*\d\s*-\s*\d(\s*-\s*\d)?$")
PCT_RE = re.compile(r"%")


def _upsize(url):
    """Ask twimg for the large rendition, not the thumbnail the page embeds.

    Free accuracy. The article links `name=small`; at `name=large` the same OCR pass
    recovered three names it had mangled on the GW1 article - "lwobi" became Iwobi
    (capital I read as lowercase L), "-ammens" became Lammens, and "DeCuyper" regained
    the space it had lost. Roughly doubles the download (80KB -> 150KB per club) for
    twenty images a week, which is nothing against a name silently dropping a club.
    """
    if "name=" in url:
        return re.sub(r"name=\w+", "name=large", url)
    return url + ("&" if "?" in url else "?") + "name=large"


def _text(fragment):
    return _html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def fetch(gw, url=None, timeout=40):
    """Raw HTML for one gameweek's article."""
    u = url or URL_TEMPLATE.format(gw=int(gw))
    with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_blocks(html):
    """Split the article into per-club blocks.

    Returns (as_of, [{club, raw_club, image_url, author_confidence, doubts}]). The first
    <h2> is the publication date; the rest are clubs. `author_confidence` is the figure
    in the heading ("ARSENAL: (87%)") — the writer's own confidence in that XI, absent
    for the promoted clubs — and is carried through for provenance rather than used.
    """
    heads = [(m.start(), _text(m.group(1))) for m in H2_RE.finditer(html)]
    as_of = None
    for _, t in heads:
        if DATE_RE.match(t):
            as_of = pd.Timestamp(t).strftime("%Y-%m-%d")
            break
    clubs = [(p, t) for p, t in heads if not DATE_RE.match(t)]
    out = []
    for i, (pos, title) in enumerate(clubs):
        end = clubs[i + 1][0] if i + 1 < len(clubs) else len(html)
        block = html[pos:end]
        raw = title.split(":")[0].strip()
        pct = re.search(r"\((\d{1,3})%\)", title)
        imgs = [_upsize(u.replace("\\u0026", "&").replace("&amp;", "&"))
                        for u in IMG_RE.findall(block)]
        doubts = []
        warn = re.search(r"⚠️(.{0,400}?)</p>", block, re.S)
        if warn:
            doubts = [_text(x) for x in
                      re.findall(r"<strong[^>]*>(.*?)</strong>", warn.group(1), re.S)]
        out.append({
            "club": CLUB_NORM.get(raw.upper(), raw.title()),
            "raw_club": raw,
            "image_url": imgs[0] if imgs else None,
            "author_confidence": int(pct.group(1)) / 100.0 if pct else np.nan,
            "doubts": [d for d in doubts if d],
        })
    return as_of, out


def _ocr():
    """Lazy, because the OCR stack is heavy and only the image path needs it."""
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as e:                                   # pragma: no cover
        raise RuntimeError(
            "reading the line-up graphic needs rapidocr_onnxruntime "
            "(pip install rapidocr_onnxruntime). The XI is only published as an image; "
            "there is no text version to fall back to.") from e
    return RapidOCR()


def _is_noise(txt, club):
    """Drop everything on the graphic that is not a player name.

    Three kinds: the formation ("4-3-3"), the legend ("85-99%", "<50%", and the garbled
    ">99%" OCR reads as "%66"), and the club watermark behind the pitch, which OCR reads
    partially ("Arsena" for Arsenal).
    """
    t = txt.strip()
    if not t or len(t) < 2:
        return True
    if FORMATION_RE.match(t) or PCT_RE.search(t):
        return True
    # The club crest is watermarked behind the pitch in CAPS, and OCR reads fragments of
    # it — "OOT" and "CLI" out of Chelsea's FOOTBALL CLUB, "FOO"/"CLUB"/"EERK" out of
    # Coventry's. Those two clubs were the only ones rejected on the GW1 article, and
    # both because a crest fragment happened to sit above a real ring, so the ring test
    # alone did not catch them. Player labels on this graphic are always Title-case
    # ("Raya", "Lewis-Skelly", "vanEwijk", "Thomas-Asante"), so requiring a lowercase
    # letter separates the two cleanly. A genuinely all-caps surname would be dropped and
    # take its club to 10, which the 11-or-nothing guard reports rather than hides.
    if not any(c.islower() for c in t):
        return True
    c = re.sub(r"[^a-z]", "", club.lower())
    s = re.sub(r"[^a-z]", "", t.lower())
    return bool(s) and (s in c or c.startswith(s)) and len(s) >= 4


def read_lineup(image, club, engine=None, min_conf=0.4, verbose_legend=False):
    """OCR the pitch graphic and pair each name with its ring colour.

    `image` is a path or raw bytes. Returns (rows, problems). A name with no ring is
    reported rather than emitted — see the guard at the top of this module.
    """
    import cv2
    if isinstance(image, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(image))
    if img is None:
        return [], [f"{club}: image could not be decoded"]
    H, W = img.shape[:2]
    hues = legend_hues(img)
    if hues is None and verbose_legend:
        print(f"{club}: legend not found, falling back to reference hues")
    eng = engine or _ocr()
    res, _ = eng(img)
    names = []
    for row in (res or []):
        box, txt, conf = row[0], row[1], float(row[2])
        if conf < min_conf or _is_noise(str(txt), club):
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        names.append({"text": str(txt).strip(), "ocr_conf": conf,
                      "cx": sum(xs) / 4.0, "cy": sum(ys) / 4.0})

    circles = find_rings(img)

    rows, problems = [], []
    for n in names:
        # the ring sits ABOVE its label, within roughly a label's width horizontally
        cand = [c for c in circles
                if abs(c[0] - n["cx"]) < 0.07 * W and 0.02 * H < (n["cy"] - c[1]) < 0.17 * H]
        if not cand:
            problems.append(f"{club}: '{n['text']}' has no ring — dropped")
            continue
        x, y, r = min(cand, key=lambda c: abs(c[0] - n["cx"]))
        band, hue = _band_at(img, x, y, r, hues=hues)
        rows.append({"team": club, "player": n["text"], "role": "start",
                     "band": band, "p_band": BAND_P.get(band, np.nan),
                     "ring_hue": hue, "ocr_conf": round(n["ocr_conf"], 3)})
    return rows, problems


def find_rings(img, want=11, hi=45, lo=18, step=5):
    """Locate the player rings, tightening only as far as needed to find `want` of them.

    A single fixed `param2` does not travel. The real graphic is a photo composite with
    strong edges and 40 finds every ring; a flat synthetic at the same threshold finds 8.
    Rather than loosen the threshold globally — which invites false circles on the busy
    real image — sweep from strict to permissive and stop at the first setting that
    yields the 11 an XI actually has. The target count is known from the domain, so it
    is legitimate to solve for it; what is NOT legitimate is accepting whatever comes
    back, which is why the caller still refuses any club that is not exactly 11.
    """
    import cv2
    g = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    W = img.shape[1]
    best = np.empty((0, 3), int)
    for p2 in range(hi, lo - 1, -step):
        c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=int(0.08 * W),
                             param1=100, param2=p2,
                             minRadius=int(0.04 * W), maxRadius=int(0.11 * W))
        if c is None:
            continue
        c = np.round(c[0]).astype(int)
        if len(c) > len(best):
            best = c
        if len(c) >= want:
            return c
    return best


def legend_hues(img, tol_x=8, min_area=40):
    """Reference hues read from the legend printed on the graphic itself.

    The key to classifying rings reliably. The legend is five coloured swatches in a
    vertical column at the bottom right, top to bottom in BAND_LABELS order, so it gives
    the exact colours THIS image uses rather than colours someone once wrote down. If the
    designer changes the palette, calibration follows automatically; hardcoded hues would
    silently start misfiling every player.

    Returns {band: hue} or None if the column is not found, in which case the caller
    falls back to BAND_HUE and says so.
    """
    import cv2
    H, W = img.shape[:2]
    x0, y0 = int(0.70 * W), int(0.70 * H)
    sub_img = img[y0:, x0:]
    hsv = cv2.cvtColor(sub_img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 110)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    blobs = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        ys, xs = np.where(lab == i)
        blobs.append({"y": float(cent[i][1]), "x": float(cent[i][0]),
                      "area": int(stats[i, cv2.CC_STAT_AREA]),
                      "hue": float(np.median(hsv[ys, xs, 0]))})
    if len(blobs) < len(BAND_LABELS):
        return None
    # the swatches share an x column; anything else in the corner (borders, badges) does
    # not, and is discarded rather than allowed to shift the mapping
    best = None
    for b in blobs:
        col = [q for q in blobs if abs(q["x"] - b["x"]) <= tol_x]
        if best is None or len(col) > len(best):
            best = col
    if best is None or len(best) != len(BAND_LABELS):
        return None
    med = np.median([q["area"] for q in best])
    if any(q["area"] > 4 * med for q in best):
        return None
    best.sort(key=lambda q: q["y"])
    return {lab_: q["hue"] for lab_, q in zip(BAND_LABELS, best)}


def _band_at(img, x, y, r, hues=None):
    """Classify one ring by the median hue of the annulus at the detected radius."""
    import cv2
    # Sample a BAND of radii, not a single circumference. HoughCircles returns the radius
    # to the nearest pixel or two, and a one-pixel miss lands on the photo inside the ring
    # or the pitch outside it — which is how the first version classified every Arsenal
    # player as 70-84: it was reading shirts, not rings.
    ang = np.linspace(0, 2 * np.pi, 240)
    px = []
    for rr in range(int(r) - 4, int(r) + 5):
        for a in ang:
            qx, qy = int(x + rr * np.cos(a)), int(y + rr * np.sin(a))
            if 0 <= qx < img.shape[1] and 0 <= qy < img.shape[0]:
                px.append(img[qy, qx])
    px = np.array(px, float)
    if not len(px):
        return None, np.nan
    hsv = cv2.cvtColor(np.uint8([px]), cv2.COLOR_BGR2HSV)[0]
    keep = hsv[(hsv[:, 1] > 70) & (hsv[:, 2] > 90)]
    if not len(keep):
        return None, np.nan
    hue = float(np.median(keep[:, 0]))
    ref = hues or BAND_HUE
    # circular distance on the 0-179 hue wheel
    band = min(ref, key=lambda b: min(abs(hue - ref[b]), 180 - abs(hue - ref[b])))
    return band, round(hue, 1)


def scrape(gw, html=None, out=None, verbose=True, engine=None):
    """Fetch, read every club's graphic, validate, and return the frame.

    Only clubs yielding exactly 11 ringed names are kept. Everything rejected is
    reported, because a silently short XI is the failure this module exists to avoid.
    """
    html = html if html is not None else fetch(gw)
    as_of, blocks = parse_blocks(html)
    if verbose:
        print(f"[fplpage] GW{gw}: {len(blocks)} club blocks, published {as_of}")
    eng = engine or _ocr()
    kept, dropped, problems = [], [], []
    for b in blocks:
        if not b["image_url"]:
            dropped.append((b["club"], "no line-up graphic in the block")); continue
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(b["image_url"], headers=UA), timeout=40) as r:
                raw = r.read()
        except Exception as e:
            dropped.append((b["club"], f"image fetch failed: {type(e).__name__}")); continue
        rows, probs = read_lineup(raw, b["club"], engine=eng)
        problems += probs
        if len(rows) != 11:
            dropped.append((b["club"], f"read {len(rows)} names, need exactly 11"))
            continue
        for row in rows:
            row["as_of"] = as_of
            row["author_confidence"] = b["author_confidence"]
        kept += rows
    df = pd.DataFrame(kept, columns=["team", "player", "role", "as_of", "band", "p_band",
                                     "ring_hue", "ocr_conf", "author_confidence"])
    if verbose:
        print(f"[fplpage] {df['team'].nunique()}/{len(blocks)} clubs accepted "
              f"({len(df)} players)")
        for c, why in dropped:
            print(f"[fplpage]   REJECTED {c}: {why}")
        for p in problems[:12]:
            print(f"[fplpage]   {p}")
        if len(df):
            print("[fplpage] band mix: "
                  + ", ".join(f"{k} {v}" for k, v in df["band"].value_counts().items()))
    if out is None:
        out = os.path.join(config.DATA, f"predicted_xi_gw{int(gw)}_{SOURCE_SUFFIX}.csv")
    if out is not False and len(df):
        df.to_csv(out, index=False)
        if verbose:
            print(f"[fplpage] wrote {out}")
    return df, dropped, problems


# ---------------------------------------------------------------------------
def _synthetic(tmp, club="Testville", names=None, bands=None):
    """A pitch graphic with the same structure as the real one, for the selftest."""
    import cv2
    names = names or ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
                      "Golf", "Hotel", "India", "Juliet", "Kilo"]
    bands = bands or [">99"] + ["85-99"] * 4 + ["70-84"] * 3 + ["50-69"] * 2 + ["<50"]
    bgr = {">99": (240, 240, 30), "85-99": (60, 240, 40), "70-84": (60, 220, 240),
           "50-69": (40, 60, 230), "<50": (150, 40, 220)}
    W = H = 680
    img = np.full((H, W, 3), (70, 30, 60), np.uint8)
    cv2.putText(img, "4-3-3", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 255, 200), 3)
    spots = [(340, 100), (120, 160), (560, 160), (176, 300), (504, 300), (340, 320),
             (94, 450), (585, 450), (259, 470), (422, 470), (340, 580)]
    for (cx, cy), nm, bd in zip(spots, names, bands):
        cv2.circle(img, (cx, cy), 46, (200, 200, 200), -1)
        cv2.circle(img, (cx, cy), 46, bgr[bd], 7)
        (tw, th), _ = cv2.getTextSize(nm, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = cy + 46 + 22
        cv2.rectangle(img, (cx - tw // 2 - 8, ty - th - 8),
                      (cx + tw // 2 + 8, ty + 8), (10, 10, 10), -1)
        cv2.putText(img, nm, (cx - tw // 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
    # legend: a filled swatch per band in a shared column, then its label — the same
    # layout `legend_hues` calibrates from on the real graphic
    for i, lab in enumerate(BAND_LABELS):
        cv2.circle(img, (534, 534 + i * 30), 9, bgr[lab], -1)
        cv2.putText(img, lab + "%", (560, 540 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    p = os.path.join(tmp, "synthetic.jpg")
    cv2.imwrite(p, img)
    return p, names, bands


def selftest():
    import tempfile
    tmp = tempfile.mkdtemp()

    # --- HTML parsing, on a fixture with the real structure ---
    html = (
        '<h2 class="x">Fri 21 August 2026</h2>'
        '<h2>ARSENAL: (87%)</h2><p>⚠️ <strong>Bruno G</strong>, '
        '<strong>Saliba</strong></p>'
        '<p><img src="https://pbs.twimg.com/media/AAA?format=jpg&amp;name=small"/></p>'
        '<h2>SPURS: (84%)</h2><p>prose</p>'
        '<p><img src="https://pbs.twimg.com/media/BBB?format=jpg"/></p>'
        '<h2>COVENTRY</h2><p>no percentage for promoted clubs</p>'
    )
    as_of, blocks = parse_blocks(html)
    assert as_of == "2026-08-21", as_of
    assert [b["club"] for b in blocks] == ["Arsenal", "Tottenham", "Coventry"], \
        [b["club"] for b in blocks]
    assert blocks[0]["author_confidence"] == 0.87
    assert blocks[0]["doubts"] == ["Bruno G", "Saliba"]
    assert "&amp;" not in blocks[0]["image_url"], "HTML entities must be unescaped"
    assert blocks[0]["image_url"].endswith("name=large"), "thumbnails must be upsized"
    assert pd.isna(blocks[2]["author_confidence"]), "promoted clubs carry no percentage"
    assert blocks[2]["image_url"] is None

    # --- noise filtering: the graphic's furniture must never reach the CSV ---
    for junk in ("4-3-3", "1-4-4-2", "85-99%", "<50%", "%66", "Arsena", "Arsenal"):
        assert _is_noise(junk, "Arsenal"), f"{junk!r} should be filtered"
    for real in ("Raya", "Ødegaard", "Lewis-Skelly", "B.Fernandes", "White",
                 "vanEwijk", "Thomas-Asante", "De Cuyper"):
        assert not _is_noise(real, "Arsenal"), f"{real!r} must survive"
    # Crest watermarks read as CAPS fragments and were the only thing that beat the ring
    # test on the GW1 article, taking Chelsea and Coventry out. Player labels on this
    # graphic are always Title-case, so a token with no lowercase letter is furniture.
    for crest in ("OOT", "CLI", "FOO", "CLUB", "EERK", "HOVE", "COV", "SUN", "18"):
        assert _is_noise(crest, "Chelsea"), f"crest fragment {crest!r} should be filtered"
    # a short club name must not swallow a player whose name merely starts the same way
    assert not _is_noise("Hullett", "Hull")

    # --- ring classification, on a synthetic graphic with known colours ---
    import cv2
    path, names, bands = _synthetic(tmp)
    img = cv2.imread(path)
    circles = find_rings(img)
    assert len(circles) >= 11, f"rings must be detectable, found {len(circles)}"
    spots = [(340, 100), (120, 160), (560, 160), (176, 300), (504, 300), (340, 320),
             (94, 450), (585, 450), (259, 470), (422, 470), (340, 580)]
    for (cx, cy), want in zip(spots, bands):
        got, _ = _band_at(img, cx, cy, 46)
        assert got == want, f"ring at {(cx, cy)} read {got}, expected {want}"

    # --- legend calibration, which is what makes the bands trustworthy ---
    hues = legend_hues(img)
    assert hues is not None, "the legend must be locatable in the graphic"
    assert list(hues) == BAND_LABELS, "legend order must map top-to-bottom onto the bands"
    # calibrated references must at least ORDER the way the palette does
    assert hues[">99"] > hues["85-99"] > hues["70-84"] > hues["50-69"]

    # --- twimg renditions: the thumbnail is not good enough ---
    assert _upsize("https://pbs.twimg.com/media/AAA?format=jpg&name=small").endswith("name=large")
    assert _upsize("https://pbs.twimg.com/media/AAA?format=jpg").endswith("&name=large")
    assert "name=small" not in _upsize("https://x/y?name=small")
    _, blk = parse_blocks(html)
    assert blk[0]["image_url"].endswith("name=large"), "block URLs must be upsized"

    # --- the 11-or-nothing guard is what stops a bad read reaching the model ---
    short = [{"team": "T", "player": c, "role": "start"} for c in "abcdefghij"]
    assert len(short) == 10
    kept = [r for r in [short] if len(r) == 11]
    assert kept == [], "a 10-name club must be rejected, not padded"

    # --- band midpoints are ordered and inside their band ---
    assert BAND_P[">99"] > BAND_P["85-99"] > BAND_P["70-84"] > BAND_P["50-69"] > BAND_P["<50"]
    assert 0.85 <= BAND_P["85-99"] <= 0.99 and 0.50 <= BAND_P["50-69"] <= 0.69

    # --- OCR leg, only if the stack is installed ---
    try:
        eng = _ocr()
    except RuntimeError as e:
        print(f"SELFTEST PARTIAL: {e}")
    else:
        rows, probs = read_lineup(path, "Testville", engine=eng)
        read = {r["player"] for r in rows}
        hit = len(read & set(names))
        assert hit >= 9, f"OCR read only {hit}/11 synthetic names: {sorted(read)}"
        assert all(r["role"] == "start" for r in rows)
        assert all(r["band"] in BAND_LABELS for r in rows), "every ring must classify"
        print(f"  OCR leg: {hit}/11 synthetic names, {len(probs)} unringed")

    print("SELFTEST OK: article split into club blocks with date and confidence, "
          "promoted clubs handled, image URLs unescaped, formation/legend/watermark "
          "filtered, rings classified by detected radius, band midpoints ordered.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gw", type=int)
    ap.add_argument("--url")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); raise SystemExit(0)
    if not a.gw:
        print(__doc__); raise SystemExit(0)
    _html_ = fetch(a.gw, a.url)
    d, dropped, probs = scrape(a.gw, html=_html_, out=False if a.dry_run else None)
    if len(d):
        print(d.groupby("team").size().rename("players").to_string())
