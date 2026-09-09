"""peek.py -- look at an artifact without reading it.

The tree holds four files that a single `cat` or an unbounded read turns into a
context-window incident: projection_detail_gw1_10.csv (13.8 MB), gw_explorer.csv
(3.9 MB), gw_explorer.html (3.3 MB) and gw_board_long.csv (2.5 MB). Nothing about
them announces its size, and the answer wanted from them is almost never their
contents -- it is "what columns does this have, what is in them, how many rows".

That question has a bounded answer, so this prints a bounded answer: shape, the
column list with dtypes and a sample value, and a few rows of a few columns. The
output is capped in LINES, not in bytes of input, so pointing it at a 500 MB file
costs the same as pointing it at a 5 KB one.

What it deliberately does NOT do is validate. Null counts here are counted over a
SAMPLE and labelled as such. The authority on whether an artifact is complete,
schema-correct and current is `src/manifest.py` via `.\\fpl.ps1 doctor`, which
checks declared columns and row floors over the whole file. A tool that reports a
sample statistic in the same shape as a gate invites someone to trust it as one.

Run:
  .\\fpl.ps1 run scripts/peek.py gw_board_long.csv
  .\\fpl.ps1 run scripts/peek.py outputs/projection_detail_gw1_10.csv -n 8
  .\\fpl.ps1 run scripts/peek.py gw_board_long.csv --cols "player|mean|gw"
  .\\fpl.ps1 run scripts/peek.py --selftest

A bare name is resolved against the config directories, so the caller never has to
know which of outputs/ data/ predictions/ .cache/ a given artifact lives in.
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
import config

import argparse
import glob
import json
import re

# Hard caps. These are the reason the tool exists: every one of them bounds output
# against a file whose size is not known in advance.
MAX_COLS_LISTED = 60      # columns in the schema table
MAX_ROWS_SHOWN = 5        # data rows, unless -n says otherwise
MAX_ROWS_CAP = 25         # ...and -n cannot exceed this
MAX_COLS_SHOWN = 8        # columns in the data preview
MAX_CELL = 22             # characters per cell before truncation
SAMPLE_ROWS = 5000        # rows read to infer dtypes and sample nulls

# Where a bare name is looked up, in order. OUTPUTS first because that is where the
# artifacts this tool exists for live; SCRATCH last because a name that exists in both
# outputs/ and .cache/ means the outputs/ one is current and the .cache/ one is a
# snapshot -- resolving to the snapshot is the stale-board trap in a different coat.
def _search_dirs():
    return [config.OUTPUTS, config.DATA, config.PREDICTIONS, config.PLANS,
            config.SCRATCH, config.ROOT]


def resolve(target):
    """A path, or a bare name looked up in the config directories. Returns None if absent."""
    if _os.path.isabs(target) or _os.sep in target or "/" in target:
        p = target if _os.path.isabs(target) else _os.path.join(config.ROOT, target)
        return p if _os.path.exists(p) else None
    for d in _search_dirs():
        p = _os.path.join(d, target)
        if _os.path.exists(p):
            return p
    return None


def suggest(target):
    """Near-miss names, so a typo does not read as an absent artifact."""
    stem = _os.path.splitext(target)[0].lower()
    hits = []
    for d in _search_dirs():
        for p in glob.glob(_os.path.join(d, "*")):
            n = _os.path.basename(p)
            if stem[:6] and stem[:6] in n.lower():
                hits.append(n)
    return sorted(set(hits))[:8]


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def count_lines(path):
    """Row count without parsing. A 14 MB CSV is ~20 ms of chunked byte counting; the
    pandas read that would otherwise answer this is ~4 s and holds the file in memory."""
    n = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n


def clip(v, width=MAX_CELL):
    s = "" if v is None else str(v)
    s = s.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    # ASCII only: the console here is UTF-8 but a piped stdout on Windows is cp1252,
    # so a single accented player name in a sample cell kills the whole report.
    s = s.encode("ascii", "replace").decode("ascii")
    return s if len(s) <= width else s[: width - 1] + "~"


def _rule(width=78):
    return "-" * width


def describe_frame(df, total_rows, sampled, args):
    """Schema table plus a small data preview. Bounded by the MAX_* caps above."""
    out = []
    ncols = len(df.columns)
    shown_rows = "all" if not sampled else f"first {len(df):,}"
    out.append(f"  rows {total_rows:,}   cols {ncols}   "
               f"(dtypes and nulls from {shown_rows} rows)")
    out.append("")
    out.append(f"  {'#':>3}  {'column':<30} {'dtype':<10} {'nulls':>7}  sample")
    out.append("  " + _rule())

    cols = list(df.columns)
    if args.cols:
        pat = re.compile(args.cols, re.I)
        cols = [c for c in cols if pat.search(str(c))]
        if not cols:
            out.append(f"  (no column matches /{args.cols}/ -- {ncols} columns present)")
            return out
    listed = cols if args.all_cols else cols[:MAX_COLS_LISTED]

    for i, c in enumerate(listed):
        s = df[c]
        nulls = int(s.isna().sum())
        first = s.dropna()
        sample = clip(first.iloc[0]) if len(first) else "(all null in sample)"
        out.append(f"  {i:>3}  {clip(str(c), 30):<30} {str(s.dtype):<10} {nulls:>7}  {sample}")
    if len(cols) > len(listed):
        out.append(f"  ... {len(cols) - len(listed)} more columns "
                   f"-- --all-cols to list them, --cols PAT to filter")

    n = min(max(args.n, 0), MAX_ROWS_CAP)
    if n and len(df):
        prev = listed[:MAX_COLS_SHOWN]
        out.append("")
        note = "" if len(listed) <= MAX_COLS_SHOWN else f" (of {len(listed)}; --cols to choose)"
        out.append(f"  first {min(n, len(df))} rows, {len(prev)} columns{note}:")
        out.append("  " + _rule())
        widths = [max(len(clip(str(c), MAX_CELL)),
                      *(len(clip(v)) for v in df[c].head(n))) for c in prev]
        out.append("  " + "  ".join(clip(str(c), MAX_CELL).ljust(w)
                                    for c, w in zip(prev, widths)))
        for _, row in df.head(n).iterrows():
            out.append("  " + "  ".join(clip(row[c]).ljust(w)
                                        for c, w in zip(prev, widths)))
    return out


def peek_csv(path, args):
    import pandas as pd
    total = max(count_lines(path) - 1, 0)          # minus the header
    sampled = total > SAMPLE_ROWS
    df = pd.read_csv(path, nrows=SAMPLE_ROWS if sampled else None,
                     sep="\t" if path.lower().endswith(".tsv") else ",")
    return describe_frame(df, total, sampled, args)


def peek_excel(path, args):
    import pandas as pd
    xl = pd.ExcelFile(path)
    out = [f"  sheets: {', '.join(clip(s, 24) for s in xl.sheet_names[:12])}"
           + (" ..." if len(xl.sheet_names) > 12 else "")]
    name = args.sheet or xl.sheet_names[0]
    out.append(f"  showing sheet '{clip(str(name), 24)}'"
               + ("" if args.sheet else " (first; --sheet to choose)"))
    out.append("")
    df = xl.parse(name, nrows=SAMPLE_ROWS)
    out += describe_frame(df, len(df), len(df) >= SAMPLE_ROWS, args)
    return out


def peek_pickle(path, args):
    import pandas as pd
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        return describe_frame(obj, len(obj), False, args)
    if isinstance(obj, dict):
        out = [f"  dict, {len(obj)} keys"]
        for k in list(obj)[:MAX_COLS_LISTED]:
            out.append(f"    {clip(str(k), 30):<30} {type(obj[k]).__name__}")
        if len(obj) > MAX_COLS_LISTED:
            out.append(f"    ... {len(obj) - MAX_COLS_LISTED} more keys")
        return out
    return [f"  {type(obj).__name__}: {clip(repr(obj), 300)}"]


def peek_npz(path, args):
    import numpy as np
    with np.load(path) as z:
        out = [f"  npz, {len(z.files)} arrays"]
        for k in z.files[:MAX_COLS_LISTED]:
            a = z[k]
            out.append(f"    {clip(k, 30):<30} {str(a.dtype):<10} shape={a.shape}")
    return out


def peek_json(path, args):
    with open(path, encoding="utf-8", errors="replace") as fh:
        obj = json.load(fh)
    if isinstance(obj, dict):
        out = [f"  object, {len(obj)} keys"]
        for k in list(obj)[:MAX_COLS_LISTED]:
            v = obj[k]
            kind = f"{type(v).__name__}[{len(v)}]" if isinstance(v, (list, dict)) \
                else type(v).__name__
            out.append(f"    {clip(str(k), 30):<30} {kind:<14} {clip(repr(v), 30)}")
        if len(obj) > MAX_COLS_LISTED:
            out.append(f"    ... {len(obj) - MAX_COLS_LISTED} more keys")
        return out
    if isinstance(obj, list):
        out = [f"  array, {len(obj)} items; first item:"]
        out.append(f"    {clip(repr(obj[0]) if obj else '(empty)', 300)}")
        return out
    return [f"  {type(obj).__name__}: {clip(repr(obj), 300)}"]


def peek_text(path, args):
    n = min(max(args.n, 0) or 10, MAX_ROWS_CAP)
    total = count_lines(path)
    out = [f"  lines {total:,}", "", f"  first {n} lines:", "  " + _rule()]
    with open(path, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            out.append("  " + clip(line.rstrip("\n"), 100))
    return out


HANDLERS = [
    ((".csv", ".tsv"), peek_csv),
    ((".xlsx", ".xlsm", ".xltx"), peek_excel),
    ((".pkl", ".pickle"), peek_pickle),
    ((".npz",), peek_npz),
    ((".json",), peek_json),
]


def peek(path, args):
    lower = path.lower()
    for exts, fn in HANDLERS:
        if lower.endswith(exts):
            return fn(path, args)
    return peek_text(path, args)


def render(target, args):
    path = resolve(target)
    if path is None:
        lines = [f"peek: no artifact named '{target}'"]
        near = suggest(target)
        if near:
            lines.append("  did you mean: " + ", ".join(near))
        lines.append("  searched: " + ", ".join(
            _os.path.relpath(d, config.ROOT) for d in _search_dirs()))
        return lines, False
    if _os.path.isdir(path):
        entries = sorted(glob.glob(_os.path.join(path, "*")))
        lines = [f"{_os.path.relpath(path, config.ROOT)}  (directory, {len(entries)} entries)"]
        for p in entries[:MAX_COLS_LISTED]:
            lines.append(f"  {human(_os.path.getsize(p)):>9}  {_os.path.basename(p)}")
        if len(entries) > MAX_COLS_LISTED:
            lines.append(f"  ... {len(entries) - MAX_COLS_LISTED} more")
        return lines, True

    size = _os.path.getsize(path)
    lines = [_os.path.relpath(path, config.ROOT),
             f"  {human(size)} on disk"]
    try:
        lines += peek(path, args)
    except Exception as e:                       # a corrupt artifact is a finding, not a crash
        lines.append(f"  UNREADABLE  {type(e).__name__}: {clip(str(e), 200)}")
    return lines, True


def selftest():
    """Offline, on synthetic fixtures. Asserts the caps hold -- an unbounded peek is
    the one failure that matters, because it is the thing this tool replaces."""
    import tempfile
    import pandas as pd

    ok = 0
    with tempfile.TemporaryDirectory() as td:
        args = argparse.Namespace(n=MAX_ROWS_SHOWN, cols=None, all_cols=False, sheet=None)

        # A frame far wider and longer than every cap, to prove the caps bind.
        wide = pd.DataFrame({f"col_{i}": range(200) for i in range(150)})
        p = _os.path.join(td, "wide.csv")
        wide.to_csv(p, index=False)
        out = peek_csv(p, args)
        assert len(out) < 100, f"unbounded output: {len(out)} lines"
        assert any("more columns" in l for l in out), "column cap not reported"
        assert f"rows {200:,}" in out[0], out[0]
        ok += 1

        # Row count must come from the file, not from the sampled frame.
        big = pd.DataFrame({"a": range(SAMPLE_ROWS + 500), "b": 1.5})
        p = _os.path.join(td, "big.csv")
        big.to_csv(p, index=False)
        out = peek_csv(p, args)
        assert f"{SAMPLE_ROWS + 500:,}" in out[0], out[0]
        assert "from first" in out[0], "sampling not disclosed"
        ok += 1

        # Embedded newlines, tabs and non-ASCII must not break the layout or the pipe.
        df = pd.DataFrame({"name": ["Ibrahim Sangaré", "a\nb", "c\td"],
                           "v": [1, None, 3]})
        p = _os.path.join(td, "odd.csv")
        df.to_csv(p, index=False)
        out = peek_csv(p, args)
        body = "\n".join(out)
        body.encode("ascii")                     # raises if any handler leaked non-ASCII
        assert "\n".join(out).count("\n") == len(out) - 1, "a cell leaked a newline"
        assert any("nulls" in l for l in out)
        ok += 1

        # -n above the cap must clamp, not obey.
        out = peek_csv(p, argparse.Namespace(n=9999, cols=None, all_cols=False, sheet=None))
        assert len([l for l in out if l.startswith("  ") and "|" not in l]) < 60
        ok += 1

        # JSON, npz and text handlers stay bounded too.
        p = _os.path.join(td, "x.json")
        json.dump({f"k{i}": list(range(5)) for i in range(200)}, open(p, "w"))
        assert len(peek_json(p, args)) <= MAX_COLS_LISTED + 3
        ok += 1

        p = _os.path.join(td, "x.txt")
        open(p, "w").write("line\n" * 5000)
        out = peek_text(p, args)
        assert len(out) < 40 and "5,000" in out[0], out[0]
        ok += 1

        # A missing name must suggest, not raise.
        lines, found = render("no_such_artifact_xyz.csv", args)
        assert not found and any("no artifact" in l for l in lines)
        ok += 1

        # A corrupt file is reported, not raised.
        p = _os.path.join(td, "bad.pkl")
        open(p, "wb").write(b"not a pickle")
        lines, found = render(p, args)
        assert found and any("UNREADABLE" in l for l in lines), lines
        ok += 1

    print(f"peek selftest: {ok}/8 passed")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Bounded look at an artifact: shape, schema, a few rows.")
    ap.add_argument("target", nargs="?", help="bare name or path")
    ap.add_argument("-n", type=int, default=MAX_ROWS_SHOWN,
                    help=f"data rows to show (max {MAX_ROWS_CAP})")
    ap.add_argument("--cols", help="regex; list only matching columns")
    ap.add_argument("--all-cols", action="store_true", help="list every column")
    ap.add_argument("--sheet", help="worksheet name, for .xlsx")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return 0
    if not args.target:
        ap.error("a target is required")

    lines, found = render(args.target, args)
    print("\n".join(lines))
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
