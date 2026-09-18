import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import config
"""
export_gw_explorer.py — build the interactive gameweek view over the current board
==================================================================================
A thin runner. All the logic, and its selftest, live in `src/gw_explorer.py`; this
exists so the view is regenerated the same way every other output is.

It is a READER. It does not run the model, so it costs seconds rather than the board's
minutes, and it can be re-run freely after any board change. It is NOT a reader of the
board alone, though: the per-fixture lambdas and club table come from
`team_projections_gw1_38.csv` / `team_projections_season.csv`, and the wildcard fifteens
from `wildcard_xi.csv`. Each is built by its own script and can be older than the board,
which is why all three are manifest inputs of the explorer (2026-09-10) and `doctor`
reports the page STALE when any of them lags.

  python scripts/gw_board.py                       # produces the board (GW1-38)
  python scripts/export_team_projections.py        # per-fixture lambdas + club table (GW1-38)
  python scripts/export_wildcard_xi.py             # wildcard fifteens, from the board
  python scripts/export_gw_explorer.py             # builds the view

  --board PATH   read a different board (e.g. .cache/boardbak/gw_board_long.gw18.csv)
                 — useful for holding an older run beside the current one. Dated and
                 .bak snapshots live in .cache/boardbak/, never in outputs/, so a glob
                 for the canonical board cannot return a stale one.
  --out PATH     write the HTML somewhere other than outputs/gw_explorer.html
"""
import argparse
import gw_explorer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default=None,
                    help="board long CSV (default: outputs/gw_board_long.csv)")
    ap.add_argument("--out", default=None, help="output HTML path")
    ap.add_argument("--csv", default=None, help="output long CSV path")
    a = ap.parse_args()
    gw_explorer.build(board_path=a.board, out_html=a.out, out_csv=a.csv)


if __name__ == "__main__":
    main()
