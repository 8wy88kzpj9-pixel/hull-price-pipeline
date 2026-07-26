#!/usr/bin/env python3
"""
build_weekly.py — self-healing weekly closes + Hull(55) for hull-price-pipeline.

WHY THIS EXISTS
---------------
The previous pipeline appended one row per run and committed unconditionally.
When the price fetch silently failed (equities returned empty while BTC still
worked), it wrote a blank row and reported success. Result: data stopped at
2026-07-03, nobody noticed for 3 weeks, and Hull went dark exactly when the
regime turned. Two design errors:
    1. INCREMENTAL  -> one bad run permanently loses a week.
    2. UNCONDITIONAL COMMIT -> failure looks identical to success.

This script fixes both:
    1. IDEMPOTENT REBUILD - every run downloads full history and rebuilds the
       whole file. A week missed on Monday is automatically backfilled Tuesday.
       Self-healing; no manual patching, ever.
    2. VALIDATE-THEN-WRITE - nothing is written unless the new data passes
       coverage + regression checks. A failed fetch exits non-zero, the old
       (good) file stays untouched, and the Actions run turns RED.

OUTPUTS (all under data/)
    weekly_closes.csv  - Friday closes per ticker (unchanged schema)
    hull_weekly.csv    - asof,ticker,close,hma55,hma55_prev,margin_pct,color,
                         near_flip   <- tracker reads verdicts from here
    _status.json       - asof date, coverage %, run timestamp, per-ticker nulls

HULL CONVENTION (calibrated against the committed seed on 2026-07-03: all 28
tickers matched on color, and the near-flip set matched WEEKLY_LOG exactly):
    HMA(55) = WMA(2*WMA(c,28) - WMA(c,55), 7)
    color   = green if HMA_t > HMA_t-1 else red      (binary, no neutral)
    margin  = (HMA_t - HMA_t-1)/HMA_t-1 * 100
    near_flip = |margin| < 0.30  -> hysteresis rule: CHART-VERIFY before use.

USAGE
    pip install "yfinance>=0.2.60" pandas numpy
    python build_weekly.py            # normal run
    python build_weekly.py --dry-run  # validate only, write nothing

v1.1: SYMBOL_MAP for vendor tickers (BTC-USD/^GSPC/^SET.BK) +
      Friday week-ending labels to match the seed convention.
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config ----
DATA = Path("data")
CLOSES = DATA / "weekly_closes.csv"
HULL = DATA / "hull_weekly.csv"
STATUS = DATA / "_status.json"

HISTORY = "3y"          # 3y of weekly bars ≈ 156 rows: ample for HMA55 warmup
HMA_PERIOD = 55
NEAR_FLIP_ABS = 0.30

MIN_COVERAGE = 0.90     # latest row must have >=90% of tickers populated
MIN_ROWS = 60           # sanity floor; HMA55 needs ~62 rows to emit 2 values

# Column name in weekly_closes.csv -> yfinance symbol. Column names are the
# system of record (tracker/TICKER_MAP read them); only the fetch layer needs
# the vendor spelling. Add here when a fetch returns empty for one ticker.
SYMBOL_MAP = {
    "BTCUSD": "BTC-USD",
    "GSPC": "^GSPC",
    "SETBK": "^SET.BK",
}


# ------------------------------------------------------------ hull engine ---
def wma(s: pd.Series, n: int) -> pd.Series:
    n = int(n)
    w = np.arange(1.0, n + 1.0)
    return s.rolling(n).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)


def hma(s: pd.Series, n: int = HMA_PERIOD) -> pd.Series:
    half, root = int(round(n / 2.0)), int(round(math.sqrt(n)))
    return wma(2.0 * wma(s, half) - wma(s, n), root)


def hull_row(series: pd.Series):
    """(close, hma, hma_prev, margin, color, near_flip) or Nones. Null means
    NO SIGNAL — never a default. Downstream must not treat it as neutral."""
    s = series.dropna().astype(float)
    h = hma(s).dropna()
    if len(h) < 2 or len(s) == 0:
        return (None,) * 6
    ht, hp = float(h.iloc[-1]), float(h.iloc[-2])
    if hp == 0:
        return (None,) * 6
    m = (ht - hp) / hp * 100.0
    return (round(float(s.iloc[-1]), 4), round(ht, 4), round(hp, 4),
            round(m, 2), "green" if m > 0 else "red", abs(m) < NEAR_FLIP_ABS)


# ------------------------------------------------------------- download -----
def fetch_weekly(tickers: list) -> pd.DataFrame:
    """Weekly closes for all tickers. Batch first; per-ticker retry for any
    column that comes back empty (batch mode fails silently far too often —
    that is precisely how this pipeline broke)."""
    import yfinance as yf

    out = {}
    ysyms = [SYMBOL_MAP.get(t, t) for t in tickers]
    try:
        raw = yf.download(" ".join(ysyms), period=HISTORY, interval="1wk",
                          auto_adjust=False, group_by="ticker", threads=True,
                          progress=False)
        for t in tickers:
            sym = SYMBOL_MAP.get(t, t)
            try:
                col = raw[sym]["Close"] if len(tickers) > 1 else raw["Close"]
                if col.notna().sum() >= MIN_ROWS:
                    out[t] = col
            except Exception:
                pass
    except Exception as e:
        print(f"WARN batch download failed: {e}", file=sys.stderr)

    missing = [t for t in tickers if t not in out]
    for t in missing:                                   # per-ticker fallback
        try:
            h = yf.Ticker(SYMBOL_MAP.get(t, t)).history(
                period=HISTORY, interval="1wk", auto_adjust=False)["Close"]
            if h.notna().sum() >= MIN_ROWS:
                out[t] = h
            else:
                print(f"WARN {t}: only {h.notna().sum()} rows", file=sys.stderr)
        except Exception as e:
            print(f"WARN {t} retry failed: {e}", file=sys.stderr)

    if not out:
        sys.exit("FATAL: no ticker returned data — refusing to write anything")

    df = pd.DataFrame(out)
    idx = pd.to_datetime(df.index)
    try:
        idx = idx.tz_localize(None)
    except TypeError:
        idx = idx.tz_convert(None)
    # yfinance labels a weekly bar with its MONDAY. The seed/WEEKLY_LOG label
    # weeks by the FRIDAY close. Shift +4 days so date identity matches the
    # system of record — mismatched labels are how wrong-date errors start.
    df.index = idx.normalize() + pd.Timedelta(days=4)
    df.index.name = "week_ending"
    return df.sort_index()


# ------------------------------------------------------------- validate -----
def validate(new: pd.DataFrame, tickers: list) -> dict:
    """Gate before any write. Returns report; raises SystemExit on failure so
    the Actions run goes RED instead of silently committing garbage."""
    rep = {}
    last = new.iloc[-1]
    cov = float(last.notna().sum()) / len(tickers)
    rep["asof"] = str(new.index[-1].date())
    rep["coverage"] = round(cov, 3)
    rep["rows"] = int(len(new))
    rep["missing_tickers"] = [t for t in tickers if t not in new.columns
                              or pd.isna(last.get(t))]

    fail = []
    if cov < MIN_COVERAGE:
        fail.append(f"coverage {cov:.0%} < {MIN_COVERAGE:.0%} "
                    f"(missing: {rep['missing_tickers']})")
    if len(new) < MIN_ROWS:
        fail.append(f"only {len(new)} rows (< {MIN_ROWS})")

    if CLOSES.exists():                                  # regression guard
        old = pd.read_csv(CLOSES)
        if len(new) < len(old) - 1:
            fail.append(f"row count regression {len(old)} -> {len(new)}")
        # price sanity: no ticker may move >60% w/w (split/bad-tick guard)
        if len(new) >= 2:
            chg = (new.iloc[-1] / new.iloc[-2] - 1).abs()
            wild = chg[chg > 0.60].dropna().index.tolist()
            if wild:
                fail.append(f"implausible w/w move: {wild}")

    if fail:
        print("VALIDATION FAILED — nothing written:", file=sys.stderr)
        for f in fail:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)
    return rep


# ----------------------------------------------------------------- main -----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not CLOSES.exists():
        sys.exit(f"FATAL: {CLOSES} not found — run from repo root")

    tickers = [c for c in pd.read_csv(CLOSES, nrows=1).columns
               if c.lower() != "week_ending"]
    print(f"universe: {len(tickers)} tickers (from existing header)")

    new = fetch_weekly(tickers)
    new = new.reindex(columns=tickers)          # preserve column order exactly
    rep = validate(new, tickers)
    print(f"validated: asof {rep['asof']}, {rep['rows']} rows, "
          f"coverage {rep['coverage']:.0%}")

    # ---- hull ----
    rows, near = [], []
    for t in tickers:
        c, ht, hp, m, col, nf = hull_row(new[t]) if t in new else (None,) * 6
        rows.append(dict(asof=rep["asof"], ticker=t, close=c, hma55=ht,
                         hma55_prev=hp, margin_pct=m, color=col, near_flip=nf))
        if nf:
            near.append(f"{t} {m:+.2f}")
    hull_df = pd.DataFrame(rows)

    if args.dry_run:
        print("\n--- DRY RUN (nothing written) ---")
        print(hull_df.to_string(index=False))
        print("near-flips (CHART-VERIFY):", near or "none")
        return

    DATA.mkdir(exist_ok=True)
    new.round(4).to_csv(CLOSES)
    hull_df.to_csv(HULL, index=False)
    STATUS.write_text(json.dumps({
        **rep,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hull_nulls": [r["ticker"] for r in rows if r["color"] is None],
        "near_flips": near,
    }, indent=2))

    print(f"WROTE {CLOSES} / {HULL} / {STATUS}")
    print("near-flips (CHART-VERIFY per hysteresis rule):", near or "none")


if __name__ == "__main__":
    main()
