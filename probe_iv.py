#!/usr/bin/env python3
"""
probe_iv.py — READ-ONLY probe. Does Yahoo serve usable option chains for the
Hartnett flow-category proxies, and which ones pass a liquidity gate?

WHY THIS EXISTS AND WHY IT WRITES NOTHING
-----------------------------------------
build_weekly.py died once already because a fetch path was assumed to work and
the failure looked like success. The correct order is: prove the fetch, measure
what comes back, THEN write the pipeline. This script only prints. It creates
no files, touches nothing under data/, and cannot break a run.

WHAT IT IS FOR
--------------
IV_DATA in HartnettTracker is meant to carry implied vol PER HARTNETT FLOW
CATEGORY — VOL_INDICES in build_weekly.py already documents that mapping
("^VXEEM  # EM (EEM) 30d IV -> em, em_total"). Today only 6 of ~28 categories
have a live number, and several of those are the wrong underlying:

    ustech/sec_tech  <- VXN   but NDX != XLK   (hardware/semi contamination)
    sec_energy       <- OVX   but CRUDE != XLE (commodity vol != equity vol)
    japan + europe   <- VXEFA one blended index covering two categories
    em/china/smallcap/HY       Yahoo stopped serving VXEEM/VXFXI/RVX/VXHYG

Computing IV30 from each proxy's OWN option chain removes every one of those
mismatches and fills the ~22 categories that have no IV at all.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* No substitution. If MCHI's chain is too thin it reports MCHI as thin. It does
  NOT quietly swap in FXI. FXI is where China option liquidity lives, but MCHI
  is the Hull proxy for the category — using FXI for IV would put TWO different
  proxies on ONE category, which is the same failure class as XLK-for-software.
  That is a decision for the operator, printed as a flag, never taken here.
* No historical rank. Nobody serves free IV history, so percentile rank has to
  be accumulated forward from first write. Until ~26 weeks exist, rank is null.
  This probe reports IV30 and IV90 so the TERM STRUCTURE is usable immediately —
  slope needs no history.
* No default values. Anything that cannot be sourced prints as null, never 0
  and never a guess.

USAGE (run on the GitHub runner, not locally — Yahoo throttles cloud IPs
differently and the whole point is to test the runner's path)
    pip install "yfinance>=0.2.60" pandas numpy
    python probe_iv.py                 # full universe
    python probe_iv.py --limit 8       # quick smoke test, first 8 only
    python probe_iv.py --json          # machine-readable to stdout
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path

import pandas as pd

# ---- universe: read it, never re-declare it --------------------------------
# build_weekly.py's own rule, verbatim from its comments: "Fed by the SAME
# ticker universe as weekly_closes.csv - never a second hardcoded list."
# A second list is how the system ends up with two definitions of one thing.
DATA = Path("data")
CLOSES = DATA / "weekly_closes.csv"
EXTRA_TICKERS = ["URTH", "EFA", "XLP", "XLI", "XBI", "UUP", "TLT", "VYM", "IAI"]
SYMBOL_MAP = {"BTCUSD": "BTC-USD", "GSPC": "^GSPC", "SETBK": "^SET.BK"}

# Instruments that structurally cannot have a US listed option chain. Excluded
# up front so they show as "n/a by construction" rather than as a fetch failure
# — the two mean different things and must not look alike.
NO_CHAIN = {
    "BTCUSD": "spot crypto — no listed options (IBIT is a different instrument)",
    "SETBK":  "Thai index — no US options",
    "GSPC":   "index level; SPX options exist under a different root, and SPY "
              "already covers the category",
}

# ---- liquidity gate on the OPTION, not the underlying ----------------------
# The KTEC precedent was an instrument with a correct mandate and no effective
# volume. An ETF can clear its ADV screen easily while its chain is deserted:
# wide markets make the printed IV meaningless AND make the position impossible
# to exit. So the gate is applied to the ATM contracts themselves.
MIN_OI_ATM = 100        # combined call+put open interest at the ATM strike
MAX_SPREAD_PCT = 15.0   # (ask-bid)/mid at ATM, percent
TARGET_DTE = (30, 90)   # near leg, far leg — far/near slope is the term signal

PAUSE = 1.2             # seconds between symbols; matches the pipeline's PAUSE
RETRIES = 2


def load_universe(limit=None):
    if not CLOSES.exists():
        print(f"FATAL: {CLOSES} not found — run this from the repo root",
              file=sys.stderr)
        sys.exit(2)
    cols = list(pd.read_csv(CLOSES, nrows=1).columns)
    tickers = [c for c in cols if c not in ("week_ending", "Unnamed: 0")]
    for t in EXTRA_TICKERS:
        if t not in tickers:
            tickers.append(t)
    return tickers[:limit] if limit else tickers


def pick_expiries(exp_strings, today):
    """Return (near, far) expiry strings closest to TARGET_DTE.

    Closest-match, not nearest-above: a chain with a 27d and a 45d expiry should
    use 27d for the near leg. The actual DTE is reported alongside so the number
    is never mistaken for a clean 30.
    """
    out = []
    parsed = []
    for e in exp_strings:
        try:
            d = date.fromisoformat(e)
        except ValueError:
            continue
        dte = (d - today).days
        if dte > 0:
            parsed.append((dte, e))
    if not parsed:
        return []
    for target in TARGET_DTE:
        dte, e = min(parsed, key=lambda p: abs(p[0] - target))
        out.append((e, dte))
    return out


def atm_read(chain, spot):
    """ATM implied vol plus the liquidity evidence behind it.

    IV is the mean of the call and put at the strike nearest spot. Taking only
    one side imports that side's skew; near ATM the two should agree closely, so
    a wide call/put gap is itself a warning and is reported as iv_gap.
    """
    calls, puts = chain.calls, chain.puts
    if calls is None or puts is None or calls.empty or puts.empty:
        return None
    k = min(calls["strike"], key=lambda x: abs(x - spot))
    c = calls[calls["strike"] == k]
    p = puts[puts["strike"] == k]
    if c.empty or p.empty:
        return None
    c, p = c.iloc[0], p.iloc[0]

    def num(v):
        try:
            v = float(v)
            return v if v == v else None      # NaN check without importing math
        except (TypeError, ValueError):
            return None

    civ, piv = num(c.get("impliedVolatility")), num(p.get("impliedVolatility"))
    ivs = [v for v in (civ, piv) if v is not None and v > 0]
    if not ivs:
        return None
    iv = sum(ivs) / len(ivs) * 100.0

    oi = int((num(c.get("openInterest")) or 0) + (num(p.get("openInterest")) or 0))
    spreads = []
    for leg in (c, p):
        b, a = num(leg.get("bid")), num(leg.get("ask"))
        if b and a and (a + b) > 0:
            spreads.append((a - b) / ((a + b) / 2) * 100.0)
    spread = max(spreads) if spreads else None

    return {
        "strike": float(k),
        "iv": round(iv, 2),
        "iv_gap": (round(abs(civ - piv) * 100, 2)
                   if civ is not None and piv is not None else None),
        "oi_atm": oi,
        "spread_pct": (round(spread, 1) if spread is not None else None),
    }


def probe(ticker):
    import yfinance as yf
    sym = SYMBOL_MAP.get(ticker, ticker)
    rec = {"ticker": ticker, "symbol": sym, "iv30": None, "iv90": None,
           "dte30": None, "dte90": None, "oi_atm": None, "spread_pct": None,
           "iv_gap": None, "slope": None, "pass": False, "note": ""}

    if ticker in NO_CHAIN:
        rec["note"] = f"n/a by construction — {NO_CHAIN[ticker]}"
        return rec

    for attempt in range(RETRIES):
        try:
            tk = yf.Ticker(sym)
            exps = list(tk.options or [])
            if not exps:
                rec["note"] = "no expiries returned"
                return rec
            legs = pick_expiries(exps, datetime.now(timezone.utc).date())
            if len(legs) < 2:
                rec["note"] = f"only {len(legs)} usable expiry"
                return rec

            hist = tk.history(period="5d")
            if hist is None or hist.empty:
                rec["note"] = "no spot price"
                return rec
            spot = float(hist["Close"].dropna().iloc[-1])

            reads = []
            for e, dte in legs:
                r = atm_read(tk.option_chain(e), spot)
                if r is None:
                    rec["note"] = f"ATM unreadable at {e}"
                    return rec
                r["dte"] = dte
                reads.append(r)
                time.sleep(0.4)

            near, far = reads
            rec.update(iv30=near["iv"], iv90=far["iv"],
                       dte30=near["dte"], dte90=far["dte"],
                       oi_atm=near["oi_atm"], spread_pct=near["spread_pct"],
                       iv_gap=near["iv_gap"])
            if near["iv"]:
                rec["slope"] = round(far["iv"] - near["iv"], 2)

            # gate on the near leg — that is the contract the IV30 read claims
            ok_oi = near["oi_atm"] >= MIN_OI_ATM
            ok_sp = (near["spread_pct"] is not None
                     and near["spread_pct"] <= MAX_SPREAD_PCT)
            rec["pass"] = bool(ok_oi and ok_sp)
            if not ok_oi:
                rec["note"] += f"OI {near['oi_atm']} < {MIN_OI_ATM}; "
            if not ok_sp:
                rec["note"] += (f"spread {near['spread_pct']}% > "
                                f"{MAX_SPREAD_PCT}%; ")
            return rec
        except Exception as e:
            if attempt == RETRIES - 1:
                rec["note"] = f"error: {str(e)[:70]}"
            else:
                time.sleep(6)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    tickers = load_universe(args.limit)
    print(f"probing {len(tickers)} tickers · gate: OI>={MIN_OI_ATM} "
          f"spread<={MAX_SPREAD_PCT}% · WRITES NOTHING", file=sys.stderr)

    rows = []
    for i, t in enumerate(tickers, 1):
        r = probe(t)
        rows.append(r)
        print(f"  [{i}/{len(tickers)}] {t:8s} "
              f"iv30={str(r['iv30']):>7} iv90={str(r['iv90']):>7} "
              f"oi={str(r['oi_atm']):>6} sp={str(r['spread_pct']):>6} "
              f"{'PASS' if r['pass'] else 'fail'} {r['note']}",
              file=sys.stderr)
        time.sleep(PAUSE)

    if args.json:
        print(json.dumps({"probed_utc": datetime.now(timezone.utc)
                          .isoformat(timespec="seconds"),
                          "gate": {"min_oi_atm": MIN_OI_ATM,
                                   "max_spread_pct": MAX_SPREAD_PCT},
                          "rows": rows}, indent=1))
        return

    ok = [r for r in rows if r["pass"]]
    thin = [r for r in rows if not r["pass"] and r["iv30"] is not None]
    dead = [r for r in rows if r["iv30"] is None]

    print("\n" + "=" * 64)
    print(f"PASS {len(ok)}  ·  THIN {len(thin)}  ·  NO CHAIN {len(dead)}"
          f"  ·  of {len(rows)}")
    print("=" * 64)
    print(f"\n{'ticker':<8}{'IV30':>8}{'IV90':>8}{'slope':>8}"
          f"{'OI':>8}{'spr%':>7}{'gap':>7}")
    for r in ok:
        print(f"{r['ticker']:<8}{r['iv30']:>8}{r['iv90']:>8}"
              f"{str(r['slope']):>8}{r['oi_atm']:>8}"
              f"{str(r['spread_pct']):>7}{str(r['iv_gap']):>7}")
    if thin:
        print("\nTHIN — chain exists but fails the gate. IV printed here is NOT "
              "usable; a wide market makes the number noise and the position "
              "hard to exit (KTEC precedent, moved one layer up to options):")
        for r in thin:
            print(f"  {r['ticker']:<8} iv30={r['iv30']} {r['note']}")
    if dead:
        print("\nNO CHAIN — null, not zero:")
        for r in dead:
            print(f"  {r['ticker']:<8} {r['note']}")

    print("\nREAD THIS BEFORE ACTING ON THE TABLE")
    print("  · slope = IV90 - IV30. Positive = contango. This is the only")
    print("    cheap/expensive signal available until ~26 weeks of history")
    print("    accumulate; it needs no history and is usable from run one.")
    print("  · IV30 alone says nothing. 25.6 is not 'cheap' or 'expensive'")
    print("    without knowing where it has been — same reason VIX 14.25 only")
    print("    means something once you know it is the 2.8th percentile.")
    print("  · iv_gap = |call IV - put IV| at ATM. Large gap on a supposedly")
    print("    liquid name means the quote is stale or skewed; distrust the IV.")
    print("  · If MCHI lands in THIN: that is real, not a bug. China option")
    print("    liquidity sits in FXI. Do NOT swap it in silently — MCHI is the")
    print("    Hull proxy for the category, and two proxies on one category is")
    print("    the exact mismatch this system forbids. Decide it explicitly.")


if __name__ == "__main__":
    main()
