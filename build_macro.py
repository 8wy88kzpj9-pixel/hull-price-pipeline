#!/usr/bin/env python3
"""
build_macro.py — FRED macro series for RegimeBreadthTracker + LiquidityImpulseTracker.

WHY
---
Every null in those two trackers that FRED can answer, FRED should answer — in the
repo, not in a chat session. Fetching by hand each week means the value depends on
which mirror happens to be fresh that night (we have already hit a cache serving
2025 data for a 2026 field, and a mirror stopping 2 days short). A committed file
removes that entire class of failure: same numbers for every reader, every session.

DESIGN (identical contract to build_weekly.py — deliberately)
  1. IDEMPOTENT REBUILD  — full history every run; a missed day self-heals tomorrow.
  2. VALIDATE-THEN-WRITE — nothing is written unless the data passes the gates;
     failure exits non-zero so the Actions run goes RED and the good file survives.
  3. SOURCED-OR-NULL     — a series that fails to fetch is written as empty, never
     carried forward, never defaulted. Nulls are visible in _macro_status.json.

OUTPUTS (data/)
  macro_daily.csv    — date + one column per series, full daily history
  macro_weekly.csv   — Friday-labelled weekly view: last observation on/before each
                       Friday (matches weekly_closes.csv date convention exactly)
  _macro_status.json — asof per series, staleness, coverage, run timestamp

SERIES (all free, no API key; FRED publishes plain-text tables at /data/<ID>.txt)
  Credit   BAMLH0A0HYM2  HY OAS            -> RegimeBreadth.hyOas   [markdown leg 3]
           BAMLC0A0CM    IG OAS            -> RegimeBreadth.igOas
  Rates    DGS2 DGS10    UST 2y/10y        -> LiquidityImpulse.ust2/ust10
           DGS30         UST 30y           -> 5.20 tripwire
           DFII10        10y REAL yield    -> LiquidityImpulse.real10
           T10Y2Y        2s10s spread      -> curve context
  Fed      WALCL         balance sheet     -> LiquidityImpulse.walcl
           WTREGEN       Treasury General  -> LiquidityImpulse.tga
           RRPONTSYD     overnight RRP     -> LiquidityImpulse.rrp
  Vol      VIXCLS        VIX close         -> Hartnett.vix cross-check

BREADTH (block 2) — computed, not fetched
  %>200dma / %>50dma and 52-week new highs/lows are computed directly from S&P 500
  constituent prices (yfinance). This is the same DEFINITION StockCharts uses for
  $SPXA200R / $SPXA50R, but not the same SERIES: membership drift and adjustment
  conventions will produce small differences. By explicit decision (2026-07-27) the
  values are written to the existing pctAbove200dma / pctAbove50dma fields rather
  than to shadow fields, so every row carries source="computed" and the status file
  records the constituent count — provenance stays traceable even though the field
  is shared. The 3-point hysteresis rule still applies: within 3pts of 70/50/20,
  chart-verify before acting.

  New highs/lows are labelled spx_new_highs / spx_new_lows, NOT nyseNewHighs:
  different universe (500 vs ~2,800), so they must never be written into the NYSE
  fields. Same separation principle as EPFR crypto vs US spot ETF flows.

  NOT AUTOMATED BY DECISION: SKEW and Cboe equity put/call have no free API and
  would require scraping a layout-fragile page. They stay null in the rule-facing
  fields and are carried in the tracker volAux block with their own as-of dates.

USAGE
  pip install pandas requests
  python build_macro.py            # normal
  python build_macro.py --dry-run  # print, write nothing
"""

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

DATA = Path("data")
DAILY = DATA / "macro_daily.csv"
WEEKLY = DATA / "macro_weekly.csv"
STATUS = DATA / "_macro_status.json"

FRED_TXT = "https://fred.stlouisfed.org/data/{sid}.txt"
TIMEOUT = 30
UA = {"User-Agent": "Mozilla/5.0 (compatible; macro-pipeline/1.0)"}

# series id -> (column name, max acceptable staleness in days)
# Weekly Fed series get a longer window: H.4.1 covers a week and posts Thursdays.
SERIES = {
    "BAMLH0A0HYM2": ("hy_oas", 6),
    "BAMLC0A0CM":   ("ig_oas", 6),
    "DGS2":         ("ust2", 6),
    "DGS10":        ("ust10", 6),
    "DGS30":        ("ust30", 6),
    "DFII10":       ("real10", 6),
    "T10Y2Y":       ("spread_2s10s", 6),
    "WALCL":        ("walcl", 12),
    "WTREGEN":      ("tga", 12),
    "RRPONTSYD":    ("rrp", 6),
    "VIXCLS":       ("vix", 6),
}

MIN_ROWS = 200          # ~10 months of daily obs; refuse a truncated download
MIN_SERIES_OK = 0.80    # at least 80% of series must fetch, else fail loud

# ---- breadth block ----
MEMBERS = DATA / "sp500_members.csv"   # self-healing cache of the constituent list
WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
MIN_MEMBERS = 400       # below this the breadth percentage is not trustworthy
BREADTH_HISTORY = "3y"  # need 200dma warmup + a year of highs/lows


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_series(sid: str) -> pd.Series:
    """FRED /data/<ID>.txt = header lines, then 'DATE VALUE' rows. '.' means no obs."""
    r = requests.get(FRED_TXT.format(sid=sid), headers=UA, timeout=TIMEOUT)
    r.raise_for_status()
    lines = r.text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) == 2 and parts[0][:4].isdigit() and "-" in parts[0]:
            start = i
            break
    if start is None:
        raise RuntimeError(f"{sid}: no data rows found (format changed?)")
    df = pd.read_csv(
        io.StringIO("\n".join(lines[start:])),
        sep=r"\s+", header=None, names=["date", "value"], engine="python",
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # '.' -> NaN
    s = df.dropna(subset=["date"]).set_index("date")["value"].sort_index()
    if s.notna().sum() < MIN_ROWS:
        raise RuntimeError(f"{sid}: only {s.notna().sum()} valid obs (< {MIN_ROWS})")
    return s



# ─────────────────────────── breadth block ───────────────────────────────────
def load_members() -> list:
    """S&P 500 constituents. Try Wikipedia; on failure fall back to the committed
    cache. On success, refresh the cache — so a Wikipedia outage never breaks a run
    and the list never silently goes years stale."""
    tickers = []
    try:
        tables = pd.read_html(requests.get(WIKI_SP500, headers=UA, timeout=TIMEOUT).text)
        for t in tables:
            if "Symbol" in t.columns:
                tickers = [str(x).replace(".", "-").strip() for x in t["Symbol"]]
                break
    except Exception as e:
        print(f"  members: wikipedia failed ({str(e)[:80]}), using cache", file=sys.stderr)

    if len(tickers) >= MIN_MEMBERS:
        DATA.mkdir(exist_ok=True)
        pd.DataFrame({"ticker": tickers}).to_csv(MEMBERS, index=False)
        return tickers
    if MEMBERS.exists():
        cached = pd.read_csv(MEMBERS)["ticker"].astype(str).tolist()
        print(f"  members: using cached list ({len(cached)})")
        return cached
    return []


def compute_breadth(tickers: list) -> pd.DataFrame:
    """Daily %>200dma, %>50dma, and 52-week new highs/lows across the given universe.
    Percentages are computed over tickers with a VALID moving average that day, so a
    late-listing name cannot silently drag the denominator."""
    import yfinance as yf

    raw = yf.download(" ".join(tickers), period=BREADTH_HISTORY, interval="1d",
                      auto_adjust=False, group_by="ticker", threads=True,
                      progress=False)
    closes = {}
    for t in tickers:
        try:
            c = raw[t]["Close"] if len(tickers) > 1 else raw["Close"]
            if c.notna().sum() > 250:
                closes[t] = c
        except Exception:
            continue
    if len(closes) < MIN_MEMBERS:
        raise RuntimeError(f"breadth: only {len(closes)} usable tickers (< {MIN_MEMBERS})")

    px = pd.DataFrame(closes).sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()

    ma200, ma50 = px.rolling(200).mean(), px.rolling(50).mean()
    above200 = (px > ma200).where(ma200.notna())
    above50 = (px > ma50).where(ma50.notna())

    roll_max = px.rolling(252).max()
    roll_min = px.rolling(252).min()
    new_hi = (px >= roll_max).where(roll_max.notna())
    new_lo = (px <= roll_min).where(roll_min.notna())

    # denominators go to NaN (not 0) before the warmup window fills, otherwise the
    # first ~200 rows divide by zero — caught by the offline test, not in production
    den200 = above200.notna().sum(axis=1).replace(0, pd.NA)
    den50 = above50.notna().sum(axis=1).replace(0, pd.NA)

    out = pd.DataFrame({
        "pct_above_200dma": above200.sum(axis=1) / den200 * 100,
        "pct_above_50dma":  above50.sum(axis=1) / den50 * 100,
        "spx_new_highs":    new_hi.sum(axis=1),
        "spx_new_lows":     new_lo.sum(axis=1),
        "breadth_universe": above200.notna().sum(axis=1),
    })
    return out.dropna(subset=["pct_above_200dma"])


def to_friday_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Last observation on or before each Friday. Matches weekly_closes.csv labels.
    Uses as-of logic (not resample-last) so a Friday holiday still reports the
    Thursday print rather than silently emitting a null."""
    if daily.empty:
        return daily
    fridays = pd.date_range(daily.index.min(), daily.index.max(), freq="W-FRI")
    out = {}
    for col in daily.columns:
        s = daily[col].dropna()
        out[col] = [
            (s.loc[:f].iloc[-1] if len(s.loc[:f]) else None) for f in fridays
        ]
    wk = pd.DataFrame(out, index=fridays)
    wk.index.name = "week_ending"
    return wk


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    frames, meta, failed = {}, {}, []
    today = datetime.now(timezone.utc).date()

    for sid, (col, max_stale) in SERIES.items():
        try:
            s = fetch_series(sid)
            frames[col] = s
            last_valid = s.dropna()
            asof = last_valid.index[-1].date()
            stale = (today - asof).days
            meta[col] = {
                "series_id": sid,
                "asof": str(asof),
                "latest": round(float(last_valid.iloc[-1]), 4),
                "staleness_days": stale,
                "stale": stale > max_stale,
                "obs": int(len(last_valid)),
            }
            flag = "  <-- STALE" if stale > max_stale else ""
            print(f"  {col:<14} {sid:<14} {asof}  {last_valid.iloc[-1]:>10.4f}{flag}")
        except Exception as e:
            failed.append(col)
            meta[col] = {"series_id": sid, "asof": None, "latest": None,
                         "staleness_days": None, "stale": True, "error": str(e)[:200]}
            print(f"  {col:<14} {sid:<14} FAILED: {e}", file=sys.stderr)

    # ---- block 2: computed breadth (independent failure) ----
    breadth_meta = {"status": "skipped", "source": "computed"}
    try:
        members = load_members()
        if len(members) < MIN_MEMBERS:
            raise RuntimeError(f"only {len(members)} members available")
        print(f"  breadth: computing over {len(members)} S&P 500 members ...")
        b = compute_breadth(members)
        for col in b.columns:
            frames[col] = b[col]
        last = b.dropna().iloc[-1]
        asof_b = b.dropna().index[-1].date()
        breadth_meta = {
            "status": "ok",
            "source": "computed",
            "method": "S&P 500 constituent closes via yfinance; same definition as "
                      "$SPXA200R/$SPXA50R but NOT the same series (membership drift)",
            "asof": str(asof_b),
            "staleness_days": (today - asof_b).days,
            "universe": int(last["breadth_universe"]),
            "pct_above_200dma": round(float(last["pct_above_200dma"]), 1),
            "pct_above_50dma": round(float(last["pct_above_50dma"]), 1),
            "spx_new_highs": int(last["spx_new_highs"]),
            "spx_new_lows": int(last["spx_new_lows"]),
            "hysteresis_warning": bool(
                min(abs(last["pct_above_200dma"] - t) for t in (70, 50, 20)) < 3
            ),
        }
        print(f"  breadth        computed       {asof_b}  "
              f"200dma {last['pct_above_200dma']:.1f}%  50dma {last['pct_above_50dma']:.1f}%  "
              f"NH/NL {int(last['spx_new_highs'])}/{int(last['spx_new_lows'])}")
        if breadth_meta["hysteresis_warning"]:
            print("  breadth: WITHIN 3pt OF A THRESHOLD — chart-verify before acting")
    except Exception as e:
        breadth_meta = {"status": "failed", "source": "computed", "error": str(e)[:200]}
        print(f"  breadth        FAILED: {e}", file=sys.stderr)

    ok_ratio = 1 - len(failed) / len(SERIES)
    if ok_ratio < MIN_SERIES_OK:
        print(f"VALIDATION FAILED: only {ok_ratio:.0%} of series fetched "
              f"(failed: {failed}) — nothing written", file=sys.stderr)
        sys.exit(1)

    daily = pd.DataFrame(frames).sort_index()
    daily.index.name = "date"
    weekly = to_friday_weekly(daily)

    # regression guard: never shrink the committed history
    if DAILY.exists():
        old_len = len(pd.read_csv(DAILY))
        if len(daily) < old_len - 5:
            print(f"VALIDATION FAILED: row regression {old_len} -> {len(daily)}",
                  file=sys.stderr)
            sys.exit(1)

    status = {
        "run_utc": now_utc(),
        "series_ok": len(SERIES) - len(failed),
        "series_total": len(SERIES),
        "failed": failed,
        "stale_series": [k for k, v in meta.items() if v.get("stale")],
        "daily_rows": int(len(daily)),
        "weekly_rows": int(len(weekly)),
        "latest_friday": str(weekly.index[-1].date()) if len(weekly) else None,
        "series": meta,
        "breadth": breadth_meta,
        "note": ("Breadth is COMPUTED from S&P 500 constituents, not fetched from "
                 "StockCharts — same definition, different series. Written to the "
                 "pctAbove200dma/pctAbove50dma fields by explicit decision; every "
                 "row carries source=computed. spx_new_highs/lows are a 500-name "
                 "universe and must NOT be written into nyseNewHighs/Lows. SKEW and "
                 "Cboe equity put/call have no free API and stay manual (volAux)."),
    }

    if args.dry_run:
        print("\n--- DRY RUN (nothing written) ---")
        print(weekly.tail(4).to_string())
        print(json.dumps({k: v for k, v in status.items() if k != "series"}, indent=1))
        return

    DATA.mkdir(exist_ok=True)
    daily.round(4).to_csv(DAILY)
    weekly.round(4).to_csv(WEEKLY)
    STATUS.write_text(json.dumps(status, indent=1))
    print(f"\nWROTE {DAILY} ({len(daily)} rows) / {WEEKLY} ({len(weekly)} rows) / {STATUS}")
    if status["stale_series"]:
        print(f"stale series (source lag, not an error): {status['stale_series']}")


if __name__ == "__main__":
    main()
