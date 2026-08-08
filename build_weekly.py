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
    vol_indices.csv    - week_ending x index WIDE history (upsert, never clobber)
    vol_snapshot.csv   - asof,index,close,pct_52w,lo_52w,hi_52w  <- IV_DATA source
    hull3d.csv         - HMA(55) on 3-trading-day bars, one row per ticker

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

v1.8: drop_incomplete_week() — ตัดแท่งสัปดาห์ที่ยังไม่ปิดก่อนคำนวณ Hull
      cron เดิม (จ.-ศ.) ทำให้ 4 ใน 5 รอบผลิตกระดานจากแท่งครึ่งใบ และ
      validate() จับไม่ได้เพราะ coverage ยังขึ้น 100% ทุก ticker มีค่าครบ
      แค่เป็นค่ากลางสัปดาห์ ยืนยันด้วยการวัดจริง 2026-08-07: INDA พลิก
      -0.05 -> +0.04 เพราะรันตอนตลาด US ยังไม่เปิด
v2.1: SKEW + equity put/call ดึงจาก cdn.cboe.com โดยตรง (Cboe เป็นผู้คำนวณ\n      Yahoo แค่ mirror และขาดค่าวันศุกร์เป็นครั้งคราว) · drop_lagging_columns()\n      เขียน null แทนค่าวันก่อนหน้าเมื่อซีรีส์มาช้า · board.json รวมทุกอย่าง\n      ไว้ไฟล์เดียวสำหรับเสิร์ฟผ่าน GitHub Pages\nv1.9: vol_indices.csv rebuild เต็มไฟล์จากรายวัน 2y (idempotent) แทน upsert
      v1.6 เขียน snapshot long ทับประวัติ wide สำเร็จจริง 2026-08-07T01:50Z
      (f61ae7e) ประวัติตั้งแต่ 2025-05-16 หายเกลี้ยง rebuild กู้คืนได้เองโดย
      ไม่ต้องขุด git · เพิ่ม VVIX กลับเข้า VOL_INDICES (เคยอยู่ในไฟล์เดิมแต่
      ไม่เคยถูกดึง) · percentile คิดจาก 252 วันล่าสุดเท่านั้น ไม่ใช่ทั้ง 2y
v1.7: vol_indices.csv is now UPSERTED, not overwritten. v1.4-1.6 wrote the long
      snapshot straight over a wide week_ending x index history — one successful
      run would have destroyed every vol level back to 2025-05-16, silently,
      because the block is non-fatal. The 52-week percentile snapshot moved to
      its own file (vol_snapshot.csv) which IS safe to replace. Also:
      to_csv(CLOSES, index_label="week_ending") — without it an unnamed index
      writes a blank header cell, the next run reads it as "Unnamed: 0", and
      that string enters the ticker universe.
v1.6: 45s cool-down between the weekly and daily fetch phases. hull3d doubles
      the request count in a single run, and Yahoo answered the first attempt
      with empty frames (30/30 "fetched", 3% coverage) — the validate-then-write
      gate caught it and preserved the good file, but the fetch itself needs to
      stop crowding the limiter.
v1.5: hull3d folded in as a third block rather than a separate repo/workflow —
      one ticker universe, one fetch discipline, one commit. A second workflow
      would have hit the same Yahoo rate limiter from the same runner pool and
      forced a duplicate ticker list to drift out of sync.
v1.4: Cboe volatility indices (VIX/VXN/GVZ/RVX/OVX/VXEEM/VXFXI/VXEWZ/VXEFA/
      VXHYG/SKEW) fetched here rather than FRED — FRED read-times-out from
      GitHub runners (cloud-IP throttling, observed 2026-07-27..08-02) while
      Yahoo serves them fine. Writes vol_indices.csv with the 52-week percentile
      each index sits at, so the tracker's IV_DATA stops going stale by hand.
      This block never fails the run: a missing index is written null.
v1.3: full hull board printed to the run log (CDN cache made log-only reads necessary).
v1.2: chunked fetch + backoff + per-ticker sweep (Yahoo rate-limit resilience).
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
VOLIDX = DATA / "vol_indices.csv"    # WIDE history, week_ending x index — UPSERT ONLY
VOLSNAP = DATA / "vol_snapshot.csv"  # long snapshot w/ 52w percentile — safe to overwrite
HULL3D = DATA / "hull3d.csv"
BOARD = DATA / "board.json"          # v2.1: ทุกอย่างในไฟล์เดียว เสิร์ฟผ่าน GitHub Pages

# ---- Cboe โดยตรง (ไม่ผ่าน Yahoo) -------------------------------------------
# Cboe เป็นผู้คำนวณดัชนีเหล่านี้เอง Yahoo แค่ mirror มา และ mirror มีช่องโหว่:
# ถ้า Yahoo ขาดค่าวันศุกร์ resample("W-FRI").last() จะหยิบค่าวันพฤหัสมาแล้ว
# ติดป้ายว่าเป็นศุกร์ โดยไม่มีสัญญาณเตือน — SKEW เพี้ยนด้วยกลไกนี้
#
# equitypc.csv คือชุด EQUITY เท่านั้น (ตัด ETP ออกตั้งแต่ 2012-06-11) ตรงกับที่
# Protocol B.1 ต้องการ และปิดกับดัก Total-P/C ที่ทำให้ค่า >= 0.85 ถูก REJECT
# มาหลายรอบ — ปัญหาเดิมคือดึงจาก YCharts ด้วยมือแล้วหยิบผิดชุด/ข้อมูลมาช้า
CBOE_SKEW_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv"
CBOE_PCR_URL = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/equitypc.csv"
PCR_REJECT_HI = 0.85     # Protocol B.1: ค่าสูงกว่านี้ = น่าจะหยิบชุด Total มาผิด

HISTORY = "3y"          # 3y of weekly bars ≈ 156 rows: ample for HMA55 warmup
HMA_PERIOD = 55
NEAR_FLIP_ABS = 0.30

MIN_COVERAGE = 0.90     # latest row must have >=90% of tickers populated
CHUNK = 6               # symbols per request — large batches trip Yahoo's limiter
PAUSE = 1.5             # seconds between requests
RETRIES = 3             # batch passes before falling back to one-by-one
BACKOFF = 20            # base seconds between retry passes (doubles each time)

# ---- hull3d block: HMA(55) on 3-trading-day bars ---------------------------
# Same period as the weekly system by explicit decision. 55 bars x 3D is about
# eight calendar months, so it reacts roughly 40% faster than HMA55 weekly while
# still filtering the noise a daily read would let through. Fed by the SAME
# ticker universe as weekly_closes.csv - never a second hardcoded list.
BAR_DAYS_3D = 3
DAILY_HISTORY = "2y"    # 55+28+7 3D bars needs ~270 trading days; 2y is ample
EST_FLAG_3D = "parallel-validation"   # clear only via a versioned commit

# Margin is a PER-BAR slope, so it does not transfer across timeframes: on an
# identical price path a 3-day bar produces a smaller margin than a 5-day weekly
# bar simply because it spans less time (measured 0.25 vs 0.42 on a linear ramp).
# The 0.30 weekly hysteresis band would therefore over-flag on 3D. Scaled by the
# bar-length ratio 3/5: 0.30 * 0.6 = 0.18.
NEAR_FLIP_3D = 0.18
COOLDOWN_3D = 45        # seconds between the weekly and the daily fetch phase
MIN_ROWS = 60           # sanity floor; HMA55 needs ~62 rows to emit 2 values

# ---- Cboe volatility indices (daily) -> data/vol_indices.csv ----------------
# Column name in the tracker's IV_DATA -> Yahoo symbol.
VOL_INDICES = {
    "VIX": "^VIX",        # S&P 500 30d IV      -> us_largecap, global_stocks
    "VXN": "^VXN",        # Nasdaq 100 30d IV   -> ustech, sec_tech
    "GVZ": "^GVZ",        # Gold (GLD) 30d IV   -> gold
    "RVX": "^RVX",        # Russell 2000 30d IV -> us_smallcap
    "OVX": "^OVX",        # Crude oil 30d IV    -> sec_energy (proxy)
    "VXEEM": "^VXEEM",    # EM (EEM) 30d IV     -> em, em_total
    "VXFXI": "^VXFXI",    # China (FXI) 30d IV  -> china_stocks, em_china
    "VXEWZ": "^VXEWZ",    # Brazil (EWZ) 30d IV -> em_brazil
    "VXEFA": "^VXEFA",    # EAFE 30d IV         -> japan/europe (REGIONAL proxy)
    "VXHYG": "^VXHYG",    # HY credit 30d IV    -> bonds
    "VVIX": "^VVIX",      # vol-of-vol          -> เคยอยู่ในไฟล์เดิม ต้องดึงเองไม่งั้นหาย
    # SKEW ย้ายไปดึงจาก Cboe โดยตรง (fetch_cboe_series) — Yahoo ขาดค่าวันศุกร์
    # เป็นครั้งคราวแล้วทำให้ค่าวันพฤหัสถูกติดป้ายเป็นศุกร์
}
VOL_HISTORY = "2y"      # v1.9: rebuild ประวัติ weekly ทั้งชุดจากรายวัน
                        # ไฟล์เดิมเริ่ม 2025-05-16 — 2y ครอบคลุมเกินนั้น

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

# ───────────────────────── hull3d: 3-day-bar Hull ────────────────────────────
def to_3d_closes(daily_close: pd.Series, bar_days: int = BAR_DAYS_3D) -> pd.Series:
    """Group daily closes into N-trading-day bars anchored at the most recent
    COMPLETED day, so the latest bar is always full. The oldest partial group is
    dropped.

    NOTE ON BOUNDARIES: TradingView anchors 3D bars from the start of the series,
    so its bar edges can sit +/-1 day away from these depending on how much chart
    history is loaded. Margins near zero will therefore disagree occasionally —
    the hysteresis rule applies to hull3d exactly as it does to weekly."""
    s = daily_close.dropna().sort_index()
    n = len(s)
    if n < bar_days:
        return pd.Series(dtype=float)
    idx_from_end = np.arange(n - 1, -1, -1)
    bar_id = idx_from_end // bar_days
    counts = pd.Series(bar_id, index=s.index).value_counts()
    keep = set(counts[counts == bar_days].index)
    frame = pd.DataFrame({"close": s, "bar": bar_id})
    frame = frame[frame["bar"].isin(keep)]
    if frame.empty:
        return pd.Series(dtype=float)
    grp = frame.groupby("bar")
    closes = grp["close"].last()
    dates = grp.apply(lambda g: g.index.max())
    out = pd.Series(closes.values, index=pd.to_datetime(dates.values))
    out = out.sort_index()
    out.index.name = "bar_end"
    return out


def fetch_daily(tickers: list) -> dict:
    """Daily closes, chunked with backoff — identical discipline to the weekly
    fetch, because the failure that took this pipeline down was a rate limit,
    not a bad symbol."""
    import time
    import yfinance as yf

    out = {}
    for attempt in range(RETRIES):
        pending = [t for t in tickers if t not in out]
        if not pending:
            break
        if attempt:
            wait = BACKOFF * (2 ** (attempt - 1))
            print(f"  hull3d retry {attempt}: {len(pending)} missing, waiting {wait}s")
            time.sleep(wait)
        for i in range(0, len(pending), CHUNK):
            grp = pending[i:i + CHUNK]
            syms = [SYMBOL_MAP.get(t, t) for t in grp]
            try:
                raw = yf.download(" ".join(syms), period=DAILY_HISTORY, interval="1d",
                                  auto_adjust=False, group_by="ticker",
                                  threads=False, progress=False)
                for t in grp:
                    sym = SYMBOL_MAP.get(t, t)
                    try:
                        col = raw[sym]["Close"] if len(grp) > 1 else raw["Close"]
                        if col is not None and col.notna().sum() > 250:
                            out[t] = col
                    except Exception:
                        pass
            except Exception as e:
                print(f"  hull3d chunk {grp[0]}..{grp[-1]}: {str(e)[:80]}",
                      file=sys.stderr)
            time.sleep(PAUSE)
    return out


def build_hull3d(tickers: list) -> pd.DataFrame:
    """One row per ticker: HMA(55) on 3D bars. Null means no signal, never a
    default — the tracker must be able to tell 'red' from 'unknown'."""
    # Cool-down between the weekly and daily pulls. Without it this run fires
    # ~60 requests back to back and Yahoo starts returning empty frames — the
    # exact signature that silently broke the pipeline on 2026-07-10 and failed
    # the 2026-08-03 run at 3% coverage.
    import time
    print(f"  hull3d: cooling down {COOLDOWN_3D}s before the daily pull ...")
    time.sleep(COOLDOWN_3D)
    daily = fetch_daily(tickers)
    print(f"  hull3d fetched {len(daily)}/{len(tickers)} tickers (daily)")
    rows = []
    for t in tickers:
        c = ht = hp = m = col = nf = None
        bars_used = 0
        asof = None
        if t in daily:
            bars = to_3d_closes(drop_incomplete_day(daily[t]))
            bars_used = int(bars.dropna().shape[0])
            if bars_used:
                asof = pd.to_datetime(bars.index.max()).date().isoformat()
            c, ht, hp, m, col, _ = hull_row(bars)
            # re-derive near_flip against the 3D-scaled band, not weekly's
            nf = (abs(m) < NEAR_FLIP_3D) if m is not None else None
        rows.append(dict(asof=asof, ticker=t, bars_3d_used=bars_used,
                         close_3d=c, hma55_3d=ht, hma55_3d_prev=hp,
                         margin_pct=m, color=col, near_flip=nf,
                         est_flag=EST_FLAG_3D))
    return pd.DataFrame(rows)


def fetch_weekly(tickers: list) -> pd.DataFrame:
    """Weekly closes for all tickers.

    Yahoo rate-limits bursty requests: a single 30-symbol batch can come back
    empty for every equity while crypto still resolves (observed 2026-07-10 and
    again 2026-07-27 — the exact signature that took this pipeline down). So:
    small chunks, backoff between them, then a slow per-ticker sweep for anything
    still missing. Slower by design; a run that takes 90s and succeeds beats one
    that takes 8s and returns nothing."""
    import time
    import yfinance as yf

    out = {}

    def keep(t, col):
        if col is not None and col.notna().sum() >= MIN_ROWS:
            out[t] = col
            return True
        return False

    # ---- pass 1: chunked batches with backoff ----
    for attempt in range(RETRIES):
        pending = [t for t in tickers if t not in out]
        if not pending:
            break
        if attempt:
            wait = BACKOFF * (2 ** (attempt - 1))
            print(f"  retry {attempt}: {len(pending)} missing, waiting {wait}s")
            time.sleep(wait)
        for i in range(0, len(pending), CHUNK):
            grp = pending[i:i + CHUNK]
            syms = [SYMBOL_MAP.get(t, t) for t in grp]
            try:
                raw = yf.download(" ".join(syms), period=HISTORY, interval="1wk",
                                  auto_adjust=False, group_by="ticker",
                                  threads=False, progress=False)
                for t in grp:
                    sym = SYMBOL_MAP.get(t, t)
                    try:
                        col = raw[sym]["Close"] if len(grp) > 1 else raw["Close"]
                        keep(t, col)
                    except Exception:
                        pass
            except Exception as e:
                print(f"  chunk {grp[0]}..{grp[-1]} failed: {str(e)[:90]}",
                      file=sys.stderr)
            time.sleep(PAUSE)

    # ---- pass 2: per-ticker sweep for stragglers ----
    for t in [t for t in tickers if t not in out]:
        try:
            h = yf.Ticker(SYMBOL_MAP.get(t, t)).history(
                period=HISTORY, interval="1wk", auto_adjust=False)["Close"]
            if not keep(t, h):
                print(f"WARN {t}: only {h.notna().sum()} rows", file=sys.stderr)
        except Exception as e:
            print(f"WARN {t} single fetch failed: {str(e)[:90]}", file=sys.stderr)
        time.sleep(PAUSE)

    print(f"  fetched {len(out)}/{len(tickers)} tickers")

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

# ────────────────────── Cboe volatility indices ──────────────────────────────
def fetch_vol_indices() -> tuple:
    """Daily closes for the Cboe vol complex, plus where each sits in its own
    52-week range. Percentile matters more than level: VIX 16 means something
    different in a 12-30 year than in a 15-45 one, and the tracker's putBuy /
    putThresh bands are range-relative.

    Failure is per-index and never fatal — a missing index is written null so
    the tracker can keep its sourced-or-null contract."""
    import time
    import yfinance as yf

    rows = []
    daily = {}
    for name, sym in VOL_INDICES.items():
        val = lo = hi = pct = None
        asof = None
        try:
            h = yf.Ticker(sym).history(period=VOL_HISTORY, interval="1d",
                                       auto_adjust=False)["Close"].dropna()
            if len(h) >= 30:
                daily[name] = h
                w52 = h.iloc[-252:] if len(h) > 252 else h   # percentile = 52wk เท่านั้น
                val = round(float(h.iloc[-1]), 2)
                lo, hi = round(float(w52.min()), 2), round(float(w52.max()), 2)
                pct = round(float((w52 <= h.iloc[-1]).mean() * 100), 1)
                asof = pd.to_datetime(h.index[-1]).date().isoformat()
            else:
                print(f"  vol {name:<6} only {len(h)} obs — null", file=sys.stderr)
        except Exception as e:
            print(f"  vol {name:<6} FAILED: {str(e)[:70]}", file=sys.stderr)
        rows.append(dict(asof=asof, index=name, symbol=sym, close=val,
                         pct_52w=pct, lo_52w=lo, hi_52w=hi))
        time.sleep(PAUSE)
    return pd.DataFrame(rows), daily


def rebuild_vol_history(daily: dict) -> tuple:
    """สร้าง data/vol_indices.csv ใหม่ทั้งไฟล์จากข้อมูลรายวัน — idempotent

    ทำไมต้อง rebuild ไม่ใช่ upsert: v1.4-v1.6 เขียน snapshot รูปแบบ long ทับ
    ไฟล์ประวัติรูปแบบ wide ทั้งไฟล์ ทำสำเร็จจริงเมื่อ 2026-08-07T01:50Z
    (commit f61ae7e) ประวัติตั้งแต่ 2025-05-16 หายทั้งหมด เหลือ 11 แถว
    และเพราะ block นี้เป็น non-fatal จึงไม่มีสัญญาณอะไรเลย

    upsert แก้อาการ แต่ไม่กู้ของที่หายไปแล้ว rebuild จากรายวัน 2y กู้คืนได้
    ทั้งหมดโดยไม่ต้องขุด git และป้องกันไม่ให้เกิดซ้ำได้ถาวร — หลักการเดียว
    กับ weekly_closes.csv ที่ docstring หัวไฟล์อธิบายไว้

    คอลัมน์ที่ดึงไม่ได้จะถูกรักษาไว้จากไฟล์เดิม (combine_first) ไม่ทับด้วย null
    """
    if not daily:
        raise RuntimeError("ไม่มี index ไหนคืนข้อมูลรายวันเลย — ไม่แตะไฟล์")

    cols, obs_last = {}, {}
    for name, s in daily.items():
        idx = pd.to_datetime(s.index)
        try:
            idx = idx.tz_localize(None)
        except TypeError:
            idx = idx.tz_convert(None)
        ser = pd.Series(s.values, index=idx).sort_index()
        obs_last[name] = ser.index[-1].normalize()
        cols[name] = ser.resample("W-FRI").last().dropna().round(2)

    new = pd.DataFrame(cols)
    new.index = pd.to_datetime(new.index).normalize()
    new.index.name = "week_ending"
    new = drop_incomplete_week(new)          # แท่งสัปดาห์ปัจจุบันยังไม่ปิด
    new = drop_lagging_columns(new, obs_last)   # ซีรีส์ที่ข้อมูลมาช้า -> null ไม่ใช่ค่าเก่า

    before = 0
    if VOLIDX.exists():
        old = pd.read_csv(VOLIDX)
        if "week_ending" in old.columns:     # ไฟล์รูปแบบ wide เดิม -> รักษาไว้
            old["week_ending"] = pd.to_datetime(old["week_ending"])
            old = old.set_index("week_ending").sort_index()
            before = len(old)
            new = new.combine_first(old)     # ค่าใหม่ชนะ ของเดิมเติมช่องว่าง
        else:                                # ไฟล์ long ที่ถูกเขียนทับ -> ทิ้ง
            print(f"  {VOLIDX} เป็น schema snapshot (cols={list(old.columns)[:4]}) "
                  f"— สร้างประวัติใหม่ทับ", file=sys.stderr)

    new = new.sort_index()
    new.index.name = "week_ending"
    new.to_csv(VOLIDX)
    return len(new), before, str(new.index[-1].date())


def drop_incomplete_week(df: pd.DataFrame) -> pd.DataFrame:
    """ตัดแท่งสัปดาห์ที่ยังไม่ปิดทิ้ง ก่อนคำนวณ Hull

    yfinance คืนแท่งของสัปดาห์ที่กำลังเดินอยู่ด้วย สคริปต์ติดป้ายเป็นวันศุกร์
    ผลคือรันวันพุธจะได้แถว 'ศุกร์' ที่มีแค่ราคาปิด จ.-อ. แล้ว HMA55 ถูกคำนวณ
    ทับลงบนแท่งครึ่งใบ — coverage ยังขึ้น 100% เพราะทุก ticker มีค่า
    gate เดิมจึงจับไม่ได้เลย

    วัดผลจริง 2026-08-07 (รันเช้าวันศุกร์ ตลาด US ยังไม่เปิด):
        INDA -0.05 -> +0.04  (พลิกสี)
        XLY  -0.19 -> -0.09
        IWF  +0.17 -> +0.24
    ทั้งหมดเป็นสัญญาณผีจากแท่งที่ยังไม่จบ

    แท่งที่ป้ายเป็นศุกร์ F ถือว่าปิดแล้วเมื่อเวลาปัจจุบัน >= F 21:00 UTC
    (ตลาดหุ้นสหรัฐปิด 20:00 UTC ช่วง EDT / 21:00 UTC ช่วง EST — ใช้ 21:00
    เป็นขอบปลอดภัยตลอดปี)

    ตัดทิ้ง ไม่ใช่ทำให้รันล้ม: ข้อมูลสัปดาห์ก่อนหน้ายังถูกต้องและใช้งานได้"""
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    while len(df) and (df.index[-1] + pd.Timedelta(hours=21)) > now:
        print(f"  ตัดแท่งที่ยังไม่ปิด: {df.index[-1].date()} "
              f"(ปิดจริง {(df.index[-1] + pd.Timedelta(hours=21))} UTC)")
        df = df.iloc[:-1]
    if df.empty:
        # ต้องเป็น RuntimeError ไม่ใช่ sys.exit: SystemExit สืบทอดจาก BaseException
        # จึงทะลุ `except Exception` ของ vol block ที่ระบุว่า non-fatal แล้วฆ่าทั้ง run
        raise RuntimeError("ไม่เหลือสัปดาห์ที่ปิดแล้วเลย")
    return df


def drop_incomplete_day(s: pd.Series) -> pd.Series:
    """ตัดแท่งรายวันของวันที่ยังไม่ปิดออก ก่อนจัดกลุ่มเป็นแท่ง 3 วัน

    to_3d_closes() หยิบ 3 วันทำการล่าสุดมาเป็นแท่งใหม่สุดเสมอ ถ้ารันตอนตลาด
    ยังเปิดอยู่ วันนั้นเป็นราคากลางคัน แท่ง 3 วันจึงปนราคาที่ยังไม่นิ่ง —
    โรคเดียวกับ drop_incomplete_week() ในฝั่ง weekly

    cron เสาร์/อาทิตย์ไม่เจอปัญหานี้ แต่ workflow_dispatch กลางสัปดาห์เจอเต็ม ๆ"""
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    idx = pd.to_datetime(s.index)
    try:
        idx = idx.tz_localize(None)
    except TypeError:
        idx = idx.tz_convert(None)
    out = pd.Series(s.values, index=idx).sort_index()
    while len(out) and (out.index[-1].normalize() + pd.Timedelta(hours=21)) > now:
        out = out.iloc[:-1]
    return out


def append_hull3d_history(h3: pd.DataFrame, weekly_colors: dict) -> tuple:
    """สะสม hull3d ลงไฟล์แทนการเขียนทับ — upsert ตาม (asof, ticker)

    v1.5-v1.9 เขียนทับทุกรอบ เหลือ snapshot แถวเดียวต่อ ticker ผลคือรันสำเร็จ
    มาตั้งแต่ 2026-08-04 แต่ประวัติที่ใช้ปลดธง est_flag ยังเป็นศูนย์ ทั้งที่
    parallel-validation ต้องเทียบ 3D กับ weekly ย้อนหลังสี่สัปดาห์

    บันทึก weekly_color ในแถวเดียวกันด้วย เพื่อให้เทียบย้อนหลังได้จากไฟล์เดียว
    ไม่ต้องประกบสองไฟล์ทีหลัง"""
    cur = h3.copy()
    cur["weekly_color"] = cur["ticker"].map(weekly_colors)
    cur["agree"] = [
        None if (pd.isna(r["color"]) or not r["weekly_color"])
        else bool(r["color"] == r["weekly_color"])
        for _, r in cur.iterrows()
    ]

    before = 0
    if HULL3D.exists():
        old = pd.read_csv(HULL3D)
        if "asof" in old.columns and "ticker" in old.columns:
            before = len(old)
            keep = cur["asof"].dropna().unique().tolist()
            old = old[~old["asof"].isin(keep)]          # upsert: รอบเดิมวันเดียวกันถูกแทน
            cur = pd.concat([old, cur], ignore_index=True)

    cur = cur.sort_values(["asof", "ticker"], na_position="last").reset_index(drop=True)
    cur.to_csv(HULL3D, index=False)
    weeks = cur["asof"].nunique()
    agree = cur[cur["agree"].notna()]["agree"]
    rate = f"{agree.mean()*100:.0f}%" if len(agree) else "n/a"
    return before, len(cur), weeks, rate


def fetch_cboe_series(url: str, date_col: str, val_col: str, label: str) -> pd.Series:
    """ดึง CSV จาก cdn.cboe.com เป็น Series รายวัน (index=วันที่, value=float)

    Cboe วางหัวตารางไว้ไม่ตรงบรรทัดแรกเสมอ — ไล่หาบรรทัดที่มีชื่อคอลัมน์จริง
    แล้วอ่านจากตรงนั้น ล้มเหลว = คืน Series ว่าง ไม่ใช่ค่าเดา"""
    import io
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    except Exception as e:
        print(f"  cboe {label}: ดึงไม่ได้ — {str(e)[:80]}", file=sys.stderr)
        return pd.Series(dtype=float)

    lines = raw.splitlines()
    hdr = next((i for i, l in enumerate(lines[:40])
                if date_col.lower() in l.lower() and val_col.lower() in l.lower()), None)
    if hdr is None:
        print(f"  cboe {label}: หาหัวตาราง '{date_col}'/'{val_col}' ไม่เจอ", file=sys.stderr)
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
        df.columns = [str(c).strip() for c in df.columns]
        dcol = next(c for c in df.columns if date_col.lower() in c.lower())
        vcol = next(c for c in df.columns if c.strip().lower() == val_col.lower())
        s = pd.Series(pd.to_numeric(df[vcol], errors="coerce").values,
                      index=pd.to_datetime(df[dcol], errors="coerce")).dropna()
        s = s[~s.index.isna()].sort_index()
        print(f"  cboe {label}: {len(s)} วัน ล่าสุด {s.index[-1].date()} = {s.iloc[-1]}")
        return s
    except Exception as e:
        print(f"  cboe {label}: parse ล้มเหลว — {str(e)[:80]}", file=sys.stderr)
        return pd.Series(dtype=float)


def drop_lagging_columns(wide: pd.DataFrame, obs_last: dict) -> pd.DataFrame:
    """ถ้าซีรีส์ไหนมีวันสังเกตล่าสุดตามหลังกลุ่ม = ข้อมูลมาช้า -> เขียน null
    ในสัปดาห์ล่าสุด แทนที่จะปล่อยให้ค่าวันก่อนหน้าถูกติดป้ายเป็นวันศุกร์

    ถ้าทุกซีรีส์ตามหลังเท่ากัน = วันหยุดตลาด -> เก็บไว้ตามปกติ
    กฎนี้แยกวันหยุดออกจากข้อมูลมาช้าได้เองโดยไม่ต้องมีปฏิทินวันหยุด"""
    if wide.empty or not obs_last:
        return wide
    newest = max(obs_last.values())
    last_wk = wide.index[-1]
    for col, d in obs_last.items():
        if col in wide.columns and d < newest:
            print(f"  {col}: ข้อมูลล่าสุด {d.date()} ตามหลังกลุ่ม ({newest.date()}) "
                  f"-> null ที่ {last_wk.date()}", file=sys.stderr)
            wide.at[last_wk, col] = pd.NA
    return wide


def write_board(rep: dict, hull_rows: list, h3: pd.DataFrame,
                vol_snap: pd.DataFrame, pcr: dict) -> None:
    """รวมทุกอย่างที่ต้องใช้ไว้ในไฟล์เดียว เสิร์ฟผ่าน GitHub Pages

    เหตุผล: หน้า blob ของ CSV บน github.com เรนเดอร์ตารางด้วย JS จึงอ่านจาก
    ภายนอกไม่ได้ ส่วน raw.githubusercontent ติด robots และ CDN cache ค้างนาน
    ไฟล์ JSON เดียวบน github.io แก้ทั้งสองปัญหา และไม่ต้อง copy ทีละไฟล์"""
    board = {
        "asof": rep["asof"],
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coverage": rep["coverage"],
        "rows": rep["rows"],
        "hull_weekly": [
            {k: r[k] for k in ("ticker", "close", "margin_pct", "color", "near_flip")}
            for r in hull_rows
        ],
        "hull3d": [
            {"ticker": r["ticker"], "margin_pct": r["margin_pct"], "color": r["color"],
             "near_flip": r["near_flip"], "weekly_color": r.get("weekly_color"),
             "agree": r.get("agree"), "est_flag": r["est_flag"]}
            for _, r in h3.iterrows()
        ] if h3 is not None and not h3.empty else [],
        "vol": [
            {"index": r["index"], "close": (None if pd.isna(r["close"]) else r["close"]),
             "pct_52w": (None if pd.isna(r["pct_52w"]) else r["pct_52w"]),
             "asof": (None if pd.isna(r["asof"]) else r["asof"])}
            for _, r in vol_snap.iterrows()
        ] if vol_snap is not None and not vol_snap.empty else [],
        "put_call_equity": pcr,
    }
    BOARD.write_text(json.dumps(board, indent=1, default=str))
    print(f"WROTE {BOARD}")


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
    try:                                        # v1.8: กันกระดานผีจากแท่งครึ่งใบ
        new = drop_incomplete_week(new)
    except RuntimeError as e:
        sys.exit(f"FATAL: {e} — ไม่เขียนอะไรทั้งสิ้น")
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
    new.round(4).to_csv(CLOSES, index_label="week_ending")
    hull_df.to_csv(HULL, index=False)
    STATUS.write_text(json.dumps({
        **rep,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hull_nulls": [r["ticker"] for r in rows if r["color"] is None],
        "near_flips": near,
    }, indent=2))

    # ---- hull3d (independent failure; never blocks the weekly hull) ----
    try:
        h3 = build_hull3d(tickers)
        ok = int(h3["color"].notna().sum())
        a3 = h3["asof"].dropna().max()
        wk_colors = {r["ticker"]: r["color"] for r in rows if r["color"]}
        n0, n1, weeks, rate = append_hull3d_history(h3, wk_colors)
        print(f"WROTE {HULL3D} ({ok}/{len(h3)} tickers, asof {a3}) "
              f"| ประวัติสะสม {n0} -> {n1} แถว, {weeks} วันที่, ตรงกับ weekly {rate}")
        print(f"\nHULL3D BOARD (asof {a3}) [{EST_FLAG_3D}]:")
        for _, r in h3.iterrows():
            if pd.notna(r["color"]):
                flag = "  **NEAR**" if r["near_flip"] else ""
                print(f"  {r['ticker']:<8}{r['color']:<6}{r['margin_pct']:+.2f}{flag}")
            else:
                print(f"  {r['ticker']:<8}null")
        # disagreement with the weekly read is the whole point of running both
        wk = {r["ticker"]: r["color"] for r in rows if r["color"]}
        diff = [f"{r['ticker']} 3D:{r['color']} vs W:{wk[r['ticker']]}"
                for _, r in h3.iterrows()
                if pd.notna(r["color"]) and wk.get(r["ticker"])
                and r["color"] != wk[r["ticker"]]]
        print(f"3D-vs-WEEKLY disagreements ({len(diff)}): {diff or 'none'}")
    except Exception as e:
        print(f"hull3d block failed (non-fatal): {str(e)[:140]}", file=sys.stderr)

    # ---- Cboe vol indices (independent failure; never blocks the price run) ----
    try:
        vol, daily_vol = fetch_vol_indices()
        # SKEW จาก Cboe โดยตรง — ผู้คำนวณตัวจริง ไม่ใช่ mirror ที่มีช่องโหว่
        skew = fetch_cboe_series(CBOE_SKEW_URL, "DATE", "SKEW", "SKEW")
        if not skew.empty:
            daily_vol["SKEW"] = skew
            w52 = skew.iloc[-252:]
            vol = pd.concat([vol, pd.DataFrame([dict(
                asof=skew.index[-1].date().isoformat(), index="SKEW", symbol="cboe:SKEW",
                close=round(float(skew.iloc[-1]), 2),
                pct_52w=round(float((w52 <= skew.iloc[-1]).mean()*100), 1),
                lo_52w=round(float(w52.min()), 2), hi_52w=round(float(w52.max()), 2))])],
                ignore_index=True)
        vol.to_csv(VOLSNAP, index=False)          # derived snapshot: safe to replace
        ok = vol["close"].notna().sum()
        print(f"WROTE {VOLSNAP} ({ok}/{len(vol)} indices)")
        n1, n0, last = rebuild_vol_history(daily_vol)  # history: rebuild เต็ม idempotent
        print(f"WROTE {VOLIDX} (rebuild {n0} -> {n1} สัปดาห์, ล่าสุด {last})")
        print("\nVOL INDICES:")
        for _, v in vol.iterrows():
            if pd.notna(v["close"]):
                print(f"  {v['index']:<7}{v['close']:>7.2f}  {v['pct_52w']:>5.1f}%ile"
                      f"  range {v['lo_52w']}-{v['hi_52w']}  asof {v['asof']}")
            else:
                print(f"  {v['index']:<7}   null")
    except Exception as e:
        print(f"vol indices block failed (non-fatal): {str(e)[:120]}", file=sys.stderr)

    print(f"WROTE {CLOSES} / {HULL} / {STATUS}")
    print("near-flips (CHART-VERIFY per hysteresis rule):", near or "none")

    # Full board in the run's own log. The committed CSV can sit behind a CDN
    # cache for several minutes after a push, so the log must be sufficient on
    # its own to reconstruct every signal without a second channel.
    # ---- Cboe equity put/call (non-fatal) ----
    pcr = {"value": None, "asof": None, "series": "cboe_equity", "status": "not_fetched"}
    try:
        s_pcr = fetch_cboe_series(CBOE_PCR_URL, "DATE", "P/C Ratio", "EQUITY P/C")
        if not s_pcr.empty:
            v = round(float(s_pcr.iloc[-1]), 4)
            pcr = {"value": v, "asof": s_pcr.index[-1].date().isoformat(),
                   "series": "cboe_equity", "status": "ok",
                   "w5_avg": round(float(s_pcr.iloc[-5:].mean()), 4)}
            if v >= PCR_REJECT_HI:          # Protocol B.1: กับดักชุด Total
                pcr["status"] = "REJECT_check_series"
                print(f"::warning::equity P/C {v} >= {PCR_REJECT_HI} — "
                      f"ตรวจว่าหยิบชุด Total มาผิดหรือไม่", file=sys.stderr)
            print(f"EQUITY P/C = {v} (asof {pcr['asof']}, 5d avg {pcr['w5_avg']})")
        else:
            pcr["status"] = "fetch_failed"
    except Exception as e:
        pcr["status"] = f"error: {str(e)[:60]}"
        print(f"put/call block failed (non-fatal): {str(e)[:100]}", file=sys.stderr)

    try:
        write_board(rep, rows, locals().get("h3"), locals().get("vol"), pcr)
    except Exception as e:
        print(f"board.json failed (non-fatal): {str(e)[:100]}", file=sys.stderr)

    print(f"\nHULL BOARD (asof {rep['asof']}):")
    for r in rows:
        if r["color"]:
            flag = "  **NEAR**" if r["near_flip"] else ""
            print(f"  {r['ticker']:<8}{r['color']:<6}{r['margin_pct']:+.2f}{flag}")
        else:
            print(f"  {r['ticker']:<8}null")


if __name__ == "__main__":
    main()
