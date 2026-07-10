#!/usr/bin/env python3
"""Daily fetch + weekly consolidation — full data layer for all 3 trackers.
rev 1.2 pipeline. Single provider per series (Yahoo prices/vol, FRED macro).
Outputs of record:
  data/weekly_closes.csv  - 28 tickers, W-FRI closes, last 70 wk (Hull input)
  data/vol_indices.csv    - Cboe vol indices, W-FRI, last 60 wk (IV + 52wk %ile)
  data/fred_series.csv    - FRED daily/weekly series, last 80 obs (credit + liquidity)
  data/_manifest.csv      - freshness + FAIL audit (sourced-or-null at the source)"""
import datetime as dt
from pathlib import Path
import pandas as pd
import yfinance as yf

HULL_TICKERS = [
    "SPY","ACWI","EEM","MCHI","EWZ","INDA","EWH","THD",
    "SOXX","XLK","GLD","GLTR","HYG","LQD","BTC-USD",
    "EWJ","VGK","IWM","IWF","IWD",
    "XLC","XLF","XLV","XLU","XLRE","XLE","XLB","XLY",
]
VOL_TICKERS = [
    "^VIX","^VXN","^GVZ","^OVX","^VVIX",
    "^VXEEM","^VXEFA","^RVX",   # ตัวที่ Cboe เลิกเผยแพร่จะขึ้น FAIL ใน manifest เอง
]
FRED = ["WALCL","WTREGEN","RRPONTSYD","DGS2","DGS10",
        "BAMLH0A0HYM2","BAMLC0A0CM"]

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
start = (dt.date.today() - dt.timedelta(days=740)).isoformat()
manifest = []

def pull(t, min_rows):
    df = yf.download(t, start=start, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) < min_rows:
        manifest.append(f"{t},FAIL,{0 if df is None else len(df)},")
        return None
    close = df["Close"].dropna()
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    manifest.append(f"{t},OK,{len(close)},{close.index[-1].date()}")
    return close

def clean(t):
    return t.replace("-", "").replace("^", "")

# ---- 1) Hull prices ----
weekly = {}
for t in HULL_TICKERS:
    c = pull(t, 300)
    if c is not None:
        weekly[clean(t)] = c.resample("W-FRI").last().dropna().tail(70)
pd.DataFrame(weekly).to_csv(OUT / "weekly_closes.csv",
                            float_format="%.4f", index_label="week_ending")

# ---- 2) Cboe vol indices ----
vols = {}
for t in VOL_TICKERS:
    c = pull(t, 200)
    if c is not None:
        vols[clean(t)] = c.resample("W-FRI").last().dropna().tail(60)
pd.DataFrame(vols).to_csv(OUT / "vol_indices.csv",
                          float_format="%.2f", index_label="week_ending")

# ---- 3) FRED series (fredgraph.csv, no key needed) ----
fred_out = {}
for sid in FRED:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    try:
        s = pd.read_csv(url, index_col=0, parse_dates=True).iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna().tail(80)
        fred_out[sid] = s
        manifest.append(f"{sid},OK,{len(s)},{s.index[-1].date()}")
    except Exception as e:
        manifest.append(f"{sid},FAIL,0,{type(e).__name__}")
pd.DataFrame(fred_out).to_csv(OUT / "fred_series.csv",
                              float_format="%.2f", index_label="date")

# ---- 4) manifest (เขียนท้ายสุดเสมอ — ครบทก series ทั้งสามกลุม) ----
(OUT / "_manifest.csv").write_text(
    "ticker,status,rows,last_date\n" + "\n".join(manifest) + "\n")
print("\n".join(manifest))
