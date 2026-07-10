#!/usr/bin/env python3
"""Daily fetch + weekly consolidation — Hull(55) proxy + Cboe IV feed.
rev 1.2 pipeline. Single provider (Yahoo/yfinance) for the whole series.
Outputs of record:
  data/weekly_closes.csv  — W-FRI closes, last 70 wk (Hull input)
  data/vol_indices.csv    — Cboe vol index W-FRI closes, last 60 wk (IV + 52wk %ile)
  data/_manifest.csv      — freshness + FAIL audit (sourced-or-null at the source)"""
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
    "^VXEEM","^VXEFA","^RVX",   # อาจถูก Cboe เลิกเผยแพร่ — manifest จะฟ้องเอง
]

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

weekly = {}
for t in HULL_TICKERS:
    c = pull(t, 300)
    if c is not None:
        weekly[clean(t)] = c.resample("W-FRI").last().dropna().tail(70)
pd.DataFrame(weekly).to_csv(OUT / "weekly_closes.csv",
                            float_format="%.4f", index_label="week_ending")

vols = {}
for t in VOL_TICKERS:
    c = pull(t, 200)
    if c is not None:
        vols[clean(t)] = c.resample("W-FRI").last().dropna().tail(60)
pd.DataFrame(vols).to_csv(OUT / "vol_indices.csv",
                          float_format="%.2f", index_label="week_ending")

(OUT / "_manifest.csv").write_text(
    "ticker,status,rows,last_date\n" + "\n".join(manifest) + "\n")
print("\n".join(manifest))
