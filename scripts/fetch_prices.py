#!/usr/bin/env python3
"""Daily price fetch for HartnettTracker Hull(55) proxy — rev 1.2 pipeline.
Yahoo via yfinance, single provider for the whole series (instructions §4).
Raw daily closes only; weekly resample + HMA happen on Claude's side."""
import datetime as dt
from pathlib import Path
import yfinance as yf

# Mirrors TICKER_MAP in HartnettTracker v2.2 (confirmed 2026-07-05)
TICKERS = [
    "SPY","ACWI","EEM","MCHI","EWZ","INDA","EWH","THD",
    "SOXX","XLK","GLD","GLTR","HYG","LQD","BTC-USD",
    "EWJ","VGK","IWM","IWF","IWD",
    "XLC","XLF","XLV","XLU","XLRE","XLE","XLB","XLY",
]

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
start = (dt.date.today() - dt.timedelta(days=740)).isoformat()

manifest = []
for t in TICKERS:
    df = yf.download(t, start=start, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) < 300:
        manifest.append(f"{t},FAIL,{0 if df is None else len(df)}")
        continue
    df = df[["Close"]].dropna()
    df.to_csv(OUT / f"{t.replace('-','')}.csv")
    manifest.append(f"{t},OK,{len(df)},{df.index[-1].date()}")

(OUT / "_manifest.csv").write_text(
    "ticker,status,rows,last_date\n" + "\n".join(manifest) + "\n")
print("\n".join(manifest))
