#!/usr/bin/env python3
"""Daily fetch + weekly consolidation for Hull(55) proxy — rev 1.2 pipeline.
Single provider (Yahoo/yfinance) for the whole series. Output of record:
data/weekly_closes.csv — one wide CSV, W-FRI closes, last 70 weeks."""
import datetime as dt
from pathlib import Path
import pandas as pd
import yfinance as yf

TICKERS = [
    "SPY","ACWI","EEM","MCHI","EWZ","INDA","EWH","THD",
    "SOXX","XLK","GLD","GLTR","HYG","LQD","BTC-USD",
    "EWJ","VGK","IWM","IWF","IWD",
    "XLC","XLF","XLV","XLU","XLRE","XLE","XLB","XLY",
]

OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(exist_ok=True)
start = (dt.date.today() - dt.timedelta(days=740)).isoformat()

weekly, manifest = {}, []
for t in TICKERS:
    df = yf.download(t, start=start, interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or len(df) < 300:
        manifest.append(f"{t},FAIL,{0 if df is None else len(df)}")
        continue
    close = df["Close"].dropna()
    if hasattr(close, "columns"):        # yfinance multiindex guard
        close = close.iloc[:, 0]
    wk = close.resample("W-FRI").last().dropna().tail(70)
    weekly[t.replace("-", "")] = wk
    manifest.append(f"{t},OK,{len(close)},{close.index[-1].date()}")

pd.DataFrame(weekly).to_csv(OUT / "weekly_closes.csv",
                            float_format="%.4f", index_label="week_ending")
(OUT / "_manifest.csv").write_text(
    "ticker,status,rows,last_date\n" + "\n".join(manifest) + "\n")
print("\n".join(manifest))
