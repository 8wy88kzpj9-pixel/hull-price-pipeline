#!/usr/bin/env python3
"""Daily fetch — full data layer for all 3 trackers + computed breadth.
rev 1.2 pipeline. Single provider per series (Yahoo prices/vol, FRED macro).
Outputs:
  data/weekly_closes.csv - 28 tickers, W-FRI closes, 70 wk (Hull input)
  data/vol_indices.csv   - Cboe vol indices, W-FRI, 60 wk (IV + 52wk %ile)
  data/fred_series.csv   - FRED series, last 80 obs (credit + liquidity)
  data/breadth.csv       - COMPUTED % of S&P500 above 200/50DMA, W-FRI, 30 wk
                           (PROXY of $SPXA200R/$SPXA50R - estFlag, verify near thresholds)
  data/_manifest.csv     - freshness + FAIL audit"""
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
VOL_TICKERS = ["^VIX","^VXN","^GVZ","^OVX","^VVIX","^VXEEM","^VXEFA","^RVX"]
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

# ---- 3) FRED series ----
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

# ---- 4) COMPUTED BREADTH: % of S&P500 members above own 200/50 DMA ----
# PROXY of StockCharts $SPXA200R/$SPXA50R. Constituents from Wikipedia
# (may lag index changes by a few names). Yahoo adjusted closes.
# Expect +/-1-3pp vs StockCharts - verify on chart near 20/50/70 thresholds.
try:
    wiki = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    syms = (pd.read_html(wiki)[0]["Symbol"]
            .str.replace(".", "-", regex=False).tolist())
    manifest.append(f"SP500_LIST,OK,{len(syms)},wikipedia")
    px = yf.download(syms, start=start, interval="1d",
                     auto_adjust=True, progress=False)["Close"]
    px = px.dropna(axis=1, thresh=250)          # ตัดตัวที่ข้อมูลขาดหนัก
    valid = px.shape[1]
    if valid < 450:
        raise ValueError(f"only {valid} valid tickers")
    ma200 = px.rolling(200).mean()
    ma50 = px.rolling(50).mean()
    ab200 = (px > ma200).sum(axis=1) / valid * 100
    ab50 = (px > ma50).sum(axis=1) / valid * 100
    br = pd.DataFrame({"pctAbove200dma": ab200, "pctAbove50dma": ab50})
    br = br[ma200.notna().sum(axis=1) > valid * 0.9]   # รอ MA200 อุ่นเครื่องครบ
    br = br.resample("W-FRI").last().dropna().tail(30).round(1)
    br.to_csv(OUT / "breadth.csv", index_label="week_ending")
    manifest.append(f"BREADTH,OK,{len(br)},{br.index[-1].date()},valid={valid}")
except Exception as e:
    manifest.append(f"BREADTH,FAIL,0,{type(e).__name__}")

# ---- 5) manifest (ท้ายสุดเสมอ) ----
(OUT / "_manifest.csv").write_text(
    "ticker,status,rows,last_date\n" + "\n".join(manifest) + "\n")
print("\n".join(manifest))
