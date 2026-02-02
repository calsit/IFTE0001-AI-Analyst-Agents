# tools_market_data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Caching
# -----------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = _PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _cache_path(ticker: str, years: int, interval: str, auto_adjust: bool) -> Path:
    safe = ticker.replace("/", "_").replace(" ", "_")
    aa = "adj" if auto_adjust else "raw"
    return CACHE_DIR / f"ohlcv_{safe}_{years}y_{interval}_{aa}.csv"


# -----------------------------
# Data fetching / cleaning
# -----------------------------
def fetch_ohlcv(
    ticker: str,
    years: int = 10,
    *,
    interval: str = "1d",
    auto_adjust: bool = False,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV using yfinance with disk cache.
    Returns a dataframe with columns:
      Date, Open, High, Low, Close, Adj Close, Volume
    (Date is a column, NOT index)

    Notes:
    - yfinance sometimes returns MultiIndex columns; we normalize it.
    - We keep Adj Close if available; if not available, we create it from Close.
    """
    p = _cache_path(ticker, years, interval, auto_adjust)
    if p.exists() and not force:
        df = pd.read_csv(p)
        df["Date"] = pd.to_datetime(df["Date"])
        return df

    df = yf.download(
        ticker,
        period=f"{years}y",
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",  # closer to standard OHLCV layout
        threads=True,
    )

    if df is None or len(df) == 0:
        raise RuntimeError(f"No data returned for ticker={ticker}. Check symbol or network.")

    # If MultiIndex columns (sometimes includes ticker level), drop to OHLCV level
    if isinstance(df.columns, pd.MultiIndex):
        # find the level that contains ticker, slice it out
        for lvl in range(df.columns.nlevels):
            if ticker in df.columns.get_level_values(lvl):
                df = df.xs(ticker, level=lvl, axis=1)
                break
        # still multiindex -> flatten
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] for c in df.columns]

    # Ensure Date is a column
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.reset_index()

    # Standardize column names
    rename_map = {}
    for c in df.columns:
        if c.lower() == "adj close":
            rename_map[c] = "Adj Close"
        elif c.lower() == "open":
            rename_map[c] = "Open"
        elif c.lower() == "high":
            rename_map[c] = "High"
        elif c.lower() == "low":
            rename_map[c] = "Low"
        elif c.lower() == "close":
            rename_map[c] = "Close"
        elif c.lower() == "volume":
            rename_map[c] = "Volume"
        elif c == "date" or c == "Date":
            rename_map[c] = "Date"
    df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns from yfinance: {missing}. Got: {list(df.columns)}")

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    # Sort, drop duplicates
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    # Persist cache
    df.to_csv(p, index=False)
    return df


# -----------------------------
# Indicators
# -----------------------------
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothing (EMA-like with alpha=1/window).
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()

    rs = roll_up / (roll_down + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


@dataclass(frozen=True)
class IndicatorConfig:
    ma_window: int = 200
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    vol_window: int = 20  # for rolling vol estimate
    ann_factor: int = 252  # daily bars assumption


def add_indicators(
    df: pd.DataFrame,
    cfg: IndicatorConfig = IndicatorConfig(),
) -> pd.DataFrame:
    """
    Adds:
      ret, log_ret,
      MA{ma_window},
      RSI{rsi_window},
      MACD, MACD_signal, MACD_hist,
      vol (annualized rolling)
    """
    out = df.copy()
    close = out["Close"].astype(float)
    adj = out["Adj Close"].astype(float)

    out["ret"] = adj.pct_change()
    out["log_ret"] = np.log(adj).diff()

    ma_name = f"MA{cfg.ma_window}"
    out[ma_name] = close.rolling(cfg.ma_window, min_periods=cfg.ma_window).mean()

    rsi_name = f"RSI{cfg.rsi_window}"
    out[rsi_name] = rsi_wilder(close, window=cfg.rsi_window)

    ema_fast = _ema(close, cfg.macd_fast)
    ema_slow = _ema(close, cfg.macd_slow)
    out["MACD"] = ema_fast - ema_slow
    out["MACD_signal"] = _ema(out["MACD"], cfg.macd_signal)
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    # rolling annualized vol
    daily_vol = out["ret"].rolling(cfg.vol_window, min_periods=cfg.vol_window).std()
    out["vol"] = daily_vol * np.sqrt(cfg.ann_factor)

    return out


def latest_snapshot(df: pd.DataFrame, *, ma_window: int = 200, rsi_window: int = 14) -> Dict:
    """
    Returns a compact dict snapshot for the latest row.
    Useful for agent prompt and trade-note generation.
    """
    if len(df) == 0:
        raise ValueError("Empty dataframe")

    row = df.iloc[-1]
    ma_name = f"MA{ma_window}"
    rsi_name = f"RSI{rsi_window}"

    snap = {
        "as_of": str(pd.to_datetime(row["Date"]).date()),
        "close": float(row["Close"]),
        "adj_close": float(row.get("Adj Close", row["Close"])),
        "ma": float(row[ma_name]) if ma_name in df.columns and pd.notna(row[ma_name]) else None,
        "rsi": float(row[rsi_name]) if rsi_name in df.columns and pd.notna(row[rsi_name]) else None,
        "macd_hist": float(row["MACD_hist"]) if "MACD_hist" in df.columns and pd.notna(row["MACD_hist"]) else None,
        "vol": float(row["vol"]) if "vol" in df.columns and pd.notna(row["vol"]) else None,
        "n_rows": int(len(df)),
        "start": str(pd.to_datetime(df["Date"].iloc[0]).date()),
        "end": str(pd.to_datetime(df["Date"].iloc[-1]).date()),
    }
    return snap


# -----------------------------
# Quick self-test
# -----------------------------
if __name__ == "__main__":
    d = fetch_ohlcv("MSFT", 10)
    d2 = add_indicators(d)
    print(d2.tail(2))
    print(latest_snapshot(d2))
