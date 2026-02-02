# agent/valuation_multiples.py
"""
This module provides a simple, reproducible valuation "quick check" using:
- Trailing P/E vs a reference multiple from config
- P/FCF using market cap and latest available FCF from the financials table

It is intentionally lightweight: it produces a transparent table + a small
summary dictionary that downstream memo generation can consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _latest_non_null(series: pd.Series) -> Optional[float]:
    """Return the most recent non-null value from a Series (assumes year index)."""
    if series is None or series.empty:
        return None
    s = series.dropna()
    if s.empty:
        return None
    # series index is fiscal year, so last() is latest year after sort_index
    return _safe_float(s.iloc[-1])


def _get_market_cap_and_pe(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    """Fetch market cap and trailing P/E from yfinance info (best-effort)."""
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    market_cap = _safe_float(info.get("marketCap"))
    trailing_pe = _safe_float(info.get("trailingPE"))
    return market_cap, trailing_pe


def run_multiples_valuation(
    ticker: str,
    config: Dict[str, Any],
    financials: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    """
    Run the multiples quick check.

    Returns:
        table_df: pd.DataFrame with rows describing valuation metrics
        summary: dict with headline outputs (e.g., pe_upside_pct, meets_buy_threshold)
        warnings: list of warnings to surface in memo/logs
    """
    warnings: List[str] = []

    valuation_cfg = config.get("valuation", {})
    pe_reference = _safe_float(valuation_cfg.get("pe_reference"))
    min_upside = _safe_float(valuation_cfg.get("min_upside_pct_for_buy"))

    if pe_reference is None:
        warnings.append("Config missing valuation.pe_reference (cannot compute P/E upside).")
    if min_upside is None:
        warnings.append("Config missing valuation.min_upside_pct_for_buy (cannot check BUY hurdle).")

    market_cap, trailing_pe = _get_market_cap_and_pe(ticker)

    if market_cap is None:
        warnings.append("Market cap unavailable from yfinance (P/FCF may be unavailable).")
    if trailing_pe is None:
        warnings.append("Trailing P/E unavailable from yfinance (P/E upside may be unavailable).")

    # Use latest available FCF from financials table
    fcf_used = None
    if isinstance(financials, pd.DataFrame) and (not financials.empty) and ("free_cash_flow" in financials.columns):
        fin_sorted = financials.sort_index()
        fcf_used = _latest_non_null(fin_sorted["free_cash_flow"])
    else:
        warnings.append("Financials missing free_cash_flow column (cannot compute P/FCF).")

    if fcf_used is None:
        warnings.append("Free cash flow unavailable for latest available years (P/FCF may be unavailable).")

    # Compute P/FCF
    p_fcf = None
    if market_cap is not None and fcf_used is not None and fcf_used != 0:
        p_fcf = market_cap / fcf_used
    elif market_cap is not None and fcf_used == 0:
        warnings.append("FCF is zero; cannot compute P/FCF.")

    # Compute P/E implied upside (directional)
    pe_upside_pct = None
    meets_buy_threshold = None
    if trailing_pe is not None and pe_reference is not None and trailing_pe != 0:
        pe_upside_pct = (pe_reference / trailing_pe - 1.0) * 100.0
        if min_upside is not None:
            meets_buy_threshold = pe_upside_pct >= min_upside
    elif trailing_pe == 0:
        warnings.append("Trailing P/E is zero; cannot compute P/E upside.")

    # Build table output (keep it readable for marking)
    rows = [
        {"metric": "Trailing P/E", "current": trailing_pe, "reference": pe_reference},
        {"metric": "P/FCF", "current": p_fcf, "reference": None},
        {"metric": "Market Cap (USD)", "current": market_cap, "reference": None},
        {"metric": "FCF used (USD)", "current": fcf_used, "reference": None},
    ]

    table_df = pd.DataFrame(rows, columns=["metric", "current", "reference"])

    summary: Dict[str, Any] = {
        "pe_upside_pct": pe_upside_pct,
        "meets_buy_threshold": meets_buy_threshold,
        "min_upside_pct_for_buy": min_upside,
        "pe_reference": pe_reference,
    }

    return table_df, summary, warnings


def save_valuation_table(table_df: pd.DataFrame, out_dir: str) -> str:
    """Save valuation multiples table to CSV; return saved path."""
    out_path = Path(out_dir) / "valuation_multiples.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(out_path, index=False)
    return str(out_path)
