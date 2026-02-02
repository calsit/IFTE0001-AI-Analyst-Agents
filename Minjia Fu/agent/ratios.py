# agent/ratios.py
"""
Computes a small, coursework-friendly set of ratios across:
- Profitability
- Growth
- Leverage
- Efficiency

Inputs:
- Cleaned annual financials table indexed by fiscal year.

Outputs:
- ratios_df: year-indexed table (oldest -> newest)
- ratios_summary_df: one-row summary (latest year + CAGRs)
- warnings: list of data/coverage issues (used in memo transparency)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class RatiosResult:
    ratios_df: pd.DataFrame
    summary_df: pd.DataFrame
    warnings: List[str]


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Elementwise division with safe handling for 0/NaN denominators."""
    out = a.astype(float) / b.astype(float)
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    return out


def _cagr(series: pd.Series) -> Optional[float]:
    """
    CAGR using first and last non-null values in the series.
    Returns None if insufficient data or invalid values.
    """
    s = series.dropna().astype(float)
    if len(s) < 2:
        return None
    first = float(s.iloc[0])
    last = float(s.iloc[-1])
    if first <= 0 or last <= 0:
        return None
    n_periods = len(s) - 1
    try:
        return (last / first) ** (1.0 / n_periods) - 1.0
    except Exception:
        return None


def compute_ratios(financials: pd.DataFrame) -> RatiosResult:
    """
    Compute ratios from cleaned financials.

    Required columns (best effort; will warn if missing):
    - revenue, ebit, net_income, free_cash_flow, total_assets, total_equity, total_debt, cash
    """
    warnings: List[str] = []

    if financials is None or financials.empty:
        warnings.append("Financials table is empty; cannot compute ratios.")
        empty = pd.DataFrame()
        return RatiosResult(empty, pd.DataFrame(), warnings)

    df = financials.copy()
    df = df.sort_index()

    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Helper getters (Series aligned to index)
    revenue = df.get("revenue")
    ebit = df.get("ebit")
    net_income = df.get("net_income")
    fcf = df.get("free_cash_flow")
    assets = df.get("total_assets")
    equity = df.get("total_equity")
    debt = df.get("total_debt")
    cash = df.get("cash")

    # Check expected fields
    expected = ["revenue", "ebit", "net_income", "free_cash_flow", "total_assets", "total_equity", "total_debt", "cash"]
    for f in expected:
        if f not in df.columns:
            warnings.append(f"Missing field '{f}' in financials; some ratios may be unavailable.")

    out = pd.DataFrame(index=df.index)

    # Profitability margins
    if revenue is not None and ebit is not None:
        out["ebit_margin"] = _safe_div(ebit, revenue)
    else:
        out["ebit_margin"] = pd.NA

    if revenue is not None and net_income is not None:
        out["net_margin"] = _safe_div(net_income, revenue)
    else:
        out["net_margin"] = pd.NA

    if revenue is not None and fcf is not None:
        out["fcf_margin"] = _safe_div(fcf, revenue)
    else:
        out["fcf_margin"] = pd.NA

    # Leverage
    if equity is not None and debt is not None:
        out["debt_to_equity"] = _safe_div(debt, equity)
    else:
        out["debt_to_equity"] = pd.NA

    if debt is not None and cash is not None:
        out["net_debt"] = debt.astype(float) - cash.astype(float)
    else:
        out["net_debt"] = pd.NA

    # Efficiency / returns
    if equity is not None and net_income is not None:
        out["roe"] = _safe_div(net_income, equity)
    else:
        out["roe"] = pd.NA

    if assets is not None and revenue is not None:
        out["asset_turnover"] = _safe_div(revenue, assets)
    else:
        out["asset_turnover"] = pd.NA

    # Growth (CAGRs shown as constants across years for readability)
    rev_cagr = _cagr(revenue) if revenue is not None else None
    fcf_cagr = _cagr(fcf) if fcf is not None else None

    out["revenue_cagr"] = rev_cagr
    out["fcf_cagr"] = fcf_cagr

    # Clean obvious infinities
    out = out.replace([float("inf"), float("-inf")], pd.NA)

    # Summary (latest year snapshot + CAGRs)
    latest_year = int(out.index.max())
    summary: Dict[str, Any] = {"year": latest_year}

    for col in ["ebit_margin", "net_margin", "fcf_margin", "debt_to_equity", "roe", "asset_turnover"]:
        v = out.loc[latest_year, col]
        summary[col] = None if pd.isna(v) else float(v)

    summary["revenue_cagr"] = rev_cagr
    summary["fcf_cagr"] = fcf_cagr

    summary_df = pd.DataFrame([summary]).set_index("year")

    # Warnings if key ratios are fully missing
    for col in ["ebit_margin", "net_margin", "fcf_margin"]:
        if out[col].isna().all():
            warnings.append(f"{col} unavailable (missing required inputs across all years).")

    return RatiosResult(ratios_df=out, summary_df=summary_df, warnings=warnings)


def save_ratios_table(ratios_df: pd.DataFrame, out_dir: str) -> str:
    out_path = Path(out_dir) / "ratios.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ratios_df.to_csv(out_path, index=True)
    return str(out_path)


def save_ratios_summary(summary_df: pd.DataFrame, out_dir: str) -> str:
    out_path = Path(out_dir) / "ratios_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=True)
    return str(out_path)
