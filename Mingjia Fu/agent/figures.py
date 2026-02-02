# agent/figures.py
"""

Produces TWO analyst-style charts to support an investment thesis:

1) Revenue & Free Cash Flow trend (financial scale + cash generation)
2) Margin profile (EBIT / Net / FCF margins)

Design notes:
- This project intentionally does NOT produce a DCF sensitivity grid figure.
- All functions are best-effort: they never raise on missing inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class FigureResult:
    paths: List[str]
    warnings: List[str]


def _ensure_dir(out_dir: str) -> Path:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_bn(x: pd.Series) -> pd.Series:
    # Convert to billions for readability
    return x.astype(float) / 1e9


def _safe_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if df is None or df.empty or col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.isna().all():
        return None
    return s


def plot_revenue_fcf(
    financials_df: pd.DataFrame,
    out_dir: str = "outputs/figures",
    filename: str = "fig1_revenue_fcf.png",
) -> Tuple[Optional[str], List[str]]:
    warnings: List[str] = []
    out_path = _ensure_dir(out_dir) / filename

    if financials_df is None or financials_df.empty:
        warnings.append("Figure 1 skipped: financials_df is empty.")
        return None, warnings

    rev = _safe_series(financials_df, "revenue")
    fcf = _safe_series(financials_df, "free_cash_flow")

    if rev is None and fcf is None:
        warnings.append("Figure 1 skipped: revenue and free_cash_flow are unavailable.")
        return None, warnings

    years = financials_df.index.astype(int)

    plt.figure()
    if rev is not None:
        plt.plot(years, _to_bn(rev), marker="o", label="Revenue (USD bn)")
    else:
        warnings.append("Figure 1: revenue unavailable; plotted only FCF.")

    if fcf is not None:
        plt.plot(years, _to_bn(fcf), marker="o", label="Free Cash Flow (USD bn)")
    else:
        warnings.append("Figure 1: free_cash_flow unavailable; plotted only revenue.")

    plt.title("Revenue and Free Cash Flow Trend")
    plt.xlabel("Fiscal year")
    plt.ylabel("USD (bn)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return str(out_path), warnings


def plot_margin_profile(
    ratios_df: pd.DataFrame,
    out_dir: str = "outputs/figures",
    filename: str = "fig2_margin_profile.png",
) -> Tuple[Optional[str], List[str]]:
    warnings: List[str] = []
    out_path = _ensure_dir(out_dir) / filename

    if ratios_df is None or ratios_df.empty:
        warnings.append("Figure 2 skipped: ratios_df is empty.")
        return None, warnings

    years = ratios_df.index.astype(int)
    ebit_m = _safe_series(ratios_df, "ebit_margin")
    net_m = _safe_series(ratios_df, "net_margin")
    fcf_m = _safe_series(ratios_df, "fcf_margin")

    if ebit_m is None and net_m is None and fcf_m is None:
        warnings.append("Figure 2 skipped: margin columns are unavailable.")
        return None, warnings

    plt.figure()
    if ebit_m is not None:
        plt.plot(years, ebit_m * 100, marker="o", label="EBIT margin (%)")
    else:
        warnings.append("Figure 2: ebit_margin unavailable.")

    if net_m is not None:
        plt.plot(years, net_m * 100, marker="o", label="Net margin (%)")
    else:
        warnings.append("Figure 2: net_margin unavailable.")

    if fcf_m is not None:
        plt.plot(years, fcf_m * 100, marker="o", label="FCF margin (%)")
    else:
        warnings.append("Figure 2: fcf_margin unavailable.")

    plt.title("Margin Profile")
    plt.xlabel("Fiscal year")
    plt.ylabel("Percent (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return str(out_path), warnings


def generate_all_figures(
    *,
    financials_df: pd.DataFrame,
    ratios_df: pd.DataFrame,
    dcf_df: Optional[pd.DataFrame] = None,  # kept for compatibility; ignored intentionally
    out_dir: str = "outputs/figures",
) -> FigureResult:
    """
    Convenience wrapper used by run_demo.py.
    Never raises. Returns saved paths + warnings.

    Note:
    - Only Fig1 and Fig2 are generated in this project.
    """
    paths: List[str] = []
    warnings: List[str] = []

    p1, w1 = plot_revenue_fcf(financials_df, out_dir=out_dir)
    if p1:
        paths.append(p1)
    warnings.extend(w1)

    p2, w2 = plot_margin_profile(ratios_df, out_dir=out_dir)
    if p2:
        paths.append(p2)
    warnings.extend(w2)

    return FigureResult(paths=paths, warnings=warnings)
