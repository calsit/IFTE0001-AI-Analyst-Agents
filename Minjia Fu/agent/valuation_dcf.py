#agent/valuation_dcf.py
"""
This module implements a basic, transparent discounted cash flow (DCF)
valuation intended to provide an intrinsic value anchor rather than a
precise point estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DCFResult:
    dcf_table: pd.DataFrame
    summary: Dict[str, float]
    warnings: List[str]


def run_simple_dcf(
    financials: pd.DataFrame,
    config: dict,
    shares_outstanding: Optional[float],
) -> DCFResult:
    """
    Run a simplified multi-stage DCF.

    Inputs:
        financials: processed financials table (indexed by year)
        config: loaded config.json
        shares_outstanding: shares outstanding from ingestion metadata

    Returns:
        DCFResult with projection table, summary metrics, and warnings
    """
    warnings: List[str] = []

    if financials is None or financials.empty:
        warnings.append("Financials table is empty; cannot run DCF.")
        return DCFResult(pd.DataFrame(), {}, warnings)

    if "free_cash_flow" not in financials.columns:
        warnings.append("Free cash flow not found in financials.")
        return DCFResult(pd.DataFrame(), {}, warnings)

    # --- Base inputs ---
    latest_fcf = financials["free_cash_flow"].dropna().iloc[-1]

    if latest_fcf <= 0:
        warnings.append("Latest free cash flow is non-positive; DCF may be unreliable.")

    dcf_cfg = config.get("dcf", {})
    projection_years = int(dcf_cfg.get("projection_years", 5))
    wacc = float(dcf_cfg.get("wacc", 0.09))
    terminal_g = float(dcf_cfg.get("terminal_growth", 0.025))
    growth_cap = float(dcf_cfg.get("fcf_growth_cap", 0.15))

    # Estimate growth using historical CAGR (if possible)
    fcf_series = financials["free_cash_flow"].dropna()
    if len(fcf_series) >= 2:
        years = len(fcf_series) - 1
        hist_growth = (fcf_series.iloc[-1] / fcf_series.iloc[0]) ** (1 / years) - 1
        growth_rate = min(hist_growth, growth_cap)
    else:
        growth_rate = min(0.05, growth_cap)
        warnings.append("Insufficient FCF history; using conservative growth assumption.")

    # --- Projection ---
    projection = []
    fcf = latest_fcf

    for t in range(1, projection_years + 1):
        fcf = fcf * (1 + growth_rate)
        discount_factor = 1 / ((1 + wacc) ** t)
        projection.append(
            {
                "year": t,
                "projected_fcf": fcf,
                "discount_factor": discount_factor,
                "present_value": fcf * discount_factor,
            }
        )

    proj_df = pd.DataFrame(projection)

    # --- Terminal value ---
    terminal_value = (
        proj_df.iloc[-1]["projected_fcf"] * (1 + terminal_g) / (wacc - terminal_g)
    )
    terminal_pv = terminal_value / ((1 + wacc) ** projection_years)

    # --- Valuation ---
    enterprise_value = proj_df["present_value"].sum() + terminal_pv

    if shares_outstanding and shares_outstanding > 0:
        intrinsic_value_per_share = enterprise_value / shares_outstanding
    else:
        intrinsic_value_per_share = np.nan
        warnings.append("Shares outstanding missing; cannot compute per-share value.")

    # --- Output table ---
    dcf_table = proj_df.copy()
    dcf_table["terminal_value_pv"] = 0.0
    dcf_table.loc[dcf_table.index[-1], "terminal_value_pv"] = terminal_pv

    summary = {
        "enterprise_value": enterprise_value,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "wacc": wacc,
        "terminal_growth": terminal_g,
        "fcf_growth_used": growth_rate,
    }

    return DCFResult(dcf_table=dcf_table, summary=summary, warnings=warnings)


def save_dcf_table(dcf_table: pd.DataFrame, out_dir: str) -> str:
    """
    Save DCF projection table to CSV.
    """
    path = f"{out_dir.rstrip('/')}/dcf_projection.csv"
    dcf_table.to_csv(path, index=False)
    return path
