# agent/processing.py
"""
Takes the raw multi-year statement table from ingestion and prepares a clean,
consistent dataset for downstream ratios, valuation, and memo generation.

Key behaviours:
- keeps numbers as numbers (coerces non-numeric to NaN)
- sorts by fiscal year
- drops fully-empty fiscal years (common when the data source returns < N years)
- produces clear warnings that can be surfaced in the memo
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


def process_financials(financials: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Clean and validate the ingested financials table.

    Args:
        financials: DataFrame indexed by fiscal year (int), produced by ingestion.

    Returns:
        cleaned: cleaned DataFrame (still indexed by fiscal year)
        warnings: list of issues to surface later (memo transparency)
    """
    warnings: List[str] = []

    if financials is None or financials.empty:
        warnings.append("Financials table is empty after ingestion.")
        return financials, warnings

    df = financials.copy()

    # Sort by fiscal year (oldest -> newest)
    try:
        df = df.sort_index()
    except Exception:
        warnings.append("Could not sort financials by fiscal year index.")

    # Coerce all columns to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop fiscal years that are completely empty across all fields
    all_nan_rows = df.isna().all(axis=1)
    if all_nan_rows.any():
        dropped_years = [int(y) for y in df.index[all_nan_rows].tolist()]
        df = df.loc[~all_nan_rows].copy()
        warnings.append(
            f"Dropped empty fiscal years with no statement data: {dropped_years}"
        )

    if df.empty:
        warnings.append("Financials table became empty after dropping empty years.")
        return df, warnings

    # Fields downstream components usually expect
    required_fields = [
        "revenue",
        "net_income",
        "free_cash_flow",
        "total_assets",
        "total_equity",
    ]

    for field in required_fields:
        if field not in df.columns:
            warnings.append(f"Missing expected field '{field}' in financials table.")
        elif df[field].isna().all():
            warnings.append(f"Field '{field}' is missing for all available years.")

    # Year-level diagnostics (useful for memo transparency)
    for year, row in df.iterrows():
        year_label = int(year) if isinstance(year, (int, float)) else year

        if pd.isna(row.get("revenue")):
            warnings.append(f"{year_label}: Revenue missing.")
        if pd.isna(row.get("net_income")):
            warnings.append(f"{year_label}: Net income missing.")
        if pd.isna(row.get("free_cash_flow")):
            warnings.append(f"{year_label}: Free cash flow missing.")
        if pd.isna(row.get("total_assets")):
            warnings.append(f"{year_label}: Total assets missing.")
        if pd.isna(row.get("total_equity")):
            warnings.append(f"{year_label}: Total equity missing.")

    return df, warnings
