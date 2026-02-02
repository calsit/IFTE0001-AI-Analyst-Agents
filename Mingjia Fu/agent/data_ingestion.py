# agent/data_ingestion.py
"""

Fetch annual financial statements from Yahoo Finance (via yfinance) and map
the fields used by downstream processing, ratios and valuation.

Notes:
- Yahoo / yfinance field labels are not fully stable, so this module uses
  a small set of row-label fallbacks.
- This module avoids year-by-year warnings to prevent duplicated diagnostics.
  Year-level checks are handled in processing.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf


@dataclass
class IngestionResult:
    """
    Output of the ingestion step.

    metadata:
        Basic context: ticker, company_name, currency, shares_outstanding, years, source.
    financials:
        Annual table indexed by fiscal year (int), sorted oldest -> newest.
    warnings:
        Non-fatal issues about availability/coverage (not year-by-year diagnostics).
    """

    metadata: Dict[str, Any]
    financials: pd.DataFrame
    warnings: List[str]


# yfinance labels may vary across versions/tickers, so we try a few candidates.
ROW_LABELS = {
    "revenue": ["Total Revenue", "TotalRevenue", "Revenue"],
    "ebit": ["EBIT", "Ebit", "Operating Income", "OperatingIncome"],
    "net_income": ["Net Income", "NetIncome", "Net Income Common Stockholders"],
    "operating_cash_flow": [
        "Operating Cash Flow",
        "OperatingCashFlow",
        "Total Cash From Operating Activities",
    ],
    "capex": [
        "Capital Expenditure",
        "CapitalExpenditure",
        "Capital Expenditures",
        "CapitalExpenditures",
    ],
    "free_cash_flow": ["Free Cash Flow", "FreeCashFlow"],
    "cash": [
        "Cash And Cash Equivalents",
        "CashAndCashEquivalents",
        "Cash",
        "Cash Cash Equivalents And Short Term Investments",
    ],
    "short_term_debt": [
        "Short Long Term Debt",
        "ShortLongTermDebt",
        "Short Term Debt",
        "ShortTermDebt",
    ],
    "long_term_debt": ["Long Term Debt", "LongTermDebt"],
    "total_debt": ["Total Debt", "TotalDebt"],
    "total_equity": [
        "Total Stockholder Equity",
        "TotalStockholderEquity",
        "Total Equity",
        "Stockholders Equity",
    ],
    "total_assets": ["Total Assets", "TotalAssets"],
}

OUTPUT_COLUMNS = [
    "revenue",
    "ebit",
    "net_income",
    "operating_cash_flow",
    "capex",
    "free_cash_flow",
    "cash",
    "short_term_debt",
    "long_term_debt",
    "total_debt",
    "total_equity",
    "total_assets",
]


def _get_statement(ticker_obj: yf.Ticker, attr_candidates: List[str]) -> Optional[pd.DataFrame]:
    """Return the first non-empty statement DataFrame from a list of attribute names."""
    for name in attr_candidates:
        try:
            df = getattr(ticker_obj, name)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception:
            continue
    return None


def _annual_columns(stmt: Optional[pd.DataFrame]) -> List[Any]:
    """Return annual statement columns sorted newest -> oldest."""
    if stmt is None or stmt.empty:
        return []
    cols = list(stmt.columns)
    try:
        return sorted(cols, reverse=True)
    except Exception:
        return cols


def _value(stmt: Optional[pd.DataFrame], row_candidates: List[str], col: Any) -> Optional[float]:
    """Try multiple row labels and return a numeric value for a given statement column."""
    if stmt is None or stmt.empty:
        return None

    for r in row_candidates:
        if r in stmt.index and col in stmt.columns:
            v = stmt.at[r, col]
            try:
                if pd.isna(v):
                    return None
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def fetch_financials_5y(ticker: str, years: int = 5) -> IngestionResult:
    """
    Fetch up to `years` annual periods of key financial statement fields.

    - DataFrame is indexed by fiscal year (int), sorted oldest -> newest.
    - FCF prefers a direct line item; otherwise approximates as OCF + Capex
      (Capex is usually negative on yfinance).
    """
    warnings: List[str] = []
    t = yf.Ticker(ticker)

    # Basic metadata (non-fatal if missing)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    company_name = info.get("shortName") or info.get("longName") or ticker
    currency = info.get("currency") or "N/A"
    shares_outstanding = info.get("sharesOutstanding")

    # Annual statements (attribute names differ across yfinance versions)
    income = _get_statement(t, ["income_stmt", "financials"])
    balance = _get_statement(t, ["balance_sheet", "balancesheet"])
    cashflow = _get_statement(t, ["cashflow_stmt", "cashflow"])

    income_cols = _annual_columns(income)
    balance_cols = _annual_columns(balance)
    cashflow_cols = _annual_columns(cashflow)

    all_cols = sorted(set(income_cols + balance_cols + cashflow_cols), reverse=True)[:years]

    if not all_cols:
        warnings.append("No annual statement data returned from yfinance (empty columns).")
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty.index.name = "year"
        return IngestionResult(
            metadata={
                "ticker": ticker,
                "company_name": company_name,
                "currency": currency,
                "shares_outstanding": shares_outstanding,
                "years": [],
                "source": "yfinance (Yahoo Finance)",
            },
            financials=empty,
            warnings=warnings,
        )

    records: List[Dict[str, Any]] = []

    # Build oldest -> newest for readability
    for col in all_cols[::-1]:
        try:
            year = pd.to_datetime(col).year
        except Exception:
            year = None

        revenue = _value(income, ROW_LABELS["revenue"], col)
        ebit = _value(income, ROW_LABELS["ebit"], col)
        net_income = _value(income, ROW_LABELS["net_income"], col)

        ocf = _value(cashflow, ROW_LABELS["operating_cash_flow"], col)
        capex = _value(cashflow, ROW_LABELS["capex"], col)
        fcf_direct = _value(cashflow, ROW_LABELS["free_cash_flow"], col)

        if fcf_direct is not None:
            free_cash_flow = fcf_direct
        elif ocf is not None and capex is not None:
            free_cash_flow = ocf + capex
        else:
            free_cash_flow = None

        cash = _value(balance, ROW_LABELS["cash"], col)
        st_debt = _value(balance, ROW_LABELS["short_term_debt"], col)
        lt_debt = _value(balance, ROW_LABELS["long_term_debt"], col)
        total_debt = _value(balance, ROW_LABELS["total_debt"], col)

        if total_debt is None and (st_debt is not None or lt_debt is not None):
            total_debt = float(st_debt or 0.0) + float(lt_debt or 0.0)

        total_equity = _value(balance, ROW_LABELS["total_equity"], col)
        total_assets = _value(balance, ROW_LABELS["total_assets"], col)

        records.append(
            {
                "year": year,
                "revenue": revenue,
                "ebit": ebit,
                "net_income": net_income,
                "operating_cash_flow": ocf,
                "capex": capex,
                "free_cash_flow": free_cash_flow,
                "cash": cash,
                "short_term_debt": st_debt,
                "long_term_debt": lt_debt,
                "total_debt": total_debt,
                "total_equity": total_equity,
                "total_assets": total_assets,
            }
        )

    df = pd.DataFrame.from_records(records)

    if df["year"].isna().any():
        warnings.append("Some annual columns were not date-like; year labels may be incomplete.")

    df = df.dropna(subset=["year"]).drop_duplicates(subset=["year"], keep="last")
    df["year"] = df["year"].astype(int)
    df = df.set_index("year").sort_index()
    df.index.name = "year"

    # Coverage warnings (high level, not year-by-year)
    def warn_if_all_missing(field: str, label: str) -> None:
        if field in df.columns and df[field].isna().all():
            warnings.append(f"Field '{label}' is missing for all years in the statements.")

    warn_if_all_missing("revenue", "Revenue")
    warn_if_all_missing("net_income", "Net income")
    warn_if_all_missing("ebit", "EBIT / Operating income")
    warn_if_all_missing("free_cash_flow", "Free cash flow")
    warn_if_all_missing("total_assets", "Total assets")
    warn_if_all_missing("total_equity", "Total equity")

    metadata = {
        "ticker": ticker,
        "company_name": company_name,
        "currency": currency,
        "shares_outstanding": shares_outstanding,
        "years": df.index.astype(int).tolist() if not df.empty else [],
        "source": "yfinance (Yahoo Finance)",
    }

    return IngestionResult(metadata=metadata, financials=df, warnings=warnings)


def save_financials_table(financials: pd.DataFrame, out_dir: str) -> str:
    """Save the multi-year financial table to CSV and return the saved path."""
    out_path = Path(out_dir) / "financials_5y.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    financials.to_csv(out_path, index=True)
    return str(out_path)
