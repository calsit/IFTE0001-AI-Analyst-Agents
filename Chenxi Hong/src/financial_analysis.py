"""
Financial Analysis Module for Fundamental Analysis

Features:
1. Analyze three major financial statements (Income Statement, Balance Sheet, Cash Flow Statement)
2. Calculate financial ratios (profitability, leverage, growth, efficiency)
3. Perform trend analysis
4. Assess financial health

The agent evaluates both level and trend of each ratio, flagging structural improvements or deteriorations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== Helper Functions ====================

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert to float, handling None and string "None"
    
    Args:
        value: Value to convert
        default: Default value
    
    Returns:
        Converted float value
    """
    if value is None or value == "None" or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_get_value(df: pd.DataFrame, column: str, index: int = -1) -> float:
    """
    Safely get value from DataFrame
    
    Args:
        df: DataFrame
        column: Column name
        index: Index position (default -1, i.e., last row)
    
    Returns:
        Value (returns 0 if not found)
    """
    if df.empty or column not in df.columns:
        return 0.0
    try:
        value = df[column].iloc[index]
        return safe_float(value, 0.0)
    except (IndexError, KeyError):
        return 0.0

def calculate_cagr(values: pd.Series, years: int) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR)
    
    Args:
        values: Time series values (sorted by time, newest first)
        years: Number of years
    
    Returns:
        CAGR (percentage)
    """
    if len(values) < 2:
        return 0.0
    
    # Ensure all values are positive
    values = values[values > 0]
    if len(values) < 2:
        return 0.0
    
    start_value = values.iloc[-1]  # Earliest value
    end_value = values.iloc[0]      # Latest value
    
    if start_value <= 0:
        return 0.0
    
    cagr = ((end_value / start_value) ** (1.0 / years) - 1) * 100
    return cagr

# ==================== Profitability Analysis ====================

def calculate_profitability_ratios(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate profitability ratios
    
    Includes:
    - Gross Margin
    - Operating Margin
    - Net Margin
    - ROE (Return on Equity)
    - ROIC (Return on Invested Capital)
    
    Args:
        income_df: Income statement DataFrame
        balance_df: Balance sheet DataFrame
    
    Returns:
        Dictionary of profitability ratios
    """
    ratios = {}
    
    if income_df.empty:
        return ratios
    
    try:
        # Get latest fiscal year data
        revenue = safe_get_value(income_df, "totalRevenue")
        gross_profit = safe_get_value(income_df, "grossProfit")
        operating_income = safe_get_value(income_df, "operatingIncome")
        net_income = safe_get_value(income_df, "netIncome")
        
        # Gross margin
        if revenue > 0:
            ratios["gross_margin"] = (gross_profit / revenue) * 100
        else:
            ratios["gross_margin"] = None
        
        # Operating margin
        if revenue > 0:
            ratios["operating_margin"] = (operating_income / revenue) * 100 if operating_income else None
        else:
            ratios["operating_margin"] = None
        
        # Net margin
        if revenue > 0:
            ratios["net_margin"] = (net_income / revenue) * 100
        else:
            ratios["net_margin"] = None
        
        # ROE (Return on Equity)
        if not balance_df.empty:
            total_equity = safe_get_value(balance_df, "totalShareholderEquity")
            if total_equity > 0:
                ratios["roe"] = (net_income / total_equity) * 100
            else:
                ratios["roe"] = None
        else:
            ratios["roe"] = None
        
        # ROIC (Return on Invested Capital)
        # ROIC = (NOPAT) / (Total Debt + Total Equity - Cash)
        if not balance_df.empty:
            total_assets = safe_get_value(balance_df, "totalAssets")
            total_liabilities = safe_get_value(balance_df, "totalLiabilities")
            cash = safe_get_value(balance_df, "cashAndCashEquivalentsAtCarryingValue")
            total_equity = safe_get_value(balance_df, "totalShareholderEquity")
            
            # Calculate invested capital
            invested_capital = total_assets - cash  # Simplified calculation
            # Or: invested_capital = total_liabilities + total_equity - cash
            
            # NOPAT = Operating Income * (1 - Tax Rate)
            # Simplified: use Operating Income as approximation
            tax_rate = safe_get_value(income_df, "incomeTaxExpense") / max(operating_income, 1) if operating_income > 0 else 0.21  # Default 21%
            nopat = operating_income * (1 - tax_rate) if operating_income else 0
            
            if invested_capital > 0:
                ratios["roic"] = (nopat / invested_capital) * 100
            else:
                ratios["roic"] = None
        else:
            ratios["roic"] = None
        
        # Calculate historical trends (if multi-year data available)
        if len(income_df) > 1:
            trends = {}
            
            # Gross margin trend
            if "totalRevenue" in income_df.columns and "grossProfit" in income_df.columns:
                revenue_series = income_df["totalRevenue"]
                gross_profit_series = income_df["grossProfit"]
                gross_margin_series = (gross_profit_series / revenue_series * 100).where(revenue_series > 0)
                if not gross_margin_series.isna().all():
                    trends["gross_margin_trend"] = "improving" if gross_margin_series.iloc[0] > gross_margin_series.iloc[-1] else "declining"
            
            # Net margin trend
            if "totalRevenue" in income_df.columns and "netIncome" in income_df.columns:
                revenue_series = income_df["totalRevenue"]
                net_income_series = income_df["netIncome"]
                net_margin_series = (net_income_series / revenue_series * 100).where(revenue_series > 0)
                if not net_margin_series.isna().all():
                    trends["net_margin_trend"] = "improving" if net_margin_series.iloc[0] > net_margin_series.iloc[-1] else "declining"
            
            ratios["trends"] = trends
    
    except Exception as e:
        print(f"⚠️ Error calculating profitability ratios: {e}")
    
    return ratios

# ==================== Leverage and Solvency Analysis ====================

def calculate_leverage_ratios(
    balance_df: pd.DataFrame,
    income_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate leverage and solvency ratios
    
    Includes:
    - Debt/Equity
    - Net Debt/EBITDA
    - Interest Coverage
    
    Args:
        balance_df: Balance sheet DataFrame
        income_df: Income statement DataFrame
    
    Returns:
        Dictionary of leverage and solvency ratios
    """
    ratios = {}
    
    if balance_df.empty:
        return ratios
    
    try:
        # Get latest fiscal year data
        total_liabilities = safe_get_value(balance_df, "totalLiabilities")
        total_equity = safe_get_value(balance_df, "totalShareholderEquity")
        cash = safe_get_value(balance_df, "cashAndCashEquivalentsAtCarryingValue")
        
        # Try to get long-term debt
        long_term_debt = safe_get_value(balance_df, "longTermDebt")
        if long_term_debt == 0:
            # If no long-term debt field, try other fields
            long_term_debt = safe_get_value(balance_df, "longTermDebtNoncurrent")
        
        # Try to get short-term debt
        short_term_debt = safe_get_value(balance_df, "shortTermDebt")
        if short_term_debt == 0:
            short_term_debt = safe_get_value(balance_df, "currentDebt")
        
        total_debt = long_term_debt + short_term_debt
        
        # Debt/Equity
        if total_equity > 0:
            ratios["debt_to_equity"] = total_debt / total_equity
        else:
            ratios["debt_to_equity"] = None
        
        # Net Debt
        net_debt = total_debt - cash
        
        # Net Debt/EBITDA
        if not income_df.empty:
            ebitda = safe_get_value(income_df, "ebitda")
            if ebitda == 0:
                # If no EBITDA field, calculate: EBITDA = Operating Income + Depreciation
                operating_income = safe_get_value(income_df, "operatingIncome")
                depreciation = safe_get_value(income_df, "depreciationAndAmortization")
                ebitda = operating_income + depreciation
            
            if ebitda > 0:
                ratios["net_debt_to_ebitda"] = net_debt / ebitda
            else:
                ratios["net_debt_to_ebitda"] = None
            
            # Interest Coverage
            interest_expense = safe_get_value(income_df, "interestExpense")
            if interest_expense == 0:
                # Try other field names
                interest_expense = safe_get_value(income_df, "interestAndDebtExpense")
            
            operating_income = safe_get_value(income_df, "operatingIncome")
            if interest_expense > 0:
                ratios["interest_coverage"] = operating_income / interest_expense
            else:
                ratios["interest_coverage"] = None  # No interest expense, cannot calculate
        else:
            ratios["net_debt_to_ebitda"] = None
            ratios["interest_coverage"] = None
        
        # Debt to Assets Ratio
        total_assets = safe_get_value(balance_df, "totalAssets")
        if total_assets > 0:
            ratios["debt_to_assets"] = (total_liabilities / total_assets) * 100
        else:
            ratios["debt_to_assets"] = None
        
        # Current Ratio
        current_assets = safe_get_value(balance_df, "totalCurrentAssets")
        current_liabilities = safe_get_value(balance_df, "totalCurrentLiabilities")
        if current_liabilities > 0:
            ratios["current_ratio"] = current_assets / current_liabilities
        else:
            ratios["current_ratio"] = None
    
    except Exception as e:
        print(f"⚠️ Error calculating leverage ratios: {e}")
    
    return ratios

# ==================== Growth Analysis ====================

def calculate_growth_ratios(
    income_df: pd.DataFrame,
    years: int = 5
) -> Dict[str, Any]:
    """
    Calculate growth ratios
    
    Includes:
    - Revenue CAGR (5y)
    - EBITDA CAGR
    - EPS CAGR (if available)
    
    Args:
        income_df: Income statement DataFrame (sorted by time, newest first)
        years: Number of years for CAGR calculation (default 5 years)
    
    Returns:
        Dictionary of growth ratios
    """
    ratios = {}
    
    if income_df.empty or len(income_df) < 2:
        return ratios
    
    try:
        # Revenue CAGR
        if "totalRevenue" in income_df.columns:
            revenue_series = income_df["totalRevenue"].dropna()
            if len(revenue_series) >= 2:
                # Ensure sufficient data points
                actual_years = min(years, len(revenue_series) - 1)
                if actual_years > 0:
                    ratios["revenue_cagr"] = calculate_cagr(revenue_series, actual_years)
                else:
                    ratios["revenue_cagr"] = None
            else:
                ratios["revenue_cagr"] = None
        else:
            ratios["revenue_cagr"] = None
        
        # EBITDA CAGR
        if "ebitda" in income_df.columns:
            ebitda_series = income_df["ebitda"].dropna()
        else:
            # Calculate EBITDA = Operating Income + Depreciation
            operating_income = income_df["operatingIncome"] if "operatingIncome" in income_df.columns else pd.Series()
            depreciation = income_df["depreciationAndAmortization"] if "depreciationAndAmortization" in income_df.columns else pd.Series()
            if not operating_income.empty and not depreciation.empty:
                ebitda_series = (operating_income + depreciation).dropna()
            else:
                ebitda_series = pd.Series()
        
        if len(ebitda_series) >= 2:
            actual_years = min(years, len(ebitda_series) - 1)
            if actual_years > 0:
                ratios["ebitda_cagr"] = calculate_cagr(ebitda_series, actual_years)
            else:
                ratios["ebitda_cagr"] = None
        else:
            ratios["ebitda_cagr"] = None
        
        # EPS CAGR (if available)
        if "reportedEPS" in income_df.columns:
            eps_series = income_df["reportedEPS"].dropna()
            if len(eps_series) >= 2:
                actual_years = min(years, len(eps_series) - 1)
                if actual_years > 0:
                    ratios["eps_cagr"] = calculate_cagr(eps_series, actual_years)
                else:
                    ratios["eps_cagr"] = None
            else:
                ratios["eps_cagr"] = None
        else:
            ratios["eps_cagr"] = None
        
        # Year-over-year growth rate (latest year)
        if "totalRevenue" in income_df.columns and len(income_df) >= 2:
            latest_revenue = safe_get_value(income_df, "totalRevenue", 0)
            prev_revenue = safe_get_value(income_df, "totalRevenue", 1)
            if prev_revenue > 0:
                ratios["revenue_growth_yoy"] = ((latest_revenue - prev_revenue) / prev_revenue) * 100
            else:
                ratios["revenue_growth_yoy"] = None
        
        if "netIncome" in income_df.columns and len(income_df) >= 2:
            latest_net_income = safe_get_value(income_df, "netIncome", 0)
            prev_net_income = safe_get_value(income_df, "netIncome", 1)
            if prev_net_income != 0:
                ratios["net_income_growth_yoy"] = ((latest_net_income - prev_net_income) / abs(prev_net_income)) * 100
            else:
                ratios["net_income_growth_yoy"] = None
    
    except Exception as e:
        print(f"⚠️ Error calculating growth ratios: {e}")
    
    return ratios

# ==================== Operational Efficiency Analysis ====================

def calculate_efficiency_ratios(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate operational efficiency ratios
    
    Includes:
    - Asset Turnover
    - Inventory Turnover
    - Receivables Turnover
    - Cash Conversion Cycle
    
    Args:
        income_df: Income statement DataFrame
        balance_df: Balance sheet DataFrame
    
    Returns:
        Dictionary of operational efficiency ratios
    """
    ratios = {}
    
    if income_df.empty or balance_df.empty:
        return ratios
    
    try:
        # Get latest fiscal year data
        revenue = safe_get_value(income_df, "totalRevenue")
        total_assets = safe_get_value(balance_df, "totalAssets")
        inventory = safe_get_value(balance_df, "inventory")
        receivables = safe_get_value(balance_df, "netReceivables")
        if receivables == 0:
            receivables = safe_get_value(balance_df, "accountsReceivable")
        
        # Asset Turnover
        if total_assets > 0:
            ratios["asset_turnover"] = revenue / total_assets
        else:
            ratios["asset_turnover"] = None
        
        # Inventory Turnover
        # Need cost data, use approximation if not available
        cost_of_revenue = safe_get_value(income_df, "costOfRevenue")
        if cost_of_revenue == 0:
            cost_of_revenue = safe_get_value(income_df, "costofGoodsAndServicesSold")
        
        if inventory > 0 and cost_of_revenue > 0:
            ratios["inventory_turnover"] = cost_of_revenue / inventory
        else:
            ratios["inventory_turnover"] = None
        
        # Receivables Turnover
        if receivables > 0:
            ratios["receivables_turnover"] = revenue / receivables
        else:
            ratios["receivables_turnover"] = None
        
        # Cash Conversion Cycle (days)
        # CCC = DIO + DSO - DPO
        # DIO = Days Inventory Outstanding = 365 / Inventory Turnover
        # DSO = Days Sales Outstanding = 365 / Receivables Turnover
        # DPO = Days Payable Outstanding = 365 / Payables Turnover
        
        if ratios.get("inventory_turnover") and ratios["inventory_turnover"] > 0:
            dio = 365 / ratios["inventory_turnover"]
        else:
            dio = None
        
        if ratios.get("receivables_turnover") and ratios["receivables_turnover"] > 0:
            dso = 365 / ratios["receivables_turnover"]
        else:
            dso = None
        
        # Calculate DPO
        payables = safe_get_value(balance_df, "accountPayables")
        if payables == 0:
            payables = safe_get_value(balance_df, "accountsPayable")
        
        if payables > 0 and cost_of_revenue > 0:
            payables_turnover = cost_of_revenue / payables
            dpo = 365 / payables_turnover
        else:
            dpo = None
        
        # Calculate CCC
        if dio is not None and dso is not None and dpo is not None:
            ratios["cash_conversion_cycle"] = dio + dso - dpo
        elif dio is not None and dso is not None:
            ratios["cash_conversion_cycle"] = dio + dso  # Simplified calculation
        else:
            ratios["cash_conversion_cycle"] = None
        
        # Save intermediate calculation results
        ratios["days_inventory_outstanding"] = dio
        ratios["days_sales_outstanding"] = dso
        ratios["days_payable_outstanding"] = dpo
    
    except Exception as e:
        print(f"⚠️ Error calculating operational efficiency ratios: {e}")
    
    return ratios

# ==================== Comprehensive Analysis Functions ====================

def analyze_financial_statements(
    financial_data: Dict[str, Any],
    years: int = 5
) -> Dict[str, Any]:
    """
    Comprehensively analyze three major financial statements, calculate all financial ratios
    
    Args:
        financial_data: Data from data_ingestion.fetch_financial_statements()
        years: Number of years for CAGR calculation (default 5 years)
    
    Returns:
        Dictionary containing all financial ratios and analysis results
    """
    result = {
        "symbol": financial_data.get("symbol", "UNKNOWN"),
        "profitability": {},
        "leverage": {},
        "growth": {},
        "efficiency": {},
        "summary": {}
    }
    
    try:
        # Get DataFrames
        income_annual = financial_data.get("income_statement", {}).get("annual", pd.DataFrame())
        balance_annual = financial_data.get("balance_sheet", {}).get("annual", pd.DataFrame())
        cashflow_annual = financial_data.get("cash_flow", {}).get("annual", pd.DataFrame())
        
        # Ensure data is sorted by time (newest first)
        if not income_annual.empty and isinstance(income_annual.index, pd.DatetimeIndex):
            income_annual = income_annual.sort_index(ascending=False)
        if not balance_annual.empty and isinstance(balance_annual.index, pd.DatetimeIndex):
            balance_annual = balance_annual.sort_index(ascending=False)
        if not cashflow_annual.empty and isinstance(cashflow_annual.index, pd.DatetimeIndex):
            cashflow_annual = cashflow_annual.sort_index(ascending=False)
        
        # 1. Profitability analysis
        print("  📊 Calculating profitability ratios...")
        result["profitability"] = calculate_profitability_ratios(income_annual, balance_annual)
        
        # 2. Leverage and solvency analysis
        print("  📊 Calculating leverage and solvency ratios...")
        result["leverage"] = calculate_leverage_ratios(balance_annual, income_annual)
        
        # 3. Growth analysis
        print("  📊 Calculating growth ratios...")
        result["growth"] = calculate_growth_ratios(income_annual, years=years)
        
        # 4. Operational efficiency analysis
        print("  📊 Calculating operational efficiency ratios...")
        result["efficiency"] = calculate_efficiency_ratios(income_annual, balance_annual)
        
        # 5. Generate summary
        result["summary"] = generate_financial_summary(result)
        
        print("  ✅ Financial ratio analysis completed")
    
    except Exception as e:
        print(f"⚠️ Error analyzing financial statements: {e}")
        import traceback
        traceback.print_exc()
    
    return result

def generate_financial_summary(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate financial analysis summary
    
    Args:
        analysis_result: Results from analyze_financial_statements
    
    Returns:
        Summary dictionary
    """
    summary = {
        "key_metrics": {},
        "trends": [],
        "strengths": [],
        "concerns": []
    }
    
    try:
        profitability = analysis_result.get("profitability", {})
        leverage = analysis_result.get("leverage", {})
        growth = analysis_result.get("growth", {})
        efficiency = analysis_result.get("efficiency", {})
        
        # Key metrics
        if profitability.get("roic"):
            summary["key_metrics"]["ROIC"] = f"{profitability['roic']:.2f}%"
        if profitability.get("net_margin"):
            summary["key_metrics"]["Net Margin"] = f"{profitability['net_margin']:.2f}%"
        if growth.get("revenue_cagr"):
            summary["key_metrics"]["Revenue CAGR (5y)"] = f"{growth['revenue_cagr']:.2f}%"
        if leverage.get("debt_to_equity"):
            summary["key_metrics"]["Debt/Equity"] = f"{leverage['debt_to_equity']:.2f}"
        
        # Trend analysis
        if profitability.get("trends"):
            trends = profitability["trends"]
            if trends.get("gross_margin_trend") == "improving":
                summary["trends"].append("Gross margin showing upward trend")
            elif trends.get("gross_margin_trend") == "declining":
                summary["trends"].append("Gross margin showing downward trend")
        
        # Strengths
        if profitability.get("roic") and profitability["roic"] > 15:
            summary["strengths"].append("ROIC exceeds 15%, excellent capital return")
        if profitability.get("net_margin") and profitability["net_margin"] > 20:
            summary["strengths"].append("Net margin exceeds 20%, strong profitability")
        if growth.get("revenue_cagr") and growth["revenue_cagr"] > 10:
            summary["strengths"].append("Revenue CAGR exceeds 10%, strong growth")
        
        # Concerns
        if leverage.get("debt_to_equity") and leverage["debt_to_equity"] > 1.0:
            summary["concerns"].append(f"Debt/equity ratio {leverage['debt_to_equity']:.2f}, high leverage level")
        if leverage.get("interest_coverage") and leverage["interest_coverage"] < 2.0:
            summary["concerns"].append(f"Interest coverage {leverage['interest_coverage']:.2f}, solvency needs attention")
        if profitability.get("net_margin") and profitability["net_margin"] < 5:
            summary["concerns"].append("Net margin below 5%, weak profitability")
    
    except Exception as e:
        print(f"⚠️ Error generating financial summary: {e}")
    
    return summary

# ==================== Main Function (for testing) ====================

if __name__ == "__main__":
    # Test example
    print("=" * 60)
    print("Financial Analysis Module - Financial Analysis Test")
    print("=" * 60)
    
    # Need to fetch data first
    from data_ingestion import fetch_financial_statements
    
    symbol = "NVDA"
    print(f"\nFetching financial statement data for {symbol}...")
    financial_data = fetch_financial_statements(symbol, years=5)
    
    print(f"\nAnalyzing financial ratios for {symbol}...")
    analysis = analyze_financial_statements(financial_data, years=5)
    
    # Display results
    print("\n" + "=" * 60)
    print("Analysis Results:")
    print("=" * 60)
    
    print("\n📊 Profitability:")
    for key, value in analysis["profitability"].items():
        if key != "trends" and value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value}")
    
    print("\n📊 Leverage & Solvency:")
    for key, value in analysis["leverage"].items():
        if value is not None:
            if isinstance(value, float):
                if "ratio" in key or "coverage" in key:
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value:.2f}%")
            else:
                print(f"  {key}: {value}")
    
    print("\n📊 Growth:")
    for key, value in analysis["growth"].items():
        if value is not None:
            print(f"  {key}: {value:.2f}%")
    
    print("\n📊 Operational Efficiency:")
    for key, value in analysis["efficiency"].items():
        if value is not None:
            if "cycle" in key or "days" in key:
                print(f"  {key}: {value:.1f} days")
            else:
                print(f"  {key}: {value:.2f}")
    
    print("\n📋 Summary:")
    summary = analysis["summary"]
    print("  Key Metrics:")
    for key, value in summary["key_metrics"].items():
        print(f"    {key}: {value}")
    
    if summary["strengths"]:
        print("  Strengths:")
        for strength in summary["strengths"]:
            print(f"    ✅ {strength}")
    
    if summary["concerns"]:
        print("  Concerns:")
        for concern in summary["concerns"]:
            print(f"    ⚠️ {concern}")
    
    print("\n✅ Test completed!")

