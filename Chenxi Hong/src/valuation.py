"""
Valuation Module for Fundamental Analysis

Features:
1. DCF Model (Discounted Cash Flow Model)
2. Comparable Multiples Analysis (EV/EBITDA, P/E, EV/Sales)
3. Peer Comparison

Assumptions are scenario-based rather than point estimates, improving robustness.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ==================== Helper Functions ====================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float"""
    if value is None or value == "None" or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_get_value(df: pd.DataFrame, column: str, index: int = -1) -> float:
    """Safely get value from DataFrame"""
    if df.empty or column not in df.columns:
        return 0.0
    try:
        value = df[column].iloc[index]
        return safe_float(value, 0.0)
    except (IndexError, KeyError):
        return 0.0

# ==================== DCF Model ====================

def calculate_wacc(
    equity_market_value: float,
    debt_market_value: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float = 0.21
) -> float:
    """
    Calculate Weighted Average Cost of Capital (WACC)
    
    WACC = (E/V) * Re + (D/V) * Rd * (1 - Tc)
    Where:
    - E = Equity market value
    - D = Debt market value
    - V = E + D (Total value)
    - Re = Cost of equity
    - Rd = Cost of debt
    - Tc = Tax rate
    
    Args:
        equity_market_value: Equity market value
        debt_market_value: Debt market value
        cost_of_equity: Cost of equity (typically using CAPM model, simplified here)
        cost_of_debt: Cost of debt (interest expense / total debt)
        tax_rate: Tax rate (default 21%)
    
    Returns:
        WACC (percentage)
    """
    total_value = equity_market_value + debt_market_value
    if total_value == 0:
        return 0.0
    
    equity_weight = equity_market_value / total_value
    debt_weight = debt_market_value / total_value
    
    wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
    return wacc * 100

def estimate_cost_of_equity(
    risk_free_rate: float = 0.04,
    beta: float = 1.0,
    market_risk_premium: float = 0.06
) -> float:
    """
    Estimate cost of equity (using CAPM model)
    
    Re = Rf + β * (Rm - Rf)
    
    Args:
        risk_free_rate: Risk-free rate (default 4%)
        beta: Beta coefficient (default 1.0)
        market_risk_premium: Market risk premium (default 6%)
    
    Returns:
        Cost of equity (percentage)
    """
    cost_of_equity = risk_free_rate + beta * market_risk_premium
    return cost_of_equity * 100

def forecast_revenue(
    historical_revenue: pd.Series,
    growth_rate: float,
    years: int = 5
) -> pd.Series:
    """
    Forecast future revenue
    
    Args:
        historical_revenue: Historical revenue series (newest first)
        growth_rate: Growth rate (percentage, e.g., 0.15 means 15%)
        years: Number of years to forecast
    
    Returns:
        Forecasted revenue series
    """
    if historical_revenue.empty:
        return pd.Series()
    
    latest_revenue = historical_revenue.iloc[0]
    forecasted = []
    
    for year in range(1, years + 1):
        revenue = latest_revenue * ((1 + growth_rate) ** year)
        forecasted.append(revenue)
    
    return pd.Series(forecasted, index=range(1, years + 1))

def calculate_dcf_valuation(
    financial_data: Dict[str, Any],
    revenue_growth_rate: float = None,
    operating_margin: float = None,
    tax_rate: float = 0.21,
    wacc: float = None,
    terminal_growth_rate: float = 0.03,
    forecast_years: int = 5
) -> Dict[str, Any]:
    """
    Calculate DCF valuation model
    
    DCF Model Structure:
    1. Revenue Forecast (3-5 years)
    2. Operating Margin Assumption
    3. Tax Rate
    4. CapEx & Depreciation
    5. ΔWorking Capital
    6. WACC
    7. Terminal Value (Gordon Growth)
    
    Args:
        financial_data: Financial data from data_ingestion
        revenue_growth_rate: Revenue growth rate (if None, calculated based on historical data)
        operating_margin: Operating margin (if None, calculated based on historical data)
        tax_rate: Tax rate (default 21%)
        wacc: WACC (if None, automatically calculated)
        terminal_growth_rate: Terminal growth rate (default 3%)
        forecast_years: Number of years to forecast (default 5 years)
    
    Returns:
        DCF valuation results dictionary
    """
    result = {
        "symbol": financial_data.get("symbol", "UNKNOWN"),
        "assumptions": {},
        "forecast": {},
        "dcf_value": None,
        "dcf_per_share": None,
        "sensitivity_analysis": {}
    }
    
    try:
        # Get data
        income_df = financial_data.get("income_statement", {}).get("annual", pd.DataFrame())
        balance_df = financial_data.get("balance_sheet", {}).get("annual", pd.DataFrame())
        cashflow_df = financial_data.get("cash_flow", {}).get("annual", pd.DataFrame())
        overview = financial_data.get("overview", {})
        
        # Try to get real-time quote (for current price)
        quote_data = None
        try:
            try:
                from data_ingestion import get_quote
            except ImportError:
                from .data_ingestion import get_quote
            quote_data = get_quote(financial_data.get("symbol", ""))
        except:
            pass
        
        if income_df.empty:
            print("⚠️ Income statement data is empty, unable to perform DCF valuation")
            return result
        
        # Ensure data is sorted by time (newest first)
        if isinstance(income_df.index, pd.DatetimeIndex):
            income_df = income_df.sort_index(ascending=False)
        if not balance_df.empty and isinstance(balance_df.index, pd.DatetimeIndex):
            balance_df = balance_df.sort_index(ascending=False)
        if not cashflow_df.empty and isinstance(cashflow_df.index, pd.DatetimeIndex):
            cashflow_df = cashflow_df.sort_index(ascending=False)
        
        # 1. Get base data
        latest_revenue = safe_get_value(income_df, "totalRevenue", 0)
        latest_operating_income = safe_get_value(income_df, "operatingIncome", 0)
        latest_net_income = safe_get_value(income_df, "netIncome", 0)
        latest_depreciation = safe_get_value(income_df, "depreciationAndAmortization", 0)
        latest_capex = abs(safe_get_value(cashflow_df, "capitalExpenditures", 0)) if not cashflow_df.empty else 0
        
        # 2. Determine assumptions
        # Revenue growth rate
        if revenue_growth_rate is None:
            # Calculate average growth rate based on historical data
            if "totalRevenue" in income_df.columns and len(income_df) >= 2:
                revenue_series = income_df["totalRevenue"].dropna()
                if len(revenue_series) >= 2:
                    # Calculate average growth rate for recent years
                    growth_rates = []
                    for i in range(min(3, len(revenue_series) - 1)):
                        if revenue_series.iloc[i+1] > 0:
                            growth = (revenue_series.iloc[i] - revenue_series.iloc[i+1]) / revenue_series.iloc[i+1]
                            growth_rates.append(growth)
                    if growth_rates:
                        revenue_growth_rate = np.mean(growth_rates)
                    else:
                        revenue_growth_rate = 0.10  # Default 10%
                else:
                    revenue_growth_rate = 0.10
            else:
                revenue_growth_rate = 0.10
        result["assumptions"]["revenue_growth_rate"] = revenue_growth_rate * 100
        
        # Operating margin
        if operating_margin is None:
            if latest_revenue > 0:
                operating_margin = latest_operating_income / latest_revenue
            else:
                operating_margin = 0.20  # Default 20%
        result["assumptions"]["operating_margin"] = operating_margin * 100
        
        # Tax rate
        if latest_operating_income > 0:
            tax_expense = safe_get_value(income_df, "incomeTaxExpense", 0)
            if tax_expense > 0:
                tax_rate = tax_expense / latest_operating_income
        result["assumptions"]["tax_rate"] = tax_rate * 100
        
        # 3. Forecast future cash flows
        forecast_df = pd.DataFrame()
        
        # Forecast revenue
        if "totalRevenue" in income_df.columns:
            historical_revenue = income_df["totalRevenue"].dropna()
            forecasted_revenue = forecast_revenue(historical_revenue, revenue_growth_rate, forecast_years)
            forecast_df["revenue"] = forecasted_revenue.values
        
        # Forecast operating income
        forecast_df["operating_income"] = forecast_df["revenue"] * operating_margin
        
        # Forecast NOPAT (Net Operating Profit After Tax)
        forecast_df["nopat"] = forecast_df["operating_income"] * (1 - tax_rate)
        
        # Forecast depreciation (assume proportional to revenue)
        if latest_revenue > 0:
            depreciation_rate = latest_depreciation / latest_revenue
        else:
            depreciation_rate = 0.05  # Default 5%
        forecast_df["depreciation"] = forecast_df["revenue"] * depreciation_rate
        
        # Forecast capital expenditures (assume proportional to revenue)
        if latest_revenue > 0:
            capex_rate = latest_capex / latest_revenue
        else:
            capex_rate = 0.08  # Default 8%
        forecast_df["capex"] = forecast_df["revenue"] * capex_rate
        
        # Forecast working capital changes (simplified: assume proportional to revenue growth)
        if not balance_df.empty and len(income_df) >= 2:
            latest_working_capital = (
                safe_get_value(balance_df, "totalCurrentAssets", 0) -
                safe_get_value(balance_df, "totalCurrentLiabilities", 0)
            )
            prev_revenue = safe_get_value(income_df, "totalRevenue", 1)
            if prev_revenue > 0:
                working_capital_rate = latest_working_capital / latest_revenue
            else:
                working_capital_rate = 0.10  # Default 10%
        else:
            working_capital_rate = 0.10
        
        # Calculate working capital changes
        forecast_df["working_capital_change"] = 0.0
        for i in range(len(forecast_df)):
            if i == 0:
                prev_revenue = latest_revenue
            else:
                prev_revenue = forecast_df["revenue"].iloc[i-1]
            current_revenue = forecast_df["revenue"].iloc[i]
            forecast_df.loc[forecast_df.index[i], "working_capital_change"] = (
                (current_revenue - prev_revenue) * working_capital_rate
            )
        
        # Calculate Free Cash Flow (FCF)
        forecast_df["fcf"] = (
            forecast_df["nopat"] +
            forecast_df["depreciation"] -
            forecast_df["capex"] -
            forecast_df["working_capital_change"]
        )
        
        result["forecast"] = forecast_df.to_dict('records')
        
        # 4. Calculate WACC
        if wacc is None:
            # Get market data
            market_cap = safe_float(overview.get("MarketCapitalization", 0))
            shares_outstanding = safe_float(overview.get("SharesOutstanding", 0))
            
            # Calculate equity market value
            if market_cap > 0:
                equity_market_value = market_cap
            elif shares_outstanding > 0:
                current_price = safe_float(overview.get("52WeekHigh", 0))
                if current_price == 0:
                    current_price = safe_float(overview.get("52WeekLow", 0))
                equity_market_value = shares_outstanding * current_price
            else:
                equity_market_value = 0
            
            # Calculate debt market value (simplified: use book value)
            total_debt = 0
            if not balance_df.empty:
                long_term_debt = safe_get_value(balance_df, "longTermDebt", 0)
                short_term_debt = safe_get_value(balance_df, "shortTermDebt", 0)
                total_debt = long_term_debt + short_term_debt
            
            # Estimate cost of equity (using CAPM)
            beta = safe_float(overview.get("Beta", 1.0))
            cost_of_equity = estimate_cost_of_equity(beta=beta) / 100
            
            # Estimate cost of debt
            if total_debt > 0 and latest_operating_income > 0:
                interest_expense = safe_get_value(income_df, "interestExpense", 0)
                if interest_expense == 0:
                    interest_expense = safe_get_value(income_df, "interestAndDebtExpense", 0)
                cost_of_debt = interest_expense / total_debt if total_debt > 0 else 0.05
            else:
                cost_of_debt = 0.05  # Default 5%
            
            wacc = calculate_wacc(
                equity_market_value=equity_market_value,
                debt_market_value=total_debt,
                cost_of_equity=cost_of_equity,
                cost_of_debt=cost_of_debt,
                tax_rate=tax_rate
            ) / 100
        else:
            wacc = wacc / 100
        
        result["assumptions"]["wacc"] = wacc * 100
        result["assumptions"]["terminal_growth_rate"] = terminal_growth_rate * 100
        
        # 5. Calculate present value (PV)
        pv_fcf = []
        for i, fcf in enumerate(forecast_df["fcf"]):
            pv = fcf / ((1 + wacc) ** (i + 1))
            pv_fcf.append(pv)
        
        sum_pv_fcf = sum(pv_fcf)
        
        # 6. Calculate terminal value
        terminal_fcf = forecast_df["fcf"].iloc[-1]
        terminal_value = terminal_fcf * (1 + terminal_growth_rate) / (wacc - terminal_growth_rate)
        pv_terminal_value = terminal_value / ((1 + wacc) ** forecast_years)
        
        # 7. Calculate enterprise value
        enterprise_value = sum_pv_fcf + pv_terminal_value
        
        # 8. Calculate equity value
        # Equity Value = Enterprise Value - Net Debt
        # Where: Net Debt = Total Debt - Cash
        # Expanded: Equity Value = Enterprise Value - Total Debt + Cash
        if not balance_df.empty:
            cash = safe_get_value(balance_df, "cashAndCashEquivalentsAtCarryingValue", 0)
            long_term_debt = safe_get_value(balance_df, "longTermDebt", 0)
            short_term_debt = safe_get_value(balance_df, "shortTermDebt", 0)
            total_debt = long_term_debt + short_term_debt
            net_debt = total_debt - cash
        else:
            cash = 0
            total_debt = 0
            net_debt = 0
        
        equity_value = enterprise_value - net_debt
        
        result["dcf_value"] = equity_value
        result["enterprise_value"] = enterprise_value
        result["sum_pv_fcf"] = sum_pv_fcf
        result["pv_terminal_value"] = pv_terminal_value
        result["net_debt"] = net_debt
        
        # 9. Calculate per-share value
        shares_outstanding = safe_float(overview.get("SharesOutstanding", 0))
        if shares_outstanding > 0:
            result["dcf_per_share"] = equity_value / shares_outstanding
            result["shares_outstanding"] = shares_outstanding
            result["target_price"] = equity_value / shares_outstanding
        else:
            result["dcf_per_share"] = None
            result["target_price"] = None
        
        # 10. Get current stock price
        current_price = None
        
        # Method 1: Get from real-time quote (most accurate)
        if quote_data and "Global Quote" in quote_data:
            quote = quote_data["Global Quote"]
            current_price = safe_float(quote.get("05. price", 0))
        
        # Method 2: Calculate from MarketCapitalization and SharesOutstanding
        if not current_price or current_price == 0:
            market_cap = safe_float(overview.get("MarketCapitalization", 0))
            if market_cap > 0 and shares_outstanding > 0:
                current_price = market_cap / shares_outstanding
        
        # Method 3: Use average of 52WeekHigh and 52WeekLow
        if not current_price or current_price == 0:
            high = safe_float(overview.get("52WeekHigh", 0))
            low = safe_float(overview.get("52WeekLow", 0))
            if high > 0 and low > 0:
                current_price = (high + low) / 2
        
        # Method 4: Use 52WeekHigh
        if not current_price or current_price == 0:
            current_price = safe_float(overview.get("52WeekHigh", 0))
        
        # Method 5: Use 52WeekLow (last attempt)
        if not current_price or current_price == 0:
            current_price = safe_float(overview.get("52WeekLow", 0))
        
        result["current_price"] = current_price if current_price and current_price > 0 else None
        
        # 11. Calculate upside potential
        if result.get("target_price") and result.get("current_price"):
            upside = ((result["target_price"] - result["current_price"]) / result["current_price"]) * 100
            result["upside_potential"] = upside
        else:
            result["upside_potential"] = None
        
        # 12. Sensitivity analysis
        result["sensitivity_analysis"] = perform_sensitivity_analysis(
            forecast_df, wacc, terminal_growth_rate, net_debt, shares_outstanding
        )
        
        print(f"  ✅ DCF valuation completed: Per-share value ${result['dcf_per_share']:.2f}" if result['dcf_per_share'] else "  ✅ DCF valuation completed")
    
    except Exception as e:
        print(f"⚠️ Error calculating DCF valuation: {e}")
        import traceback
        traceback.print_exc()
    
    return result

def perform_sensitivity_analysis(
    forecast_df: pd.DataFrame,
    base_wacc: float,
    base_terminal_growth: float,
    net_debt: float,
    shares_outstanding: float
) -> Dict[str, Any]:
    """
    Perform sensitivity analysis
    
    Args:
        forecast_df: Forecast data
        base_wacc: Base WACC
        base_terminal_growth: Base terminal growth rate
        net_debt: Net debt
        shares_outstanding: Shares outstanding
    
    Returns:
        Sensitivity analysis results
    """
    sensitivity = {}
    
    try:
        # WACC sensitivity (±2%)
        wacc_scenarios = [base_wacc - 0.02, base_wacc, base_wacc + 0.02]
        terminal_fcf = forecast_df["fcf"].iloc[-1]
        
        wacc_values = []
        for wacc in wacc_scenarios:
            sum_pv = sum([fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(forecast_df["fcf"])])
            terminal_value = terminal_fcf * (1 + base_terminal_growth) / (wacc - base_terminal_growth)
            pv_terminal = terminal_value / ((1 + wacc) ** len(forecast_df))
            equity_value = sum_pv + pv_terminal - net_debt
            per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
            wacc_values.append(per_share)
        
        sensitivity["wacc_sensitivity"] = {
            "low": wacc_values[0],
            "base": wacc_values[1],
            "high": wacc_values[2]
        }
        
        # Terminal growth rate sensitivity (±1%)
        growth_scenarios = [base_terminal_growth - 0.01, base_terminal_growth, base_terminal_growth + 0.01]
        growth_values = []
        for growth in growth_scenarios:
            sum_pv = sum([fcf / ((1 + base_wacc) ** (i + 1)) for i, fcf in enumerate(forecast_df["fcf"])])
            terminal_value = terminal_fcf * (1 + growth) / (base_wacc - growth)
            pv_terminal = terminal_value / ((1 + base_wacc) ** len(forecast_df))
            equity_value = sum_pv + pv_terminal - net_debt
            per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0
            growth_values.append(per_share)
        
        sensitivity["terminal_growth_sensitivity"] = {
            "low": growth_values[0],
            "base": growth_values[1],
            "high": growth_values[2]
        }
    
    except Exception as e:
        print(f"⚠️ Error in sensitivity analysis: {e}")
    
    return sensitivity

# ==================== Comparable Multiples Analysis ====================

def calculate_multiples_valuation(
    financial_data: Dict[str, Any],
    peer_data: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Calculate comparable multiples valuation
    
    Includes:
    - EV/EBITDA
    - P/E
    - EV/Sales
    
    Args:
        financial_data: Target company's financial data
        peer_data: List of peer company data (optional)
    
    Returns:
        Multiples valuation results dictionary
    """
    result = {
        "symbol": financial_data.get("symbol", "UNKNOWN"),
        "multiples": {},
        "peer_comparison": {}
    }
    
    try:
        # Get data
        income_df = financial_data.get("income_statement", {}).get("annual", pd.DataFrame())
        balance_df = financial_data.get("balance_sheet", {}).get("annual", pd.DataFrame())
        overview = financial_data.get("overview", {})
        
        if income_df.empty:
            print("⚠️ Income statement data is empty, unable to perform multiples valuation")
            return result
        
        # Ensure data is sorted by time (newest first)
        if isinstance(income_df.index, pd.DatetimeIndex):
            income_df = income_df.sort_index(ascending=False)
        if not balance_df.empty and isinstance(balance_df.index, pd.DatetimeIndex):
            balance_df = balance_df.sort_index(ascending=False)
        
        # Get latest fiscal year data
        revenue = safe_get_value(income_df, "totalRevenue", 0)
        operating_income = safe_get_value(income_df, "operatingIncome", 0)
        net_income = safe_get_value(income_df, "netIncome", 0)
        depreciation = safe_get_value(income_df, "depreciationAndAmortization", 0)
        ebitda = operating_income + depreciation
        
        # Get market data
        market_cap = safe_float(overview.get("MarketCapitalization", 0))
        shares_outstanding = safe_float(overview.get("SharesOutstanding", 0))
        
        # Try to get real-time quote (for current price)
        quote_data = None
        try:
            try:
                from data_ingestion import get_quote
            except ImportError:
                from .data_ingestion import get_quote
            quote_data = get_quote(financial_data.get("symbol", ""))
        except:
            pass
        
        # Get current stock price
        current_price = None
        
        # Method 1: Get from real-time quote (most accurate)
        if quote_data and "Global Quote" in quote_data:
            quote = quote_data["Global Quote"]
            current_price = safe_float(quote.get("05. price", 0))
        
        # Method 2: Calculate from MarketCapitalization and SharesOutstanding
        if not current_price or current_price == 0:
            if market_cap > 0 and shares_outstanding > 0:
                current_price = market_cap / shares_outstanding
        
        # Method 3: Use average of 52WeekHigh and 52WeekLow
        if not current_price or current_price == 0:
            high = safe_float(overview.get("52WeekHigh", 0))
            low = safe_float(overview.get("52WeekLow", 0))
            if high > 0 and low > 0:
                current_price = (high + low) / 2
        
        # Method 4: Use 52WeekHigh
        if not current_price or current_price == 0:
            current_price = safe_float(overview.get("52WeekHigh", 0))
        
        # Method 5: Use 52WeekLow (last attempt)
        if not current_price or current_price == 0:
            current_price = safe_float(overview.get("52WeekLow", 0))
        
        # Calculate enterprise value (EV)
        if not balance_df.empty:
            cash = safe_get_value(balance_df, "cashAndCashEquivalentsAtCarryingValue", 0)
            long_term_debt = safe_get_value(balance_df, "longTermDebt", 0)
            short_term_debt = safe_get_value(balance_df, "shortTermDebt", 0)
            total_debt = long_term_debt + short_term_debt
        else:
            cash = 0
            total_debt = 0
        
        if market_cap > 0:
            equity_value = market_cap
        elif shares_outstanding > 0 and current_price > 0:
            equity_value = shares_outstanding * current_price
        else:
            equity_value = 0
        
        enterprise_value = equity_value + total_debt - cash
        
        # Calculate multiples
        # EV/EBITDA
        if ebitda > 0:
            ev_ebitda = enterprise_value / ebitda
            result["multiples"]["ev_ebitda"] = ev_ebitda
        else:
            result["multiples"]["ev_ebitda"] = None
        
        # P/E
        if net_income > 0 and shares_outstanding > 0:
            eps = net_income / shares_outstanding
            if current_price > 0:
                result["multiples"]["pe"] = current_price / eps
                result["multiples"]["pe_ratio"] = current_price / eps  # Add alias
            else:
                result["multiples"]["pe"] = None
                result["multiples"]["pe_ratio"] = None
        else:
            pe_ratio = safe_float(overview.get("PERatio", None))
            result["multiples"]["pe"] = pe_ratio
            result["multiples"]["pe_ratio"] = pe_ratio  # Add alias
        
        # EV/Sales
        if revenue > 0:
            result["multiples"]["ev_sales"] = enterprise_value / revenue
        else:
            result["multiples"]["ev_sales"] = None
        
        # Save base data
        result["enterprise_value"] = enterprise_value
        result["market_cap"] = equity_value
        result["ebitda"] = ebitda
        result["revenue"] = revenue
        result["net_income"] = net_income
        
        # Peer comparison
        if peer_data:
            result["peer_comparison"] = compare_with_peers(
                result["multiples"],
                peer_data,
                result["symbol"]
            )
        
        print(f"  ✅ Multiples valuation completed")
    
    except Exception as e:
        print(f"⚠️ Error calculating multiples valuation: {e}")
        import traceback
        traceback.print_exc()
    
    return result

def compare_with_peers(
    target_multiples: Dict[str, Any],
    peer_data: List[Dict[str, Any]],
    target_symbol: str
) -> Dict[str, Any]:
    """
    Compare with peer companies
    
    Args:
        target_multiples: Target company's multiples
        peer_data: List of peer company data
        target_symbol: Target company symbol
    
    Returns:
        Peer comparison results
    """
    comparison = {
        "peer_count": len(peer_data),
        "ev_ebitda": {"target": target_multiples.get("ev_ebitda"), "peer_median": None, "peer_mean": None},
        "pe": {"target": target_multiples.get("pe"), "peer_median": None, "peer_mean": None},
        "ev_sales": {"target": target_multiples.get("ev_sales"), "peer_median": None, "peer_mean": None}
    }
    
    try:
        # Collect peer multiples
        peer_ev_ebitda = []
        peer_pe = []
        peer_ev_sales = []
        
        for peer in peer_data:
            peer_multiples = peer.get("multiples", {})
            if peer_multiples.get("ev_ebitda"):
                peer_ev_ebitda.append(peer_multiples["ev_ebitda"])
            if peer_multiples.get("pe"):
                peer_pe.append(peer_multiples["pe"])
            if peer_multiples.get("ev_sales"):
                peer_ev_sales.append(peer_multiples["ev_sales"])
        
        # Calculate median and mean
        if peer_ev_ebitda:
            comparison["ev_ebitda"]["peer_median"] = np.median(peer_ev_ebitda)
            comparison["ev_ebitda"]["peer_mean"] = np.mean(peer_ev_ebitda)
        
        if peer_pe:
            comparison["pe"]["peer_median"] = np.median(peer_pe)
            comparison["pe"]["peer_mean"] = np.mean(peer_pe)
        
        if peer_ev_sales:
            comparison["ev_sales"]["peer_median"] = np.median(peer_ev_sales)
            comparison["ev_sales"]["peer_mean"] = np.mean(peer_ev_sales)
        
        # Calculate relative valuation
        if comparison["ev_ebitda"]["target"] and comparison["ev_ebitda"]["peer_median"]:
            comparison["ev_ebitda"]["premium_discount"] = (
                (comparison["ev_ebitda"]["target"] - comparison["ev_ebitda"]["peer_median"]) /
                comparison["ev_ebitda"]["peer_median"] * 100
            )
        
        if comparison["pe"]["target"] and comparison["pe"]["peer_median"]:
            comparison["pe"]["premium_discount"] = (
                (comparison["pe"]["target"] - comparison["pe"]["peer_median"]) /
                comparison["pe"]["peer_median"] * 100
            )
        
        if comparison["ev_sales"]["target"] and comparison["ev_sales"]["peer_median"]:
            comparison["ev_sales"]["premium_discount"] = (
                (comparison["ev_sales"]["target"] - comparison["ev_sales"]["peer_median"]) /
                comparison["ev_sales"]["peer_median"] * 100
            )
    
    except Exception as e:
        print(f"⚠️ Error in peer comparison: {e}")
    
    return comparison

# ==================== Comprehensive Valuation Analysis ====================

def comprehensive_valuation(
    financial_data: Dict[str, Any],
    dcf_params: Dict[str, Any] = None,
    peer_data: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Comprehensive valuation analysis (DCF + Multiples)
    
    Args:
        financial_data: Financial data
        dcf_params: DCF model parameters (optional)
        peer_data: Peer company data (optional)
    
    Returns:
        Comprehensive valuation results
    """
    result = {
        "symbol": financial_data.get("symbol", "UNKNOWN"),
        "dcf_valuation": {},
        "multiples_valuation": {},
        "summary": {}
    }
    
    print(f"\n📊 Starting valuation analysis: {result['symbol']}")
    
    # 1. DCF valuation
    print("  📈 Calculating DCF valuation...")
    if dcf_params:
        dcf_result = calculate_dcf_valuation(financial_data, **dcf_params)
    else:
        dcf_result = calculate_dcf_valuation(financial_data)
    result["dcf_valuation"] = dcf_result
    
    # 2. Multiples valuation
    print("  📊 Calculating multiples valuation...")
    multiples_result = calculate_multiples_valuation(financial_data, peer_data)
    result["multiples_valuation"] = multiples_result
    
    # 3. Generate summary
    result["summary"] = generate_valuation_summary(dcf_result, multiples_result)
    
    print(f"\n✅ {result['symbol']} valuation analysis completed")
    
    return result

def generate_valuation_summary(
    dcf_result: Dict[str, Any],
    multiples_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate valuation summary
    
    Args:
        dcf_result: DCF valuation results
        multiples_result: Multiples valuation results
    
    Returns:
        Valuation summary
    """
    summary = {
        "dcf_value_per_share": dcf_result.get("dcf_per_share"),
        "current_multiples": multiples_result.get("multiples", {}),
        "valuation_upside": None,
        "key_insights": []
    }
    
    try:
        # Calculate valuation upside
        dcf_per_share = dcf_result.get("dcf_per_share")
        current_price = multiples_result.get("multiples", {}).get("pe")
        
        if dcf_per_share and current_price:
            # Need to get current price from overview
            pass
        
        # Generate insights
        if dcf_result.get("dcf_per_share"):
            summary["key_insights"].append(f"DCF Valuation: ${dcf_result['dcf_per_share']:.2f}/share")
        
        if multiples_result.get("multiples", {}).get("ev_ebitda"):
            ev_ebitda = multiples_result["multiples"]["ev_ebitda"]
            summary["key_insights"].append(f"EV/EBITDA: {ev_ebitda:.2f}x")
        
        if multiples_result.get("peer_comparison", {}).get("ev_ebitda", {}).get("premium_discount"):
            premium = multiples_result["peer_comparison"]["ev_ebitda"]["premium_discount"]
            if premium > 0:
                summary["key_insights"].append(f"Premium vs peers: {premium:.1f}%")
            else:
                summary["key_insights"].append(f"Discount vs peers: {abs(premium):.1f}%")
    
    except Exception as e:
        print(f"⚠️ Error generating valuation summary: {e}")
    
    return summary

# ==================== Main Function (for testing) ====================

if __name__ == "__main__":
    # Test example
    print("=" * 60)
    print("Valuation Module - Valuation Model Test")
    print("=" * 60)
    
    # Need to fetch data first
    from data_ingestion import fetch_financial_statements
    
    symbol = "NVDA"
    print(f"\nFetching financial statement data for {symbol}...")
    financial_data = fetch_financial_statements(symbol, years=5)
    
    print(f"\nPerforming comprehensive valuation analysis...")
    valuation = comprehensive_valuation(financial_data)
    
    # Display results
    print("\n" + "=" * 60)
    print("Valuation Results:")
    print("=" * 60)
    
    print("\n📈 DCF Valuation:")
    dcf = valuation["dcf_valuation"]
    if dcf.get("dcf_per_share"):
        print(f"  Per-share value: ${dcf['dcf_per_share']:.2f}")
    print(f"  Enterprise value: ${dcf.get('enterprise_value', 0)/1e9:.2f}B")
    print(f"  Assumptions:")
    for key, value in dcf.get("assumptions", {}).items():
        print(f"    {key}: {value:.2f}%")
    
    print("\n📊 Multiples Valuation:")
    multiples = valuation["multiples_valuation"]["multiples"]
    for key, value in multiples.items():
        if value:
            print(f"  {key.upper()}: {value:.2f}")
    
    print("\n📋 Summary:")
    for insight in valuation["summary"].get("key_insights", []):
        print(f"  • {insight}")
    
    print("\n✅ Test completed!")

