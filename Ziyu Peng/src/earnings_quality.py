"""
Earnings Quality Analysis Module for Fundamental Analysis

Features:
1. Cash vs Profit (CFO/Net Income)
2. Accrual Quality (Accruals = NI - CFO)
3. Profit Volatility (Margin Stability)
4. One-time Items Dependency
5. Capital Structure Support

Earnings appear largely cash-backed and operationally driven, supporting sustainability.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import other modules
try:
    from financial_analysis import safe_float, safe_get_value
except ImportError:
    from .financial_analysis import safe_float, safe_get_value

# Import LLM module (for one-time items analysis)
try:
    from talk2ai import OpenAIChat, get_config_from_env
    OPENAI_AVAILABLE = True
except ImportError:
    try:
        from .talk2ai import OpenAIChat, get_config_from_env
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        print("OpenAI module not found, one-time items analysis will be unavailable")
        OpenAIChat = None
        get_config_from_env = None

# ==================== 1. Cash vs Profit Analysis ====================

def analyze_cash_vs_profit(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze cash vs profit (CFO/Net Income)
    
    Assess the cash support level of earnings
    
    Args:
        income_df: Income statement DataFrame
        cashflow_df: Cash flow statement DataFrame
    
    Returns:
        Cash vs profit analysis results
    """
    analysis = {
        "cfo_to_net_income": {},
        "cash_conversion_ratio": {},
        "trend_analysis": {},
        "assessment": ""
    }
    
    if income_df.empty or cashflow_df.empty:
        return {"error": "Insufficient data"}
    
    try:
        # Ensure data is sorted by time (newest first)
        if isinstance(income_df.index, pd.DatetimeIndex):
            income_df = income_df.sort_index(ascending=False)
        if isinstance(cashflow_df.index, pd.DatetimeIndex):
            cashflow_df = cashflow_df.sort_index(ascending=False)
        
        # Get latest fiscal year data
        net_income = safe_get_value(income_df, "netIncome", 0)
        operating_cf = safe_get_value(cashflow_df, "operatingCashflow", 0)
        
        # CFO/Net Income ratio
        if net_income != 0:
            cfo_to_ni_ratio = operating_cf / net_income
            analysis["cfo_to_net_income"]["latest"] = cfo_to_ni_ratio
            analysis["cfo_to_net_income"]["net_income"] = net_income
            analysis["cfo_to_net_income"]["operating_cf"] = operating_cf
        else:
            analysis["cfo_to_net_income"]["latest"] = None
            analysis["cfo_to_net_income"]["net_income"] = net_income
            analysis["cfo_to_net_income"]["operating_cf"] = operating_cf
        
        # Calculate historical trends (if multi-year data available)
        if "netIncome" in income_df.columns and "operatingCashflow" in cashflow_df.columns:
            # Align time indices
            common_dates = income_df.index.intersection(cashflow_df.index)
            if len(common_dates) >= 2:
                ni_series = income_df.loc[common_dates, "netIncome"]
                cfo_series = cashflow_df.loc[common_dates, "operatingCashflow"]
                
                # Calculate ratio series
                ratio_series = (cfo_series / ni_series).where(ni_series != 0)
                ratio_series = ratio_series.dropna()
                
                if len(ratio_series) >= 2:
                    # Average value
                    analysis["cfo_to_net_income"]["average"] = ratio_series.mean()
                    analysis["cfo_to_net_income"]["median"] = ratio_series.median()
                    analysis["cfo_to_net_income"]["min"] = ratio_series.min()
                    analysis["cfo_to_net_income"]["max"] = ratio_series.max()
                    analysis["cfo_to_net_income"]["std"] = ratio_series.std()
                    
                    # Trend
                    latest_ratio = ratio_series.iloc[0]
                    prev_ratio = ratio_series.iloc[-1]
                    if latest_ratio > prev_ratio:
                        analysis["trend_analysis"]["cfo_ni_trend"] = "improving"
                    elif latest_ratio < prev_ratio:
                        analysis["trend_analysis"]["cfo_ni_trend"] = "declining"
                    else:
                        analysis["trend_analysis"]["cfo_ni_trend"] = "stable"
        
        # Cash Conversion Ratio
        # Defined as: CFO / Revenue
        if "totalRevenue" in income_df.columns:
            revenue = safe_get_value(income_df, "totalRevenue", 0)
            if revenue > 0:
                cash_conversion_ratio = operating_cf / revenue
                analysis["cash_conversion_ratio"]["latest"] = cash_conversion_ratio
                analysis["cash_conversion_ratio"]["revenue"] = revenue
        
        # Assessment
        if analysis["cfo_to_net_income"].get("latest"):
            ratio = analysis["cfo_to_net_income"]["latest"]
            if ratio > 1.0:
                analysis["assessment"] = "Excellent: Operating cash flow significantly exceeds net income, high earnings quality"
            elif ratio > 0.8:
                analysis["assessment"] = "Good: Operating cash flow matches net income well"
            elif ratio > 0.5:
                analysis["assessment"] = "Average: Operating cash flow below net income, needs attention"
            else:
                analysis["assessment"] = "Poor: Operating cash flow far below net income, earnings quality needs caution"
        else:
            analysis["assessment"] = "Cannot assess: Insufficient data"
    
    except Exception as e:
        print(f"⚠️ Error analyzing cash vs profit: {e}")
        import traceback
        traceback.print_exc()
        analysis["error"] = str(e)
    
    return analysis

# ==================== 2. Accrual Quality Analysis ====================

def analyze_accrual_quality(
    income_df: pd.DataFrame,
    cashflow_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze accrual quality (Accruals = NI - CFO)
    
    Accruals = Net Income - Operating Cash Flow
    Higher accrual quality indicates earnings are more cash-dependent, which is better
    
    Args:
        income_df: Income statement DataFrame
        cashflow_df: Cash flow statement DataFrame
    
    Returns:
        Accrual quality analysis results
    """
    analysis = {
        "accruals": {},
        "accrual_ratio": {},
        "trend_analysis": {},
        "assessment": ""
    }
    
    if income_df.empty or cashflow_df.empty:
        return {"error": "Insufficient data"}
    
    try:
        # Ensure data is sorted by time (newest first)
        if isinstance(income_df.index, pd.DatetimeIndex):
            income_df = income_df.sort_index(ascending=False)
        if isinstance(cashflow_df.index, pd.DatetimeIndex):
            cashflow_df = cashflow_df.sort_index(ascending=False)
        
        # Align time indices
        common_dates = income_df.index.intersection(cashflow_df.index)
        if len(common_dates) < 1:
            return {"error": "Data time mismatch"}
        
        # Get latest fiscal year data
        net_income = safe_get_value(income_df, "netIncome", 0)
        operating_cf = safe_get_value(cashflow_df, "operatingCashflow", 0)
        
        # Calculate accruals
        accruals = net_income - operating_cf
        analysis["accruals"]["latest"] = accruals
        analysis["accruals"]["net_income"] = net_income
        analysis["accruals"]["operating_cf"] = operating_cf
        
        # Accrual Ratio (Accrual Ratio = Accruals / Total Assets)
        # Here we simplify by using revenue as the denominator
        if "totalRevenue" in income_df.columns:
            revenue = safe_get_value(income_df, "totalRevenue", 0)
            if revenue > 0:
                accrual_ratio = accruals / revenue
                analysis["accrual_ratio"]["latest"] = accrual_ratio
        
        # Calculate historical trends
        if "netIncome" in income_df.columns and "operatingCashflow" in cashflow_df.columns:
            ni_series = income_df.loc[common_dates, "netIncome"]
            cfo_series = cashflow_df.loc[common_dates, "operatingCashflow"]
            
            accruals_series = ni_series - cfo_series
            accruals_series = accruals_series.dropna()
            
            if len(accruals_series) >= 2:
                analysis["accruals"]["average"] = accruals_series.mean()
                analysis["accruals"]["median"] = accruals_series.median()
                analysis["accruals"]["std"] = accruals_series.std()
                
                # Trend analysis
                latest_accruals = accruals_series.iloc[0]
                prev_accruals = accruals_series.iloc[-1]
                
                # Smaller accruals are better (indicates earnings are more cash-dependent)
                if abs(latest_accruals) < abs(prev_accruals):
                    analysis["trend_analysis"]["accrual_trend"] = "improving"
                elif abs(latest_accruals) > abs(prev_accruals):
                    analysis["trend_analysis"]["accrual_trend"] = "deteriorating"
                else:
                    analysis["trend_analysis"]["accrual_trend"] = "stable"
        
        # Assessment
        if analysis["accruals"].get("latest") is not None:
            accruals = analysis["accruals"]["latest"]
            if abs(accruals) < abs(net_income) * 0.2:  # Accruals less than 20% of net income
                analysis["assessment"] = "Excellent: Small accruals, earnings primarily cash-dependent"
            elif abs(accruals) < abs(net_income) * 0.5:
                analysis["assessment"] = "Good: Moderate accruals, earnings quality acceptable"
            else:
                analysis["assessment"] = "Needs attention: Large accruals, low cash dependency of earnings"
        else:
            analysis["assessment"] = "Cannot assess: Insufficient data"
    
    except Exception as e:
        print(f"⚠️ Error analyzing accrual quality: {e}")
        import traceback
        traceback.print_exc()
        analysis["error"] = str(e)
    
    return analysis

# ==================== 3. Profit Volatility Analysis ====================

def analyze_profit_volatility(
    income_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze profit volatility (Margin stability)
    
    Assess the stability and sustainability of earnings
    
    Args:
        income_df: Income statement DataFrame
    
    Returns:
        Profit volatility analysis results
    """
    analysis = {
        "margin_stability": {},
        "profit_volatility": {},
        "trend_analysis": {},
        "assessment": ""
    }
    
    if income_df.empty:
        return {"error": "Insufficient data"}
    
    try:
        # Ensure data is sorted by time (newest first)
        if isinstance(income_df.index, pd.DatetimeIndex):
            income_df = income_df.sort_index(ascending=False)
        
        # Calculate various profit margins
        if "totalRevenue" in income_df.columns and "netIncome" in income_df.columns:
            revenue_series = income_df["totalRevenue"].dropna()
            net_income_series = income_df["netIncome"].dropna()
            
            # Align time
            common_dates = revenue_series.index.intersection(net_income_series.index)
            if len(common_dates) >= 2:
                revenue_aligned = revenue_series.loc[common_dates]
                ni_aligned = net_income_series.loc[common_dates]
                
                # Calculate net margin series
                net_margin_series = (ni_aligned / revenue_aligned * 100).where(revenue_aligned > 0)
                net_margin_series = net_margin_series.dropna()
                
                if len(net_margin_series) >= 2:
                    # Statistical indicators
                    analysis["margin_stability"]["latest"] = net_margin_series.iloc[0]
                    analysis["margin_stability"]["average"] = net_margin_series.mean()
                    analysis["margin_stability"]["median"] = net_margin_series.median()
                    analysis["margin_stability"]["std"] = net_margin_series.std()
                    analysis["margin_stability"]["min"] = net_margin_series.min()
                    analysis["margin_stability"]["max"] = net_margin_series.max()
                    
                    # Coefficient of Variation
                    if net_margin_series.mean() != 0:
                        cv = (net_margin_series.std() / abs(net_margin_series.mean())) * 100
                        analysis["profit_volatility"]["coefficient_of_variation"] = cv
                        
                        # Assess stability
                        if cv < 20:
                            analysis["profit_volatility"]["stability"] = "Very stable"
                        elif cv < 40:
                            analysis["profit_volatility"]["stability"] = "Relatively stable"
                        elif cv < 60:
                            analysis["profit_volatility"]["stability"] = "Average"
                        else:
                            analysis["profit_volatility"]["stability"] = "Unstable"
                    
                    # Trend analysis
                    if len(net_margin_series) >= 3:
                        recent_3y = net_margin_series.iloc[:3]
                        if recent_3y.iloc[0] > recent_3y.iloc[-1]:
                            analysis["trend_analysis"]["margin_trend"] = "improving"
                        elif recent_3y.iloc[0] < recent_3y.iloc[-1]:
                            analysis["trend_analysis"]["margin_trend"] = "declining"
                        else:
                            analysis["trend_analysis"]["margin_trend"] = "stable"
        
        # Analyze operating margin stability
        if "totalRevenue" in income_df.columns and "operatingIncome" in income_df.columns:
            revenue_series = income_df["totalRevenue"].dropna()
            op_income_series = income_df["operatingIncome"].dropna()
            
            common_dates = revenue_series.index.intersection(op_income_series.index)
            if len(common_dates) >= 2:
                revenue_aligned = revenue_series.loc[common_dates]
                op_income_aligned = op_income_series.loc[common_dates]
                
                op_margin_series = (op_income_aligned / revenue_aligned * 100).where(revenue_aligned > 0)
                op_margin_series = op_margin_series.dropna()
                
                if len(op_margin_series) >= 2:
                    analysis["margin_stability"]["operating_margin_std"] = op_margin_series.std()
                    analysis["margin_stability"]["operating_margin_cv"] = (
                        (op_margin_series.std() / abs(op_margin_series.mean())) * 100
                        if op_margin_series.mean() != 0 else None
                    )
        
        # Comprehensive assessment
        cv = analysis["profit_volatility"].get("coefficient_of_variation")
        if cv is not None:
            if cv < 20:
                analysis["assessment"] = "Excellent: Profit margin very stable, strong earnings sustainability"
            elif cv < 40:
                analysis["assessment"] = "Good: Profit margin relatively stable, good earnings sustainability"
            elif cv < 60:
                analysis["assessment"] = "Average: Profit margin volatility is high, needs attention"
            else:
                analysis["assessment"] = "Poor: Profit margin volatility is very high, earnings sustainability questionable"
        else:
            analysis["assessment"] = "Cannot assess: Insufficient data"
    
    except Exception as e:
        print(f"⚠️ Error analyzing profit volatility: {e}")
        import traceback
        traceback.print_exc()
        analysis["error"] = str(e)
    
    return analysis

# ==================== 4. One-time Items Dependency Analysis ====================

def analyze_one_time_items(
    income_df: pd.DataFrame,
    financial_data: Dict[str, Any] = None,
    news_data: Dict[str, Any] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Analyze one-time items dependency
    
    Identify one-time items in income statement (such as asset disposals, restructuring charges, etc.)
    Requires LLM analysis combining financial data and news
    
    Args:
        income_df: Income statement DataFrame
        financial_data: Complete financial data (optional, for additional context)
        news_data: News data (optional, for LLM analysis)
        api_key: OpenAI API Key
        base_url: OpenAI Base URL
        model: AI model name
    
    Returns:
        One-time items analysis results
    """
    analysis = {
        "one_time_items_detected": [],
        "special_items_ratio": {},
        "llm_analysis": {},
        "assessment": ""
    }
    
    if income_df.empty:
        return {"error": "Insufficient data"}
    
    try:
        # Ensure data is sorted by time (newest first)
        if isinstance(income_df.index, pd.DatetimeIndex):
            income_df = income_df.sort_index(ascending=False)
        
        # 1. Structured data analysis: Identify possible one-time item fields
        # Fields that may be included in Alpha Vantage income statement:
        # - discontinuedOperations
        # - extraordinaryItems
        # - otherNonOperatingIncome
        # - otherNonOperatingExpenses
        
        one_time_fields = [
            "discontinuedOperations",
            "extraordinaryItems",
            "otherNonOperatingIncome",
            "otherNonOperatingExpenses",
            "gainLossOnSaleOfAssets",
            "restructuringCharges"
        ]
        
        detected_items = []
        latest_net_income = safe_get_value(income_df, "netIncome", 0)
        
        for field in one_time_fields:
            if field in income_df.columns:
                value = safe_get_value(income_df, field, 0)
                if abs(value) > 0:
                    detected_items.append({
                        "item": field,
                        "value": value,
                        "ratio_to_ni": (value / abs(latest_net_income) * 100) if latest_net_income != 0 else None
                    })
        
        analysis["one_time_items_detected"] = detected_items
        
        # Calculate one-time items ratio
        if detected_items and latest_net_income != 0:
            total_one_time = sum(abs(item["value"]) for item in detected_items)
            one_time_ratio = (total_one_time / abs(latest_net_income)) * 100
            analysis["special_items_ratio"]["total_one_time_items"] = total_one_time
            analysis["special_items_ratio"]["ratio_to_net_income"] = one_time_ratio
            analysis["special_items_ratio"]["net_income"] = latest_net_income
        
        # 2. LLM analysis (if news data available)
        if news_data and news_data.get("news_count", 0) > 0 and OPENAI_AVAILABLE:
            llm_result = analyze_one_time_items_with_llm(
                symbol=financial_data.get("symbol", "UNKNOWN") if financial_data else "UNKNOWN",
                income_df=income_df,
                news_data=news_data,
                detected_items=detected_items,
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            analysis["llm_analysis"] = llm_result
        
        # 3. Assessment
        one_time_ratio = analysis["special_items_ratio"].get("ratio_to_net_income", 0)
        if one_time_ratio < 5:
            analysis["assessment"] = "Excellent: One-time items ratio is very small, earnings mainly from core business"
        elif one_time_ratio < 15:
            analysis["assessment"] = "Good: One-time items ratio is moderate, earnings quality acceptable"
        elif one_time_ratio < 30:
            analysis["assessment"] = "Needs attention: One-time items ratio is relatively high, earnings quality needs caution"
        else:
            analysis["assessment"] = "Poor: One-time items ratio is very high, earnings sustainability questionable"
    
    except Exception as e:
        print(f"⚠️ Error analyzing one-time items: {e}")
        import traceback
        traceback.print_exc()
        analysis["error"] = str(e)
    
    return analysis

def analyze_one_time_items_with_llm(
    symbol: str,
    income_df: pd.DataFrame,
    news_data: Dict[str, Any],
    detected_items: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Analyze one-time items using LLM
    
    Args:
        symbol: Stock symbol
        income_df: Income statement DataFrame
        news_data: News data
        detected_items: Detected one-time items
        api_key: OpenAI API Key
        base_url: OpenAI Base URL
        model: AI model name
    
    Returns:
        LLM analysis results
    """
    result = {
        "one_time_items_identified": [],
        "impact_assessment": {},
        "sustainability_concerns": []
    }
    
    if not OPENAI_AVAILABLE:
        return {"error": "LLM not available"}
    
    # Set default API key and base_url (if not provided)
    if not api_key:
        try:
            from talk2ai import get_config_from_env
            env_api_key, env_base_url = get_config_from_env()
            api_key = env_api_key
            if not base_url:
                base_url = env_base_url or "https://api.openai.com/v1"
        except:
            api_key = None
            if not base_url:
                base_url = "https://api.openai.com/v1"
        
        # Validate API Key
        if not api_key:
            result["error"] = "OpenAI API Key is required"
            return result
    
    try:
        # Prepare data
        latest_income = income_df.iloc[0].to_dict() if not income_df.empty else {}
        net_income = latest_income.get("netIncome", 0)
        
        # Format news
        news_items = news_data.get("news_items", [])[:10]  # Use latest 10 news items
        news_text = ""
        for item in news_items:
            title = item.get("title", "")
            summary = item.get("summary", "") or item.get("content", "")
            news_text += f"Title: {title}\nSummary: {summary[:200]}...\n\n"
        
        # Build Prompt
        prompt = f"""Please analyze the earnings quality of {symbol}, with special focus on the impact of one-time items on earnings.

## Financial Data Summary

Latest Fiscal Year Net Income: ${net_income/1e9:.2f}B

Detected One-time Items:
{json.dumps(detected_items, ensure_ascii=False, indent=2) if detected_items else "No obvious one-time item fields detected"}

## Related News

{news_text[:2000] if news_text else "No news data"}

## Analysis Requirements

Please analyze from the following perspectives:

1. **One-time Items Identification**:
   - Identify one-time events that may affect earnings from news (such as asset disposals, restructuring, litigation, government subsidies, etc.)
   - Assess the impact level of these events on earnings

2. **Earnings Quality Assessment**:
   - Assess whether earnings are primarily from core business operations
   - Identify if there is reliance on one-time items to "polish" earnings

3. **Sustainability Judgment**:
   - If one-time items are removed, what is the core earnings level?
   - What is the sustainability of future earnings?

Please return in JSON format, containing the following fields:

```json
{{
    "one_time_items_identified": [
        {{
            "type": "Asset Disposal/Restructuring/Subsidy, etc.",
            "description": "Detailed description",
            "impact_amount": "Impact amount (if estimable)",
            "impact_on_ni": "Impact on net income (High/Medium/Low)",
            "evidence": "Supporting evidence (from news or data)"
        }}
    ],
    "core_earnings_estimate": {{
        "reported_net_income": 0,
        "estimated_one_time_items": 0,
        "estimated_core_earnings": 0,
        "core_earnings_margin": 0
    }},
    "sustainability_assessment": {{
        "reliance_on_one_time_items": "High/Medium/Low",
        "core_earnings_quality": "Excellent/Good/Average/Poor",
        "sustainability_concerns": [
            "Concern 1",
            "Concern 2"
        ],
        "key_insights": [
            "Insight 1",
            "Insight 2"
        ]
    }}
}}
```"""

        # Call LLM
        chat_client = OpenAIChat(api_key=api_key, base_url=base_url)
        messages = [
            {
                "role": "system",
                "content": """You are a senior financial analyst specializing in earnings quality analysis.
Your task is to identify and analyze the impact of one-time items on earnings, and assess earnings sustainability.
Please base your analysis on facts and data, avoid subjective speculation."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        llm_response = chat_client.chat(messages, model=model)
        
        if llm_response:
            # Parse response
            parsed = parse_llm_one_time_items_response(llm_response)
            result.update(parsed)
            result["raw_llm_response"] = llm_response
    
    except Exception as e:
        print(f"⚠️ LLM analysis of one-time items failed: {e}")
        result["error"] = str(e)
    
    return result

def parse_llm_one_time_items_response(llm_response: str) -> Dict[str, Any]:
    """Parse LLM one-time items analysis response"""
    result = {
        "one_time_items_identified": [],
        "core_earnings_estimate": {},
        "sustainability_assessment": {}
    }
    
    try:
        # Extract JSON
        json_str = None
        if "```json" in llm_response:
            json_start = llm_response.find("```json") + 7
            json_end = llm_response.find("```", json_start)
            if json_end > json_start:
                json_str = llm_response[json_start:json_end].strip()
        elif "```" in llm_response:
            json_start = llm_response.find("```") + 3
            json_end = llm_response.find("```", json_start)
            if json_end > json_start:
                json_str = llm_response[json_start:json_end].strip()
        else:
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
        
        if json_str:
            parsed = json.loads(json_str)
            result["one_time_items_identified"] = parsed.get("one_time_items_identified", [])
            result["core_earnings_estimate"] = parsed.get("core_earnings_estimate", {})
            result["sustainability_assessment"] = parsed.get("sustainability_assessment", {})
    
    except Exception as e:
        print(f"⚠️ Failed to parse LLM response: {e}")
        result["parse_error"] = str(e)
    
    return result

# ==================== 5. Capital Structure Support Analysis ====================

def analyze_capital_structure_support(
    income_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    cashflow_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyze capital structure support
    
    Assess whether earnings are "propped up" by increasing leverage
    
    Args:
        income_df: Income statement DataFrame
        balance_df: Balance sheet DataFrame
        cashflow_df: Cash flow statement DataFrame
    
    Returns:
        Capital structure support analysis results
    """
    analysis = {
        "leverage_trend": {},
        "debt_support_analysis": {},
        "interest_burden": {},
        "assessment": ""
    }
    
    if income_df.empty or balance_df.empty:
        return {"error": "Insufficient data"}
    
    try:
        # Ensure data is sorted by time (newest first)
        if isinstance(income_df.index, pd.DatetimeIndex):
            income_df = income_df.sort_index(ascending=False)
        if isinstance(balance_df.index, pd.DatetimeIndex):
            balance_df = balance_df.sort_index(ascending=False)
        if isinstance(cashflow_df.index, pd.DatetimeIndex):
            cashflow_df = cashflow_df.sort_index(ascending=False)
        
        # Get latest fiscal year data
        net_income = safe_get_value(income_df, "netIncome", 0)
        total_assets = safe_get_value(balance_df, "totalAssets", 0)
        total_liabilities = safe_get_value(balance_df, "totalLiabilities", 0)
        total_equity = safe_get_value(balance_df, "totalShareholderEquity", 0)
        
        # Calculate debt
        long_term_debt = safe_get_value(balance_df, "longTermDebt", 0)
        short_term_debt = safe_get_value(balance_df, "shortTermDebt", 0)
        total_debt = long_term_debt + short_term_debt
        
        # 1. Leverage trend analysis
        if len(balance_df) >= 2:
            # Calculate historical debt-to-equity ratio
            debt_to_equity_series = []
            for i in range(min(3, len(balance_df))):
                equity = safe_get_value(balance_df, "totalShareholderEquity", i)
                lt_debt = safe_get_value(balance_df, "longTermDebt", i)
                st_debt = safe_get_value(balance_df, "shortTermDebt", i)
                debt = lt_debt + st_debt
                if equity > 0:
                    debt_to_equity = debt / equity
                    debt_to_equity_series.append(debt_to_equity)
            
            if len(debt_to_equity_series) >= 2:
                latest_dte = debt_to_equity_series[0]
                prev_dte = debt_to_equity_series[-1]
                
                analysis["leverage_trend"]["latest_debt_to_equity"] = latest_dte
                analysis["leverage_trend"]["previous_debt_to_equity"] = prev_dte
                analysis["leverage_trend"]["change"] = latest_dte - prev_dte
                
                if latest_dte > prev_dte:
                    analysis["leverage_trend"]["trend"] = "increasing"
                elif latest_dte < prev_dte:
                    analysis["leverage_trend"]["trend"] = "decreasing"
                else:
                    analysis["leverage_trend"]["trend"] = "stable"
        
        # 2. Debt support analysis
        # Assess whether profit growth is mainly from leverage increase
        if len(income_df) >= 2 and len(balance_df) >= 2:
            latest_ni = safe_get_value(income_df, "netIncome", 0)
            prev_ni = safe_get_value(income_df, "netIncome", 1)
            
            latest_debt = total_debt
            prev_debt = (
                safe_get_value(balance_df, "longTermDebt", 1) +
                safe_get_value(balance_df, "shortTermDebt", 1)
            )
            
            if prev_ni != 0 and prev_debt > 0:
                ni_growth = ((latest_ni - prev_ni) / abs(prev_ni)) * 100
                debt_growth = ((latest_debt - prev_debt) / prev_debt) * 100
                
                analysis["debt_support_analysis"]["net_income_growth"] = ni_growth
                analysis["debt_support_analysis"]["debt_growth"] = debt_growth
                
                # If debt growth is significantly faster than profit growth, leverage support may exist
                if debt_growth > ni_growth * 1.5 and debt_growth > 10:
                    analysis["debt_support_analysis"]["leverage_concern"] = True
                    analysis["debt_support_analysis"]["concern_level"] = "High"
                elif debt_growth > ni_growth * 1.2 and debt_growth > 5:
                    analysis["debt_support_analysis"]["leverage_concern"] = True
                    analysis["debt_support_analysis"]["concern_level"] = "Medium"
                else:
                    analysis["debt_support_analysis"]["leverage_concern"] = False
        
        # 3. Interest burden analysis
        interest_expense = safe_get_value(income_df, "interestExpense", 0)
        if interest_expense == 0:
            interest_expense = safe_get_value(income_df, "interestAndDebtExpense", 0)
        
        operating_income = safe_get_value(income_df, "operatingIncome", 0)
        
        if operating_income > 0:
            interest_coverage = operating_income / interest_expense if interest_expense > 0 else None
            analysis["interest_burden"]["interest_expense"] = interest_expense
            analysis["interest_burden"]["interest_coverage"] = interest_coverage
            
            if interest_coverage:
                if interest_coverage < 2:
                    analysis["interest_burden"]["burden_level"] = "High"
                elif interest_coverage < 5:
                    analysis["interest_burden"]["burden_level"] = "Medium"
                else:
                    analysis["interest_burden"]["burden_level"] = "Low"
        
        # 4. Comprehensive assessment
        leverage_concern = analysis["debt_support_analysis"].get("leverage_concern", False)
        interest_burden = analysis["interest_burden"].get("burden_level", "")
        debt_to_equity = analysis["leverage_trend"].get("latest_debt_to_equity", 0)
        
        if leverage_concern or interest_burden == "High" or debt_to_equity > 1.0:
            analysis["assessment"] = "Needs attention: Signs of leverage supporting earnings, or heavy interest burden"
        elif debt_to_equity > 0.5 or interest_burden == "Medium":
            analysis["assessment"] = "Average: Moderate leverage level, needs ongoing attention"
        else:
            analysis["assessment"] = "Good: Healthy capital structure, earnings not dependent on leverage"
    
    except Exception as e:
        print(f"⚠️ Error analyzing capital structure support: {e}")
        import traceback
        traceback.print_exc()
        analysis["error"] = str(e)
    
    return analysis

# ==================== Comprehensive Earnings Quality Analysis ====================

def comprehensive_earnings_quality_analysis(
    financial_data: Dict[str, Any],
    news_data: Dict[str, Any] = None,
    financial_analysis: Dict[str, Any] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    generate_llm_report: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive earnings sustainability analysis
    
    Integrate all 5 dimensions of analysis, optionally generate LLM earnings analysis report
    
    Args:
        financial_data: Financial data (from data_ingestion)
        news_data: News data (optional, for one-time items analysis)
        financial_analysis: Financial analysis results (optional, for LLM report generation)
        api_key: OpenAI API Key (optional)
        base_url: OpenAI Base URL (optional)
        model: AI model name
        generate_llm_report: Whether to generate LLM earnings analysis report (default False)
    
    Returns:
        Comprehensive earnings quality analysis results (includes LLM report if generate_llm_report=True)
    """
    result = {
        "symbol": financial_data.get("symbol", "UNKNOWN"),
        "cash_vs_profit": {},
        "accrual_quality": {},
        "profit_volatility": {},
        "one_time_items": {},
        "capital_structure_support": {},
        "overall_score": 0,
        "overall_assessment": "",
        "summary": {}
    }
    
    print(f"\n📊 Starting earnings sustainability analysis: {result['symbol']}")
    
    try:
        # Get data
        income_df = financial_data.get("income_statement", {}).get("annual", pd.DataFrame())
        balance_df = financial_data.get("balance_sheet", {}).get("annual", pd.DataFrame())
        cashflow_df = financial_data.get("cash_flow", {}).get("annual", pd.DataFrame())
        
        if income_df.empty:
            print("  ⚠️ Income statement data is empty, cannot perform analysis")
            return result
        
        # 1. Cash vs Profit analysis
        print("  💵 Analyzing cash vs profit...")
        result["cash_vs_profit"] = analyze_cash_vs_profit(income_df, cashflow_df)
        
        # 2. Accrual quality analysis
        print("  📊 Analyzing accrual quality...")
        result["accrual_quality"] = analyze_accrual_quality(income_df, cashflow_df)
        
        # 3. Profit volatility analysis
        print("  📈 Analyzing profit volatility...")
        result["profit_volatility"] = analyze_profit_volatility(income_df)
        
        # 4. One-time items dependency analysis
        print("  🔍 Analyzing one-time items dependency...")
        result["one_time_items"] = analyze_one_time_items(
            income_df=income_df,
            financial_data=financial_data,
            news_data=news_data,
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        
        # 5. Capital structure support analysis
        print("  🏗️ Analyzing capital structure support...")
        result["capital_structure_support"] = analyze_capital_structure_support(
            income_df, balance_df, cashflow_df
        )
        
        # 6. Comprehensive scoring
        result["overall_score"], result["overall_assessment"] = calculate_earnings_quality_score(result)
        
        # 7. Generate summary
        result["summary"] = generate_earnings_quality_summary(result)
        
        print(f"  ✅ Earnings sustainability analysis completed: {result['overall_assessment']} ({result['overall_score']}/100)")
        
        # 8. Generate LLM earnings analysis report (if requested)
        if generate_llm_report and OPENAI_AVAILABLE:
            print("  📝 Generating LLM earnings analysis report...")
            llm_report = generate_earnings_analysis_report_with_llm(
                earnings_quality_result=result,
                financial_data=financial_data,
                financial_analysis=financial_analysis,
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            result["llm_report"] = llm_report
    
    except Exception as e:
        print(f"⚠️ Earnings sustainability analysis failed: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
    
    return result

def calculate_earnings_quality_score(analysis_result: Dict[str, Any]) -> Tuple[int, str]:
    """
    Calculate comprehensive earnings quality score
    
    Args:
        analysis_result: Analysis results
    
    Returns:
        (Score, Rating)
    """
    score = 0
    max_score = 100
    
    try:
        # 1. Cash vs Profit (25 points)
        cash_profit = analysis_result.get("cash_vs_profit", {})
        cfo_ni_ratio = cash_profit.get("cfo_to_net_income", {}).get("latest")
        if cfo_ni_ratio:
            if cfo_ni_ratio > 1.0:
                score += 25
            elif cfo_ni_ratio > 0.8:
                score += 20
            elif cfo_ni_ratio > 0.5:
                score += 15
            else:
                score += 10
        
        # 2. Accrual Quality (20 points)
        accrual = analysis_result.get("accrual_quality", {})
        accruals = accrual.get("accruals", {}).get("latest")
        net_income = accrual.get("accruals", {}).get("net_income", 0)
        if accruals is not None and net_income != 0:
            accrual_ratio = abs(accruals) / abs(net_income)
            if accrual_ratio < 0.2:
                score += 20
            elif accrual_ratio < 0.5:
                score += 15
            else:
                score += 10
        
        # 3. Profit Volatility (20 points)
        volatility = analysis_result.get("profit_volatility", {})
        cv = volatility.get("profit_volatility", {}).get("coefficient_of_variation")
        if cv is not None:
            if cv < 20:
                score += 20
            elif cv < 40:
                score += 15
            elif cv < 60:
                score += 10
            else:
                score += 5
        
        # 4. One-time Items Dependency (20 points)
        one_time = analysis_result.get("one_time_items", {})
        one_time_ratio = one_time.get("special_items_ratio", {}).get("ratio_to_net_income", 100)
        if one_time_ratio < 5:
            score += 20
        elif one_time_ratio < 15:
            score += 15
        elif one_time_ratio < 30:
            score += 10
        else:
            score += 5
        
        # 5. Capital Structure Support (15 points)
        capital = analysis_result.get("capital_structure_support", {})
        leverage_concern = capital.get("debt_support_analysis", {}).get("leverage_concern", False)
        interest_burden = capital.get("interest_burden", {}).get("burden_level", "")
        if not leverage_concern and interest_burden != "High":
            score += 15
        elif interest_burden == "Medium":
            score += 10
        else:
            score += 5
        
        # Rating
        if score >= 80:
            grade = "Excellent"
        elif score >= 60:
            grade = "Good"
        elif score >= 40:
            grade = "Average"
        else:
            grade = "Needs Attention"
        
        return score, grade
    
    except Exception as e:
        print(f"⚠️ Error calculating earnings quality score: {e}")
        return 0, "Cannot Assess"

def generate_earnings_quality_summary(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate earnings quality summary"""
    summary = {
        "key_metrics": {},
        "strengths": [],
        "concerns": [],
        "sustainability_conclusion": "",
        "detailed_metrics": {},
        "historical_trends": {}
    }
    
    try:
        # Key metrics
        cash_profit = analysis_result.get("cash_vs_profit", {})
        cfo_ni_ratio = cash_profit.get("cfo_to_net_income", {}).get("latest")
        if cfo_ni_ratio:
            summary["key_metrics"]["CFO/Net Income"] = f"{cfo_ni_ratio:.2f}"
            summary["detailed_metrics"]["cfo_ni_ratio"] = {
                "latest": cfo_ni_ratio,
                "average": cash_profit.get("cfo_to_net_income", {}).get("average"),
                "median": cash_profit.get("cfo_to_net_income", {}).get("median"),
                "min": cash_profit.get("cfo_to_net_income", {}).get("min"),
                "max": cash_profit.get("cfo_to_net_income", {}).get("max"),
                "trend": cash_profit.get("trend_analysis", {}).get("cfo_ni_trend")
            }
        
        # Cash conversion ratio
        cash_conversion = cash_profit.get("cash_conversion_ratio", {}).get("latest")
        if cash_conversion:
            summary["detailed_metrics"]["cash_conversion_ratio"] = cash_conversion
        
        # Accrual quality detailed metrics
        accrual = analysis_result.get("accrual_quality", {})
        accruals = accrual.get("accruals", {}).get("latest")
        if accruals is not None:
            summary["detailed_metrics"]["accruals"] = {
                "latest": accruals,
                "average": accrual.get("accruals", {}).get("average"),
                "median": accrual.get("accruals", {}).get("median"),
                "std": accrual.get("accruals", {}).get("std"),
                "trend": accrual.get("trend_analysis", {}).get("accrual_trend")
            }
            accrual_ratio = accrual.get("accrual_ratio", {}).get("latest")
            if accrual_ratio is not None:
                summary["key_metrics"]["Accrual Ratio"] = f"{accrual_ratio:.4f}"
        
        # Profit volatility detailed metrics
        volatility = analysis_result.get("profit_volatility", {})
        cv = volatility.get("profit_volatility", {}).get("coefficient_of_variation")
        if cv is not None:
            summary["key_metrics"]["Profit Margin Coefficient of Variation"] = f"{cv:.1f}%"
            margin_stability = volatility.get("margin_stability", {})
            summary["detailed_metrics"]["profit_volatility"] = {
                "coefficient_of_variation": cv,
                "stability": volatility.get("profit_volatility", {}).get("stability"),
                "latest_margin": margin_stability.get("latest"),
                "average_margin": margin_stability.get("average"),
                "min_margin": margin_stability.get("min"),
                "max_margin": margin_stability.get("max"),
                "margin_std": margin_stability.get("std"),
                "trend": volatility.get("trend_analysis", {}).get("margin_trend")
            }
        
        # One-time items detailed metrics
        one_time = analysis_result.get("one_time_items", {})
        one_time_ratio = one_time.get("special_items_ratio", {}).get("ratio_to_net_income")
        if one_time_ratio is not None:
            summary["key_metrics"]["One-time Items Ratio"] = f"{one_time_ratio:.1f}%"
            summary["detailed_metrics"]["one_time_items"] = {
                "ratio_to_ni": one_time_ratio,
                "total_amount": one_time.get("special_items_ratio", {}).get("total_one_time_items"),
                "detected_items_count": len(one_time.get("one_time_items_detected", []))
            }
        
        # Capital structure detailed metrics
        capital = analysis_result.get("capital_structure_support", {})
        leverage_trend = capital.get("leverage_trend", {})
        if leverage_trend.get("latest_debt_to_equity") is not None:
            summary["detailed_metrics"]["capital_structure"] = {
                "debt_to_equity": leverage_trend.get("latest_debt_to_equity"),
                "debt_to_equity_trend": leverage_trend.get("trend"),
                "leverage_concern": capital.get("debt_support_analysis", {}).get("leverage_concern", False),
                "interest_coverage": capital.get("interest_burden", {}).get("interest_coverage"),
                "interest_burden_level": capital.get("interest_burden", {}).get("burden_level")
            }
        
        # Historical trend data
        summary["historical_trends"] = {
            "cfo_ni_trend": cash_profit.get("trend_analysis", {}).get("cfo_ni_trend"),
            "accrual_trend": accrual.get("trend_analysis", {}).get("accrual_trend"),
            "margin_trend": volatility.get("trend_analysis", {}).get("margin_trend"),
            "leverage_trend": leverage_trend.get("trend")
        }
        
        # Strengths
        if cfo_ni_ratio and cfo_ni_ratio > 1.0:
            summary["strengths"].append("Operating cash flow significantly exceeds net income, high earnings quality")
        
        if cv and cv < 20:
            summary["strengths"].append("Profit margin very stable, strong earnings sustainability")
        
        if one_time_ratio and one_time_ratio < 5:
            summary["strengths"].append("One-time items ratio very small, earnings mainly from core business")
        
        if capital.get("debt_support_analysis", {}).get("leverage_concern", False) == False:
            summary["strengths"].append("Healthy capital structure, earnings not dependent on leverage")
        
        # Concerns
        if cfo_ni_ratio and cfo_ni_ratio < 0.5:
            summary["concerns"].append("Operating cash flow far below net income, earnings quality needs caution")
        
        if cv and cv > 60:
            summary["concerns"].append("Profit margin volatility very high, earnings sustainability questionable")
        
        if one_time_ratio and one_time_ratio > 30:
            summary["concerns"].append("One-time items ratio very high, earnings sustainability questionable")
        
        if capital.get("debt_support_analysis", {}).get("leverage_concern", False):
            summary["concerns"].append("Signs of leverage supporting earnings")
        
        if capital.get("interest_burden", {}).get("burden_level") == "High":
            summary["concerns"].append("Heavy interest burden, may affect earnings sustainability")
        
        # Sustainability conclusion
        overall_score = analysis_result.get("overall_score", 0)
        if overall_score >= 80:
            summary["sustainability_conclusion"] = "Excellent earnings quality, strong sustainability"
        elif overall_score >= 60:
            summary["sustainability_conclusion"] = "Good earnings quality, relatively good sustainability"
        elif overall_score >= 40:
            summary["sustainability_conclusion"] = "Average earnings quality, needs ongoing attention"
        else:
            summary["sustainability_conclusion"] = "Earnings quality questionable, sustainability needs caution"
    
    except Exception as e:
        print(f"⚠️ Error generating earnings quality summary: {e}")
    
    return summary

# ==================== LLM Earnings Analysis Report Generation ====================

def generate_earnings_analysis_report_with_llm(
    earnings_quality_result: Dict[str, Any],
    financial_data: Dict[str, Any] = None,
    financial_analysis: Dict[str, Any] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Generate professional earnings analysis report using LLM
    
    Generate an in-depth, professional earnings sustainability analysis report, including:
    1. Executive Summary
    2. Comprehensive earnings quality assessment
    3. Detailed analysis of five dimensions
    4. Historical trend analysis
    5. Earnings sustainability judgment
    6. Investment recommendations and risk warnings
    
    Args:
        earnings_quality_result: Earnings quality analysis results
        financial_data: Financial data (optional, for additional context)
        financial_analysis: Financial analysis results (optional, for additional context)
        api_key: OpenAI API Key
        base_url: OpenAI Base URL
        model: AI model name
    
    Returns:
        Dictionary containing complete earnings analysis report
    """
    result = {
        "symbol": earnings_quality_result.get("symbol", "UNKNOWN"),
        "report": "",
        "executive_summary": "",
        "detailed_analysis": {},
        "investment_implications": "",
        "risk_warnings": [],
        "raw_llm_response": ""
    }
    
    if not OPENAI_AVAILABLE:
        result["error"] = "LLM not available"
        return result
    
    # Set default API key and base_url (if not provided)
    if not api_key:
        try:
            from talk2ai import get_config_from_env
            env_api_key, env_base_url = get_config_from_env()
            api_key = env_api_key
            if not base_url:
                base_url = env_base_url or "https://api.openai.com/v1"
        except:
            api_key = None
            if not base_url:
                base_url = "https://api.openai.com/v1"
        
        # Validate API Key
        if not api_key:
            result["error"] = "OpenAI API Key is required"
            return result
    
    try:
        print(f"\n📝 Starting to generate earnings analysis report for {result['symbol']}...")
        
        # Format analysis data
        analysis_data = format_earnings_quality_data_for_llm(
            earnings_quality_result=earnings_quality_result,
            financial_data=financial_data,
            financial_analysis=financial_analysis
        )
        
        # Build Prompt
        prompt = build_earnings_analysis_report_prompt(
            symbol=result['symbol'],
            analysis_data=analysis_data
        )
        
        # Call LLM
        chat_client = OpenAIChat(api_key=api_key, base_url=base_url)
        messages = [
            {
                "role": "system",
                "content": """You are a senior financial analyst with over 15 years of earnings quality analysis experience.
Your expertise lies in assessing the health and sustainability of company earnings, identifying earnings quality risks, and providing professional investment recommendations.

Your analysis should:
1. Be based on facts and data, deeply analyze all dimensions of earnings
2. Identify earnings quality strengths and risk points
3. Assess earnings sustainability
4. Provide clear, actionable investment recommendations
5. Use professional financial terminology and analytical frameworks

Your report should meet the professional standards of buy-side research departments - deep, comprehensive, and insightful."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        print("  🤖 Calling LLM to generate earnings analysis report...")
        llm_response = chat_client.chat(messages, model=model)
        
        if llm_response:
            # Parse response
            parsed = parse_earnings_analysis_report_response(llm_response)
            result.update(parsed)
            result["raw_llm_response"] = llm_response
            
            # Generate full report
            result["report"] = format_full_earnings_report(result, earnings_quality_result)
            
            print(f"  ✅ Earnings analysis report generation completed")
        else:
            result["error"] = "LLM response is empty"
    
    except Exception as e:
        print(f"⚠️ Failed to generate earnings analysis report: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
    
    return result

def format_earnings_quality_data_for_llm(
    earnings_quality_result: Dict[str, Any],
    financial_data: Dict[str, Any] = None,
    financial_analysis: Dict[str, Any] = None
) -> str:
    """Format earnings quality analysis results into LLM-readable text"""
    formatted_text = f"# {earnings_quality_result.get('symbol', 'UNKNOWN')} Earnings Quality Analysis Data\n\n"
    
    # Overall score
    overall_score = earnings_quality_result.get("overall_score", 0)
    overall_assessment = earnings_quality_result.get("overall_assessment", "N/A")
    formatted_text += f"## Comprehensive Assessment\n\n"
    formatted_text += f"- Overall Score: {overall_score}/100\n"
    formatted_text += f"- Rating: {overall_assessment}\n\n"
    
    # 1. Cash vs Profit
    cash_profit = earnings_quality_result.get("cash_vs_profit", {})
    if cash_profit:
        formatted_text += "## 1. Cash vs Profit Analysis\n\n"
        cfo_ni = cash_profit.get("cfo_to_net_income", {})
        if cfo_ni.get("latest"):
            formatted_text += f"- CFO/Net Income Ratio: {cfo_ni['latest']:.2f}\n"
            if cfo_ni.get("average"):
                formatted_text += f"- Historical Average: {cfo_ni['average']:.2f}\n"
            if cfo_ni.get("median"):
                formatted_text += f"- Historical Median: {cfo_ni['median']:.2f}\n"
            formatted_text += f"- Latest Net Income: ${cfo_ni.get('net_income', 0)/1e9:.2f}B\n"
            formatted_text += f"- Latest Operating Cash Flow: ${cfo_ni.get('operating_cf', 0)/1e9:.2f}B\n"
        cash_conv = cash_profit.get("cash_conversion_ratio", {})
        if cash_conv.get("latest"):
            formatted_text += f"- Cash Conversion Ratio (CFO/Revenue): {cash_conv['latest']:.2f}\n"
        trend = cash_profit.get("trend_analysis", {}).get("cfo_ni_trend")
        if trend:
            formatted_text += f"- Trend: {trend}\n"
        formatted_text += f"- Assessment: {cash_profit.get('assessment', 'N/A')}\n\n"
    
    # 2. Accrual Quality
    accrual = earnings_quality_result.get("accrual_quality", {})
    if accrual:
        formatted_text += "## 2. Accrual Quality Analysis\n\n"
        accruals_dict = accrual.get("accruals", {})
        if accruals_dict.get("latest") is not None:
            formatted_text += f"- Accruals (NI - CFO): ${accruals_dict['latest']/1e9:.2f}B\n"
            if accruals_dict.get("average"):
                formatted_text += f"- Historical Average: ${accruals_dict['average']/1e9:.2f}B\n"
            if accruals_dict.get("median"):
                formatted_text += f"- Historical Median: ${accruals_dict['median']/1e9:.2f}B\n"
        accrual_ratio = accrual.get("accrual_ratio", {})
        if accrual_ratio.get("latest") is not None:
            formatted_text += f"- Accrual Ratio (Accruals/Revenue): {accrual_ratio['latest']:.4f}\n"
        trend = accrual.get("trend_analysis", {}).get("accrual_trend")
        if trend:
            formatted_text += f"- Trend: {trend}\n"
        formatted_text += f"- Assessment: {accrual.get('assessment', 'N/A')}\n\n"
    
    # 3. Profit Volatility
    volatility = earnings_quality_result.get("profit_volatility", {})
    if volatility:
        formatted_text += "## 3. Profit Volatility Analysis\n\n"
        margin_stability = volatility.get("margin_stability", {})
        if margin_stability.get("latest") is not None:
            formatted_text += f"- Latest Net Margin: {margin_stability['latest']:.2f}%\n"
            if margin_stability.get("average"):
                formatted_text += f"- Historical Average Net Margin: {margin_stability['average']:.2f}%\n"
            if margin_stability.get("min"):
                formatted_text += f"- Minimum Net Margin: {margin_stability['min']:.2f}%\n"
            if margin_stability.get("max"):
                formatted_text += f"- Maximum Net Margin: {margin_stability['max']:.2f}%\n"
        profit_vol = volatility.get("profit_volatility", {})
        cv = profit_vol.get("coefficient_of_variation")
        if cv is not None:
            formatted_text += f"- Margin Coefficient of Variation: {cv:.1f}%\n"
            formatted_text += f"- Stability Rating: {profit_vol.get('stability', 'N/A')}\n"
        trend = volatility.get("trend_analysis", {}).get("margin_trend")
        if trend:
            formatted_text += f"- Trend: {trend}\n"
        formatted_text += f"- Assessment: {volatility.get('assessment', 'N/A')}\n\n"
    
    # 4. One-time Items
    one_time = earnings_quality_result.get("one_time_items", {})
    if one_time:
        formatted_text += "## 4. One-time Items Dependency Analysis\n\n"
        special_items = one_time.get("special_items_ratio", {})
        one_time_ratio = special_items.get("ratio_to_net_income")
        if one_time_ratio is not None:
            formatted_text += f"- One-time Items Ratio: {one_time_ratio:.1f}%\n"
            formatted_text += f"- Total One-time Items: ${special_items.get('total_one_time_items', 0)/1e9:.2f}B\n"
        detected_items = one_time.get("one_time_items_detected", [])
        if detected_items:
            formatted_text += f"- Number of Detected One-time Items: {len(detected_items)}\n"
            for item in detected_items[:5]:  # Maximum 5
                formatted_text += f"  - {item.get('item', 'N/A')}: ${item.get('value', 0)/1e9:.2f}B\n"
        llm_analysis = one_time.get("llm_analysis", {})
        if llm_analysis.get("one_time_items_identified"):
            formatted_text += f"- LLM-Identified One-time Items: {len(llm_analysis['one_time_items_identified'])} items\n"
        formatted_text += f"- Assessment: {one_time.get('assessment', 'N/A')}\n\n"
    
    # 5. Capital Structure Support
    capital = earnings_quality_result.get("capital_structure_support", {})
    if capital:
        formatted_text += "## 5. Capital Structure Support Analysis\n\n"
        leverage_trend = capital.get("leverage_trend", {})
        if leverage_trend.get("latest_debt_to_equity") is not None:
            formatted_text += f"- Latest Debt/Equity Ratio: {leverage_trend['latest_debt_to_equity']:.2f}\n"
            if leverage_trend.get("previous_debt_to_equity") is not None:
                formatted_text += f"- Previous Debt/Equity Ratio: {leverage_trend['previous_debt_to_equity']:.2f}\n"
            formatted_text += f"- Trend: {leverage_trend.get('trend', 'N/A')}\n"
        debt_support = capital.get("debt_support_analysis", {})
        if debt_support.get("leverage_concern") is not None:
            formatted_text += f"- Leverage Concern: {'Yes' if debt_support['leverage_concern'] else 'No'}\n"
            if debt_support.get("concern_level"):
                formatted_text += f"- Concern Level: {debt_support['concern_level']}\n"
        interest_burden = capital.get("interest_burden", {})
        if interest_burden.get("interest_coverage"):
            formatted_text += f"- Interest Coverage: {interest_burden['interest_coverage']:.2f}x\n"
            formatted_text += f"- Interest Burden Level: {interest_burden.get('burden_level', 'N/A')}\n"
        formatted_text += f"- Assessment: {capital.get('assessment', 'N/A')}\n\n"
    
    # Summary
    summary = earnings_quality_result.get("summary", {})
    if summary:
        formatted_text += "## Comprehensive Summary\n\n"
        key_metrics = summary.get("key_metrics", {})
        if key_metrics:
            formatted_text += "### Key Metrics\n"
            for key, value in key_metrics.items():
                formatted_text += f"- {key}: {value}\n"
            formatted_text += "\n"
        
        strengths = summary.get("strengths", [])
        if strengths:
            formatted_text += "### Strengths\n"
            for strength in strengths:
                formatted_text += f"- ✅ {strength}\n"
            formatted_text += "\n"
        
        concerns = summary.get("concerns", [])
        if concerns:
            formatted_text += "### Concerns\n"
            for concern in concerns:
                formatted_text += f"- ⚠️ {concern}\n"
            formatted_text += "\n"
        
        formatted_text += f"### Sustainability Conclusion\n{summary.get('sustainability_conclusion', 'N/A')}\n\n"
    
    # If financial analysis data is available, add some context
    if financial_analysis:
        profitability = financial_analysis.get("profitability", {})
        if profitability:
            formatted_text += "## Supplementary Financial Data\n\n"
            formatted_text += f"- ROE: {profitability.get('roe', 0) or 0:.2f}%\n"
            formatted_text += f"- ROIC: {profitability.get('roic', 0) or 0:.2f}%\n"
            formatted_text += f"- Net Margin: {profitability.get('net_margin', 0) or 0:.2f}%\n\n"
    
    return formatted_text

def build_earnings_analysis_report_prompt(symbol: str, analysis_data: str) -> str:
    """Build earnings analysis report prompt"""
    prompt = f"""Please generate a professional, in-depth, and comprehensive earnings analysis report based on the following earnings quality analysis data.

## Analysis Data

{analysis_data}

## Report Requirements

Please generate a structured earnings analysis report containing the following sections:

### 1. Executive Summary
- Overall score and rating
- Core conclusions on earnings quality
- 3-5 key findings

### 2. Comprehensive Earnings Quality Assessment
- Comprehensive assessment of five dimensions
- Earnings quality strengths and risk points
- Earnings sustainability judgment

### 3. Detailed Analysis of Five Dimensions
Deep analysis of each dimension:
- **Cash vs Profit**: Analyze the relationship between operating cash flow and net income, assess the cash support level of earnings
- **Accrual Quality**: Analyze accruals, assess earnings' dependence on cash
- **Profit Volatility**: Analyze margin stability, assess earnings predictability
- **One-time Items Dependency**: Identify one-time items, assess core earnings quality
- **Capital Structure Support**: Analyze leverage trends, assess whether earnings are "propped up" by increasing leverage

### 4. Historical Trend Analysis
- Historical trends of each metric
- Analysis of reasons for trend changes
- Future trend predictions

### 5. Earnings Sustainability Judgment
- Comprehensive assessment of earnings sustainability
- Identify key factors affecting sustainability
- Assess future earnings quality risks and opportunities

### 6. Investment Recommendations and Risk Warnings
- Investment recommendations based on earnings quality analysis
- Risk points to watch
- Recommended monitoring metrics

## Output Format

Please return in JSON format, containing the following fields:

```json
{{
    "executive_summary": {{
        "overall_score": 0,
        "overall_rating": "Excellent/Good/Average/Needs Attention",
        "key_findings": [
            "Finding 1",
            "Finding 2",
            "Finding 3"
        ],
        "core_conclusion": "Core conclusion (2-3 sentences)"
    }},
    "comprehensive_assessment": {{
        "five_dimensions_summary": "Comprehensive assessment of five dimensions (detailed description)",
        "strengths": [
            "Strength 1",
            "Strength 2"
        ],
        "weaknesses": [
            "Risk point 1",
            "Risk point 2"
        ],
        "sustainability_judgment": "Earnings sustainability judgment (detailed description)"
    }},
    "detailed_analysis": {{
        "cash_vs_profit": "Detailed analysis of cash vs profit (in-depth analysis)",
        "accrual_quality": "Detailed analysis of accrual quality (in-depth analysis)",
        "profit_volatility": "Detailed analysis of profit volatility (in-depth analysis)",
        "one_time_items": "Detailed analysis of one-time items dependency (in-depth analysis)",
        "capital_structure": "Detailed analysis of capital structure support (in-depth analysis)"
    }},
    "historical_trends": {{
        "trend_analysis": "Comprehensive historical trend analysis (detailed description)",
        "trend_drivers": [
            "Trend driver 1",
            "Trend driver 2"
        ],
        "future_outlook": "Future trend outlook (detailed description)"
    }},
    "investment_implications": {{
        "investment_advice": "Investment recommendations based on earnings quality (detailed description)",
        "key_monitoring_metrics": [
            "Monitoring metric 1",
            "Monitoring metric 2"
        ],
        "risk_warnings": [
            "Risk warning 1",
            "Risk warning 2"
        ]
    }}
}}
```

Please ensure:
1. Analysis is deep, professional, and insightful
2. Analysis is based on provided data, avoid subjective speculation
3. Use professional financial terminology
4. Provide actionable investment recommendations
5. Identify key risks and opportunities"""
    
    return prompt

def parse_earnings_analysis_report_response(llm_response: str) -> Dict[str, Any]:
    """Parse LLM earnings analysis report response"""
    result = {
        "executive_summary": {},
        "comprehensive_assessment": {},
        "detailed_analysis": {},
        "historical_trends": {},
        "investment_implications": {}
    }
    
    try:
        # Extract JSON
        json_str = None
        if "```json" in llm_response:
            json_start = llm_response.find("```json") + 7
            json_end = llm_response.find("```", json_start)
            if json_end > json_start:
                json_str = llm_response[json_start:json_end].strip()
        elif "```" in llm_response:
            json_start = llm_response.find("```") + 3
            json_end = llm_response.find("```", json_start)
            if json_end > json_start:
                json_str = llm_response[json_start:json_end].strip()
        else:
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
        
        if json_str:
            parsed = json.loads(json_str)
            result["executive_summary"] = parsed.get("executive_summary", {})
            result["comprehensive_assessment"] = parsed.get("comprehensive_assessment", {})
            result["detailed_analysis"] = parsed.get("detailed_analysis", {})
            result["historical_trends"] = parsed.get("historical_trends", {})
            result["investment_implications"] = parsed.get("investment_implications", {})
    
    except Exception as e:
        print(f"⚠️ Failed to parse earnings analysis report: {e}")
        result["parse_error"] = str(e)
    
    return result

def format_full_earnings_report(
    parsed_result: Dict[str, Any],
    earnings_quality_result: Dict[str, Any]
) -> str:
    """Format complete earnings analysis report"""
    symbol = earnings_quality_result.get("symbol", "UNKNOWN")
    overall_score = earnings_quality_result.get("overall_score", 0)
    overall_assessment = earnings_quality_result.get("overall_assessment", "N/A")
    
    report = f"""
{'='*80}
{symbol} Earnings Sustainability Analysis Report
{'='*80}

Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Overall Score: {overall_score}/100
Rating: {overall_assessment}

{'-'*80}
1. Executive Summary
{'-'*80}

"""
    
    exec_summary = parsed_result.get("executive_summary", {})
    if exec_summary:
        report += f"Overall Score: {exec_summary.get('overall_score', overall_score)}/100\n"
        report += f"Rating: {exec_summary.get('overall_rating', overall_assessment)}\n\n"
        report += f"Core Conclusion:\n{exec_summary.get('core_conclusion', 'N/A')}\n\n"
        key_findings = exec_summary.get("key_findings", [])
        if key_findings:
            report += "Key Findings:\n"
            for i, finding in enumerate(key_findings, 1):
                report += f"{i}. {finding}\n"
            report += "\n"
    
    report += f"""
{'-'*80}
2. Comprehensive Earnings Quality Assessment
{'-'*80}

"""
    
    comp_assessment = parsed_result.get("comprehensive_assessment", {})
    if comp_assessment:
        report += f"{comp_assessment.get('five_dimensions_summary', 'N/A')}\n\n"
        
        strengths = comp_assessment.get("strengths", [])
        if strengths:
            report += "Strengths:\n"
            for strength in strengths:
                report += f"  ✅ {strength}\n"
            report += "\n"
        
        weaknesses = comp_assessment.get("weaknesses", [])
        if weaknesses:
            report += "Risk Points:\n"
            for weakness in weaknesses:
                report += f"  ⚠️ {weakness}\n"
            report += "\n"
        
        report += f"Earnings Sustainability Judgment:\n{comp_assessment.get('sustainability_judgment', 'N/A')}\n\n"
    
    report += f"""
{'-'*80}
3. Detailed Analysis of Five Dimensions
{'-'*80}

"""
    
    detailed = parsed_result.get("detailed_analysis", {})
    if detailed:
        report += "3.1 Cash vs Profit Analysis\n"
        report += f"{detailed.get('cash_vs_profit', 'N/A')}\n\n"
        
        report += "3.2 Accrual Quality Analysis\n"
        report += f"{detailed.get('accrual_quality', 'N/A')}\n\n"
        
        report += "3.3 Profit Volatility Analysis\n"
        report += f"{detailed.get('profit_volatility', 'N/A')}\n\n"
        
        report += "3.4 One-time Items Dependency Analysis\n"
        report += f"{detailed.get('one_time_items', 'N/A')}\n\n"
        
        report += "3.5 Capital Structure Support Analysis\n"
        report += f"{detailed.get('capital_structure', 'N/A')}\n\n"
    
    report += f"""
{'-'*80}
4. Historical Trend Analysis
{'-'*80}

"""
    
    trends = parsed_result.get("historical_trends", {})
    if trends:
        report += f"{trends.get('trend_analysis', 'N/A')}\n\n"
        
        trend_drivers = trends.get("trend_drivers", [])
        if trend_drivers:
            report += "Trend Drivers:\n"
            for driver in trend_drivers:
                report += f"  • {driver}\n"
            report += "\n"
        
        report += f"Future Trend Outlook:\n{trends.get('future_outlook', 'N/A')}\n\n"
    
    report += f"""
{'-'*80}
5. Investment Recommendations and Risk Warnings
{'-'*80}

"""
    
    implications = parsed_result.get("investment_implications", {})
    if implications:
        report += f"Investment Recommendations:\n{implications.get('investment_advice', 'N/A')}\n\n"
        
        monitoring_metrics = implications.get("key_monitoring_metrics", [])
        if monitoring_metrics:
            report += "Key Monitoring Metrics:\n"
            for metric in monitoring_metrics:
                report += f"  • {metric}\n"
            report += "\n"
        
        risk_warnings = implications.get("risk_warnings", [])
        if risk_warnings:
            report += "Risk Warnings:\n"
            for warning in risk_warnings:
                report += f"  ⚠️ {warning}\n"
            report += "\n"
    
    report += f"""
{'='*80}
End of Report
{'='*80}
"""
    
    return report

# ==================== Main Function (for testing) ====================

if __name__ == "__main__":
    # Test example
    print("=" * 60)
    print("Earnings Quality Analysis Module - Earnings Sustainability Analysis Test")
    print("=" * 60)
    
    try:
        from data_ingestion import fetch_financial_statements
        from qualitative_analysis import fetch_stock_news
    except ImportError:
        from .data_ingestion import fetch_financial_statements
        from .qualitative_analysis import fetch_stock_news
    
    symbol = "NVDA"
    print(f"\nFetching financial statement data for {symbol}...")
    financial_data = fetch_financial_statements(symbol, years=5)
    
    print(f"\nFetching news data for {symbol}...")
    news_data = fetch_stock_news(symbol, limit=30)
    
    print(f"\nPerforming earnings sustainability analysis...")
    analysis = comprehensive_earnings_quality_analysis(
        financial_data=financial_data,
        news_data=news_data
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Analysis Results:")
    print("=" * 60)
    
    print(f"\n💵 Cash vs Profit:")
    cash_profit = analysis.get("cash_vs_profit", {})
    cfo_ni = cash_profit.get("cfo_to_net_income", {})
    if cfo_ni.get("latest"):
        print(f"  CFO/Net Income: {cfo_ni['latest']:.2f}")
    print(f"  Assessment: {cash_profit.get('assessment', 'N/A')}")
    
    print(f"\n📊 Accrual Quality:")
    accrual = analysis.get("accrual_quality", {})
    if accrual.get("accruals", {}).get("latest") is not None:
        print(f"  Accruals: ${accrual['accruals']['latest']/1e9:.2f}B")
    print(f"  Assessment: {accrual.get('assessment', 'N/A')}")
    
    print(f"\n📈 Profit Volatility:")
    volatility = analysis.get("profit_volatility", {})
    cv = volatility.get("profit_volatility", {}).get("coefficient_of_variation")
    if cv:
        print(f"  Coefficient of Variation: {cv:.1f}%")
        print(f"  Stability: {volatility['profit_volatility'].get('stability', 'N/A')}")
    print(f"  Assessment: {volatility.get('assessment', 'N/A')}")
    
    print(f"\n🔍 One-time Items:")
    one_time = analysis.get("one_time_items", {})
    one_time_ratio = one_time.get("special_items_ratio", {}).get("ratio_to_net_income")
    if one_time_ratio is not None:
        print(f"  One-time Items Ratio: {one_time_ratio:.1f}%")
    if one_time.get("llm_analysis", {}).get("one_time_items_identified"):
        print(f"  LLM-Identified One-time Items: {len(one_time['llm_analysis']['one_time_items_identified'])} items")
    print(f"  Assessment: {one_time.get('assessment', 'N/A')}")
    
    print(f"\n🏗️ Capital Structure Support:")
    capital = analysis.get("capital_structure_support", {})
    leverage_trend = capital.get("leverage_trend", {})
    if leverage_trend.get("latest_debt_to_equity"):
        print(f"  Debt-to-Equity Ratio: {leverage_trend['latest_debt_to_equity']:.2f}")
        print(f"  Trend: {leverage_trend.get('trend', 'N/A')}")
    print(f"  Assessment: {capital.get('assessment', 'N/A')}")
    
    print(f"\n📋 Comprehensive Assessment:")
    print(f"  Overall Score: {analysis.get('overall_score', 0)}/100")
    print(f"  Rating: {analysis.get('overall_assessment', 'N/A')}")
    
    summary = analysis.get("summary", {})
    if summary.get("strengths"):
        print("  Strengths:")
        for strength in summary["strengths"]:
            print(f"    ✅ {strength}")
    
    if summary.get("concerns"):
        print("  Concerns:")
        for concern in summary["concerns"]:
            print(f"    ⚠️ {concern}")
    
    print(f"\n  Sustainability Conclusion: {summary.get('sustainability_conclusion', 'N/A')}")
    
    print("\n✅ Test completed!")

