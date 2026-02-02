"""
Investment Memo Generator Module for Fundamental Analysis

Features:
Integrates results from all analysis modules, uses LLM to generate professional investment memos, including:
1. Investment Thesis
2. Business Overview
3. Financial Performance & Trends
4. Valuation Summary
5. Peer Comparison
6. Catalysts
7. Risks
8. Recommendation (Buy/Hold/Sell + rationale)

Based on the convergence of valuation upside, improving fundamentals, and manageable risk profile, the agent recommends...
"""

import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import LLM module
try:
    from talk2ai import OpenAIChat, get_config_from_env
    OPENAI_AVAILABLE = True
except ImportError:
    try:
        from .talk2ai import OpenAIChat, get_config_from_env
        OPENAI_AVAILABLE = True
    except ImportError:
        OPENAI_AVAILABLE = False
        print("OpenAI module not found, investment memo generation will be unavailable")
        OpenAIChat = None
        get_config_from_env = None

# ==================== Data Integration and Formatting ====================

def format_analysis_data_for_llm(
    symbol: str,
    financial_data: Dict[str, Any],
    financial_analysis: Dict[str, Any],
    valuation_result: Dict[str, Any],
    comprehensive_analysis: Dict[str, Any],
    qualitative_analysis: Dict[str, Any],
    earnings_quality: Dict[str, Any]
) -> str:
    """
    Format all analysis results into detailed text understandable by LLM
    
    Note: This function includes detailed information from all analysis data to ensure LLM can fully utilize all analysis results
    
    Args:
        symbol: Stock symbol
        financial_data: Financial data
        financial_analysis: Financial analysis results
        valuation_result: Valuation results
        comprehensive_analysis: Comprehensive analysis results
        qualitative_analysis: Qualitative analysis results
        earnings_quality: Earnings quality analysis results
    
    Returns:
        Formatted text
    """
    formatted_text = f"""
# {symbol} Complete Fundamental Analysis Data
Data Source: Alpha Vantage API, Financial Statement Analysis, Valuation Models, Peer Comparison, News Analysis, Earnings Quality Analysis
Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

================================================================================
Important Note: The following data is derived from comprehensive analysis. Please fully utilize this data in the investment memo.
Each section should have specific data support and cite data sources.
================================================================================

"""
    
    # 1. Company basic information (detailed)
    overview = financial_data.get("overview", {})
    if overview:
        formatted_text += "## 1. Company Basic Information (Data Source: Alpha Vantage Overview API)\n\n"
        formatted_text += f"- **Company Name**: {overview.get('Name', 'N/A')}\n"
        formatted_text += f"- **Stock Symbol**: {symbol}\n"
        formatted_text += f"- **Industry**: {overview.get('Industry', 'N/A')}\n"
        formatted_text += f"- **Sector**: {overview.get('Sector', 'N/A')}\n"
        market_cap = overview.get('MarketCapitalization', 0)
        if market_cap and market_cap != 'None':
            formatted_text += f"- **Market Cap**: ${float(market_cap)/1e9:.2f}B\n"
        shares_out = overview.get('SharesOutstanding', 0)
        if shares_out and shares_out != 'None':
            formatted_text += f"- **Shares Outstanding**: {float(shares_out)/1e9:.2f}B shares\n"
        pe_ratio = overview.get('PERatio', 'N/A')
        formatted_text += f"- **P/E Ratio**: {pe_ratio}\n"
        dividend_yield = overview.get('DividendYield', 'N/A')
        formatted_text += f"- **Dividend Yield**: {dividend_yield}\n"
        beta = overview.get('Beta', 'N/A')
        formatted_text += f"- **Beta**: {beta}\n"
        week_high = overview.get('52WeekHigh', 'N/A')
        week_low = overview.get('52WeekLow', 'N/A')
        formatted_text += f"- **52-Week Price Range**: ${week_low} - ${week_high}\n"
        description = overview.get('Description', '')
        if description and description != 'None':
            formatted_text += f"- **Company Description**: {description[:800]}...\n"
        formatted_text += "\n"
    
    # 2. Financial Performance & Trends (detailed data)
    formatted_text += "## 2. Financial Performance & Trends (Data Source: Financial Statement Analysis, based on 5-year historical data)\n\n"
    
    # Profitability (detailed)
    profitability = financial_analysis.get("profitability", {})
    if profitability:
        formatted_text += "### 2.1 Profitability Metrics\n\n"
        gross_margin = profitability.get('gross_margin', 0) or 0
        formatted_text += f"- **Gross Margin**: {gross_margin:.2f}%\n"
        operating_margin = profitability.get('operating_margin', 0) or 0
        formatted_text += f"- **Operating Margin**: {operating_margin:.2f}%\n"
        net_margin = profitability.get('net_margin', 0) or 0
        formatted_text += f"- **Net Margin**: {net_margin:.2f}%\n"
        roe = profitability.get('roe', 0) or 0
        formatted_text += f"- **ROE (Return on Equity)**: {roe:.2f}%\n"
        roic = profitability.get('roic', 0) or 0
        formatted_text += f"- **ROIC (Return on Invested Capital)**: {roic:.2f}%\n"
        
        # Trend analysis
        trends = profitability.get('trends', {})
        if trends:
            formatted_text += "\n**Profitability Trends**:\n"
            if trends.get('gross_margin_trend'):
                formatted_text += f"- Gross Margin Trend: {trends['gross_margin_trend']}\n"
            if trends.get('net_margin_trend'):
                formatted_text += f"- Net Margin Trend: {trends['net_margin_trend']}\n"
        formatted_text += "\n"
    
    # Growth (detailed)
    growth = financial_analysis.get("growth", {})
    if growth:
        formatted_text += "### 2.2 Growth Metrics\n\n"
        revenue_cagr = growth.get('revenue_cagr_5y', 0) or growth.get('revenue_cagr', 0) or 0
        formatted_text += f"- **Revenue 5-Year CAGR**: {revenue_cagr:.2f}%\n"
        ebitda_cagr = growth.get('ebitda_cagr_5y', 0) or growth.get('ebitda_cagr', 0) or 0
        formatted_text += f"- **EBITDA 5-Year CAGR**: {ebitda_cagr:.2f}%\n"
        eps_cagr = growth.get('eps_cagr_5y', 0) or growth.get('eps_cagr', 0) or 0
        if eps_cagr:
            formatted_text += f"- **EPS 5-Year CAGR**: {eps_cagr:.2f}%\n"
        revenue_growth_yoy = growth.get('revenue_growth_yoy', {})
        if isinstance(revenue_growth_yoy, dict):
            revenue_growth = revenue_growth_yoy.get('latest', 0) or 0
        else:
            revenue_growth = revenue_growth_yoy or 0
        if revenue_growth:
            formatted_text += f"- **Latest Fiscal Year Revenue Growth YoY**: {revenue_growth:.2f}%\n"
        net_income_growth = growth.get('net_income_growth_yoy', {})
        if isinstance(net_income_growth, dict):
            ni_growth = net_income_growth.get('latest', 0) or 0
        else:
            ni_growth = net_income_growth or 0
        if ni_growth:
            formatted_text += f"- **Latest Fiscal Year Net Income Growth YoY**: {ni_growth:.2f}%\n"
        formatted_text += "\n"
    
    # Leverage & Solvency (detailed)
    leverage = financial_analysis.get("leverage", {})
    if leverage:
        formatted_text += "### 2.3 Leverage & Solvency Metrics\n\n"
        debt_to_equity = leverage.get('debt_to_equity', 0) or 0
        formatted_text += f"- **Debt to Equity**: {debt_to_equity:.2f}\n"
        net_debt_ebitda = leverage.get('net_debt_to_ebitda', 0) or 0
        formatted_text += f"- **Net Debt/EBITDA**: {net_debt_ebitda:.2f}\n"
        interest_coverage = leverage.get('interest_coverage', 0) or 0
        formatted_text += f"- **Interest Coverage**: {interest_coverage:.2f}x\n"
        debt_to_assets = leverage.get('debt_to_assets', 0) or 0
        if debt_to_assets:
            formatted_text += f"- **Debt to Assets**: {debt_to_assets:.2f}%\n"
        current_ratio = leverage.get('current_ratio', 0) or 0
        if current_ratio:
            formatted_text += f"- **Current Ratio**: {current_ratio:.2f}\n"
        formatted_text += "\n"
    
    # Operational Efficiency (detailed)
    efficiency = financial_analysis.get("efficiency", {})
    if efficiency:
        formatted_text += "### 2.4 Operational Efficiency Metrics\n\n"
        asset_turnover = efficiency.get('asset_turnover', 0) or 0
        formatted_text += f"- **Asset Turnover**: {asset_turnover:.2f}x\n"
        inventory_turnover = efficiency.get('inventory_turnover', 0) or 0
        if inventory_turnover:
            formatted_text += f"- **Inventory Turnover**: {inventory_turnover:.2f}x\n"
        receivables_turnover = efficiency.get('receivables_turnover', 0) or 0
        if receivables_turnover:
            formatted_text += f"- **Receivables Turnover**: {receivables_turnover:.2f}x\n"
        cash_conversion_cycle = efficiency.get('cash_conversion_cycle', 0) or 0
        if cash_conversion_cycle:
            formatted_text += f"- **Cash Conversion Cycle**: {cash_conversion_cycle:.2f} days\n"
        formatted_text += "\n"
    
    # 3. Valuation Summary (detailed data)
    formatted_text += "## 3. Valuation Summary (Data Source: DCF Valuation Model, Multiples Valuation Model)\n\n"
    
    if valuation_result:
        # DCF Valuation (detailed)
        dcf = valuation_result.get("dcf_valuation", {})
        if dcf:
            formatted_text += "### 3.1 DCF Valuation Results\n\n"
            ev = dcf.get('enterprise_value', 0)
            equity_value = dcf.get('equity_value', 0)
            target_price = dcf.get('target_price', 0)
            current_price = dcf.get('current_price', 0)
            upside = dcf.get('upside_potential', 0)
            
            formatted_text += f"- **Enterprise Value**: ${ev/1e9:.2f}B\n"
            formatted_text += f"- **Equity Value**: ${equity_value/1e9:.2f}B\n"
            formatted_text += f"- **Target Price**: ${target_price:.2f}\n"
            formatted_text += f"- **Current Price**: ${current_price:.2f}\n"
            formatted_text += f"- **Upside Potential**: {upside:.2f}%\n"
            
            # DCF Assumptions
            assumptions = dcf.get("assumptions", {})
            if assumptions:
                formatted_text += "\n**DCF Model Key Assumptions**:\n"
                if assumptions.get("revenue_growth_rate"):
                    formatted_text += f"- Revenue Growth Rate: {assumptions['revenue_growth_rate']:.2f}%\n"
                if assumptions.get("operating_margin"):
                    formatted_text += f"- Operating Margin: {assumptions['operating_margin']:.2f}%\n"
                if assumptions.get("wacc"):
                    formatted_text += f"- WACC: {assumptions['wacc']:.2f}%\n"
                if assumptions.get("terminal_growth_rate"):
                    formatted_text += f"- Terminal Growth Rate: {assumptions['terminal_growth_rate']:.2f}%\n"
            formatted_text += "\n"
        
        # Comparable Multiples Valuation (detailed)
        multiples_valuation = valuation_result.get("multiples_valuation", {})
        if multiples_valuation:
            formatted_text += "### 3.2 Comparable Multiples Valuation Results\n\n"
            multiples = multiples_valuation.get("multiples", {})
            if multiples:
                ev_ebitda = multiples.get('ev_ebitda') or multiples.get('ev_ebitda_ratio')
                if ev_ebitda:
                    formatted_text += f"- **EV/EBITDA**: {ev_ebitda:.2f}x\n"
                pe = multiples.get('pe') or multiples.get('pe_ratio')
                if pe:
                    formatted_text += f"- **P/E Ratio**: {pe:.2f}x\n"
                ev_sales = multiples.get('ev_sales') or multiples.get('ev_sales_ratio')
                if ev_sales:
                    formatted_text += f"- **EV/Sales**: {ev_sales:.2f}x\n"
            formatted_text += "\n"
        
        # Valuation vs Peers (detailed)
        peer_comparison = valuation_result.get("peer_comparison", {})
        if peer_comparison:
            formatted_text += "### 3.3 Valuation vs Peer Comparison\n\n"
            ev_ebitda_dict = peer_comparison.get('ev_ebitda', {})
            if isinstance(ev_ebitda_dict, dict):
                target_ev_ebitda = ev_ebitda_dict.get('target', 'N/A')
                peer_median = ev_ebitda_dict.get('peer_median', 'N/A')
                premium_discount = ev_ebitda_dict.get('premium_discount_pct', 'N/A')
                formatted_text += f"- **EV/EBITDA**: Target {target_ev_ebitda:.2f}x vs Peer Median {peer_median:.2f}x"
                if premium_discount and premium_discount != 'N/A':
                    formatted_text += f" ({premium_discount:+.1f}%)\n"
                else:
                    formatted_text += "\n"
            pe_dict = peer_comparison.get('pe', {}) or peer_comparison.get('pe_ratio', {})
            if isinstance(pe_dict, dict):
                target_pe = pe_dict.get('target', 'N/A')
                peer_median_pe = pe_dict.get('peer_median', 'N/A')
                premium_discount_pe = pe_dict.get('premium_discount_pct', 'N/A')
                formatted_text += f"- **P/E**: Target {target_pe:.2f}x vs Peer Median {peer_median_pe:.2f}x"
                if premium_discount_pe and premium_discount_pe != 'N/A':
                    formatted_text += f" ({premium_discount_pe:+.1f}%)\n"
                else:
                    formatted_text += "\n"
            formatted_text += "\n"
    
    # 4. Peer Comparison (comprehensive analysis, detailed data)
    formatted_text += "## 4. Peer Comparison Analysis (Data Source: Comprehensive Analysis Module, based on peer company financial data comparison)\n\n"
    
    peer_comp = comprehensive_analysis.get("peer_comparison", {})
    if peer_comp:
        peer_count = peer_comp.get("peer_count", 0)
        formatted_text += f"**Number of Peer Companies**: {peer_count}\n\n"
        
        profitability_comp = peer_comp.get("profitability_comparison", {})
        if profitability_comp:
            formatted_text += "### 4.1 Profitability vs Peers\n\n"
            for metric_name, metric_key in [("Gross Margin", "gross_margin"), ("Operating Margin", "operating_margin"), 
                                           ("Net Margin", "net_margin"), ("ROE", "roe"), ("ROIC", "roic")]:
                metric_data = profitability_comp.get(metric_key, {})
                if isinstance(metric_data, dict):
                    target_val = metric_data.get('target', 'N/A')
                    peer_median = metric_data.get('peer_median', 'N/A')
                    peer_mean = metric_data.get('peer_mean', 'N/A')
                    premium_discount = metric_data.get('premium_discount_pct', 'N/A')
                    percentile = metric_data.get('percentile_rank', 'N/A')
                    if target_val != 'N/A':
                        formatted_text += f"- **{metric_name}**: Target {target_val:.2f}%"
                        if peer_median != 'N/A':
                            formatted_text += f" vs Peer Median {peer_median:.2f}%"
                        if premium_discount != 'N/A':
                            formatted_text += f" ({premium_discount:+.1f}%)"
                        if percentile != 'N/A':
                            formatted_text += f", Percentile Rank: {percentile:.1f}%"
                        formatted_text += "\n"
            formatted_text += "\n"
        
        growth_comp = peer_comp.get("growth_comparison", {})
        if growth_comp:
            formatted_text += "### 4.2 Growth vs Peers\n\n"
            for metric_name, metric_key in [("Revenue 5Y CAGR", "revenue_cagr"), ("EBITDA 5Y CAGR", "ebitda_cagr"), 
                                           ("EPS 5Y CAGR", "eps_cagr"), ("Revenue Growth YoY", "revenue_growth_yoy")]:
                metric_data = growth_comp.get(metric_key, {})
                if isinstance(metric_data, dict):
                    target_val = metric_data.get('target', 'N/A')
                    peer_median = metric_data.get('peer_median', 'N/A')
                    premium_discount = metric_data.get('premium_discount_pct', 'N/A')
                    percentile = metric_data.get('percentile_rank', 'N/A')
                    if target_val != 'N/A':
                        formatted_text += f"- **{metric_name}**: Target {target_val:.2f}%"
                        if peer_median != 'N/A':
                            formatted_text += f" vs Peer Median {peer_median:.2f}%"
                        if premium_discount != 'N/A':
                            formatted_text += f" ({premium_discount:+.1f}%)"
                        if percentile != 'N/A':
                            formatted_text += f", Percentile Rank: {percentile:.1f}%"
                        formatted_text += "\n"
            formatted_text += "\n"
        
        valuation_comp = peer_comp.get("valuation_comparison", {})
        if valuation_comp:
            formatted_text += "### 4.3 Valuation vs Peers\n\n"
            for metric_name, metric_key in [("EV/EBITDA", "ev_ebitda"), ("P/E", "pe"), ("EV/Sales", "ev_sales")]:
                metric_data = valuation_comp.get(metric_key, {})
                if isinstance(metric_data, dict):
                    target_val = metric_data.get('target', 'N/A')
                    peer_median = metric_data.get('peer_median', 'N/A')
                    premium_discount = metric_data.get('premium_discount_pct', 'N/A')
                    if target_val != 'N/A':
                        formatted_text += f"- **{metric_name}**: Target {target_val:.2f}x"
                        if peer_median != 'N/A':
                            formatted_text += f" vs Peer Median {peer_median:.2f}x"
                        if premium_discount != 'N/A':
                            formatted_text += f" ({premium_discount:+.1f}%)"
                        formatted_text += "\n"
            formatted_text += "\n"
        
        structure_comp = peer_comp.get("financial_structure_comparison", {})
        if structure_comp:
            formatted_text += "### 4.4 Financial Structure vs Peers\n\n"
            for metric_name, metric_key in [("Debt to Equity", "debt_to_equity"), ("Debt to Assets", "debt_to_assets"), 
                                           ("Current Ratio", "current_ratio"), ("Interest Coverage", "interest_coverage")]:
                metric_data = structure_comp.get(metric_key, {})
                if isinstance(metric_data, dict):
                    target_val = metric_data.get('target', 'N/A')
                    peer_median = metric_data.get('peer_median', 'N/A')
                    premium_discount = metric_data.get('premium_discount_pct', 'N/A')
                    if target_val != 'N/A':
                        formatted_text += f"- **{metric_name}**: Target {target_val:.2f}"
                        if peer_median != 'N/A':
                            formatted_text += f" vs Peer Median {peer_median:.2f}"
                        if premium_discount != 'N/A':
                            formatted_text += f" ({premium_discount:+.1f}%)"
                        formatted_text += "\n"
            formatted_text += "\n"
        
        # Relative Position Summary
        summary = peer_comp.get("summary", {})
        if summary:
            formatted_text += "### 4.5 Relative Position Summary\n\n"
            strengths = summary.get("strengths", [])
            if strengths:
                formatted_text += "**Relative Strengths**:\n"
                for strength in strengths:
                    formatted_text += f"- ✅ {strength}\n"
                formatted_text += "\n"
            weaknesses = summary.get("weaknesses", [])
            if weaknesses:
                formatted_text += "**Relative Weaknesses**:\n"
                for weakness in weaknesses:
                    formatted_text += f"- ⚠️ {weakness}\n"
                formatted_text += "\n"
        
        # LLM Peer Comparison Deep Analysis Report (if available)
        llm_peer_report = peer_comp.get("llm_report", {})
        if llm_peer_report and not llm_peer_report.get("error"):
            formatted_text += "### 4.6 LLM Peer Comparison Deep Analysis (Data Source: LLM Analysis)\n\n"
            
            comp_assessment = llm_peer_report.get("comprehensive_assessment", {})
            if comp_assessment:
                overall_position = comp_assessment.get('overall_competitive_position', 'N/A')
                formatted_text += f"**Overall Competitive Position**: {overall_position}\n\n"
                
                competitive_summary = comp_assessment.get('competitive_summary', '')
                if competitive_summary:
                    formatted_text += f"{competitive_summary}\n\n"
                
                advantages = comp_assessment.get("core_competitive_advantages", [])
                if advantages:
                    formatted_text += "**Core Competitive Advantages**:\n"
                    for advantage in advantages:
                        formatted_text += f"- ✅ {advantage}\n"
                    formatted_text += "\n"
                
                weaknesses = comp_assessment.get("key_weaknesses", [])
                if weaknesses:
                    formatted_text += "**Key Weaknesses**:\n"
                    for weakness in weaknesses:
                        formatted_text += f"- ⚠️ {weakness}\n"
                    formatted_text += "\n"
            
            dimension_analysis = llm_peer_report.get("dimension_analysis", {})
            if dimension_analysis:
                formatted_text += "**Multi-Dimensional Deep Analysis**:\n\n"
                
                profitability_analysis = dimension_analysis.get('profitability_analysis', '')
                if profitability_analysis:
                    formatted_text += f"Profitability Analysis: {profitability_analysis[:500]}...\n\n"
                
                growth_analysis = dimension_analysis.get('growth_analysis', '')
                if growth_analysis:
                    formatted_text += f"Growth Analysis: {growth_analysis[:500]}...\n\n"
                
                valuation_analysis = dimension_analysis.get('valuation_analysis', '')
                if valuation_analysis:
                    formatted_text += f"Valuation Analysis: {valuation_analysis[:500]}...\n\n"
            
            investment_insights = llm_peer_report.get("investment_insights", {})
            if investment_insights:
                formatted_text += "**Investment Insights**:\n\n"
                
                key_insights = investment_insights.get("key_insights", [])
                if key_insights:
                    formatted_text += "Key Insights:\n"
                    for insight in key_insights:
                        formatted_text += f"- {insight}\n"
                    formatted_text += "\n"
                
                highlights = investment_insights.get("investment_highlights", [])
                if highlights:
                    formatted_text += "Investment Highlights:\n"
                    for highlight in highlights:
                        formatted_text += f"- ✅ {highlight}\n"
                    formatted_text += "\n"
                
                risk_points = investment_insights.get("risk_points", [])
                if risk_points:
                    formatted_text += "Risk Points:\n"
                    for risk in risk_points:
                        formatted_text += f"- ⚠️ {risk}\n"
                    formatted_text += "\n"
    
    # Management Quality (detailed)
    mgmt_quality = comprehensive_analysis.get("management_quality", {})
    if mgmt_quality:
        formatted_text += "## 5. Management Quality Assessment (Data Source: Comprehensive Analysis Module)\n\n"
        overall_score = mgmt_quality.get('overall_score', 0)
        overall_grade = mgmt_quality.get('overall_grade', 'N/A')
        formatted_text += f"**Overall Score**: {overall_score}/100 ({overall_grade})\n\n"
        
        roic_wacc = mgmt_quality.get('roic_vs_wacc', {})
        if isinstance(roic_wacc, dict):
            roic = roic_wacc.get('roic', 'N/A')
            wacc = roic_wacc.get('wacc', 'N/A')
            spread = roic_wacc.get('spread', 'N/A')
            roic_wacc_assessment = roic_wacc.get('assessment', 'N/A')
            formatted_text += f"- **ROIC vs WACC**: ROIC {roic:.2f}% vs WACC {wacc:.2f}%"
            if spread != 'N/A':
                formatted_text += f" (Spread: {spread:.2f}%)"
            formatted_text += f" - {roic_wacc_assessment}\n"
        
        earnings_stability = mgmt_quality.get('earnings_stability', {})
        if isinstance(earnings_stability, dict):
            cv = earnings_stability.get('coefficient_of_variation', 'N/A')
            growth_trend = earnings_stability.get('growth_trend', 'N/A')
            positive_ratio = earnings_stability.get('positive_years_ratio', 'N/A')
            earnings_assessment = earnings_stability.get('assessment', 'N/A')
            formatted_text += f"- **Earnings Stability**: Coefficient of Variation {cv:.1f}%" if cv != 'N/A' else "- **Earnings Stability**:"
            if growth_trend != 'N/A':
                formatted_text += f", Growth Trend: {growth_trend}"
            if positive_ratio != 'N/A':
                formatted_text += f", Positive Years Ratio: {positive_ratio:.1f}%"
            formatted_text += f" - {earnings_assessment}\n"
        
        fcf_allocation = mgmt_quality.get('free_cashflow_allocation', {})
        if isinstance(fcf_allocation, dict):
            fcf = fcf_allocation.get('free_cashflow', 0)
            dividend_ratio = fcf_allocation.get('dividend_payout_ratio', 'N/A')
            reinvestment_ratio = fcf_allocation.get('reinvestment_ratio', 'N/A')
            fcf_assessment = fcf_allocation.get('assessment', 'N/A')
            formatted_text += f"- **Free Cash Flow Allocation**: Free Cash Flow ${fcf/1e9:.2f}B"
            if dividend_ratio != 'N/A':
                formatted_text += f", Dividend Payout Ratio: {dividend_ratio:.1f}%"
            if reinvestment_ratio != 'N/A':
                formatted_text += f", Reinvestment Ratio: {reinvestment_ratio:.1f}%"
            formatted_text += f" - {fcf_assessment}\n"
        formatted_text += "\n"
    
    # 7. Catalysts (detailed data)
    formatted_text += "## 7. Catalyst Analysis (Data Source: Qualitative Analysis, Comprehensive Analysis)\n\n"
    
    # First display news data summary (so LLM knows what news was actually fetched)
    news_data = qualitative_analysis.get("news_data", {})
    if news_data:
        news_count = news_data.get("news_count", 0)
        news_items = news_data.get("news_items", [])
        formatted_text += f"### 7.0 News Data Summary (Data Source: Alpha Vantage NEWS_SENTIMENT API)\n\n"
        formatted_text += f"- **Number of News Items Retrieved**: {news_count}\n"
        if news_items and len(news_items) > 0:
            formatted_text += f"- **News Time Range**: From latest to historical\n"
            formatted_text += f"- **Key News Headlines** (Latest 10):\n"
            for i, item in enumerate(news_items[:10], 1):
                title = item.get('title', 'N/A')
                time_published = item.get('time_published', 'N/A')
                summary = item.get('summary', '') or item.get('content', '')
                sentiment_score = item.get('overall_sentiment_score', 'N/A')
                formatted_text += f"  {i}. {title}\n"
                if time_published and time_published != 'N/A':
                    formatted_text += f"     Time: {time_published}\n"
                if summary:
                    formatted_text += f"     Summary: {summary[:150]}...\n"
                if sentiment_score and sentiment_score != 'N/A':
                    formatted_text += f"     Sentiment Score: {sentiment_score}\n"
            formatted_text += "\n"
        else:
            formatted_text += "- ⚠️ No news content retrieved\n\n"
    
    # Catalysts from qualitative analysis (detailed)
    qual_catalysts = qualitative_analysis.get("catalyst_analysis", {})
    if qual_catalysts:
        formatted_text += "### 7.1 Catalysts Identified by Qualitative Analysis (Data Source: News Analysis + LLM Analysis)\n\n"
        
        # New products/markets
        product_catalysts = qual_catalysts.get("product_market_catalysts", [])
        if product_catalysts:
            formatted_text += "**New Products/New Market Opportunities**:\n"
            for i, cat in enumerate(product_catalysts[:5], 1):  # Maximum 5
                formatted_text += f"{i}. **{cat.get('type', 'N/A')}**: {cat.get('description', 'N/A')}\n"
                formatted_text += f"   - Impact Level: {cat.get('impact', 'N/A')}\n"
                formatted_text += f"   - Timeframe: {cat.get('timeframe', 'N/A')}\n"
                if cat.get('confidence'):
                    formatted_text += f"   - Confidence: {cat.get('confidence', 'N/A')}\n"
                if cat.get('expected_timing'):
                    formatted_text += f"   - Expected Timing: {cat.get('expected_timing', 'N/A')}\n"
                if cat.get('evidence'):
                    formatted_text += f"   - Evidence: {cat.get('evidence', 'N/A')[:150]}...\n"
            formatted_text += "\n"
        
        # Cost improvements
        cost_catalysts = qual_catalysts.get("cost_improvement_catalysts", [])
        if cost_catalysts:
            formatted_text += "**Cost Improvement Opportunities**:\n"
            for i, cat in enumerate(cost_catalysts[:5], 1):
                formatted_text += f"{i}. **{cat.get('type', 'N/A')}**: {cat.get('description', 'N/A')}\n"
                formatted_text += f"   - Impact Level: {cat.get('impact', 'N/A')}\n"
                formatted_text += f"   - Timeframe: {cat.get('timeframe', 'N/A')}\n"
            formatted_text += "\n"
        
        # Industry cycles
        cycle_catalysts = qual_catalysts.get("industry_cycle_catalysts", [])
        if cycle_catalysts:
            formatted_text += "**Industry Cycle Changes**:\n"
            for i, cat in enumerate(cycle_catalysts[:5], 1):
                formatted_text += f"{i}. **{cat.get('type', 'N/A')}**: {cat.get('description', 'N/A')}\n"
                formatted_text += f"   - Impact Level: {cat.get('impact', 'N/A')}\n"
                formatted_text += f"   - Timeframe: {cat.get('timeframe', 'N/A')}\n"
            formatted_text += "\n"
        
        # Policy/interest rates
        policy_catalysts = qual_catalysts.get("policy_rate_catalysts", [])
        if policy_catalysts:
            formatted_text += "**Policy/Interest Rate Changes**:\n"
            for i, cat in enumerate(policy_catalysts[:5], 1):
                formatted_text += f"{i}. **{cat.get('type', 'N/A')}**: {cat.get('description', 'N/A')}\n"
                formatted_text += f"   - Impact Level: {cat.get('impact', 'N/A')}\n"
                formatted_text += f"   - Timeframe: {cat.get('timeframe', 'N/A')}\n"
            formatted_text += "\n"
        
        # Catalyst Summary
        cat_summary = qual_catalysts.get("summary", {})
        if cat_summary:
            formatted_text += "**Catalyst Summary**:\n"
            formatted_text += f"- Total Catalysts: {cat_summary.get('total_catalysts', 0)}\n"
            formatted_text += f"- High Impact Catalysts: {cat_summary.get('high_impact_count', 0)}\n"
            formatted_text += f"- Short-term Catalysts: {cat_summary.get('short_term_count', 0)}\n"
            formatted_text += f"- Medium-to-Long-term Catalysts: {cat_summary.get('long_term_count', 0)}\n\n"
    
    # Catalysts from comprehensive analysis (detailed)
    comp_catalysts = comprehensive_analysis.get("catalysts", {})
    if comp_catalysts:
        formatted_text += "### 7.2 Catalysts Identified by Structured Analysis (Data Source: Financial Data Analysis)\n\n"
        
        # Cost structure improvements
        cost_improvement_cats = comp_catalysts.get('cost_improvement_catalysts', [])
        if cost_improvement_cats:
            formatted_text += "**Cost Structure Improvements**:\n"
            for cat in cost_improvement_cats[:3]:
                formatted_text += f"- {cat.get('type', 'N/A')}: {cat.get('description', 'N/A')}\n"
                formatted_text += f"  Impact: {cat.get('impact', 'N/A')}, Timeframe: {cat.get('timeframe', 'N/A')}\n"
        
        # Product/market catalysts
        product_market_cats = comp_catalysts.get('product_market_catalysts', [])
        if product_market_cats:
            formatted_text += "**Product/Market Catalysts**:\n"
            for cat in product_market_cats[:3]:
                formatted_text += f"- {cat.get('type', 'N/A')}: {cat.get('description', 'N/A')}\n"
                formatted_text += f"  Impact: {cat.get('impact', 'N/A')}, Timeframe: {cat.get('timeframe', 'N/A')}\n"
        
        formatted_text += "\n"
    
    # 6. Earnings Quality (detailed data)
    formatted_text += "## 6. Earnings Sustainability Analysis (Data Source: Earnings Quality Analysis Module, 5-Dimensional Comprehensive Assessment)\n\n"
    
    if earnings_quality:
        overall_score = earnings_quality.get('overall_score', 0)
        overall_assessment = earnings_quality.get('overall_assessment', 'N/A')
        formatted_text += f"**Overall Score**: {overall_score}/100 ({overall_assessment})\n\n"
        
        # Cash vs Profit (detailed)
        cash_profit = earnings_quality.get("cash_vs_profit", {})
        if cash_profit:
            formatted_text += "### 6.1 Cash vs Profit Analysis\n\n"
            cfo_to_ni_dict = cash_profit.get("cfo_to_net_income", {})
            if isinstance(cfo_to_ni_dict, dict):
                cfo_ni = cfo_to_ni_dict.get("latest")
                if cfo_ni is not None:
                    formatted_text += f"- **CFO/Net Income Ratio**: {cfo_ni:.2f}\n"
                    formatted_text += f"  - Latest Net Income: ${cfo_to_ni_dict.get('net_income', 0)/1e9:.2f}B\n"
                    formatted_text += f"  - Latest Operating Cash Flow: ${cfo_to_ni_dict.get('operating_cf', 0)/1e9:.2f}B\n"
                    if cfo_to_ni_dict.get("average"):
                        formatted_text += f"  - Historical Average: {cfo_to_ni_dict['average']:.2f}\n"
                    if cfo_to_ni_dict.get("median"):
                        formatted_text += f"  - Historical Median: {cfo_to_ni_dict['median']:.2f}\n"
                    trend = cash_profit.get("trend_analysis", {}).get("cfo_ni_trend")
                    if trend:
                        formatted_text += f"  - Trend: {trend}\n"
            cash_conv = cash_profit.get("cash_conversion_ratio", {})
            if cash_conv.get("latest"):
                formatted_text += f"- **Cash Conversion Ratio (CFO/Revenue)**: {cash_conv['latest']:.2f}\n"
            formatted_text += f"- **Assessment**: {cash_profit.get('assessment', 'N/A')}\n\n"
        
        # Accrual Quality (detailed)
        accrual = earnings_quality.get("accrual_quality", {})
        if accrual:
            formatted_text += "### 6.2 Accrual Quality Analysis\n\n"
            accruals_dict = accrual.get("accruals", {})
            if isinstance(accruals_dict, dict):
                accruals = accruals_dict.get("latest")
                if accruals is not None:
                    formatted_text += f"- **Accruals (NI - CFO)**: ${accruals/1e9:.2f}B\n"
                    if accruals_dict.get("average"):
                        formatted_text += f"  - Historical Average: ${accruals_dict['average']/1e9:.2f}B\n"
                    if accruals_dict.get("median"):
                        formatted_text += f"  - Historical Median: ${accruals_dict['median']/1e9:.2f}B\n"
            accrual_ratio = accrual.get("accrual_ratio", {})
            if accrual_ratio.get("latest") is not None:
                formatted_text += f"- **Accrual Ratio (Accruals/Revenue)**: {accrual_ratio['latest']:.4f}\n"
            trend = accrual.get("trend_analysis", {}).get("accrual_trend")
            if trend:
                formatted_text += f"- **Trend**: {trend}\n"
            formatted_text += f"- **Assessment**: {accrual.get('assessment', 'N/A')}\n\n"
        
        # Profit Volatility (detailed)
        volatility = earnings_quality.get("profit_volatility", {})
        if volatility:
            formatted_text += "### 6.3 Profit Volatility Analysis\n\n"
            margin_stability = volatility.get("margin_stability", {})
            if margin_stability.get("latest") is not None:
                formatted_text += f"- **Latest Net Margin**: {margin_stability['latest']:.2f}%\n"
                if margin_stability.get("average"):
                    formatted_text += f"  - Historical Average Net Margin: {margin_stability['average']:.2f}%\n"
                if margin_stability.get("min"):
                    formatted_text += f"  - Minimum Net Margin: {margin_stability['min']:.2f}%\n"
                if margin_stability.get("max"):
                    formatted_text += f"  - Maximum Net Margin: {margin_stability['max']:.2f}%\n"
            profit_vol_dict = volatility.get("profit_volatility", {})
            if isinstance(profit_vol_dict, dict):
                cv = profit_vol_dict.get("coefficient_of_variation")
                if cv is not None:
                    formatted_text += f"- **Margin Coefficient of Variation**: {cv:.1f}%\n"
                    formatted_text += f"  - Stability Rating: {profit_vol_dict.get('stability', 'N/A')}\n"
            trend = volatility.get("trend_analysis", {}).get("margin_trend")
            if trend:
                formatted_text += f"- **Trend**: {trend}\n"
            formatted_text += f"- **Assessment**: {volatility.get('assessment', 'N/A')}\n\n"
        
        # One-time Items (detailed)
        one_time = earnings_quality.get("one_time_items", {})
        if one_time:
            formatted_text += "### 6.4 One-time Items Dependency Analysis\n\n"
            special_items_dict = one_time.get("special_items_ratio", {})
            if isinstance(special_items_dict, dict):
                one_time_ratio = special_items_dict.get("ratio_to_net_income")
                if one_time_ratio is not None:
                    formatted_text += f"- **One-time Items Ratio**: {one_time_ratio:.1f}%\n"
                    formatted_text += f"  - Total One-time Items: ${special_items_dict.get('total_one_time_items', 0)/1e9:.2f}B\n"
                    formatted_text += f"  - Net Income: ${special_items_dict.get('net_income', 0)/1e9:.2f}B\n"
            detected_items = one_time.get("one_time_items_detected", [])
            if detected_items:
                formatted_text += f"- **Number of One-time Items Detected**: {len(detected_items)}\n"
                for item in detected_items[:3]:
                    formatted_text += f"  - {item.get('item', 'N/A')}: ${item.get('value', 0)/1e9:.2f}B\n"
            llm_analysis = one_time.get("llm_analysis", {})
            if llm_analysis.get("one_time_items_identified"):
                formatted_text += f"- **One-time Items Identified by LLM**: {len(llm_analysis['one_time_items_identified'])}\n"
            formatted_text += f"- **Assessment**: {one_time.get('assessment', 'N/A')}\n\n"
        
        # Capital Structure (detailed)
        capital = earnings_quality.get("capital_structure_support", {})
        if capital:
            formatted_text += "### 6.5 Capital Structure Support Analysis\n\n"
            leverage_trend = capital.get("leverage_trend", {})
            if leverage_trend.get("latest_debt_to_equity") is not None:
                formatted_text += f"- **Latest Debt to Equity**: {leverage_trend['latest_debt_to_equity']:.2f}\n"
                if leverage_trend.get("previous_debt_to_equity") is not None:
                    formatted_text += f"  - Previous Debt to Equity: {leverage_trend['previous_debt_to_equity']:.2f}\n"
                formatted_text += f"  - Trend: {leverage_trend.get('trend', 'N/A')}\n"
            debt_support = capital.get("debt_support_analysis", {})
            if debt_support.get("leverage_concern") is not None:
                formatted_text += f"- **Leverage Concern**: {'Yes' if debt_support['leverage_concern'] else 'No'}\n"
                if debt_support.get("concern_level"):
                    formatted_text += f"  - Concern Level: {debt_support['concern_level']}\n"
            interest_burden = capital.get("interest_burden", {})
            if interest_burden.get("interest_coverage"):
                formatted_text += f"- **Interest Coverage**: {interest_burden['interest_coverage']:.2f}x\n"
                formatted_text += f"  - Interest Burden Level: {interest_burden.get('burden_level', 'N/A')}\n"
            formatted_text += f"- **Assessment**: {capital.get('assessment', 'N/A')}\n\n"
        
        # Summary
        summary = earnings_quality.get("summary", {})
        if summary:
            formatted_text += "### 6.6 Earnings Quality Comprehensive Summary\n\n"
            strengths = summary.get("strengths", [])
            if strengths:
                formatted_text += "**Strengths**:\n"
                for strength in strengths:
                    formatted_text += f"- ✅ {strength}\n"
                formatted_text += "\n"
            concerns = summary.get("concerns", [])
            if concerns:
                formatted_text += "**Concerns**:\n"
                for concern in concerns:
                    formatted_text += f"- ⚠️ {concern}\n"
                formatted_text += "\n"
            formatted_text += f"**Sustainability Conclusion**: {summary.get('sustainability_conclusion', 'N/A')}\n\n"
    
    # 8. Risk Factors (extracted from various analyses, detailed data)
    formatted_text += "## 8. Risk Factor Analysis (Data Source: Financial Analysis, Earnings Quality Analysis, Valuation Analysis, Peer Comparison)\n\n"
    
    # Extract risks from financial analysis (detailed)
    financial_summary = financial_analysis.get("summary", {})
    if financial_summary:
        concerns = financial_summary.get("concerns", [])
        if concerns:
            formatted_text += "### 8.1 Financial Risks (Data Source: Financial Statement Analysis)\n\n"
            for i, concern in enumerate(concerns, 1):
                formatted_text += f"{i}. {concern}\n"
            formatted_text += "\n"
    
    # Extract risks from earnings quality (detailed)
    if earnings_quality:
        eq_summary = earnings_quality.get("summary", {})
        if eq_summary:
            eq_concerns = eq_summary.get("concerns", [])
            if eq_concerns:
                formatted_text += "### 8.2 Earnings Quality Risks (Data Source: Earnings Quality Analysis)\n\n"
                for i, concern in enumerate(eq_concerns, 1):
                    formatted_text += f"{i}. {concern}\n"
                formatted_text += "\n"
    
    # Extract risks from valuation (detailed)
    if valuation_result:
        dcf = valuation_result.get("dcf_valuation", {})
        if dcf:
            risks = dcf.get("risks", [])
            if risks:
                formatted_text += "### 8.3 Valuation Risks (Data Source: DCF Valuation Model)\n\n"
                for i, risk in enumerate(risks, 1):
                    formatted_text += f"{i}. {risk}\n"
                formatted_text += "\n"
    
    # Extract risks from peer comparison
    if peer_comp:
        peer_summary = peer_comp.get("summary", {})
        if peer_summary:
            weaknesses = peer_summary.get("weaknesses", [])
            if weaknesses:
                formatted_text += "### 8.4 Relative Competitive Risks (Data Source: Peer Comparison Analysis)\n\n"
                for i, weakness in enumerate(weaknesses, 1):
                    formatted_text += f"{i}. {weakness}\n"
                formatted_text += "\n"
    
    formatted_text += """
================================================================================
Data Source Notes:
- Company Basic Information: Alpha Vantage Overview API
- Financial Performance & Trends: Financial Statement Analysis (based on 5-year historical data)
- Valuation Summary: DCF Valuation Model, Multiples Valuation Model
- Peer Comparison: Comprehensive Analysis Module (based on peer company financial data comparison)
- Management Quality: Comprehensive Analysis Module
- Catalysts: Qualitative Analysis (News + LLM), Comprehensive Analysis (Financial Data Analysis)
- Earnings Sustainability: Earnings Quality Analysis Module (5-dimensional comprehensive assessment)
- Risk Factors: Financial Analysis, Earnings Quality Analysis, Valuation Analysis, Peer Comparison
================================================================================
"""
    
    return formatted_text

# ==================== Investment Memo Generation ====================

def generate_investment_memo(
    symbol: str,
    financial_data: Dict[str, Any],
    financial_analysis: Dict[str, Any],
    valuation_result: Dict[str, Any],
    comprehensive_analysis: Dict[str, Any],
    qualitative_analysis: Dict[str, Any],
    earnings_quality: Dict[str, Any],
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Generate complete investment memo
    
    Integrate all analysis results, use LLM to generate professional investment memo
    
    Args:
        symbol: Stock symbol
        financial_data: Financial data
        financial_analysis: Financial analysis results
        valuation_result: Valuation results
        comprehensive_analysis: Comprehensive analysis results
        qualitative_analysis: Qualitative analysis results
        earnings_quality: Earnings quality analysis results
        api_key: OpenAI API Key
        base_url: OpenAI Base URL
        model: AI model name
    
    Returns:
        Investment memo dictionary
    """
    result = {
        "symbol": symbol,
        "generated_at": datetime.now().isoformat(),
        "investment_thesis": "",
        "business_overview": "",
        "financial_performance_trends": "",
        "valuation_summary": "",
        "peer_comparison": "",
        "catalysts": "",
        "risks": "",
        "recommendation": "",
        "recommendation_rationale": "",
        "full_memo": "",
        "raw_llm_response": ""
    }
    
    if not OPENAI_AVAILABLE:
        result["error"] = "LLM not available, unable to generate investment memo"
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
    
    # 验证 API Key 是否提供
    if not api_key:
        result["error"] = "OpenAI API Key is required. Please set OPENAI_API_KEY environment variable or provide it as a parameter."
        return result
    
    try:
        print(f"\n📝 Starting to generate investment memo for {symbol}...")
        
        # Format all analysis data
        analysis_data = format_analysis_data_for_llm(
            symbol=symbol,
            financial_data=financial_data,
            financial_analysis=financial_analysis,
            valuation_result=valuation_result,
            comprehensive_analysis=comprehensive_analysis,
            qualitative_analysis=qualitative_analysis,
            earnings_quality=earnings_quality
        )
        
        # Build Prompt
        prompt = build_investment_memo_prompt(symbol, analysis_data)
        
        # Call LLM
        chat_client = OpenAIChat(api_key=api_key, base_url=base_url)
        messages = [
        {
            "role": "system",
            "content": """You are a senior buy-side research analyst with over 15 years of fundamental analysis experience, having held senior research positions at top investment institutions (such as BlackRock, Vanguard, Fidelity).

Your task is to generate a professional, in-depth, and comprehensive investment memo (1-2 pages, approximately 1500-2500 words) based on the provided complete analysis data.

**Core Requirements**:

1. **Must fully utilize all provided data**:
   - Each section must cite specific data and metrics
   - Cannot omit any important analysis dimensions
   - All conclusions must be supported by data
   - Must cite data sources (e.g., "based on financial statement analysis", "based on DCF valuation model", etc.)

2. **Deep analysis requirements**:
   - Not just present data, but deeply analyze the meaning behind the data
   - Identify correlations and logical relationships between data
   - Analyze causes and impacts of trend changes
   - Assess data reliability and sustainability

3. **Professional standards**:
   - Use professional financial terminology and analytical frameworks
   - Clear logic, well-reasoned arguments
   - Balance risks and opportunities, unbiased
   - Provide actionable investment recommendations

4. **Structural completeness**:
   - All 8 sections must be included
   - Each section must have substantive content
   - Logical coherence between sections

5. **Data citation standards**:
   - When citing data, clearly state the data source
   - Example: "Based on 5-year financial statement analysis, the company's gross margin is XX% (Data Source: Financial Statement Analysis)"
   - Example: "Based on DCF valuation model, target price is $XX (Data Source: DCF Valuation Model)"

Your investment memo should meet the professional standards of buy-side research departments and be directly usable for investment decisions."""
        },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        print("  🤖 Calling LLM to generate investment memo...")
        llm_response = chat_client.chat(messages, model=model)
        
        if llm_response:
            # Parse response
            parsed = parse_investment_memo_response(llm_response)
            result.update(parsed)
            result["raw_llm_response"] = llm_response
            
            # Generate full memo
            result["full_memo"] = format_full_memo(result)
            
            print(f"  ✅ Investment memo generation completed")
        else:
            result["error"] = "LLM response is empty"
    
    except Exception as e:
        print(f"⚠️ Failed to generate investment memo: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
    
    return result

def build_investment_memo_prompt(symbol: str, analysis_data: str) -> str:
    """
    Build investment memo generation prompt (high-quality version)
    
    Args:
        symbol: Stock symbol
        analysis_data: Formatted analysis data
    
    Returns:
        Prompt text
    """
    prompt = f"""Please generate a professional investment memo (1-2 pages, approximately 1500-2500 words) for {symbol} based on the following complete analysis data.

================================================================================
Important Notes:
1. You must fully utilize all analysis data below, and not omit any important information
2. Each section must cite specific data and metrics, and cite data sources
3. Conduct deep analysis - not just present data, but analyze the meaning behind the data
4. All conclusions must be supported by data, avoid subjective speculation
5. Investment recommendations must be logically consistent with valuation, fundamentals, and risk analysis
================================================================================

## Complete Analysis Data

{analysis_data}

## Memo Structure Requirements

Please generate a well-structured, in-depth investment memo containing the following 8 sections (each section must have substantive content):

### 1. Investment Thesis
**Requirements**: Must extract 3-5 core investment theses based on all provided analysis data
- Each thesis should be concise, powerful, and data-based
- Must cite specific financial metrics, valuation data, peer comparison results, etc.
- Highlight the company's core competitive advantages and investment value
- Each thesis must cite data sources
- **Example format**: "Thesis 1: Company ROIC reaches XX%, significantly higher than WACC (Data Source: Financial Analysis, Valuation Analysis), indicating excellent capital allocation efficiency..."

### 2. Business Overview
**Requirements**: Based on company basic information data, deeply analyze business model
- Company's main business and business model (cite company description data)
- Industry position and competitive advantages (cite industry, sector data)
- Main products/services
- Market positioning and market cap size (cite market cap, shares outstanding, etc.)
- **Must cite**: All relevant data from the company basic information section

### 3. Financial Performance & Trends
**Requirements**: Deeply analyze financial data, identify trends and changes
- **Profitability Analysis**: Must cite specific values for gross margin, net margin, ROE, ROIC, analyze trend changes (Data Source: Financial Statement Analysis)
- **Growth Analysis**: Must cite specific data for revenue CAGR, EBITDA CAGR, EPS CAGR, YoY growth, etc., analyze growth quality and sustainability (Data Source: Financial Statement Analysis)
- **Financial Health**: Must cite specific data for debt-to-equity ratio, net debt/EBITDA, interest coverage, etc. (Data Source: Financial Statement Analysis)
- **Operational Efficiency**: Must cite specific data for asset turnover, inventory turnover, accounts receivable turnover, etc. (Data Source: Financial Statement Analysis)
- **Key Trends and Changes**: Analyze historical trends of each metric, identify areas of improvement or deterioration

### 4. Valuation Summary
**Requirements**: Comprehensively analyze valuation results, assess valuation reasonableness
- **DCF Valuation Results**: Must cite specific data for target price, current price, upside potential, enterprise value, etc. (Data Source: DCF Valuation Model)
- **Comparable Multiples Valuation**: Must cite specific multiples such as EV/EBITDA, P/E, EV/Sales (Data Source: Multiples Valuation Model)
- **Valuation vs Peers**: Must cite comparison data with peer median (Data Source: Peer Comparison Analysis)
- **Valuation Reasonableness Assessment**: Based on DCF and multiples valuation, comprehensively assess the reasonableness of current valuation
- **Must cite**: All specific values from the valuation summary section

### 5. Peer Comparison
**Requirements**: Deeply analyze relative position, identify competitive advantages and disadvantages
- **Profitability vs Peers**: Must cite percentile rankings and premium/discount data for gross margin, net margin, ROIC, etc. (Data Source: Peer Comparison Analysis)
- **Growth vs Peers**: Must cite percentile rankings for revenue CAGR, EBITDA CAGR, etc. (Data Source: Peer Comparison Analysis)
- **Valuation vs Peers**: Must cite relative positions for EV/EBITDA, P/E, etc. (Data Source: Peer Comparison Analysis)
- **Financial Structure vs Peers**: Must cite relative positions for debt-to-equity ratio, current ratio, etc. (Data Source: Peer Comparison Analysis)
- **Relative Strengths and Weaknesses**: Based on percentile rankings, clearly identify relative strengths and weaknesses

### 6. Catalysts
**Requirements**: Comprehensively analyze all identified catalysts
- **New Products/Market Opportunities**: Must cite specific catalysts identified by qualitative analysis and comprehensive analysis, including impact level, timeframe, confidence (Data Source: Qualitative Analysis, Comprehensive Analysis)
- **Cost Structure Improvements**: Must cite specific data such as gross margin improvement, operating expense ratio decline (Data Source: Financial Analysis, Comprehensive Analysis)
- **Industry Cycle Changes**: Analyze industry cycle position and potential changes (Data Source: Qualitative Analysis)
- **Policy/Interest Rate Impact**: Analyze impact of policy or interest rate changes (Data Source: Qualitative Analysis)
- **Must cite**: All specific catalysts and their detailed information identified in the catalysts section

### 7. Risks
**Requirements**: Comprehensively identify and assess all risk factors
- **Financial Risks**: Must cite concerns from financial analysis summary, including leverage, solvency, earnings volatility, etc. (Data Source: Financial Analysis)
- **Earnings Quality Risks**: Must cite concerns from earnings quality analysis, including cash conversion, accrual quality, one-time items, capital structure, etc. (Data Source: Earnings Quality Analysis)
- **Valuation Risks**: Analyze reasonableness of DCF assumptions, assess valuation risks (Data Source: Valuation Analysis)
- **Industry Risks**: Based on industry and sector information, analyze industry risks
- **Competitive Risks**: Based on peer comparison, analyze competitive risks
- **Must cite**: All specific risks identified in the risk factors section

### 8. Recommendation
**Requirements**: Based on all analysis, provide clear investment recommendation
- **Clear Investment Recommendation**: Buy / Hold / Sell (must be logically consistent with previous analysis)
- **Detailed Investment Rationale**: Must comprehensively cite valuation results, fundamental analysis, catalysts, risk analysis, provide sufficient rationale (2-3 paragraphs)
- **Target Price**: Must cite target price from DCF valuation (Data Source: DCF Valuation Model)
- **Investment Timeframe**: Based on catalyst timeframes, determine investment timeframe
- **Key Monitoring Metrics**: List key metrics that need continuous monitoring (based on risk analysis)

## Output Format

Please return in JSON format, containing the following fields:

```json
{{
    "investment_thesis": [
        "Thesis 1: ...",
        "Thesis 2: ...",
        "Thesis 3: ...",
        "Thesis 4: ...",
        "Thesis 5: ..."
    ],
    "business_overview": "Detailed business overview text (2-3 paragraphs)",
    "financial_performance_trends": "Detailed financial performance and trends analysis (3-4 paragraphs)",
    "valuation_summary": "Detailed valuation summary (2-3 paragraphs)",
    "peer_comparison": "Detailed peer comparison analysis (2-3 paragraphs)",
    "catalysts": "Detailed catalysts analysis (3-4 paragraphs)",
    "risks": "Detailed risk analysis (3-4 paragraphs)",
    "recommendation": "Buy/Hold/Sell",
    "recommendation_rationale": "Detailed investment rationale (2-3 paragraphs)",
    "target_price": 0.0,
    "investment_timeframe": "Short-term/Long-term",
    "key_monitoring_metrics": [
        "Monitoring metric 1",
        "Monitoring metric 2",
        "Monitoring metric 3"
    ]
}}
```

## Key Requirements (Must Strictly Follow)

1. **Fully utilize all data**:
   - Must cite specific data and metrics for each section
   - Cannot omit any important analysis dimensions
   - All conclusions must be supported by data
   - Must cite data sources (format: Data Source: XXX Analysis/Model)

2. **Deep analysis**:
   - Not just present data, but analyze the meaning behind the data
   - Identify correlations and logical relationships between data
   - Analyze causes and impacts of trend changes
   - Assess data reliability and sustainability

3. **Logical consistency**:
   - Investment recommendations must be logically consistent with valuation, fundamentals, and risk analysis
   - If valuation shows large upside but high risk, need balanced analysis
   - If earnings quality is excellent but valuation is high, need comprehensive assessment

4. **Balanced perspective**:
   - See opportunities while identifying risks
   - Highlight strengths while acknowledging weaknesses
   - Avoid excessive optimism or pessimism

5. **Professional standards**:
   - Use professional financial analysis terminology
   - Clear logic, well-reasoned arguments
   - Meet buy-side research department standards

6. **Actionability**:
   - Investment recommendations should be clear and actionable
   - Provide specific monitoring metrics
   - Clearly define investment timeframe

7. **Data citation standards**:
   - When citing data, always cite the data source
   - Example: "Based on financial statement analysis, company gross margin is XX% (Data Source: Financial Statement Analysis)"
   - Example: "Based on DCF valuation model, target price is $XX (Data Source: DCF Valuation Model)"

8. **Memo length**:
   - Total length approximately 1500-2500 words (1-2 pages)
   - Each section must have substantive content
   - Avoid empty statements, must have specific data support

## Output Format

Please return in JSON format, containing the following fields (each field must have substantive content):

```json
{{
    "investment_thesis": [
        "Thesis 1: ... (must cite specific data, cite data source)",
        "Thesis 2: ... (must cite specific data, cite data source)",
        "Thesis 3: ... (must cite specific data, cite data source)",
        "Thesis 4: ... (must cite specific data, cite data source)",
        "Thesis 5: ... (must cite specific data, cite data source)"
    ],
    "business_overview": "Detailed business overview text (2-3 paragraphs, must cite company basic information data, cite data source)",
    "financial_performance_trends": "Detailed financial performance and trends analysis (3-4 paragraphs, must cite all financial metrics' specific values, cite data source)",
    "valuation_summary": "Detailed valuation summary (2-3 paragraphs, must cite DCF and multiples valuation specific values, cite data source)",
    "peer_comparison": "Detailed peer comparison analysis (2-3 paragraphs, must cite percentile rankings and relative position data, cite data source)",
    "catalysts": "Detailed catalysts analysis (3-4 paragraphs, must cite all identified catalysts, cite data source)",
    "risks": "Detailed risk analysis (3-4 paragraphs, must cite all identified risk factors, cite data source)",
    "recommendation": "Buy/Hold/Sell",
    "recommendation_rationale": "Detailed investment rationale (2-3 paragraphs, must comprehensively cite valuation, fundamentals, catalysts, risk analysis, cite data source)",
    "target_price": 0.0,
    "investment_timeframe": "Short-term/Long-term",
    "key_monitoring_metrics": [
        "Monitoring metric 1 (based on risk analysis)",
        "Monitoring metric 2 (based on risk analysis)",
        "Monitoring metric 3 (based on risk analysis)"
    ]
}}
```

**Re-emphasis**:
- Each section must cite specific data and metrics
- Every data citation must cite the data source
- Conduct deep analysis, not just list data
- Investment recommendations must be logically consistent with all analysis

Please begin generating the investment memo."""
    
    return prompt

def parse_investment_memo_response(llm_response: str) -> Dict[str, Any]:
    """
    Parse LLM investment memo response
    
    Args:
        llm_response: LLM response text
    
    Returns:
        Parsed investment memo dictionary
    """
    result = {
        "investment_thesis": [],
        "business_overview": "",
        "financial_performance_trends": "",
        "valuation_summary": "",
        "peer_comparison": "",
        "catalysts": "",
        "risks": "",
        "recommendation": "",
        "recommendation_rationale": "",
        "target_price": 0.0,
        "investment_timeframe": "",
        "key_monitoring_metrics": []
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
            result.update(parsed)
        else:
            # If unable to parse JSON, try to extract text
            print("⚠️ Unable to parse JSON, trying to extract text content")
            # Can add text extraction logic
    
    except Exception as e:
        print(f"⚠️ Failed to parse investment memo response: {e}")
        result["parse_error"] = str(e)
    
    return result

def format_full_memo(memo_data: Dict[str, Any]) -> str:
    """
    Format complete investment memo text
    
    Args:
        memo_data: Investment memo data
    
    Returns:
        Formatted complete memo text
    """
    symbol = memo_data.get("symbol", "UNKNOWN")
    generated_at = memo_data.get("generated_at", "")
    
    memo = f"""
{'='*80}
Investment Memo
Stock Symbol: {symbol}
Generated At: {generated_at}
{'='*80}

# 1. Investment Thesis

"""
    
    thesis = memo_data.get("investment_thesis", [])
    if isinstance(thesis, list):
        for i, point in enumerate(thesis, 1):
            memo += f"{i}. {point}\n"
    else:
        memo += f"{thesis}\n"
    
    memo += f"""
# 2. Business Overview

{memo_data.get('business_overview', 'N/A')}

# 3. Financial Performance & Trends

{memo_data.get('financial_performance_trends', 'N/A')}

# 4. Valuation Summary

{memo_data.get('valuation_summary', 'N/A')}

# 5. Peer Comparison

{memo_data.get('peer_comparison', 'N/A')}

# 6. Catalysts

{memo_data.get('catalysts', 'N/A')}

# 7. Risks

{memo_data.get('risks', 'N/A')}

# 8. Recommendation

**Investment Recommendation: {memo_data.get('recommendation', 'N/A')}**

**Investment Rationale:**
{memo_data.get('recommendation_rationale', 'N/A')}

**Target Price:** ${memo_data.get('target_price', 0):.2f}

**Investment Timeframe:** {memo_data.get('investment_timeframe', 'N/A')}

**Key Monitoring Metrics:**
"""
    
    metrics = memo_data.get("key_monitoring_metrics", [])
    if isinstance(metrics, list):
        for i, metric in enumerate(metrics, 1):
            memo += f"{i}. {metric}\n"
    else:
        memo += f"{metrics}\n"
    
    memo += f"""
{'='*80}
End of Memo
{'='*80}
"""
    
    return memo

# ==================== Complete Analysis Workflow ====================

def comprehensive_fundamental_analysis(
    symbol: str,
    peer_symbols: List[str] = None,
    years: int = 5,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    news_limit: int = 50
) -> Dict[str, Any]:
    """
    Complete fundamental analysis workflow
    
    Integrate all analysis modules to generate complete investment memo
    
    Args:
        symbol: Stock symbol
        peer_symbols: List of peer company symbols (optional)
        years: Number of years to analyze (default 5 years)
        api_key: OpenAI API Key
        base_url: OpenAI Base URL
        model: AI model name
        news_limit: News quantity limit
    
    Returns:
        Complete analysis results, including investment memo
    """
    result = {
        "symbol": symbol,
        "analysis_timestamp": datetime.now().isoformat(),
        "financial_data": {},
        "financial_analysis": {},
        "valuation_result": {},
        "comprehensive_analysis": {},
        "qualitative_analysis": {},
        "earnings_quality": {},
        "investment_memo": {}
    }
    
    print("=" * 80)
    print(f"Starting complete fundamental analysis for {symbol}")
    print("=" * 80)
    
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
    
    # 验证 API Key 是否提供
    if not api_key:
        result["error"] = "OpenAI API Key is required. Please set OPENAI_API_KEY environment variable or provide it as a parameter."
        return result
    
    try:
        # Import all required modules
        try:
            from data_ingestion import fetch_financial_statements
            from financial_analysis import analyze_financial_statements
            from valuation import comprehensive_valuation
            from comprehensive_analysis import comprehensive_company_analysis
            from qualitative_analysis import comprehensive_qualitative_analysis, fetch_stock_news
            from earnings_quality import comprehensive_earnings_quality_analysis
        except ImportError:
            from .data_ingestion import fetch_financial_statements
            from .financial_analysis import analyze_financial_statements
            from .valuation import comprehensive_valuation
            from .comprehensive_analysis import comprehensive_company_analysis
            from .qualitative_analysis import comprehensive_qualitative_analysis, fetch_stock_news
            from .earnings_quality import comprehensive_earnings_quality_analysis
        
        # 1. Data Collection
        print("\n[1/7] 📊 Data Collection...")
        financial_data = fetch_financial_statements(symbol, years=years)
        result["financial_data"] = financial_data
        
        # 2. Financial Analysis
        print("\n[2/7] 📈 Financial Analysis...")
        financial_analysis = analyze_financial_statements(financial_data, years=years)
        result["financial_analysis"] = financial_analysis
        
        # 3. Valuation Analysis
        print("\n[3/7] 💰 Valuation Analysis...")
        # Get peer data for valuation
        peer_data = None
        if peer_symbols:
            try:
                from data_ingestion import fetch_multiple_symbols
            except ImportError:
                from .data_ingestion import fetch_multiple_symbols
            peer_data_dict = fetch_multiple_symbols(peer_symbols, years=years)
            # `fetch_multiple_symbols` 返回的是 {symbol: data} 的字典，这里需要转换成列表
            # 只保留成功拉取的数据（value 不为 None）
            peer_data = [data for data in peer_data_dict.values() if data]
        valuation_result = comprehensive_valuation(financial_data, peer_data=peer_data)
        result["valuation_result"] = valuation_result
        
        # 4. Comprehensive Analysis (peer comparison, management quality, catalysts)
        print("\n[4/7] 🔍 Comprehensive Analysis...")
        comprehensive_analysis = comprehensive_company_analysis(
            target_financial_data=financial_data,
            peer_symbols=peer_symbols,
            valuation_data=valuation_result,
            auto_fetch_peers=True,
            api_key=api_key,
            base_url=base_url,
            model=model,
            generate_llm_peer_report=True  # Generate LLM peer comparison deep analysis report
        )
        result["comprehensive_analysis"] = comprehensive_analysis
        
        # 5. Qualitative Analysis (News + LLM catalyst analysis)
        print("\n[5/7] 📰 Qualitative Analysis...")
        qualitative_analysis = comprehensive_qualitative_analysis(
            symbol=symbol,
            financial_analysis=financial_analysis,
            api_key=api_key,
            base_url=base_url,
            model=model,
            news_limit=news_limit
        )
        result["qualitative_analysis"] = qualitative_analysis
        
        # 6. Earnings Quality Analysis
        print("\n[6/7] 💎 Earnings Quality Analysis...")
        news_data = qualitative_analysis.get("news_data", {})
        earnings_quality = comprehensive_earnings_quality_analysis(
            financial_data=financial_data,
            news_data=news_data,
            financial_analysis=financial_analysis,
            api_key=api_key,
            base_url=base_url,
            model=model,
            generate_llm_report=True  # Generate LLM earnings analysis report
        )
        result["earnings_quality"] = earnings_quality
        
        # 7. Generate Investment Memo
        print("\n[7/7] 📝 Generating Investment Memo...")
        investment_memo = generate_investment_memo(
            symbol=symbol,
            financial_data=financial_data,
            financial_analysis=financial_analysis,
            valuation_result=valuation_result,
            comprehensive_analysis=comprehensive_analysis,
            qualitative_analysis=qualitative_analysis,
            earnings_quality=earnings_quality,
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        result["investment_memo"] = investment_memo
        
        print("\n" + "=" * 80)
        print(f"✅ {symbol} complete fundamental analysis finished!")
        print("=" * 80)
        
        # Display investment recommendation summary
        if investment_memo.get("recommendation"):
            print(f"\n📊 Investment Recommendation: {investment_memo.get('recommendation')}")
            print(f"📈 Target Price: ${investment_memo.get('target_price', 0):.2f}")
            print(f"⏰ Investment Timeframe: {investment_memo.get('investment_timeframe', 'N/A')}")
        
    except Exception as e:
        print(f"\n⚠️ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
    
    return result

# ==================== Main Function (for testing) ====================

if __name__ == "__main__":
    # Test example
    print("=" * 60)
    print("Investment Memo Generator - Test")
    print("=" * 60)
    
    print("\n⚠️ This is an integration module that requires all analysis modules to run first")
    print("Please refer to README.md for complete usage examples")
    
    # Example code
    print("\nUsage Example:")
    print("""
from .investment_memo import comprehensive_fundamental_analysis

# Complete analysis workflow
result = comprehensive_fundamental_analysis(
    symbol="NVDA",
    peer_symbols=["AMD", "INTC", "TSM"],
    years=5,
    api_key="your-openai-api-key",
    model="gpt-4o-mini"
)

# View investment memo
memo = result["investment_memo"]
print(memo["full_memo"])
print("\\nInvestment Recommendation:", memo["recommendation"])
print("Target Price:", memo["target_price"])
""")

