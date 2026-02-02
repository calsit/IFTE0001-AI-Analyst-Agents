"""
Comprehensive Analysis Module for Fundamental Analysis

Features:
1. Peer comparison (profitability, growth, valuation, financial structure)
2. Management quality assessment (ROIC vs WACC, earnings stability, free cash flow allocation)
3. Catalyst analysis (new products, cost improvements, industry cycles, policy changes)
4. LLM peer comparison deep analysis report

Management quality is inferred indirectly through capital allocation discipline and return consistency.
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
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
        print("OpenAI module not found, LLM peer comparison analysis will be unavailable")
        OpenAIChat = None
        get_config_from_env = None

# Import other modules
try:
    from financial_analysis import (
        analyze_financial_statements,
        calculate_profitability_ratios,
        calculate_growth_ratios,
        calculate_leverage_ratios
    )
except ImportError:
    from .financial_analysis import (
        analyze_financial_statements,
        calculate_profitability_ratios,
        calculate_growth_ratios,
        calculate_leverage_ratios
    )

# Lazy import to avoid circular dependencies
# from data_ingestion import fetch_financial_statements
try:
    from valuation import (
        calculate_multiples_valuation,
        calculate_wacc,
        estimate_cost_of_equity
    )
except ImportError:
    from .valuation import (
        calculate_multiples_valuation,
        calculate_wacc,
        estimate_cost_of_equity
    )

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

# ==================== Peer Comparison ====================

def compare_peer_financials(
    target_financial_data: Dict[str, Any],
    target_analysis: Dict[str, Any],
    peer_financial_data_list: List[Dict[str, Any]],
    peer_analysis_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Peer financial comparison
    
    Comparison dimensions:
    - Profitability vs peers
    - Growth vs peers
    - Valuation vs peers
    - Financial structure vs peers
    
    Args:
        target_financial_data: Target company financial data
        target_analysis: Target company financial analysis results
        peer_financial_data_list: List of peer company financial data
        peer_analysis_list: List of peer company financial analysis results
    
    Returns:
        Peer comparison results dictionary
    """
    comparison = {
        "target_symbol": target_financial_data.get("symbol", "UNKNOWN"),
        "peer_count": len(peer_financial_data_list),
        "profitability_comparison": {},
        "growth_comparison": {},
        "valuation_comparison": {},
        "financial_structure_comparison": {},
        "summary": {}
    }
    
    try:
        target_symbol = comparison["target_symbol"]
        
        # 1. Profitability comparison
        target_profitability = target_analysis.get("profitability", {})
        peer_profitability_list = [peer.get("profitability", {}) for peer in peer_analysis_list]
        
        profitability_metrics = ["gross_margin", "operating_margin", "net_margin", "roe", "roic"]
        profitability_comparison = {}
        
        for metric in profitability_metrics:
            target_value = target_profitability.get(metric)
            if target_value is None:
                continue
            
            peer_values = [peer.get(metric) for peer in peer_profitability_list if peer.get(metric) is not None]
            if not peer_values:
                continue
            
            peer_median = np.median(peer_values)
            peer_mean = np.mean(peer_values)
            peer_min = np.min(peer_values)
            peer_max = np.max(peer_values)
            
            premium_discount = ((target_value - peer_median) / peer_median * 100) if peer_median != 0 else None
            percentile_rank = (sum(1 for v in peer_values if v < target_value) / len(peer_values) * 100) if peer_values else None
            
            profitability_comparison[metric] = {
                "target": target_value,
                "peer_median": peer_median,
                "peer_mean": peer_mean,
                "peer_min": peer_min,
                "peer_max": peer_max,
                "premium_discount_pct": premium_discount,
                "percentile_rank": percentile_rank
            }
        
        comparison["profitability_comparison"] = profitability_comparison
        
        # 2. Growth comparison
        target_growth = target_analysis.get("growth", {})
        peer_growth_list = [peer.get("growth", {}) for peer in peer_analysis_list]
        
        growth_metrics = ["revenue_cagr", "ebitda_cagr", "eps_cagr", "revenue_growth_yoy"]
        growth_comparison = {}
        
        for metric in growth_metrics:
            target_value = target_growth.get(metric)
            if target_value is None:
                continue
            
            peer_values = [peer.get(metric) for peer in peer_growth_list if peer.get(metric) is not None]
            if not peer_values:
                continue
            
            peer_median = np.median(peer_values)
            peer_mean = np.mean(peer_values)
            premium_discount = ((target_value - peer_median) / peer_median * 100) if peer_median != 0 else None
            percentile_rank = (sum(1 for v in peer_values if v < target_value) / len(peer_values) * 100) if peer_values else None
            
            growth_comparison[metric] = {
                "target": target_value,
                "peer_median": peer_median,
                "peer_mean": peer_mean,
                "premium_discount_pct": premium_discount,
                "percentile_rank": percentile_rank
            }
        
        comparison["growth_comparison"] = growth_comparison
        
        # 3. Valuation comparison
        print("  💰 Calculating target company valuation multiples...")
        target_multiples = calculate_multiples_valuation(target_financial_data)
        
        print(f"  💰 Calculating valuation multiples for {len(peer_financial_data_list)} peer companies...")
        peer_multiples_list = []
        for i, peer in enumerate(peer_financial_data_list):
            peer_symbol = peer.get("symbol", f"Peer_{i+1}")
            try:
                peer_multiples = calculate_multiples_valuation(peer)
                peer_multiples_list.append(peer_multiples)
                # Check if multiples were successfully calculated
                multiples = peer_multiples.get("multiples", {})
                calculated_count = sum(1 for v in [multiples.get("ev_ebitda"), multiples.get("pe"), multiples.get("ev_sales")] if v is not None)
                if calculated_count > 0:
                    print(f"    ✅ {peer_symbol}: Successfully calculated {calculated_count}/3 multiples")
                else:
                    print(f"    ⚠️ {peer_symbol}: Unable to calculate multiples (may be missing market data)")
            except Exception as e:
                print(f"    ⚠️ {peer_symbol}: Failed to calculate multiples - {e}")
                peer_multiples_list.append({"symbol": peer_symbol, "multiples": {}})
        
        valuation_metrics = ["ev_ebitda", "pe", "ev_sales"]
        valuation_comparison = {}
        
        target_multiples_dict = target_multiples.get("multiples", {})
        for metric in valuation_metrics:
            target_value = target_multiples_dict.get(metric)
            if target_value is None:
                continue
            
            peer_values = [
                peer.get("multiples", {}).get(metric)
                for peer in peer_multiples_list
                if peer.get("multiples", {}).get(metric) is not None
            ]
            if not peer_values:
                print(f"    ⚠️ {metric}: No available peer company multiple data")
                continue
            
            peer_median = np.median(peer_values)
            peer_mean = np.mean(peer_values)
            premium_discount = ((target_value - peer_median) / peer_median * 100) if peer_median != 0 else None
            percentile_rank = (sum(1 for v in peer_values if v < target_value) / len(peer_values) * 100) if peer_values else None
            
            valuation_comparison[metric] = {
                "target": target_value,
                "peer_median": peer_median,
                "peer_mean": peer_mean,
                "premium_discount_pct": premium_discount,
                "percentile_rank": percentile_rank
            }
        
        comparison["valuation_comparison"] = valuation_comparison
        
        # 4. Financial structure comparison
        target_leverage = target_analysis.get("leverage", {})
        peer_leverage_list = [peer.get("leverage", {}) for peer in peer_analysis_list]
        
        structure_metrics = ["debt_to_equity", "debt_to_assets", "current_ratio", "interest_coverage"]
        structure_comparison = {}
        
        for metric in structure_metrics:
            target_value = target_leverage.get(metric)
            if target_value is None:
                continue
            
            peer_values = [peer.get(metric) for peer in peer_leverage_list if peer.get(metric) is not None]
            if not peer_values:
                continue
            
            peer_median = np.median(peer_values)
            peer_mean = np.mean(peer_values)
            premium_discount = ((target_value - peer_median) / peer_median * 100) if peer_median != 0 else None
            percentile_rank = (sum(1 for v in peer_values if v < target_value) / len(peer_values) * 100) if peer_values else None
            
            structure_comparison[metric] = {
                "target": target_value,
                "peer_median": peer_median,
                "peer_mean": peer_mean,
                "premium_discount_pct": premium_discount,
                "percentile_rank": percentile_rank
            }
        
        comparison["financial_structure_comparison"] = structure_comparison
        
        # 5. Generate summary (rule-based)
        comparison["summary"] = generate_peer_comparison_summary(comparison)
        
        print(f"  ✅ Peer comparison completed: {comparison['peer_count']} peer companies")
    
    except Exception as e:
        print(f"⚠️ Error in peer comparison: {e}")
        import traceback
        traceback.print_exc()
    
    return comparison

def generate_peer_comparison_summary(comparison: Dict[str, Any]) -> Dict[str, Any]:
    """Generate peer comparison summary (rule-based)"""
    summary = {
        "strengths": [],
        "weaknesses": [],
        "relative_position": {}
    }
    
    try:
        # Profitability relative position
        if comparison.get("profitability_comparison", {}).get("roic"):
            roic_comp = comparison["profitability_comparison"]["roic"]
            if roic_comp.get("percentile_rank", 0) >= 75:
                summary["strengths"].append(f"ROIC in top 25% of peers ({roic_comp['percentile_rank']:.1f} percentile)")
            elif roic_comp.get("percentile_rank", 0) < 25:
                summary["weaknesses"].append(f"ROIC below peer median")
            summary["relative_position"]["roic_percentile"] = roic_comp.get("percentile_rank")
        
        # Growth relative position
        if comparison.get("growth_comparison", {}).get("revenue_cagr"):
            growth_comp = comparison["growth_comparison"]["revenue_cagr"]
            if growth_comp.get("percentile_rank", 0) >= 75:
                summary["strengths"].append(f"Revenue growth in top 25% of peers")
            summary["relative_position"]["revenue_cagr_percentile"] = growth_comp.get("percentile_rank")
        
        # Valuation relative position
        if comparison.get("valuation_comparison", {}).get("ev_ebitda"):
            val_comp = comparison["valuation_comparison"]["ev_ebitda"]
            if val_comp.get("premium_discount_pct", 0) < -10:
                summary["strengths"].append(f"Valuation discounted {abs(val_comp['premium_discount_pct']):.1f}% relative to peers")
            elif val_comp.get("premium_discount_pct", 0) > 20:
                summary["weaknesses"].append(f"Valuation premium {val_comp['premium_discount_pct']:.1f}% relative to peers")
    
    except Exception as e:
        print(f"⚠️ Error generating peer comparison summary: {e}")
    
    return summary

# ==================== LLM Peer Comparison Deep Analysis ====================

def generate_peer_comparison_llm_report(
    comparison: Dict[str, Any],
    target_financial_data: Dict[str, Any] = None,
    target_analysis: Dict[str, Any] = None,
    peer_symbols: List[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Generate in-depth peer comparison analysis report using LLM
    
    Generate a professional peer comparison analysis report containing:
    1. Comprehensive competitive assessment
    2. Deep analysis by dimension (profitability, growth, valuation, financial structure)
    3. In-depth interpretation of relative strengths and weaknesses
    4. Investment insights and recommendations
    
    Args:
        comparison: Peer comparison results (from compare_peer_financials)
        target_financial_data: Target company financial data (optional, for additional context)
        target_analysis: Target company financial analysis results (optional, for additional context)
        peer_symbols: List of peer company symbols (optional, for displaying company names)
        api_key: OpenAI API Key
        base_url: OpenAI Base URL
        model: AI model name
    
    Returns:
        Dictionary containing LLM analysis report
    """
    result = {
        "symbol": comparison.get("target_symbol", "UNKNOWN"),
        "report": "",
        "comprehensive_assessment": "",
        "dimension_analysis": {},
        "investment_insights": "",
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
        
        # 验证 API Key
        if not api_key:
            result["error"] = "OpenAI API Key is required"
            return result
    
    try:
        print(f"  🤖 Generating LLM peer comparison analysis report...")
        
        # Format peer comparison data
        comparison_data = format_peer_comparison_data_for_llm(
            comparison=comparison,
            target_financial_data=target_financial_data,
            target_analysis=target_analysis,
            peer_symbols=peer_symbols
        )
        
        # Build Prompt
        prompt = build_peer_comparison_llm_prompt(
            symbol=result['symbol'],
            comparison_data=comparison_data,
            peer_count=comparison.get("peer_count", 0)
        )
        
        # Call LLM
        chat_client = OpenAIChat(api_key=api_key, base_url=base_url)
        messages = [
            {
                "role": "system",
                "content": """You are a senior buy-side research analyst with over 15 years of experience in peer comparison analysis.
Your expertise lies in deeply analyzing companies' competitive positions relative to peers, identifying competitive advantages and disadvantages, and providing professional investment insights.

Your analysis should:
1. Conduct deep analysis based on provided data, not just list data
2. Analyze the business logic and competitive implications behind the data
3. Identify the target company's core competitive advantages and key weaknesses
4. Provide actionable investment insights
5. Use professional financial terminology and analytical frameworks

Your analysis should meet the professional standards of buy-side research departments - deep, comprehensive, and insightful."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        print(f"  ⏳ Calling LLM for analysis...")
        llm_response = chat_client.chat(messages, model=model)
        
        if llm_response:
            # Parse response
            parsed = parse_peer_comparison_llm_response(llm_response)
            result.update(parsed)
            result["raw_llm_response"] = llm_response
            
            # Generate full report
            result["report"] = format_full_peer_comparison_report(result, comparison)
            
            print(f"  ✅ LLM peer comparison analysis report generation completed")
        else:
            result["error"] = "LLM response is empty"
    
    except Exception as e:
        print(f"⚠️ Failed to generate LLM peer comparison analysis report: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
    
    return result

def format_peer_comparison_data_for_llm(
    comparison: Dict[str, Any],
    target_financial_data: Dict[str, Any] = None,
    target_analysis: Dict[str, Any] = None,
    peer_symbols: List[str] = None
) -> str:
    """Format peer comparison data into LLM-readable text"""
    target_symbol = comparison.get("target_symbol", "UNKNOWN")
    peer_count = comparison.get("peer_count", 0)
    
    formatted_text = f"""
# {target_symbol} Peer Comparison Analysis Data
Data Source: Financial statement analysis, valuation models, peer company financial data comparison
Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Basic Information
- **Target Company**: {target_symbol}
- **Number of Peer Companies**: {peer_count}
- **Peer Company Symbols**: {', '.join(peer_symbols) if peer_symbols else 'N/A'}

================================================================================
Important Note: The following data is derived from comprehensive peer comparison analysis. 
Please fully utilize this data in the analysis report. Each section should have specific data support 
and analyze the business logic and competitive implications behind the data.
================================================================================

"""
    
    # 1. Profitability comparison (detailed)
    profitability_comp = comparison.get("profitability_comparison", {})
    if profitability_comp:
        formatted_text += "## 1. Profitability vs Peers (Data Source: Financial Statement Analysis)\n\n"
        
        for metric_name, metric_key in [("Gross Margin", "gross_margin"), ("Operating Margin", "operating_margin"), 
                                       ("Net Margin", "net_margin"), ("ROE", "roe"), ("ROIC", "roic")]:
            metric_data = profitability_comp.get(metric_key, {})
            if isinstance(metric_data, dict):
                target_val = metric_data.get('target', 'N/A')
                peer_median = metric_data.get('peer_median', 'N/A')
                peer_mean = metric_data.get('peer_mean', 'N/A')
                peer_min = metric_data.get('peer_min', 'N/A')
                peer_max = metric_data.get('peer_max', 'N/A')
                premium_discount = metric_data.get('premium_discount_pct', 'N/A')
                percentile = metric_data.get('percentile_rank', 'N/A')
                
                if target_val != 'N/A':
                    formatted_text += f"### {metric_name}\n\n"
                    formatted_text += f"- **Target Company**: {target_val:.2f}%\n"
                    if peer_median != 'N/A':
                        formatted_text += f"- **Peer Median**: {peer_median:.2f}%\n"
                    if peer_mean != 'N/A':
                        formatted_text += f"- **Peer Mean**: {peer_mean:.2f}%\n"
                    if peer_min != 'N/A' and peer_max != 'N/A':
                        formatted_text += f"- **Peer Range**: {peer_min:.2f}% - {peer_max:.2f}%\n"
                    if premium_discount != 'N/A':
                        formatted_text += f"- **Relative Position**: {premium_discount:+.1f}% (relative to peer median)\n"
                    if percentile != 'N/A':
                        formatted_text += f"- **Percentile Rank**: {percentile:.1f}% (in top {100-float(percentile):.1f}% of peers)\n"
                    formatted_text += "\n"
    
    # 2. Growth comparison (detailed)
    growth_comp = comparison.get("growth_comparison", {})
    if growth_comp:
        formatted_text += "## 2. Growth vs Peers (Data Source: Financial Statement Analysis)\n\n"
        
        for metric_name, metric_key in [("Revenue 5Y CAGR", "revenue_cagr"), ("EBITDA 5Y CAGR", "ebitda_cagr"), 
                                       ("EPS 5Y CAGR", "eps_cagr"), ("Revenue YoY Growth", "revenue_growth_yoy")]:
            metric_data = growth_comp.get(metric_key, {})
            if isinstance(metric_data, dict):
                target_val = metric_data.get('target', 'N/A')
                peer_median = metric_data.get('peer_median', 'N/A')
                peer_mean = metric_data.get('peer_mean', 'N/A')
                premium_discount = metric_data.get('premium_discount_pct', 'N/A')
                percentile = metric_data.get('percentile_rank', 'N/A')
                
                if target_val != 'N/A':
                    formatted_text += f"### {metric_name}\n\n"
                    formatted_text += f"- **Target Company**: {target_val:.2f}%\n"
                    if peer_median != 'N/A':
                        formatted_text += f"- **Peer Median**: {peer_median:.2f}%\n"
                    if peer_mean != 'N/A':
                        formatted_text += f"- **Peer Mean**: {peer_mean:.2f}%\n"
                    if premium_discount != 'N/A':
                        formatted_text += f"- **Relative Position**: {premium_discount:+.1f}% (relative to peer median)\n"
                    if percentile != 'N/A':
                        formatted_text += f"- **Percentile Rank**: {percentile:.1f}% (in top {100-float(percentile):.1f}% of peers)\n"
                    formatted_text += "\n"
    
    # 3. Valuation comparison (detailed)
    valuation_comp = comparison.get("valuation_comparison", {})
    if valuation_comp:
        formatted_text += "## 3. Valuation vs Peers (Data Source: Valuation Models)\n\n"
        
        for metric_name, metric_key in [("EV/EBITDA", "ev_ebitda"), ("P/E", "pe"), ("EV/Sales", "ev_sales")]:
            metric_data = valuation_comp.get(metric_key, {})
            if isinstance(metric_data, dict):
                target_val = metric_data.get('target', 'N/A')
                peer_median = metric_data.get('peer_median', 'N/A')
                peer_mean = metric_data.get('peer_mean', 'N/A')
                premium_discount = metric_data.get('premium_discount_pct', 'N/A')
                percentile = metric_data.get('percentile_rank', 'N/A')
                
                if target_val != 'N/A':
                    formatted_text += f"### {metric_name}\n\n"
                    formatted_text += f"- **Target Company**: {target_val:.2f}x\n"
                    if peer_median != 'N/A':
                        formatted_text += f"- **Peer Median**: {peer_median:.2f}x\n"
                    if peer_mean != 'N/A':
                        formatted_text += f"- **Peer Mean**: {peer_mean:.2f}x\n"
                    if premium_discount != 'N/A':
                        formatted_text += f"- **Relative Position**: {premium_discount:+.1f}% (relative to peer median)\n"
                    if percentile != 'N/A':
                        formatted_text += f"- **Percentile Rank**: {percentile:.1f}% (in top {100-float(percentile):.1f}% of peers)\n"
                    formatted_text += "\n"
    
    # 4. Financial structure comparison (detailed)
    structure_comp = comparison.get("financial_structure_comparison", {})
    if structure_comp:
        formatted_text += "## 4. Financial Structure vs Peers (Data Source: Financial Statement Analysis)\n\n"
        
        for metric_name, metric_key in [("Debt/Equity", "debt_to_equity"), ("Debt/Assets", "debt_to_assets"), 
                                       ("Current Ratio", "current_ratio"), ("Interest Coverage", "interest_coverage")]:
            metric_data = structure_comp.get(metric_key, {})
            if isinstance(metric_data, dict):
                target_val = metric_data.get('target', 'N/A')
                peer_median = metric_data.get('peer_median', 'N/A')
                peer_mean = metric_data.get('peer_mean', 'N/A')
                premium_discount = metric_data.get('premium_discount_pct', 'N/A')
                percentile = metric_data.get('percentile_rank', 'N/A')
                
                if target_val != 'N/A':
                    formatted_text += f"### {metric_name}\n\n"
                    formatted_text += f"- **Target Company**: {target_val:.2f}\n"
                    if peer_median != 'N/A':
                        formatted_text += f"- **Peer Median**: {peer_median:.2f}\n"
                    if peer_mean != 'N/A':
                        formatted_text += f"- **Peer Mean**: {peer_mean:.2f}\n"
                    if premium_discount != 'N/A':
                        formatted_text += f"- **Relative Position**: {premium_discount:+.1f}% (relative to peer median)\n"
                    if percentile != 'N/A':
                        formatted_text += f"- **Percentile Rank**: {percentile:.1f}% (in top {100-float(percentile):.1f}% of peers)\n"
                    formatted_text += "\n"
    
    # 5. Relative position summary
    summary = comparison.get("summary", {})
    if summary:
        formatted_text += "## 5. Relative Position Summary (Data Source: Comprehensive Analysis)\n\n"
        
        strengths = summary.get("strengths", [])
        if strengths:
            formatted_text += "### Relative Strengths\n\n"
            for strength in strengths:
                formatted_text += f"- ✅ {strength}\n"
            formatted_text += "\n"
        
        weaknesses = summary.get("weaknesses", [])
        if weaknesses:
            formatted_text += "### Relative Weaknesses\n\n"
            for weakness in weaknesses:
                formatted_text += f"- ⚠️ {weakness}\n"
            formatted_text += "\n"
        
        relative_position = summary.get("relative_position", {})
        if relative_position:
            formatted_text += "### Key Metrics Percentile Rankings\n\n"
            for key, value in relative_position.items():
                if value is not None:
                    formatted_text += f"- {key.replace('_percentile', '').replace('_', ' ').title()}: {value:.1f}%\n"
            formatted_text += "\n"
    
    # 6. Target company supplementary information (if available)
    if target_analysis:
        profitability = target_analysis.get("profitability", {})
        if profitability:
            formatted_text += "## 6. Target Company Key Financial Metrics (Data Source: Financial Statement Analysis)\n\n"
            formatted_text += f"- **Gross Margin**: {profitability.get('gross_margin', 0) or 0:.2f}%\n"
            formatted_text += f"- **Net Margin**: {profitability.get('net_margin', 0) or 0:.2f}%\n"
            formatted_text += f"- **ROE**: {profitability.get('roe', 0) or 0:.2f}%\n"
            formatted_text += f"- **ROIC**: {profitability.get('roic', 0) or 0:.2f}%\n\n"
        
        growth = target_analysis.get("growth", {})
        if growth:
            revenue_cagr = growth.get('revenue_cagr_5y', 0) or growth.get('revenue_cagr', 0) or 0
            formatted_text += f"- **Revenue 5Y CAGR**: {revenue_cagr:.2f}%\n\n"
    
    formatted_text += """
================================================================================
Data Source Description:
- Profitability Comparison: Based on financial statement analysis, calculating metrics such as gross margin, net margin, ROE, ROIC
- Growth Comparison: Based on financial statement analysis, calculating metrics such as CAGR, YoY growth
- Valuation Comparison: Based on valuation models, calculating multiples such as EV/EBITDA, P/E, EV/Sales
- Financial Structure Comparison: Based on financial statement analysis, calculating metrics such as debt-to-equity ratio, current ratio
================================================================================
"""
    
    return formatted_text

def build_peer_comparison_llm_prompt(symbol: str, comparison_data: str, peer_count: int) -> str:
    """Build LLM analysis prompt for peer comparison"""
    prompt = f"""Please generate a deep, professional peer comparison analysis report for {symbol} based on the following complete peer comparison data.

================================================================================
Important Notes:
1. You must fully utilize all peer comparison data below, and not omit any important information
2. Each section must cite specific data and metrics, and analyze the business logic behind the data
3. Conduct deep analysis - not just present data, but analyze competitive implications and investment insights
4. All conclusions must be supported by data, avoid subjective speculation
5. Identify the target company's core competitive advantages and key weaknesses
================================================================================

## Complete Peer Comparison Data

{comparison_data}

## Analysis Report Requirements

Please generate a well-structured, in-depth peer comparison analysis report containing the following 5 sections:

### 1. Comprehensive Competitive Assessment
- Based on data from all dimensions, comprehensively assess the target company's overall competitive position among peers
- Identify the target company's core competitive advantages and key weaknesses
- Assess whether the target company is in a leading position, middle tier, or lagging behind in the industry
- Must cite specific percentile rankings and relative position data

### 2. Profitability Deep Analysis
- Deeply analyze the target company's advantages and disadvantages in profitability relative to peers
- Analyze the meaning of metrics such as gross margin, net margin, ROE, ROIC
- Identify possible reasons for profitability differences (product pricing, cost control, operational efficiency, etc.)
- Assess the sustainability of profitability
- Must cite specific values, percentile rankings, and relative positions

### 3. Growth Capability Deep Analysis
- Deeply analyze the target company's advantages and disadvantages in growth relative to peers
- Analyze the meaning of metrics such as revenue growth, profit growth
- Identify possible reasons for growth differences (market expansion, product innovation, market share, etc.)
- Assess the sustainability of growth quality
- Must cite specific values, percentile rankings, and relative positions

### 4. Valuation and Financial Structure Deep Analysis
- Deeply analyze the reasonableness of the target company's valuation relative to peers
- Analyze reasons for valuation premium/discount (growth expectations, profitability, risk, etc.)
- Analyze the health and risk of financial structure
- Assess whether valuation is reasonable and whether investment opportunities exist
- Must cite specific valuation multiples and relative positions

### 5. Investment Insights and Recommendations
- Provide investment insights based on peer comparison analysis
- Identify the target company's investment highlights and risk points
- Assess whether the target company is worth investing in, and the investment logic
- Provide monitoring metrics to watch
- Must be based on previous analysis, logically consistent

## Output Format

Please return in JSON format, containing the following fields:

```json
{{
    "comprehensive_assessment": {{
        "overall_competitive_position": "Industry Leading/Middle Tier/Lagging Behind",
        "core_competitive_advantages": [
            "Advantage 1 (must cite specific data)",
            "Advantage 2 (must cite specific data)"
        ],
        "key_weaknesses": [
            "Weakness 1 (must cite specific data)",
            "Weakness 2 (must cite specific data)"
        ],
        "competitive_summary": "Comprehensive competitive assessment (2-3 paragraphs, must cite specific data)"
    }},
    "dimension_analysis": {{
        "profitability_analysis": "Profitability deep analysis (3-4 paragraphs, must cite specific data, analyze business logic behind data)",
        "growth_analysis": "Growth capability deep analysis (3-4 paragraphs, must cite specific data, analyze business logic behind data)",
        "valuation_analysis": "Valuation and financial structure deep analysis (3-4 paragraphs, must cite specific data, analyze business logic behind data)"
    }},
    "investment_insights": {{
        "key_insights": [
            "Insight 1 (must be data-based)",
            "Insight 2 (must be data-based)",
            "Insight 3 (must be data-based)"
        ],
        "investment_highlights": [
            "Investment highlight 1 (must be data-based)",
            "Investment highlight 2 (must be data-based)"
        ],
        "risk_points": [
            "Risk point 1 (must be data-based)",
            "Risk point 2 (must be data-based)"
        ],
        "monitoring_metrics": [
            "Monitoring metric 1",
            "Monitoring metric 2"
        ]
    }}
}}
```

## Key Requirements

1. **Fully utilize all data**: Must cite specific values, percentile rankings, and relative positions for each dimension
2. **Deep analysis**: Not just present data, but analyze the business logic and competitive implications behind the data
3. **Logical consistency**: Investment insights must be logically consistent with previous analysis
4. **Professional standards**: Use professional financial terminology, meet buy-side research department standards
5. **Data citation standards**: When citing data, clearly state the data source and meaning

Please begin generating the peer comparison analysis report."""
    
    return prompt

def parse_peer_comparison_llm_response(llm_response: str) -> Dict[str, Any]:
    """Parse LLM peer comparison analysis report response"""
    result = {
        "comprehensive_assessment": {},
        "dimension_analysis": {},
        "investment_insights": {}
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
            result["comprehensive_assessment"] = parsed.get("comprehensive_assessment", {})
            result["dimension_analysis"] = parsed.get("dimension_analysis", {})
            result["investment_insights"] = parsed.get("investment_insights", {})
    
    except Exception as e:
        print(f"⚠️ Failed to parse LLM response: {e}")
        result["parse_error"] = str(e)
    
    return result

def format_full_peer_comparison_report(
    parsed_result: Dict[str, Any],
    comparison: Dict[str, Any]
) -> str:
    """Format full peer comparison analysis report"""
    symbol = comparison.get("target_symbol", "UNKNOWN")
    peer_count = comparison.get("peer_count", 0)
    
    report = f"""
{'='*80}
{symbol} Peer Comparison Deep Analysis Report
{'='*80}

Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Number of Peer Companies: {peer_count}

{'-'*80}
1. Comprehensive Competitive Assessment
{'-'*80}

"""
    
    comp_assessment = parsed_result.get("comprehensive_assessment", {})
    if comp_assessment:
        overall_position = comp_assessment.get('overall_competitive_position', 'N/A')
        report += f"**Overall Competitive Position**: {overall_position}\n\n"
        
        report += f"{comp_assessment.get('competitive_summary', 'N/A')}\n\n"
        
        advantages = comp_assessment.get("core_competitive_advantages", [])
        if advantages:
            report += "**Core Competitive Advantages**:\n"
            for i, advantage in enumerate(advantages, 1):
                report += f"{i}. {advantage}\n"
            report += "\n"
        
        weaknesses = comp_assessment.get("key_weaknesses", [])
        if weaknesses:
            report += "**Key Weaknesses**:\n"
            for i, weakness in enumerate(weaknesses, 1):
                report += f"{i}. {weakness}\n"
            report += "\n"
    
    report += f"""
{'-'*80}
2. Profitability Deep Analysis
{'-'*80}

"""
    
    dimension_analysis = parsed_result.get("dimension_analysis", {})
    if dimension_analysis:
        report += f"{dimension_analysis.get('profitability_analysis', 'N/A')}\n\n"
        
        report += f"""
{'-'*80}
3. Growth Capability Deep Analysis
{'-'*80}

"""
        report += f"{dimension_analysis.get('growth_analysis', 'N/A')}\n\n"
        
        report += f"""
{'-'*80}
4. Valuation and Financial Structure Deep Analysis
{'-'*80}

"""
        report += f"{dimension_analysis.get('valuation_analysis', 'N/A')}\n\n"
    
    report += f"""
{'-'*80}
5. Investment Insights and Recommendations
{'-'*80}

"""
    
    insights = parsed_result.get("investment_insights", {})
    if insights:
        key_insights = insights.get("key_insights", [])
        if key_insights:
            report += "**Key Insights**:\n"
            for i, insight in enumerate(key_insights, 1):
                report += f"{i}. {insight}\n"
            report += "\n"
        
        highlights = insights.get("investment_highlights", [])
        if highlights:
            report += "**Investment Highlights**:\n"
            for i, highlight in enumerate(highlights, 1):
                report += f"{i}. {highlight}\n"
            report += "\n"
        
        risk_points = insights.get("risk_points", [])
        if risk_points:
            report += "**Risk Points**:\n"
            for i, risk in enumerate(risk_points, 1):
                report += f"{i}. {risk}\n"
            report += "\n"
        
        monitoring_metrics = insights.get("monitoring_metrics", [])
        if monitoring_metrics:
            report += "**Monitoring Metrics**:\n"
            for i, metric in enumerate(monitoring_metrics, 1):
                report += f"{i}. {metric}\n"
            report += "\n"
    
    report += f"""
{'='*80}
End of Report
{'='*80}
"""
    
    return report

# ==================== Management Quality Assessment ====================

def assess_management_quality(
    financial_data: Dict[str, Any],
    financial_analysis: Dict[str, Any],
    valuation_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Assess management quality
    
    Assessment dimensions:
    - ROIC vs WACC (capital allocation efficiency)
    - Earnings stability (earnings volatility)
    - Free cash flow allocation (buybacks/dividends/investment)
    
    Args:
        financial_data: Financial data
        financial_analysis: Financial analysis results
        valuation_data: Valuation data (optional, for obtaining WACC)
    
    Returns:
        Management quality assessment results
    """
    assessment = {
        "symbol": financial_data.get("symbol", "UNKNOWN"),
        "roic_vs_wacc": {},
        "earnings_stability": {},
        "free_cashflow_allocation": {},
        "overall_score": 0,
        "summary": {}
    }
    
    try:
        # Get data
        income_df = financial_data.get("income_statement", {}).get("annual", pd.DataFrame())
        balance_df = financial_data.get("balance_sheet", {}).get("annual", pd.DataFrame())
        cashflow_df = financial_data.get("cash_flow", {}).get("annual", pd.DataFrame())
        overview = financial_data.get("overview", {})
        
        if income_df.empty:
            print("⚠️ Income statement data is empty, unable to assess management quality")
            return assessment
        
        # Ensure data is sorted by time (newest first)
        if isinstance(income_df.index, pd.DatetimeIndex):
            income_df = income_df.sort_index(ascending=False)
        if not balance_df.empty and isinstance(balance_df.index, pd.DatetimeIndex):
            balance_df = balance_df.sort_index(ascending=False)
        if not cashflow_df.empty and isinstance(cashflow_df.index, pd.DatetimeIndex):
            cashflow_df = cashflow_df.sort_index(ascending=False)
        
        # 1. ROIC vs WACC
        roic = financial_analysis.get("profitability", {}).get("roic")
        if roic is None:
            roic = 0
        
        # Calculate or get WACC
        wacc = None
        if valuation_data and valuation_data.get("dcf_valuation", {}).get("assumptions", {}).get("wacc"):
            wacc = valuation_data["dcf_valuation"]["assumptions"]["wacc"] / 100
        else:
            # Automatically calculate WACC
            market_cap = safe_float(overview.get("MarketCapitalization", 0))
            shares_outstanding = safe_float(overview.get("SharesOutstanding", 0))
            current_price = safe_float(overview.get("52WeekHigh", 0))
            
            if market_cap > 0:
                equity_value = market_cap
            elif shares_outstanding > 0 and current_price > 0:
                equity_value = shares_outstanding * current_price
            else:
                equity_value = 0
            
            total_debt = 0
            if not balance_df.empty:
                long_term_debt = safe_get_value(balance_df, "longTermDebt", 0)
                short_term_debt = safe_get_value(balance_df, "shortTermDebt", 0)
                total_debt = long_term_debt + short_term_debt
            
            beta = safe_float(overview.get("Beta", 1.0))
            cost_of_equity = estimate_cost_of_equity(beta=beta) / 100
            
            interest_expense = safe_get_value(income_df, "interestExpense", 0)
            cost_of_debt = (interest_expense / total_debt) if total_debt > 0 else 0.05
            
            tax_rate = 0.21
            wacc = calculate_wacc(
                equity_market_value=equity_value,
                debt_market_value=total_debt,
                cost_of_equity=cost_of_equity,
                cost_of_debt=cost_of_debt,
                tax_rate=tax_rate
            ) / 100
        
        if wacc:
            roic_wacc_spread = roic - (wacc * 100)
            assessment["roic_vs_wacc"] = {
                "roic": roic,
                "wacc": wacc * 100,
                "spread": roic_wacc_spread,
                "assessment": "优秀" if roic_wacc_spread > 10 else "良好" if roic_wacc_spread > 5 else "一般" if roic_wacc_spread > 0 else "需关注"
            }
        else:
            assessment["roic_vs_wacc"] = {
                "roic": roic,
                "wacc": None,
                "spread": None,
                "assessment": "无法评估（缺少 WACC 数据）"
            }
        
        # 2. 盈利稳定性
        if "netIncome" in income_df.columns and len(income_df) >= 3:
            net_income_series = income_df["netIncome"].dropna()
            if len(net_income_series) >= 3:
                # 计算盈利波动性（变异系数）
                mean_ni = net_income_series.mean()
                std_ni = net_income_series.std()
                cv = (std_ni / abs(mean_ni) * 100) if mean_ni != 0 else None
                
                # 计算盈利趋势（最近3年）
                if len(net_income_series) >= 3:
                    recent_3y = net_income_series.iloc[:3]
                    growth_trend = "improving" if recent_3y.iloc[0] > recent_3y.iloc[-1] else "declining"
                else:
                    growth_trend = None
                
                # 计算盈利为正的年数占比
                positive_years = sum(1 for ni in net_income_series if ni > 0)
                positive_ratio = (positive_years / len(net_income_series) * 100) if len(net_income_series) > 0 else 0
                
                assessment["earnings_stability"] = {
                    "coefficient_of_variation": cv,
                    "growth_trend": growth_trend,
                    "positive_years_ratio": positive_ratio,
                    "assessment": "稳定" if cv and cv < 30 and positive_ratio >= 80 else "较稳定" if cv and cv < 50 else "不稳定"
                }
            else:
                assessment["earnings_stability"] = {"assessment": "数据不足"}
        else:
            assessment["earnings_stability"] = {"assessment": "数据不足"}
        
        # 3. 自由现金流分配
        if not cashflow_df.empty:
            # 获取最新财年数据
            operating_cf = safe_get_value(cashflow_df, "operatingCashflow", 0)
            capex = abs(safe_get_value(cashflow_df, "capitalExpenditures", 0))
            free_cashflow = operating_cf - capex
            
            # 分红
            dividends_paid = abs(safe_get_value(cashflow_df, "dividendsPaid", 0))
            if dividends_paid == 0:
                dividends_paid = abs(safe_get_value(cashflow_df, "dividendPayout", 0))
            
            # 股票回购（简化：从股东权益变化推断）
            # 实际应该从现金流量表中获取，但 Alpha Vantage 可能不提供
            # 这里使用简化方法
            share_repurchase = 0  # 需要额外数据源
            
            # 计算分配比例
            if free_cashflow > 0:
                dividend_payout_ratio = (dividends_paid / free_cashflow * 100) if free_cashflow > 0 else 0
                reinvestment_ratio = (capex / free_cashflow * 100) if free_cashflow > 0 else 0
            else:
                dividend_payout_ratio = 0
                reinvestment_ratio = 0
            
            assessment["free_cashflow_allocation"] = {
                "free_cashflow": free_cashflow,
                "dividends_paid": dividends_paid,
                "capex": capex,
                "dividend_payout_ratio": dividend_payout_ratio,
                "reinvestment_ratio": reinvestment_ratio,
                "assessment": "平衡" if 20 <= dividend_payout_ratio <= 50 and reinvestment_ratio >= 30 else "需关注"
            }
        else:
            assessment["free_cashflow_allocation"] = {"assessment": "数据不足"}
        
        # 4. 综合评分
        score = 0
        max_score = 100
        
        # ROIC vs WACC (40分)
        if assessment["roic_vs_wacc"].get("spread") is not None:
            spread = assessment["roic_vs_wacc"]["spread"]
            if spread > 10:
                score += 40
            elif spread > 5:
                score += 30
            elif spread > 0:
                score += 20
            else:
                score += 10
        
        # 盈利稳定性 (30分)
        if assessment["earnings_stability"].get("coefficient_of_variation") is not None:
            cv = assessment["earnings_stability"]["coefficient_of_variation"]
            positive_ratio = assessment["earnings_stability"].get("positive_years_ratio", 0)
            if cv < 30 and positive_ratio >= 80:
                score += 30
            elif cv < 50 and positive_ratio >= 60:
                score += 20
            else:
                score += 10
        
        # 自由现金流分配 (30分)
        if assessment["free_cashflow_allocation"].get("free_cashflow", 0) > 0:
            dividend_ratio = assessment["free_cashflow_allocation"].get("dividend_payout_ratio", 0)
            reinvestment_ratio = assessment["free_cashflow_allocation"].get("reinvestment_ratio", 0)
            if 20 <= dividend_ratio <= 50 and reinvestment_ratio >= 30:
                score += 30
            elif reinvestment_ratio >= 20:
                score += 20
            else:
                score += 10
        
        assessment["overall_score"] = score
        assessment["overall_grade"] = "优秀" if score >= 80 else "良好" if score >= 60 else "一般" if score >= 40 else "需关注"
        
        # 5. 生成摘要
        assessment["summary"] = generate_management_quality_summary(assessment)
        
        print(f"  ✅ Management quality assessment completed: {assessment['overall_grade']} ({score}/100)")
    
    except Exception as e:
        print(f"⚠️ Error assessing management quality: {e}")
        import traceback
        traceback.print_exc()
    
    return assessment

def generate_management_quality_summary(assessment: Dict[str, Any]) -> Dict[str, Any]:
    """Generate management quality summary"""
    summary = {
        "key_insights": [],
        "strengths": [],
        "concerns": []
    }
    
    try:
        # ROIC vs WACC
        roic_wacc = assessment.get("roic_vs_wacc", {})
        if roic_wacc.get("spread") is not None:
            spread = roic_wacc["spread"]
            if spread > 10:
                summary["strengths"].append(f"ROIC significantly higher than WACC (difference {spread:.1f}%), excellent capital allocation efficiency")
            elif spread > 0:
                summary["key_insights"].append(f"ROIC higher than WACC (difference {spread:.1f}%), reasonable capital allocation")
            else:
                summary["concerns"].append(f"ROIC lower than WACC (difference {spread:.1f}%), capital allocation efficiency needs attention")
        
        # 盈利稳定性
        earnings_stability = assessment.get("earnings_stability", {})
        if earnings_stability.get("coefficient_of_variation") is not None:
            cv = earnings_stability["coefficient_of_variation"]
            if cv < 30:
                summary["strengths"].append(f"Earnings stability high (coefficient of variation {cv:.1f}%)")
            elif cv > 50:
                summary["concerns"].append(f"Earnings volatility large (coefficient of variation {cv:.1f}%)")
        
        # 自由现金流分配
        fcf_allocation = assessment.get("free_cashflow_allocation", {})
        if fcf_allocation.get("free_cashflow", 0) > 0:
            dividend_ratio = fcf_allocation.get("dividend_payout_ratio", 0)
            reinvestment_ratio = fcf_allocation.get("reinvestment_ratio", 0)
            summary["key_insights"].append(f"Free cash flow allocation: Dividends {dividend_ratio:.1f}%, reinvestment {reinvestment_ratio:.1f}%")
    
    except Exception as e:
        print(f"⚠️ Error generating management quality summary: {e}")
    
    return summary

# ==================== 催化因素分析 ====================

def analyze_catalysts(
    financial_data: Dict[str, Any],
    financial_analysis: Dict[str, Any],
    external_data: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Analyze catalysts
    
    Catalyst types:
    - New product/new market
    - Cost structure improvement
    - Industry cycle reversal
    - Interest rate or policy changes
    
    Args:
        financial_data: Financial data
        financial_analysis: Financial analysis results
        external_data: External data (news, industry data, etc., optional)
    
    Returns:
        Catalyst analysis results
    """
    catalysts = {
        "symbol": financial_data.get("symbol", "UNKNOWN"),
        "product_market_catalysts": [],
        "cost_improvement_catalysts": [],
        "industry_cycle_catalysts": [],
        "policy_rate_catalysts": [],
        "summary": {}
    }
    
    try:
        # 获取数据
        income_df = financial_data.get("income_statement", {}).get("annual", pd.DataFrame())
        balance_df = financial_data.get("balance_sheet", {}).get("annual", pd.DataFrame())
        
        if income_df.empty:
            print("⚠️ Profit statement data is empty, cannot analyze catalysts")
            return catalysts
        
        # Ensure data is sorted by time (newest first)
        if isinstance(income_df.index, pd.DatetimeIndex):
            income_df = income_df.sort_index(ascending=False)
        if not balance_df.empty and isinstance(balance_df.index, pd.DatetimeIndex):
            balance_df = balance_df.sort_index(ascending=False)
        
        # 1. Cost structure improvement
        if len(income_df) >= 3:
            # Analyze gross margin trend
            if "totalRevenue" in income_df.columns and "grossProfit" in income_df.columns:
                revenue_series = income_df["totalRevenue"].iloc[:3]
                gross_profit_series = income_df["grossProfit"].iloc[:3]
                gross_margin_series = (gross_profit_series / revenue_series * 100).where(revenue_series > 0)
                
                if not gross_margin_series.isna().all():
                    latest_margin = gross_margin_series.iloc[0]
                    prev_margin = gross_margin_series.iloc[-1]
                    margin_improvement = latest_margin - prev_margin
                    
                    if margin_improvement > 5:
                        catalysts["cost_improvement_catalysts"].append({
                            "type": "Gross margin improvement",
                            "description": f"Gross margin improved from {prev_margin:.1f}% to {latest_margin:.1f}% by {margin_improvement:.1f} percentage points",
                            "impact": "High",
                            "timeframe": "Short-term"
                        })
            
            # 分析营业费用率趋势
            if "totalRevenue" in income_df.columns and "operatingExpenses" in income_df.columns:
                revenue_series = income_df["totalRevenue"].iloc[:3]
                op_expenses_series = income_df["operatingExpenses"].iloc[:3]
                op_expense_ratio_series = (op_expenses_series / revenue_series * 100).where(revenue_series > 0)
                
                if not op_expense_ratio_series.isna().all():
                    latest_ratio = op_expense_ratio_series.iloc[0]
                    prev_ratio = op_expense_ratio_series.iloc[-1]
                    ratio_improvement = prev_ratio - latest_ratio
                    
                    if ratio_improvement > 2:
                        catalysts["cost_improvement_catalysts"].append({
                            "type": "Operating expense ratio decrease",
                            "description": f"Operating expense ratio decreased from {prev_ratio:.1f}% to {latest_ratio:.1f}% by {ratio_improvement:.1f} percentage points",
                            "impact": "Medium",
                            "timeframe": "Short-term"
                        })
        
        # 2. Growth acceleration (may indicate new product/new market)
        growth_data = financial_analysis.get("growth", {})
        revenue_cagr = growth_data.get("revenue_cagr")
        revenue_growth_yoy = growth_data.get("revenue_growth_yoy")
        
        if revenue_cagr and revenue_cagr > 20:
            catalysts["product_market_catalysts"].append({
                "type": "High growth",
                "description": f"Revenue CAGR (5 years) reached {revenue_cagr:.1f}%, possibly benefiting from new products or new market expansion",
                "impact": "High",
                "timeframe": "Long-term"
            })
        
        if revenue_growth_yoy and revenue_growth_yoy > 30:
            catalysts["product_market_catalysts"].append({
                "type": "Growth acceleration",
                "description": f"Revenue growth YoY {revenue_growth_yoy:.1f}%，growth accelerated significantly",
                "impact": "High",
                "timeframe": "Short-term"
            })
        
        # 3. 行业周期（基于盈利波动性推断）
        earnings_stability = financial_analysis.get("profitability", {})
        if earnings_stability:
            # 如果盈利波动性较大，可能处于周期性行业
            # 这里简化处理，实际需要行业数据
            catalysts["industry_cycle_catalysts"].append({
                "type": "Industry cycle analysis",
                "description": "Need to combine industry data to determine current position in cycle, based on earnings volatility",
                "impact": "Medium",
                "timeframe": "Long-term"
            })
        
        # 4. 政策/利率（需要外部数据）
        if external_data:
            # 这里可以添加政策变化、利率变化等分析
            pass
        else:
            catalysts["policy_rate_catalysts"].append({
                "type": "Policy/rate impact",
                "description": "Need to combine external data (news, policy files, etc.) for in-depth analysis",
                "impact": "Medium",
                "timeframe": "Long-term"
            })
        
        # 5. 生成摘要
        catalysts["summary"] = generate_catalysts_summary(catalysts)
        
        total_catalysts = (
            len(catalysts["product_market_catalysts"]) +
            len(catalysts["cost_improvement_catalysts"]) +
            len(catalysts["industry_cycle_catalysts"]) +
            len(catalysts["policy_rate_catalysts"])
        )
        print(f"  ✅ Catalyst analysis completed: Identified {total_catalysts} potential catalysts")
    
    except Exception as e:
        print(f"⚠️ Error analyzing catalysts: {e}")
        import traceback
        traceback.print_exc()
    
    return catalysts

def generate_catalysts_summary(catalysts: Dict[str, Any]) -> Dict[str, Any]:
    """Generate catalysts summary"""
    summary = {
        "total_catalysts": 0,
        "high_impact_count": 0,
        "short_term_count": 0,
        "key_catalysts": []
    }
    
    try:
        all_catalysts = (
            catalysts.get("product_market_catalysts", []) +
            catalysts.get("cost_improvement_catalysts", []) +
            catalysts.get("industry_cycle_catalysts", []) +
            catalysts.get("policy_rate_catalysts", [])
        )
        
        summary["total_catalysts"] = len(all_catalysts)
        summary["high_impact_count"] = sum(1 for c in all_catalysts if c.get("impact") == "高")
        summary["short_term_count"] = sum(1 for c in all_catalysts if c.get("timeframe") == "短期")
        
        # 提取关键催化因素
        high_impact = [c for c in all_catalysts if c.get("impact") == "高"]
        summary["key_catalysts"] = high_impact[:5]  # 最多5个
    
    except Exception as e:
        print(f"⚠️ Error generating catalysts summary: {e}")
    
    return summary

# ==================== 辅助函数：自动获取同业数据 ====================

def fetch_peer_financial_data(
    peer_symbols: List[str],
    years: int = 5,
    include_quarterly: bool = True
) -> List[Dict[str, Any]]:

    try:
        from data_ingestion import fetch_financial_statements
    except ImportError:
        from .data_ingestion import fetch_financial_statements
    
    peer_data_list = []
    
    print(f"  📊 Fetching financial data for {len(peer_symbols)} peer companies...")
    for symbol in peer_symbols:
        try:
            print(f"    - Fetching data for {symbol}...")
            peer_data = fetch_financial_statements(
                symbol=symbol,
                years=years,
                include_quarterly=include_quarterly
            )
            if peer_data:
                peer_data_list.append(peer_data)
        except Exception as e:
            print(f"    ⚠️ Error fetching data for {symbol}: {e}")
            continue
    
    print(f"  ✅ Successfully fetched {len(peer_data_list)}/{len(peer_symbols)} peer company data")
    return peer_data_list

# ==================== 综合分析 ====================

def comprehensive_company_analysis(
    target_financial_data: Dict[str, Any],
    peer_financial_data_list: List[Dict[str, Any]] = None,
    peer_symbols: List[str] = None,
    valuation_data: Dict[str, Any] = None,
    external_data: Dict[str, Any] = None,
    auto_fetch_peers: bool = True,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    generate_llm_peer_report: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive company analysis
    
    Args:
        target_financial_data: Target company financial data
        peer_financial_data_list: Peer company financial data list (optional, if provided, use directly)
        peer_symbols: Peer company stock code list (optional, if provided and auto_fetch_peers=True, automatically fetch)
        valuation_data: Valuation data (optional)
        external_data: External data (optional)
        auto_fetch_peers: Whether to automatically fetch peer company data (default True, if peer_symbols are provided)
    
    Returns:
        Comprehensive analysis results
    """
    result = {
        "symbol": target_financial_data.get("symbol", "UNKNOWN"),
        "peer_comparison": {},
        "management_quality": {},
        "catalysts": {},
        "summary": {}
    }
    
    print(f"\n📊 Beginning comprehensive analysis: {result['symbol']}")
    
    try:
        # 1. 财务分析
        print("  📈 Conducting financial analysis...")
        target_analysis = analyze_financial_statements(target_financial_data, years=5)
        
        # 2. 同业比较
        # 如果提供了 peer_symbols 且 auto_fetch_peers=True，自动获取同业数据
        if peer_symbols and auto_fetch_peers and not peer_financial_data_list:
            print("  📊 Automatically fetching peer company data...")
            peer_financial_data_list = fetch_peer_financial_data(
                peer_symbols=peer_symbols,
                years=5,
                include_quarterly=True
            )
        
        if peer_financial_data_list:
            print("  📊 Conducting peer comparison...")
            peer_analysis_list = [analyze_financial_statements(peer, years=5) for peer in peer_financial_data_list]
            peer_comparison = compare_peer_financials(
                target_financial_data,
                target_analysis,
                peer_financial_data_list,
                peer_analysis_list
            )
            result["peer_comparison"] = peer_comparison
            
            # 生成 LLM 同业比较深度分析报告
            if generate_llm_peer_report and OPENAI_AVAILABLE and peer_comparison.get("peer_count", 0) > 0:
                print("  🤖 Generating LLM peer comparison deep analysis report...")
                llm_peer_report = generate_peer_comparison_llm_report(
                    comparison=peer_comparison,
                    target_financial_data=target_financial_data,
                    target_analysis=target_analysis,
                    peer_symbols=peer_symbols,
                    api_key=api_key,
                    base_url=base_url,
                    model=model
                )
                result["peer_comparison"]["llm_report"] = llm_peer_report
        else:
            print("  ⚠️ No peer data provided, skipping peer comparison")
        
        # 3. 管理层质量评估
        print("  👔 Assessing management quality...")
        result["management_quality"] = assess_management_quality(
            target_financial_data,
            target_analysis,
            valuation_data
        )
        
        # 4. 催化因素分析
        print("  🚀 Analyzing catalysts...")
        result["catalysts"] = analyze_catalysts(
            target_financial_data,
            target_analysis,
            external_data
        )
        
        # 5. 生成综合摘要
        result["summary"] = generate_comprehensive_summary(result)
        
        print(f"\n✅ {result['symbol']} Comprehensive analysis completed")
    
    except Exception as e:
        print(f"⚠️ Error during comprehensive analysis: {e}")
        import traceback
        traceback.print_exc()
    
    return result

def generate_comprehensive_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive analysis summary (integrate LLM analysis results)"""
    summary = {
        "key_insights": [],
        "investment_thesis": [],
        "risks": []
    }
    
    try:
        # 优先使用 LLM 同业比较分析的洞察
        peer_comparison = result.get("peer_comparison", {})
        llm_peer_report = peer_comparison.get("llm_report", {})
        
        if llm_peer_report and not llm_peer_report.get("error"):
            # 使用 LLM 分析的洞察
            investment_insights = llm_peer_report.get("investment_insights", {})
            if investment_insights:
                key_insights = investment_insights.get("key_insights", [])
                if key_insights:
                    summary["key_insights"].extend(key_insights)
                
                highlights = investment_insights.get("investment_highlights", [])
                if highlights:
                    summary["investment_thesis"].extend(highlights)
                
                risk_points = investment_insights.get("risk_points", [])
                if risk_points:
                    summary["risks"].extend(risk_points)
        else:
            # 回退到规则基础的摘要
            if peer_comparison.get("summary", {}).get("strengths"):
                for strength in peer_comparison["summary"]["strengths"]:
                    summary["key_insights"].append(f"Relative strengths: {strength}")
            
            if peer_comparison.get("summary", {}).get("weaknesses"):
                for weakness in peer_comparison["summary"]["weaknesses"]:
                    summary["risks"].append(f"Relative weaknesses: {weakness}")
        
        # Extract insights from management quality
        mgmt_quality = result.get("management_quality", {})
        if mgmt_quality.get("overall_grade"):
            summary["key_insights"].append(f"Management quality: {mgmt_quality['overall_grade']} ({mgmt_quality.get('overall_score', 0)}/100)")
        
        # Extract investment logic from catalysts
        catalysts = result.get("catalysts", {})
        if catalysts.get("summary", {}).get("key_catalysts"):
            for catalyst in catalysts["summary"]["key_catalysts"][:3]:
                summary["investment_thesis"].append(f"Catalysts: {catalyst.get('description', '')}")
        
        # Identify risks
        if mgmt_quality.get("summary", {}).get("concerns"):
            for concern in mgmt_quality["summary"]["concerns"]:
                summary["risks"].append(f"Management concerns: {concern}")
    
    except Exception as e:
        print(f"⚠️ Error generating comprehensive summary: {e}")
    
    return summary

# ==================== 主函数（用于测试） ====================

if __name__ == "__main__":
    # 测试示例
    print("=" * 60)
    print("Comprehensive Analysis Module - Comprehensive analysis test")
    print("=" * 60)
    
    try:
        from data_ingestion import fetch_financial_statements
    except ImportError:
        from .data_ingestion import fetch_financial_statements
    
    symbol = "NVDA"
    print(f"\nFetching financial statements data for {symbol}...")
    financial_data = fetch_financial_statements(symbol, years=5)
    
    print(f"\nConducting comprehensive analysis...")
    analysis = comprehensive_company_analysis(financial_data)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("Comprehensive analysis results:")
    print("=" * 60)
    
    print("\n📊 Peer comparison:")
    peer_comp = analysis.get("peer_comparison", {})
    if peer_comp:
        print(f"  Number of peer companies: {peer_comp.get('peer_count', 0)}")
        if peer_comp.get("summary", {}).get("strengths"):
            print("  Relative strengths:")
            for strength in peer_comp["summary"]["strengths"]:
                print(f"    ✅ {strength}")
    
    print("\n👔 Management quality:")
    mgmt = analysis.get("management_quality", {})
    print(f"  Comprehensive score: {mgmt.get('overall_score', 0)}/100 ({mgmt.get('overall_grade', 'N/A')})")
    if mgmt.get("summary", {}).get("strengths"):
        print("  Strengths:")
        for strength in mgmt["summary"]["strengths"]:
            print(f"    ✅ {strength}")
    
    print("\n🚀 Catalysts:")
    catalysts = analysis.get("catalysts", {})
    cat_summary = catalysts.get("summary", {})
    print(f"  Total number of catalysts: {cat_summary.get('total_catalysts', 0)}")
    print(f"  High impact catalysts: {cat_summary.get('high_impact_count', 0)}")
    if cat_summary.get("key_catalysts"):
        print("  Key catalysts:")
        for cat in cat_summary["key_catalysts"]:
            print(f"    • {cat.get('type', 'N/A')}: {cat.get('description', 'N/A')}")
    
    print("\n📋 Comprehensive summary:")
    summary = analysis.get("summary", {})
    if summary.get("key_insights"):
        print("  Key insights:")
        for insight in summary["key_insights"]:
            print(f"    • {insight}")
    
    print("\n✅ Test completed!")

