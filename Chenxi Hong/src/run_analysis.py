"""
Complete Fundamental Analysis Script
One-click complete all analysis and save results
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add current directory and parent directory to path
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir))

# Import analysis modules
try:
    from investment_memo import comprehensive_fundamental_analysis
except ImportError:
    try:
        from .investment_memo import comprehensive_fundamental_analysis
    except ImportError as e:
        print(f"❌ Failed to import analysis module: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {current_dir}")
        print(f"Please ensure running in Fundamental-Analyst-Agent directory")
        sys.exit(1)

def run_complete_analysis(
    symbol: str,
    peer_symbols: List[str] = None,
    years: int = 5,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    news_limit: int = 50,
    save_results: bool = True,
    output_dir: str = "analysis_results"
) -> Dict[str, Any]:
    """
    Run complete fundamental analysis
    
    Args:
        symbol: Stock symbol
        peer_symbols: List of peer company symbols
        years: Number of years to analyze
        api_key: OpenAI API Key
        base_url: OpenAI Base URL
        model: AI model name
        news_limit: News quantity limit
        save_results: Whether to save results
        output_dir: Output directory
    
    Returns:
        Complete analysis results
    """
    print("=" * 80)
    print(f"Starting complete fundamental analysis for {symbol}")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Peer companies: {peer_symbols if peer_symbols else 'None'}")
    print(f"Analysis years: {years}")
    print("=" * 80)
    
    # Run complete analysis
    result = comprehensive_fundamental_analysis(
        symbol=symbol,
        peer_symbols=peer_symbols,
        years=years,
        api_key=api_key,
        base_url=base_url,
        model=model,
        news_limit=news_limit
    )
    
    # Save results
    if save_results:
        # 使用 report/{股票代码}/ 目录结构
        symbol_clean = symbol.upper()
        output_path = Path("report") / symbol_clean
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results (JSON)
        json_file = output_path / f"complete_analysis_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            # Remove non-serializable objects like DataFrame
            result_serializable = make_serializable(result)
            json.dump(result_serializable, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n✅ Analysis results saved to: {json_file}")
        
        # Save investment memo (text)
        memo = result.get("investment_memo", {})
        if memo.get("full_memo"):
            memo_file = output_path / f"investment_memo_{timestamp}.txt"
            with open(memo_file, 'w', encoding='utf-8') as f:
                f.write(memo["full_memo"])
            print(f"✅ Investment memo saved to: {memo_file}")
        
        # Save summary (Markdown)
        summary_file = output_path / f"summary_{timestamp}.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(generate_summary_markdown(result))
        print(f"✅ Analysis summary saved to: {summary_file}")
        
        # Save peer comparison report if available
        peer_comp = result.get("peer_comparison", {})
        llm_peer_report = peer_comp.get("llm_report", {})
        if llm_peer_report.get("report"):
            peer_report_file = output_path / f"peer_comparison_report_{timestamp}.txt"
            with open(peer_report_file, 'w', encoding='utf-8') as f:
                f.write(llm_peer_report["report"])
            print(f"✅ Peer comparison report saved to: {peer_report_file}")
        
        # Save earnings quality report if available
        earnings_quality = result.get("earnings_quality", {})
        llm_report = earnings_quality.get("llm_report", {})
        if llm_report.get("report"):
            earnings_report_file = output_path / f"earnings_quality_report_{timestamp}.txt"
            with open(earnings_report_file, 'w', encoding='utf-8') as f:
                f.write(llm_report["report"])
            print(f"✅ Earnings quality report saved to: {earnings_report_file}")
    
    # Display key results
    print("\n" + "=" * 80)
    print("Analysis completed! Key results:")
    print("=" * 80)
    
    memo = result.get("investment_memo", {})
    if memo.get("recommendation"):
        print(f"\n📊 Investment recommendation: {memo.get('recommendation')}")
        print(f"📈 Target price: ${memo.get('target_price', 0):.2f}")
        print(f"⏰ Investment timeframe: {memo.get('investment_timeframe', 'N/A')}")
    
    earnings_quality = result.get("earnings_quality", {})
    if earnings_quality.get("overall_score") is not None:
        print(f"\n💎 Earnings quality score: {earnings_quality.get('overall_score')}/100 ({earnings_quality.get('overall_assessment', 'N/A')})")
    
    valuation = result.get("valuation_result", {})
    dcf = valuation.get("dcf_valuation", {})
    if dcf.get("upside_potential"):
        print(f"\n💰 Valuation upside potential: {dcf.get('upside_potential', 0):.2f}%")
    
    print("\n" + "=" * 80)
    
    return result

def make_serializable(obj: Any) -> Any:
    """Convert object to serializable format"""
    import pandas as pd
    from datetime import datetime, date
    
    if isinstance(obj, dict):
        # Convert dictionary keys to strings if they are not already serializable
        result = {}
        for k, v in obj.items():
            # Convert key to string if it's not a basic type
            if isinstance(k, (str, int, float, bool)) or k is None:
                key = k
            else:
                # Handle Timestamp, datetime, date, etc.
                if isinstance(k, (pd.Timestamp, datetime, date)):
                    key = str(k)
                else:
                    key = str(k)
            result[key] = make_serializable(v)
        return result
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (pd.Timestamp, datetime, date)):
        return str(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif hasattr(obj, 'to_dict'):
        return make_serializable(obj.to_dict())
    elif hasattr(obj, '__dict__'):
        return make_serializable(obj.__dict__)
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)

def safe_get_nested(obj, *keys, default='N/A'):
    """Safely get nested dictionary values"""
    current = obj
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key)
            if current is None:
                return default
        else:
            return default
    return current

def format_number(value, default='N/A', format_str='.2f'):
    """Format a number with fallback to default"""
    if value is None or value == 'N/A' or value == default:
        return default
    if isinstance(value, (int, float)):
        try:
            return f"{value:{format_str}}"
        except:
            return str(value)
    return str(value)

def generate_summary_markdown(result: Dict[str, Any]) -> str:
    """Generate Markdown format summary"""
    symbol = result.get("symbol", "UNKNOWN")
    timestamp = result.get("analysis_timestamp", "")
    
    md = f"""# {symbol} Fundamental Analysis Summary

**Analysis Time**: {timestamp}

## 📊 Investment Recommendation

"""
    
    memo = result.get("investment_memo", {})
    if memo.get("recommendation"):
        md += f"- **Investment Recommendation**: {memo.get('recommendation')}\n"
        md += f"- **Target Price**: ${memo.get('target_price', 0):.2f}\n"
        md += f"- **Investment Timeframe**: {memo.get('investment_timeframe', 'N/A')}\n\n"
        
        thesis = memo.get("investment_thesis", [])
        if thesis:
            md += "### Investment Thesis\n\n"
            if isinstance(thesis, list):
                for i, point in enumerate(thesis, 1):
                    md += f"{i}. {point}\n"
            else:
                md += f"{thesis}\n"
            md += "\n"
    
    # Financial performance
    financial_analysis = result.get("financial_analysis", {})
    if financial_analysis:
        md += "## 📈 Financial Performance\n\n"
        
        profitability = financial_analysis.get("profitability", {})
        if profitability:
            md += "### Profitability\n"
            # Handle different data structures
            gross_margin = safe_get_nested(profitability, 'gross_margin', 'latest')
            if gross_margin == 'N/A':
                gross_margin = profitability.get('gross_margin', 'N/A')
            
            net_margin = safe_get_nested(profitability, 'net_margin', 'latest')
            if net_margin == 'N/A':
                net_margin = profitability.get('net_margin', 'N/A')
            
            roic = safe_get_nested(profitability, 'roic', 'latest')
            if roic == 'N/A':
                roic = profitability.get('roic', 'N/A')
            
            md += f"- Gross Margin: {format_number(gross_margin)}%\n"
            md += f"- Net Margin: {format_number(net_margin)}%\n"
            md += f"- ROIC: {format_number(roic)}%\n\n"
        
        growth = financial_analysis.get("growth", {})
        if growth:
            md += "### Growth\n"
            revenue_cagr = growth.get('revenue_cagr_5y', 'N/A')
            ebitda_cagr = growth.get('ebitda_cagr_5y', 'N/A')
            md += f"- Revenue 5Y CAGR: {format_number(revenue_cagr)}%\n"
            md += f"- EBITDA 5Y CAGR: {format_number(ebitda_cagr)}%\n\n"
    
    # Valuation
    valuation = result.get("valuation_result", {})
    if valuation:
        md += "## 💰 Valuation Summary\n\n"
        
        dcf = valuation.get("dcf_valuation", {})
        if dcf:
            md += f"- Target Price: ${dcf.get('target_price', 0):.2f}\n"
            md += f"- Current Price: ${dcf.get('current_price', 0):.2f}\n"
            md += f"- Upside Potential: {dcf.get('upside_potential', 0):.2f}%\n\n"
    
    # Earnings quality
    earnings_quality = result.get("earnings_quality", {})
    if earnings_quality:
        md += "## 💎 Earnings Quality\n\n"
        md += f"- Overall Score: {earnings_quality.get('overall_score', 0)}/100\n"
        md += f"- Rating: {earnings_quality.get('overall_assessment', 'N/A')}\n\n"
        
        summary = earnings_quality.get("summary", {})
        if summary:
            strengths = summary.get("strengths", [])
            if strengths:
                md += "### Strengths\n"
                for strength in strengths:
                    md += f"- ✅ {strength}\n"
                md += "\n"
            
            concerns = summary.get("concerns", [])
            if concerns:
                md += "### Concerns\n"
                for concern in concerns:
                    md += f"- ⚠️ {concern}\n"
                md += "\n"
    
    # Risks
    md += "## ⚠️ Key Risks\n\n"
    risks_text = memo.get("risks", "")
    if risks_text:
        md += f"{risks_text}\n\n"
    
    return md

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete fundamental analysis")
    parser.add_argument("symbol", help="Stock symbol")
    parser.add_argument("--peers", nargs="+", help="List of peer company symbols", default=None)
    parser.add_argument("--years", type=int, default=5, help="Number of years to analyze (default 5 years)")
    parser.add_argument("--api-key", help="OpenAI API Key")
    parser.add_argument("--base-url", help="OpenAI Base URL")
    parser.add_argument("--model", default="gpt-4o-mini", help="AI model name (default gpt-4o-mini)")
    parser.add_argument("--news-limit", type=int, default=50, help="News quantity limit (default 50)")
    parser.add_argument("--no-save", action="store_true", help="Do not save results")
    parser.add_argument("--output-dir", default="report", help="Output base directory (default report, files will be saved to report/{symbol}/)")
    
    args = parser.parse_args()
    
    # Get API Key from environment variable or command line argument
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ Error: OpenAI API Key is required. Please set OPENAI_API_KEY environment variable or use --api-key argument.")
        sys.exit(1)
    
    # Get base URL (from argument, environment variable, or default)
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    
    # Run analysis
    result = run_complete_analysis(
        symbol=args.symbol,
        peer_symbols=args.peers,
        years=args.years,
        api_key=api_key,
        base_url=base_url,
        model=args.model,
        news_limit=args.news_limit,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
    
    print("\n✅ Analysis completed!")

