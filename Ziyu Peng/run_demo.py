"""
Demo Script for MSFT Fundamental Analysis
Hardcoded API configuration and complete analysis workflow
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add src directory to path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
parent_dir = current_dir.parent
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir))

# ==================== Hardcoded API Configuration ====================

# Alpha Vantage API Configuration
ALPHAVANTAGE_API_KEY = "sk-RgYy04OsW9vLCcnrl486s9ExWQBThgsJQ53QA7BkBxJGfpQS"
ALPHAVANTAGE_BASE_URL = "http://104.194.90.24:18081/alphavantage/query"

# OpenAI API Configuration
OPENAI_API_KEY = "sk-eRbtDCmAbT8WTklZfhOdAY5rRpmusCe61gzE66Pt6TNcRQxG"
OPENAI_BASE_URL = "https://zjuapi.com/v1"
OPENAI_MODEL = "gpt-4o-mini"
# OPENAI_MODEL = "gpt-5-nano"

# Target stock symbol
TARGET_SYMBOL = "MSFT"
PEER_COUNT = 5  # Number of peer companies to search for
ANALYSIS_YEARS = 5  # Number of years to analyze

# ==================== Set Environment Variables and Module Configuration ====================

# Set environment variables
os.environ["ALPHAVANTAGE_API_KEY"] = ALPHAVANTAGE_API_KEY
os.environ["ALPHAVANTAGE_BASE_URL"] = ALPHAVANTAGE_BASE_URL
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

# Import and configure data_ingestion module
try:
    import data_ingestion
    data_ingestion.API_KEY = ALPHAVANTAGE_API_KEY
    data_ingestion.BASE_URL = ALPHAVANTAGE_BASE_URL
    print("✅ Alpha Vantage API configuration completed")
except Exception as e:
    print(f"⚠️ Error configuring Alpha Vantage API: {e}")

# ==================== Import Analysis Modules ====================

try:
    from investment_memo import comprehensive_fundamental_analysis
    from company_search import search_company, get_peer_recommendations
    # Import save function from run_analysis
    from run_analysis import run_complete_analysis
    print("✅ Analysis modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import analysis modules: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {current_dir}")
    print("Please ensure running this script from the project root directory")
    sys.exit(1)

# ==================== Main Function ====================

def main():
    """Main function: Execute complete analysis workflow"""
    
    print("=" * 80)
    print("🚀 MSFT Fundamental Analysis Demo")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target symbol: {TARGET_SYMBOL}")
    print(f"Analysis years: {ANALYSIS_YEARS}")
    print("=" * 80)
    
    # Step 1: Search target company information
    print(f"\n[Step 1/4] 🔍 Searching {TARGET_SYMBOL} company information...")
    company_info = None
    try:
        company_info = search_company(TARGET_SYMBOL)
        if company_info:
            print(f"  ✅ Company name: {company_info.get('name', 'N/A')}")
            print(f"  ✅ Industry: {company_info.get('industry', 'N/A')}")
            print(f"  ✅ Sector: {company_info.get('sector', 'N/A')}")
        else:
            print(f"  ⚠️ Failed to get {TARGET_SYMBOL} company information, will use default peer recommendations")
    except Exception as e:
        print(f"  ⚠️ Error searching company information: {e}")
    
    # Step 2: Get peer company recommendations
    print(f"\n[Step 2/4] 👥 Searching peer companies for {TARGET_SYMBOL}...")
    peer_symbols = []
    
    try:
        # Get peer recommendations
        if company_info:
            sector = company_info.get("sector", "")
            industry = company_info.get("industry", "")
            recommended_peers = get_peer_recommendations(
                target_symbol=TARGET_SYMBOL,
                target_sector=sector,
                target_industry=industry
            )
        else:
            recommended_peers = get_peer_recommendations(target_symbol=TARGET_SYMBOL)
        
        # Limit to top N peers
        peer_symbols = recommended_peers[:PEER_COUNT] if recommended_peers else []
        
        if peer_symbols:
            print(f"  ✅ Found {len(peer_symbols)} peer companies:")
            for i, peer in enumerate(peer_symbols, 1):
                print(f"    {i}. {peer}")
        else:
            print(f"  ⚠️ Failed to find peer companies, will use default list")
            # Use common MSFT peers (from code)
            peer_symbols = ["GOOGL", "AAPL", "META", "ORCL", "CRM"]
            print(f"  📋 Using default peer list: {', '.join(peer_symbols)}")
            
    except Exception as e:
        print(f"  ⚠️ Error getting peer recommendations: {e}")
        # Use default peer list
        peer_symbols = ["GOOGL", "AAPL", "META", "ORCL", "CRM"]
        print(f"  📋 Using default peer list: {', '.join(peer_symbols)}")
    
    # Step 3: Run complete analysis
    print(f"\n[Step 3/4] 📊 Starting complete fundamental analysis...")
    print(f"  - Target company: {TARGET_SYMBOL}")
    print(f"  - Peer companies: {', '.join(peer_symbols) if peer_symbols else 'None'}")
    print(f"  - Analysis years: {ANALYSIS_YEARS}")
    print(f"  - LLM model: {OPENAI_MODEL}")
    print()
    
    try:
        # Use run_complete_analysis function, which automatically saves reports to report/{symbol}/ directory
        result = run_complete_analysis(
            symbol=TARGET_SYMBOL,
            peer_symbols=peer_symbols if peer_symbols else None,
            years=ANALYSIS_YEARS,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=OPENAI_MODEL,
            news_limit=50,
            save_results=True,  # Automatically save reports
            output_dir="report"  # Reports saved to report/{symbol}/ directory
        )
        
        # Check for errors
        if result.get("error"):
            print(f"\n❌ Error during analysis: {result.get('error')}")
            return
        
        print("\n✅ Analysis completed!")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Display key results
    print(f"\n[Step 4/4] 📈 Analysis Results Summary")
    print("=" * 80)
    
    # Investment recommendation
    memo = result.get("investment_memo", {})
    if memo.get("recommendation"):
        print(f"\n📊 Investment recommendation: {memo.get('recommendation')}")
        print(f"📈 Target price: ${memo.get('target_price', 0):.2f}")
        print(f"⏰ Investment timeframe: {memo.get('investment_timeframe', 'N/A')}")
        
        # Investment thesis
        thesis = memo.get("investment_thesis", [])
        if thesis:
            print(f"\n💡 Investment thesis:")
            if isinstance(thesis, list):
                for i, point in enumerate(thesis[:3], 1):  # Show only first 3
                    print(f"   {i}. {point}")
            else:
                print(f"   {thesis[:200]}...")
    
    # Earnings quality score
    earnings_quality = result.get("earnings_quality", {})
    if earnings_quality.get("overall_score") is not None:
        print(f"\n💎 Earnings quality score: {earnings_quality.get('overall_score')}/100")
        print(f"   Rating: {earnings_quality.get('overall_assessment', 'N/A')}")
    
    # Valuation results
    valuation = result.get("valuation_result", {})
    dcf = valuation.get("dcf_valuation", {})
    if dcf.get("upside_potential"):
        print(f"\n💰 DCF Valuation:")
        print(f"   Target price: ${dcf.get('target_price', 0):.2f}")
        print(f"   Current price: ${dcf.get('current_price', 0):.2f}")
        print(f"   Upside potential: {dcf.get('upside_potential', 0):.2f}%")
    
    # Financial performance
    financial_analysis = result.get("financial_analysis", {})
    if financial_analysis:
        profitability = financial_analysis.get("profitability", {})
        if profitability:
            print(f"\n📊 Profitability:")
            # Handle different data structures - gross_margin might be dict or float
            gross_margin_raw = profitability.get("gross_margin", "N/A")
            if isinstance(gross_margin_raw, dict):
                gross_margin = gross_margin_raw.get("latest", "N/A")
            else:
                gross_margin = gross_margin_raw
            
            net_margin_raw = profitability.get("net_margin", "N/A")
            if isinstance(net_margin_raw, dict):
                net_margin = net_margin_raw.get("latest", "N/A")
            else:
                net_margin = net_margin_raw
            
            roic_raw = profitability.get("roic", "N/A")
            if isinstance(roic_raw, dict):
                roic = roic_raw.get("latest", "N/A")
            else:
                roic = roic_raw
            
            print(f"   Gross margin: {gross_margin:.2f}%" if isinstance(gross_margin, (int, float)) else f"   Gross margin: {gross_margin}")
            print(f"   Net margin: {net_margin:.2f}%" if isinstance(net_margin, (int, float)) else f"   Net margin: {net_margin}")
            print(f"   ROIC: {roic:.2f}%" if isinstance(roic, (int, float)) else f"   ROIC: {roic}")
    
    print("\n" + "=" * 80)
    print(f"✅ Analysis completed! Reports saved to report/{TARGET_SYMBOL}/ directory")
    print("=" * 80)
    
    return result

# ==================== Run Main Function ====================

if __name__ == "__main__":
    try:
        result = main()
        if result:
            print("\n🎉 Demo completed!")
        else:
            print("\n⚠️ Demo not completed, please check error messages")
    except KeyboardInterrupt:
        print("\n\n⚠️ User interrupted execution")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

