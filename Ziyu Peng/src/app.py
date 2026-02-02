"""
Fundamental Analysis Web Application
Web interface built with Streamlit
"""

import streamlit as st
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import json

# Page configuration
st.set_page_config(
    page_title="Fundamental Analysis Agent",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add current directory and parent directory to path
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(parent_dir))

# Import analysis modules
try:
    # First try direct import (when app.py is in  directory)
    from investment_memo import comprehensive_fundamental_analysis
    from data_ingestion import fetch_financial_statements, get_overview
    from financial_analysis import analyze_financial_statements
    from valuation import comprehensive_valuation
    from comprehensive_analysis import comprehensive_company_analysis
    from qualitative_analysis import comprehensive_qualitative_analysis
    from earnings_quality import comprehensive_earnings_quality_analysis
    from company_search import search_company, get_peer_recommendations, get_common_stocks
except ImportError:
    # If direct import fails, try package import
    try:
        from .investment_memo import comprehensive_fundamental_analysis
        from .data_ingestion import fetch_financial_statements, get_overview
        from .financial_analysis import analyze_financial_statements
        from .valuation import comprehensive_valuation
        from .comprehensive_analysis import comprehensive_company_analysis
        from .qualitative_analysis import comprehensive_qualitative_analysis
        from .earnings_quality import comprehensive_earnings_quality_analysis
        from .company_search import search_company, get_peer_recommendations, get_common_stocks
    except ImportError as e:
        st.error(f"Failed to import analysis modules: {e}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Script directory: {current_dir}")
        st.error(f"Please ensure you are running from project root: cd <project-root> && streamlit run src/app.py")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# ==================== Report Saving Functions ====================

def save_all_reports(result: Dict[str, Any], symbol: str) -> Dict[str, str]:
    """
    保存所有分析报告到 report/{股票代码}/ 目录
    
    Args:
        result: 完整的分析结果字典
        symbol: 股票代码
    
    Returns:
        保存的文件路径字典
    """
    # 创建报告目录
    report_dir = Path("report") / symbol.upper()
    report_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 1. 保存投资备忘录
        memo = result.get("investment_memo", {})
        if memo.get("full_memo"):
            memo_file = report_dir / f"investment_memo_{timestamp}.txt"
            with open(memo_file, 'w', encoding='utf-8') as f:
                f.write(memo["full_memo"])
            saved_files["investment_memo"] = str(memo_file)
        
        # 2. 保存同行比较报告
        peer_comp = result.get("peer_comparison", {})
        llm_peer_report = peer_comp.get("llm_report", {})
        if llm_peer_report.get("report"):
            peer_report_file = report_dir / f"peer_comparison_report_{timestamp}.txt"
            with open(peer_report_file, 'w', encoding='utf-8') as f:
                f.write(llm_peer_report["report"])
            saved_files["peer_comparison_report"] = str(peer_report_file)
        
        # 3. 保存收益质量报告
        earnings_quality = result.get("earnings_quality", {})
        llm_report = earnings_quality.get("llm_report", {})
        if llm_report.get("report"):
            earnings_report_file = report_dir / f"earnings_quality_report_{timestamp}.txt"
            with open(earnings_report_file, 'w', encoding='utf-8') as f:
                f.write(llm_report["report"])
            saved_files["earnings_quality_report"] = str(earnings_report_file)
        
        # 4. 保存财务报表 CSV
        financial_data = result.get("financial_data", {})
        if financial_data:
            # 利润表
            income_annual = financial_data.get("income_statement", {}).get("annual", pd.DataFrame())
            if not income_annual.empty:
                income_file = report_dir / f"income_statement_annual_{timestamp}.csv"
                income_annual.to_csv(income_file, encoding="utf-8-sig")
                saved_files["income_statement_annual"] = str(income_file)
            
            income_quarterly = financial_data.get("income_statement", {}).get("quarterly", pd.DataFrame())
            if not income_quarterly.empty:
                income_q_file = report_dir / f"income_statement_quarterly_{timestamp}.csv"
                income_quarterly.to_csv(income_q_file, encoding="utf-8-sig")
                saved_files["income_statement_quarterly"] = str(income_q_file)
            
            # 资产负债表
            balance_annual = financial_data.get("balance_sheet", {}).get("annual", pd.DataFrame())
            if not balance_annual.empty:
                balance_file = report_dir / f"balance_sheet_annual_{timestamp}.csv"
                balance_annual.to_csv(balance_file, encoding="utf-8-sig")
                saved_files["balance_sheet_annual"] = str(balance_file)
            
            balance_quarterly = financial_data.get("balance_sheet", {}).get("quarterly", pd.DataFrame())
            if not balance_quarterly.empty:
                balance_q_file = report_dir / f"balance_sheet_quarterly_{timestamp}.csv"
                balance_quarterly.to_csv(balance_q_file, encoding="utf-8-sig")
                saved_files["balance_sheet_quarterly"] = str(balance_q_file)
            
            # 现金流量表
            cashflow_annual = financial_data.get("cash_flow", {}).get("annual", pd.DataFrame())
            if not cashflow_annual.empty:
                cashflow_file = report_dir / f"cashflow_annual_{timestamp}.csv"
                cashflow_annual.to_csv(cashflow_file, encoding="utf-8-sig")
                saved_files["cashflow_annual"] = str(cashflow_file)
            
            cashflow_quarterly = financial_data.get("cash_flow", {}).get("quarterly", pd.DataFrame())
            if not cashflow_quarterly.empty:
                cashflow_q_file = report_dir / f"cashflow_quarterly_{timestamp}.csv"
                cashflow_quarterly.to_csv(cashflow_q_file, encoding="utf-8-sig")
                saved_files["cashflow_quarterly"] = str(cashflow_q_file)
        
        # 5. 保存财务比率 CSV
        financial_analysis = result.get("financial_analysis", {})
        if financial_analysis:
            # 创建财务比率汇总表
            ratios_data = []
            
            # 盈利能力指标
            profitability = financial_analysis.get("profitability", {})
            if profitability:
                ratios_data.append({
                    "Category": "Profitability",
                    "Metric": "Gross Margin",
                    "Latest": profitability.get("gross_margin", {}).get("latest", "N/A"),
                    "5Y Avg": profitability.get("gross_margin", {}).get("5y_avg", "N/A"),
                    "Trend": profitability.get("gross_margin", {}).get("trend", "N/A")
                })
                ratios_data.append({
                    "Category": "Profitability",
                    "Metric": "Net Margin",
                    "Latest": profitability.get("net_margin", {}).get("latest", "N/A"),
                    "5Y Avg": profitability.get("net_margin", {}).get("5y_avg", "N/A"),
                    "Trend": profitability.get("net_margin", {}).get("trend", "N/A")
                })
                ratios_data.append({
                    "Category": "Profitability",
                    "Metric": "ROIC",
                    "Latest": profitability.get("roic", {}).get("latest", "N/A"),
                    "5Y Avg": profitability.get("roic", {}).get("5y_avg", "N/A"),
                    "Trend": profitability.get("roic", {}).get("trend", "N/A")
                })
            
            # 增长指标
            growth = financial_analysis.get("growth", {})
            if growth:
                ratios_data.append({
                    "Category": "Growth",
                    "Metric": "Revenue 5Y CAGR",
                    "Latest": growth.get("revenue_cagr_5y", "N/A"),
                    "5Y Avg": "N/A",
                    "Trend": "N/A"
                })
                ratios_data.append({
                    "Category": "Growth",
                    "Metric": "EBITDA 5Y CAGR",
                    "Latest": growth.get("ebitda_cagr_5y", "N/A"),
                    "5Y Avg": "N/A",
                    "Trend": "N/A"
                })
            
            if ratios_data:
                ratios_df = pd.DataFrame(ratios_data)
                ratios_file = report_dir / f"financial_ratios_{timestamp}.csv"
                ratios_df.to_csv(ratios_file, index=False, encoding="utf-8-sig")
                saved_files["financial_ratios"] = str(ratios_file)
        
        # 6. 保存估值数据 CSV
        valuation = result.get("valuation_result", {})
        if valuation:
            valuation_data = []
            
            dcf = valuation.get("dcf_valuation", {})
            if dcf:
                valuation_data.append({
                    "Valuation Method": "DCF",
                    "Target Price": dcf.get("target_price", "N/A"),
                    "Current Price": dcf.get("current_price", "N/A"),
                    "Upside Potential (%)": dcf.get("upside_potential", "N/A"),
                    "Fair Value": dcf.get("fair_value", "N/A")
                })
            
            multiples = valuation.get("multiples_valuation", {})
            if multiples:
                for method, data in multiples.items():
                    if isinstance(data, dict) and data.get("target_price"):
                        valuation_data.append({
                            "Valuation Method": method.upper(),
                            "Target Price": data.get("target_price", "N/A"),
                            "Current Price": data.get("current_price", "N/A"),
                            "Upside Potential (%)": data.get("upside_potential", "N/A"),
                            "Fair Value": data.get("fair_value", "N/A")
                        })
            
            if valuation_data:
                valuation_df = pd.DataFrame(valuation_data)
                valuation_file = report_dir / f"valuation_{timestamp}.csv"
                valuation_df.to_csv(valuation_file, index=False, encoding="utf-8-sig")
                saved_files["valuation"] = str(valuation_file)
        
        # 7. Save complete analysis results JSON
        def make_serializable(obj):
            """Convert object to serializable format"""
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
        
        result_clean = make_serializable(result)
        json_file = report_dir / f"complete_analysis_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(result_clean, f, ensure_ascii=False, indent=2, default=str)
        saved_files["complete_analysis_json"] = str(json_file)
        
    except Exception as e:
        print(f"Error saving reports: {e}")
        import traceback
        traceback.print_exc()
    
    return saved_files

# Page title
st.title("Fundamental Analysis Agent")
st.markdown("---")

# ==================== API Configuration Section (Top of Page) ====================
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; margin-bottom: 25px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h2 style='color: white; margin: 0 0 10px 0; font-size: 28px;'>API Settings</h2>
    <p style='color: #f0f0f0; margin: 0; font-size: 16px;'>Please configure the following API information to use the system features. The configuration will be automatically saved and applied throughout the session.</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state for API configurations
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")
if 'openai_base_url' not in st.session_state:
    st.session_state.openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
if 'alphavantage_api_key' not in st.session_state:
    st.session_state.alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
if 'alphavantage_base_url' not in st.session_state:
    st.session_state.alphavantage_base_url = os.getenv("ALPHAVANTAGE_BASE_URL", "")
if 'ai_model' not in st.session_state:
    st.session_state.ai_model = "gpt-4o-mini"
if 'config_applied' not in st.session_state:
    st.session_state.config_applied = False

# API Configuration Container
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 5px solid #1f77b4; margin-bottom: 15px;'>
            <h3 style='color: #1f77b4; margin: 0 0 10px 0;'>OpenAI API Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            key="openai_key_input",
            help="For LLM analysis and investment memo generation"
        )
        st.session_state.openai_base_url = st.text_input(
            "OpenAI Base URL",
            value=st.session_state.openai_base_url,
            key="openai_url_input",
            help="OpenAI API base URL (default: https://api.openai.com/v1)"
        )

    
        model_options = ["gpt-5-nano", "gpt-4o-mini"]
        default_index = model_options.index(st.session_state.ai_model) if st.session_state.ai_model in model_options else 0


        selected_model = st.selectbox(
            "AI Model (Preset Options)",
            options=model_options,
            index=default_index,
            key="ai_model_select",
            help="Select a preset AI model for analysis"
        )

        # 可选：自定义模型名称，如果填写则优先使用这里的模型
        custom_model = st.text_input(
            "Custom Model (Optional)",
            value="" if st.session_state.ai_model in model_options else st.session_state.ai_model,
            key="ai_model_custom_input",
            help="Enter a custom model name, e.g., gpt-4.1, if different from the preset options above"
        )

        # 逻辑：如果用户填写了自定义模型，则优先使用自定义模型，否则使用下拉框选择的模型
        if custom_model and custom_model.strip():
            st.session_state.ai_model = custom_model.strip()
        else:
            st.session_state.ai_model = selected_model
        # Display current URL
        st.caption(f"📍 Current Base URL: `{st.session_state.openai_base_url}`")
    
    with col2:
        st.markdown("""
        <div style='background-color: #fff5f0; padding: 15px; border-radius: 10px; border-left: 5px solid #ff7f0e; margin-bottom: 15px;'>
            <h3 style='color: #ff7f0e; margin: 0 0 10px 0;'>Alpha Vantage API Configuration</h3>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.alphavantage_api_key = st.text_input(
            "Alpha Vantage API Key",
            value=st.session_state.alphavantage_api_key,
            type="password",
            key="av_key_input",
            help="For fetching financial data (income statement, balance sheet, cash flow statement, etc.)"
        )
        st.session_state.alphavantage_base_url = st.text_input(
            "Alpha Vantage Base URL",
            value=st.session_state.alphavantage_base_url,
            key="av_url_input",
            help="Alpha Vantage API base URL (read from ALPHAVANTAGE_BASE_URL environment variable or enter manually)"
        )
        # Display current URL
        st.caption(f"📍 Current Base URL: `{st.session_state.alphavantage_base_url}`")

# Status indicators with enhanced styling
st.markdown("---")
status_col1, status_col2 = st.columns(2)
with status_col1:
    if st.session_state.openai_api_key and len(st.session_state.openai_api_key) > 10:
        st.markdown("""
        <div style='background-color: #d4edda; padding: 12px; border-radius: 8px; border-left: 4px solid #28a745;'>
            <strong style='color: #155724;'>OpenAI API Configured</strong>
            <p style='color: #155724; margin: 5px 0 0 0; font-size: 12px;'>API Key Length: {} characters | Base URL: {}</p>
        </div>
        """.format(len(st.session_state.openai_api_key), st.session_state.openai_base_url), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color: #fff3cd; padding: 12px; border-radius: 8px; border-left: 4px solid #ffc107;'>
            <strong style='color: #856404;'>OpenAI API Key Not Configured or Invalid</strong>
            <p style='color: #856404; margin: 5px 0 0 0; font-size: 12px;'>Please enter a valid API Key in the above input field</p>
        </div>
        """, unsafe_allow_html=True)
with status_col2:
    if st.session_state.alphavantage_api_key and len(st.session_state.alphavantage_api_key) > 10:
        st.markdown("""
        <div style='background-color: #d4edda; padding: 12px; border-radius: 8px; border-left: 4px solid #28a745;'>
            <strong style='color: #155724;'>Alpha Vantage API Configured</strong>
            <p style='color: #155724; margin: 5px 0 0 0; font-size: 12px;'>API Key Length: {} characters | Base URL: {}</p>
        </div>
        """.format(len(st.session_state.alphavantage_api_key), st.session_state.alphavantage_base_url), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background-color: #fff3cd; padding: 12px; border-radius: 8px; border-left: 4px solid #ffc107;'>
            <strong style='color: #856404;'>Alpha Vantage API Key Not Configured or Invalid</strong>
            <p style='color: #856404; margin: 5px 0 0 0; font-size: 12px;'>Please enter a valid API Key in the above input field</p>
        </div>
        """, unsafe_allow_html=True)

# Apply Configuration Button
st.markdown("---")
apply_col1, apply_col2, apply_col3 = st.columns([1, 2, 1])
with apply_col2:
    if st.button("✅ Apply Configuration", type="primary", use_container_width=True, key="apply_config_btn"):
        # Validate configurations
        errors = []
        if not st.session_state.openai_api_key or len(st.session_state.openai_api_key) < 10:
            errors.append("OpenAI API Key is required and must be at least 10 characters")
        if not st.session_state.alphavantage_api_key or len(st.session_state.alphavantage_api_key) < 10:
            errors.append("Alpha Vantage API Key is required and must be at least 10 characters")
        if not st.session_state.alphavantage_base_url:
            errors.append("Alpha Vantage Base URL is required")
        
        if errors:
            for error in errors:
                st.error(f"⚠️ {error}")
        else:
            # Apply configurations to data_ingestion module
            # Try both import methods to ensure all modules get the config
            try:
                import data_ingestion
                data_ingestion.API_KEY = st.session_state.alphavantage_api_key
                data_ingestion.BASE_URL = st.session_state.alphavantage_base_url
            except:
                pass
            
            try:
                try:
                    import data_ingestion as fa_data_ingestion
                except ImportError:
                    from .data_ingestion import data_ingestion as fa_data_ingestion
                fa_data_ingestion.API_KEY = st.session_state.alphavantage_api_key
                fa_data_ingestion.BASE_URL = st.session_state.alphavantage_base_url
            except:
                pass
            
            # Also set environment variables for modules that read from env
            os.environ["ALPHAVANTAGE_API_KEY"] = st.session_state.alphavantage_api_key
            os.environ["ALPHAVANTAGE_BASE_URL"] = st.session_state.alphavantage_base_url
            
            # Update all data_ingestion module references in sys.modules
            import sys
            for module_name in list(sys.modules.keys()):
                if 'data_ingestion' in module_name:
                    module = sys.modules[module_name]
                    if hasattr(module, 'API_KEY'):
                        module.API_KEY = st.session_state.alphavantage_api_key
                    if hasattr(module, 'BASE_URL'):
                        module.BASE_URL = st.session_state.alphavantage_base_url
            
            # Mark configuration as applied
            st.session_state.config_applied = True
            st.session_state.config_applied_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            st.success("✅ Configuration applied successfully! All API settings are now active.")
            st.info(f"📝 Configuration applied at: {st.session_state.config_applied_time}")

# Show configuration status if applied
if st.session_state.get('config_applied', False):
    st.markdown(f"""
    <div style='background-color: #e7f3ff; padding: 10px; border-radius: 8px; border-left: 4px solid #2196F3; margin-top: 10px;'>
        <p style='color: #1976D2; margin: 0; font-size: 14px;'>
            <strong>✓ Configuration Active</strong> | Applied at: {st.session_state.get('config_applied_time', 'N/A')}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Target company search and selection
    st.subheader("Target Company")
    
    # Search mode selection
    search_mode = st.radio(
        "Selection Method",
        ["Search Company", "Direct Input Code"],
        horizontal=True
    )
    
    symbol = None
    company_info = None
    
    if search_mode == "Search Company":
        # Search box
        search_query = st.text_input(
            "Search Company (Enter Stock Symbol)",
            value="",
            placeholder="e.g., NVDA, AAPL, MSFT",
            help="Enter stock symbol to search for company information"
        )
        
        if search_query:
            search_query = search_query.strip().upper()
            with st.spinner(f"Searching for {search_query}..."):
                try:
                    company_info = search_company(search_query)
                    if company_info:
                        symbol = company_info["symbol"]
                        st.success(f"Found: **{company_info['name']}** ({company_info['symbol']})")
                        col1, col2 = st.columns(2)
                        with col1:
                            sector = company_info.get('sector', 'N/A')
                            st.caption(f"**Sector**: {sector}")
                        with col2:
                            industry = company_info.get('industry', 'N/A')
                            st.caption(f"**Industry**: {industry}")
                    else:
                        st.warning(f"Company information not found, will use symbol: {search_query}")
                        symbol = search_query  # Still allow using the entered code
                except Exception as e:
                    st.warning(f"Search failed: {e}, will use symbol: {search_query}")
                    symbol = search_query
        else:
            # Display common stocks list
            st.caption("Quick Selection of Common Stocks:")
            try:
                common_stocks = get_common_stocks()
                stock_options = [f"{s['symbol']} - {s['name']}" for s in common_stocks]
                selected_stock = st.selectbox(
                    "Select Company",
                    [""] + stock_options,
                    label_visibility="visible"
                )
                if selected_stock:
                    symbol = selected_stock.split(" - ")[0]
                    try:
                        company_info = search_company(symbol)
                        if company_info:
                            st.success(f"Selected: **{company_info['name']}** ({company_info['symbol']})")
                    except:
                        pass
            except Exception as e:
                st.warning(f"Failed to load common stocks list: {e}")
    else:
        # Direct input mode
        symbol = st.text_input(
            "Stock Symbol",
            value="NVDA",
            help="Enter stock symbol to analyze (e.g., NVDA, AAPL)"
        )
        if symbol:
            symbol = symbol.strip().upper()
            # Try to get company information
            try:
                company_info = search_company(symbol)
            except:
                pass
    
    st.markdown("---")
    
    # Peer company selection
    st.subheader("Peer Companies")
    
    # Auto-recommend peer companies
    recommended_peers = []
    peer_symbols = None
    
    if symbol:
        # If company info is not available yet, try to get it
        if not company_info:
            try:
                company_info = search_company(symbol)
            except:
                pass
        
        # Try to recommend peer companies
        try:
            if company_info:
                sector = company_info.get("sector", "")
                industry = company_info.get("industry", "")
                
                st.info(f"Recommending peer companies based on sector「{sector}」and industry「{industry}」...")
                recommended_peers = get_peer_recommendations(
                    target_symbol=symbol,
                    target_sector=sector,
                    target_industry=industry
                )
            else:
                # Even without company info, try to recommend (based on common industries)
                st.info(f"Recommending peer companies for {symbol}...")
                recommended_peers = get_peer_recommendations(target_symbol=symbol)
        except Exception as e:
            st.warning(f"Error recommending peer companies: {e}")
            import traceback
            st.code(traceback.format_exc())
            recommended_peers = []
    
    # Display recommended peer companies
    if recommended_peers and len(recommended_peers) > 0:
        if company_info:
            sector = company_info.get('sector', '')
            industry = company_info.get('industry', '')
            st.caption(f"Recommended peer companies based on **{sector}** / **{industry}** (Total: {len(recommended_peers)}):")
        else:
            st.caption(f"Recommended peer companies (Total: {len(recommended_peers)}):")
        
        # Use multiselect for user selection
        selected_peers = st.multiselect(
            "Select Peer Companies (Multiple)",
            options=recommended_peers,
            default=recommended_peers[:5] if len(recommended_peers) >= 5 else recommended_peers,
            help="Select peer companies for comparison, multiple selection allowed"
        )
        
        # Display selected companies
        if selected_peers:
            st.success(f"Selected {len(selected_peers)} companies: {', '.join(selected_peers)}")
            peer_symbols = selected_peers
        else:
            st.info("Please select at least one peer company for comparison, or use manual input below")
        
        st.markdown("---")
        st.caption("Or manually enter peer company symbols:")
    
    # Manual input option (always displayed)
    peers_input = st.text_input(
        "Peer Company Symbols (Optional, comma-separated)", 
        value="",
        help="Enter peer company symbols separated by commas (e.g., AMD,INTC,TSM). If you've selected above, you can add more companies here."
    )
    
    # Merge manual input and selected companies
    manual_peers = [p.strip().upper() for p in peers_input.split(",") if p.strip()] if peers_input else []
    if manual_peers:
        if peer_symbols:
            # Merge and deduplicate
            peer_symbols = list(dict.fromkeys(peer_symbols + manual_peers))
        else:
            peer_symbols = manual_peers
    
    # Display final selected peer companies
    if peer_symbols:
        st.success(f"Will use {len(peer_symbols)} peer companies: {', '.join(peer_symbols)}")
    
    # Analysis years
    years = st.slider("Analysis Years", min_value=3, max_value=7, value=5, help="Select number of years of financial statements to analyze")
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.info("""
    This system will automatically complete:
    1. Financial statement data collection (5 years)
    2. Financial ratio calculation and analysis
    3. DCF and Multiples valuation
    4. Peer comparison and management quality assessment
    5. Catalyst identification (LLM)
    6. Earnings sustainability analysis
    7. Investment memo generation
    """)

# Initialize session state (on page load)
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'analysis_progress' not in st.session_state:
    st.session_state.analysis_progress = {}

# Main interface
if st.button("Start Analysis", type="primary", use_container_width=True):
    if not symbol:
        st.error("Please enter stock symbol")
        st.stop()
    
    # Validate API configurations
    if not st.session_state.openai_api_key or len(st.session_state.openai_api_key) < 10:
        st.error("⚠️ OpenAI API Key 未配置或无效。请在页面顶部配置区域输入有效的 API Key。")
        st.stop()
    
    if not st.session_state.alphavantage_api_key or len(st.session_state.alphavantage_api_key) < 10:
        st.error("⚠️ Alpha Vantage API Key 未配置或无效。请在页面顶部配置区域输入有效的 API Key。")
        st.stop()
    
    if not st.session_state.alphavantage_base_url:
        st.error("⚠️ Alpha Vantage Base URL 未配置。请在页面顶部配置区域输入有效的 Base URL。")
        st.stop()
    
    # Run analysis
    with st.spinner("Analyzing, please wait..."):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update progress
            def update_progress(step, total, message):
                progress = step / total
                progress_bar.progress(progress)
                status_text.text(f"Step {step}/{total}: {message}")
            
            # Update Alpha Vantage API configuration in data_ingestion module
            import data_ingestion
            import sys
            data_ingestion.API_KEY = st.session_state.alphavantage_api_key
            data_ingestion.BASE_URL = st.session_state.alphavantage_base_url
            
            # Update all data_ingestion module references in sys.modules
            for module_name in list(sys.modules.keys()):
                if 'data_ingestion' in module_name:
                    module = sys.modules[module_name]
                    if hasattr(module, 'API_KEY'):
                        module.API_KEY = st.session_state.alphavantage_api_key
                    if hasattr(module, 'BASE_URL'):
                        module.BASE_URL = st.session_state.alphavantage_base_url
            
            # Also set environment variables as fallback
            os.environ["ALPHAVANTAGE_API_KEY"] = st.session_state.alphavantage_api_key
            os.environ["ALPHAVANTAGE_BASE_URL"] = st.session_state.alphavantage_base_url
            
            # Run complete analysis
            update_progress(1, 7, "Collecting data...")
            result = comprehensive_fundamental_analysis(
                symbol=symbol,
                peer_symbols=peer_symbols if peer_symbols else None,
                years=years,
                api_key=st.session_state.openai_api_key,
                base_url=st.session_state.openai_base_url,
                model=st.session_state.get('ai_model', 'gpt-4o-mini'),
                news_limit=50
            )
            
            update_progress(7, 7, "Analysis complete!")
            st.session_state.analysis_result = result
            
            # 保存所有报告到 report/{股票代码}/ 目录
            status_text.text("Saving reports...")
            saved_files = save_all_reports(result, symbol)
            
            if saved_files:
                st.success(f"✅ Analysis complete! Reports saved to report/{symbol.upper()}/")
                st.info(f"📁 Saved {len(saved_files)} report files")
            else:
                st.success("Analysis complete!")
            
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

# Display analysis results
if st.session_state.get('analysis_result') is not None:
    result = st.session_state.analysis_result
    
    # Create tabs
    tabs = st.tabs([
        "Investment Memo",
        "Valuation Analysis",
        "Financial Analysis",
        "Peer Comparison",
        "Earnings Quality",
        "Catalysts",
        "Data Overview"
    ])
    
    # Tab 1: Investment Memo
    with tabs[0]:
        st.header("Investment Memo")
        
        memo = result.get("investment_memo", {})
        
        if memo.get("error"):
            st.error(f"{memo.get('error')}")
        elif memo.get("full_memo"):
            # Investment recommendation cards
            col1, col2, col3 = st.columns(3)
            with col1:
                recommendation = memo.get("recommendation", "N/A")
                if recommendation == "Buy":
                    st.metric("Recommendation", recommendation, delta="Buy", delta_color="normal")
                elif recommendation == "Hold":
                    st.metric("Recommendation", recommendation, delta="Hold", delta_color="off")
                else:
                    st.metric("Recommendation", recommendation, delta="Sell", delta_color="inverse")
            
            with col2:
                target_price = memo.get("target_price", 0)
                st.metric("Target Price", f"${target_price:.2f}")
            
            with col3:
                timeframe = memo.get("investment_timeframe", "N/A")
                st.metric("Investment Timeframe", timeframe)
            
            st.markdown("---")
            
            # Investment thesis
            thesis = memo.get("investment_thesis", [])
            if thesis:
                st.subheader("Investment Thesis")
                if isinstance(thesis, list):
                    for i, point in enumerate(thesis, 1):
                        st.markdown(f"{i}. {point}")
                else:
                    st.markdown(thesis)
                st.markdown("---")
            
            # Full memo
            st.subheader("Full Memo")
            st.markdown(memo.get("full_memo", "No content"))
            
            # Download button
            memo_text = memo.get("full_memo", "")
            if memo_text:
                st.download_button(
                    label="Download Investment Memo",
                    data=memo_text,
                    file_name=f"{symbol}_investment_memo_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
        else:
            st.warning("Investment memo not generated, please check API Key configuration")
    
    # Tab 2: Valuation Analysis
    with tabs[1]:
        st.header("Valuation Analysis")
        
        valuation = result.get("valuation_result", {})
        
        # DCF valuation
        dcf = valuation.get("dcf_valuation", {})
        if dcf:
            st.subheader("DCF Valuation")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Target Price", f"${dcf.get('target_price', 0):.2f}")
            with col2:
                st.metric("Current Price", f"${dcf.get('current_price', 0):.2f}")
            with col3:
                upside = dcf.get("upside_potential", 0)
                st.metric("Upside Potential", f"{upside:.2f}%", delta=f"{upside:.2f}%")
            with col4:
                st.metric("Enterprise Value", f"${dcf.get('enterprise_value', 0)/1e9:.2f}B")
            
            # DCF assumptions
            with st.expander("DCF Model Assumptions", expanded=False):
                assumptions = dcf.get("assumptions", {})
                if assumptions:
                    st.json(assumptions)
        
        # Multiples valuation
        multiples_valuation = valuation.get("multiples_valuation", {})
        if multiples_valuation:
            st.subheader("Comparable Multiples Valuation")
            multiples = multiples_valuation.get("multiples", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                ev_ebitda = multiples.get('ev_ebitda')
                if ev_ebitda:
                    st.metric("EV/EBITDA", f"{ev_ebitda:.2f}x")
                else:
                    st.metric("EV/EBITDA", "N/A")
            with col2:
                pe_ratio = multiples.get('pe_ratio') or multiples.get('pe')
                if pe_ratio:
                    st.metric("P/E", f"{pe_ratio:.2f}x")
                else:
                    st.metric("P/E", "N/A")
            with col3:
                ev_sales = multiples.get('ev_sales')
                if ev_sales:
                    st.metric("EV/Sales", f"{ev_sales:.2f}x")
                else:
                    st.metric("EV/Sales", "N/A")
    
    # Tab 3: Financial Analysis
    with tabs[2]:
        st.header("Financial Analysis")
        
        financial_analysis = result.get("financial_analysis", {})
        
        # Profitability
        profitability = financial_analysis.get("profitability", {})
        if profitability:
            st.subheader("Profitability")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                gross_margin = profitability.get('gross_margin', 0) or 0
                st.metric("Gross Margin", f"{gross_margin:.2f}%")
            with col2:
                operating_margin = profitability.get('operating_margin', 0) or 0
                st.metric("Operating Margin", f"{operating_margin:.2f}%")
            with col3:
                net_margin = profitability.get('net_margin', 0) or 0
                st.metric("Net Margin", f"{net_margin:.2f}%")
            with col4:
                roe = profitability.get('roe', 0) or 0
                st.metric("ROE", f"{roe:.2f}%")
            with col5:
                roic = profitability.get('roic', 0) or 0
                st.metric("ROIC", f"{roic:.2f}%")
        
        # Growth
        growth = financial_analysis.get("growth", {})
        if growth:
            st.subheader("Growth")
            col1, col2, col3 = st.columns(3)
            with col1:
                revenue_cagr = growth.get('revenue_cagr_5y', 0) or growth.get('revenue_cagr', 0) or 0
                st.metric("Revenue 5Y CAGR", f"{revenue_cagr:.2f}%")
            with col2:
                ebitda_cagr = growth.get('ebitda_cagr_5y', 0) or growth.get('ebitda_cagr', 0) or 0
                st.metric("EBITDA 5Y CAGR", f"{ebitda_cagr:.2f}%")
            with col3:
                eps_cagr = growth.get('eps_cagr_5y', 0) or growth.get('eps_cagr', 0) or 0
                st.metric("EPS 5Y CAGR", f"{eps_cagr:.2f}%")
        
        # Leverage
        leverage = financial_analysis.get("leverage", {})
        if leverage:
            st.subheader("Leverage & Solvency")
            col1, col2, col3 = st.columns(3)
            with col1:
                debt_to_equity = leverage.get('debt_to_equity', 0) or 0
                st.metric("Debt/Equity", f"{debt_to_equity:.2f}")
            with col2:
                net_debt_ebitda = leverage.get('net_debt_to_ebitda', 0) or 0
                st.metric("Net Debt/EBITDA", f"{net_debt_ebitda:.2f}")
            with col3:
                interest_coverage = leverage.get('interest_coverage', 0) or 0
                st.metric("Interest Coverage", f"{interest_coverage:.2f}x")
    
    # Tab 4: Peer Comparison
    with tabs[3]:
        st.header("Peer Comparison Analysis")
        
        comprehensive = result.get("comprehensive_analysis", {})
        peer_comp = comprehensive.get("peer_comparison", {})
        
        if peer_comp and peer_comp.get("peer_count", 0) > 0:
            peer_count = peer_comp.get("peer_count", 0)
            target_symbol = peer_comp.get("target_symbol", "N/A")
            
            # Top information bar
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Company", target_symbol)
            with col2:
                st.metric("Peer Companies", f"{peer_count}")
            with col3:
                summary = peer_comp.get("summary", {})
                strengths_count = len(summary.get("strengths", []))
                weaknesses_count = len(summary.get("weaknesses", []))
                st.metric("Strengths/Weaknesses", f"{strengths_count}/{weaknesses_count}")
            
            st.markdown("---")
            
            # LLM peer comparison deep analysis report
            llm_peer_report = peer_comp.get("llm_report", {})
            if llm_peer_report and not llm_peer_report.get("error"):
                st.subheader("LLM Peer Comparison Deep Analysis Report")
                
                # Comprehensive competitive assessment
                comp_assessment = llm_peer_report.get("comprehensive_assessment", {})
                if comp_assessment:
                    with st.expander("Comprehensive Competitive Assessment", expanded=True):
                        overall_position = comp_assessment.get('overall_competitive_position', 'N/A')
                        st.write(f"**Overall Competitive Position**: {overall_position}")
                        
                        competitive_summary = comp_assessment.get('competitive_summary', '')
                        if competitive_summary:
                            st.write(competitive_summary)
                        
                        advantages = comp_assessment.get("core_competitive_advantages", [])
                        if advantages:
                            st.write("**Core Competitive Advantages**:")
                            for advantage in advantages:
                                st.success(f"• {advantage}")
                        
                        weaknesses = comp_assessment.get("key_weaknesses", [])
                        if weaknesses:
                            st.write("**Key Weaknesses**:")
                            for weakness in weaknesses:
                                st.warning(f"• {weakness}")
                
                # Full report
                if llm_peer_report.get("report"):
                    with st.expander("Full Peer Comparison Analysis Report", expanded=False):
                        st.text(llm_peer_report["report"])
                    
                    # Download button
                    report_text = llm_peer_report.get("report", "")
                    if report_text:
                        st.download_button(
                            label="Download Peer Comparison Report",
                            data=report_text,
                            file_name=f"{symbol}_peer_comparison_report_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                
                st.markdown("---")
            
            # 1. Profitability comparison
            profitability_comp = peer_comp.get("profitability_comparison", {})
            if profitability_comp:
                st.subheader("Profitability vs Peers")
                
                # Create profitability comparison table
                profit_data = []
                metric_names = {
                    "gross_margin": "Gross Margin (%)",
                    "operating_margin": "Operating Margin (%)",
                    "net_margin": "Net Margin (%)",
                    "roe": "ROE (%)",
                    "roic": "ROIC (%)"
                }
                
                for metric, name in metric_names.items():
                    if metric in profitability_comp:
                        data = profitability_comp[metric]
                        target_val = data.get("target", 0)
                        peer_median = data.get("peer_median", 0)
                        peer_mean = data.get("peer_mean", 0)
                        peer_min = data.get("peer_min", 0)
                        peer_max = data.get("peer_max", 0)
                        premium_discount = data.get("premium_discount_pct", 0)
                        percentile = data.get("percentile_rank", 0)
                        
                        profit_data.append({
                            "Metric": name,
                            "Target Company": f"{target_val:.2f}",
                            "Peer Median": f"{peer_median:.2f}",
                            "Peer Mean": f"{peer_mean:.2f}",
                            "Peer Min": f"{peer_min:.2f}",
                            "Peer Max": f"{peer_max:.2f}",
                            "Premium/Discount (%)": f"{premium_discount:+.2f}" if premium_discount else "N/A",
                            "Percentile Rank": f"{percentile:.1f}%"
                        })
                
                if profit_data:
                    profit_df = pd.DataFrame(profit_data)
                    st.dataframe(profit_df, use_container_width=True, hide_index=True)
                    
                    # Display key metrics cards
                    st.markdown("#### Key Profitability Metrics")
                    key_metrics = ["gross_margin", "operating_margin", "net_margin", "roic"]
                    cols = st.columns(len(key_metrics))
                    for idx, metric in enumerate(key_metrics):
                        if metric in profitability_comp:
                            data = profitability_comp[metric]
                            target_val = data.get("target", 0)
                            peer_median = data.get("peer_median", 0)
                            premium_discount = data.get("premium_discount_pct", 0)
                            
                            with cols[idx]:
                                delta_color = "normal"
                                delta_text = ""
                                if premium_discount:
                                    if premium_discount > 10:
                                        delta_color = "normal"
                                        delta_text = f"Premium {premium_discount:.1f}%"
                                    elif premium_discount < -10:
                                        delta_color = "inverse"
                                        delta_text = f"Discount {abs(premium_discount):.1f}%"
                                
                                st.metric(
                                    metric_names.get(metric, metric),
                                    f"{target_val:.2f}%",
                                    delta=delta_text if delta_text else None,
                                    delta_color=delta_color
                                )
            
            st.markdown("---")
            
            # 2. Growth comparison
            growth_comp = peer_comp.get("growth_comparison", {})
            if growth_comp:
                st.subheader("Growth vs Peers")
                
                growth_data = []
                growth_metric_names = {
                    "revenue_cagr": "Revenue 5Y CAGR (%)",
                    "ebitda_cagr": "EBITDA 5Y CAGR (%)",
                    "eps_cagr": "EPS 5Y CAGR (%)",
                    "revenue_growth_yoy": "Revenue YoY Growth (%)"
                }
                
                for metric, name in growth_metric_names.items():
                    if metric in growth_comp:
                        data = growth_comp[metric]
                        target_val = data.get("target", 0)
                        peer_median = data.get("peer_median", 0)
                        peer_mean = data.get("peer_mean", 0)
                        premium_discount = data.get("premium_discount_pct", 0)
                        percentile = data.get("percentile_rank", 0)
                        
                        growth_data.append({
                            "Metric": name,
                            "Target Company": f"{target_val:.2f}",
                            "Peer Median": f"{peer_median:.2f}",
                            "Peer Mean": f"{peer_mean:.2f}",
                            "Premium/Discount (%)": f"{premium_discount:+.2f}" if premium_discount else "N/A",
                            "Percentile Rank": f"{percentile:.1f}%"
                        })
                
                if growth_data:
                    growth_df = pd.DataFrame(growth_data)
                    st.dataframe(growth_df, use_container_width=True, hide_index=True)
                    
                    # Display key growth metrics cards
                    st.markdown("#### Key Growth Metrics")
                    key_growth_metrics = ["revenue_cagr", "ebitda_cagr", "eps_cagr"]
                    cols = st.columns(len(key_growth_metrics))
                    for idx, metric in enumerate(key_growth_metrics):
                        if metric in growth_comp:
                            data = growth_comp[metric]
                            target_val = data.get("target", 0)
                            peer_median = data.get("peer_median", 0)
                            premium_discount = data.get("premium_discount_pct", 0)
                            
                            with cols[idx]:
                                delta_color = "normal"
                                delta_text = ""
                                if premium_discount:
                                    if premium_discount > 20:
                                        delta_color = "normal"
                                        delta_text = f"Above Peers {premium_discount:.1f}%"
                                    elif premium_discount < -20:
                                        delta_color = "inverse"
                                        delta_text = f"Below Peers {abs(premium_discount):.1f}%"
                                
                                st.metric(
                                    growth_metric_names.get(metric, metric),
                                    f"{target_val:.2f}%",
                                    delta=delta_text if delta_text else None,
                                    delta_color=delta_color
                                )
            
            st.markdown("---")
            
            # 3. Valuation comparison
            valuation_comp = peer_comp.get("valuation_comparison", {})
            if valuation_comp:
                st.subheader("Valuation vs Peers")
                
                valuation_data = []
                valuation_metric_names = {
                    "ev_ebitda": "EV/EBITDA (x)",
                    "pe": "P/E (x)",
                    "ev_sales": "EV/Sales (x)"
                }
                
                for metric, name in valuation_metric_names.items():
                    if metric in valuation_comp:
                        data = valuation_comp[metric]
                        target_val = data.get("target", 0)
                        peer_median = data.get("peer_median", 0)
                        peer_mean = data.get("peer_mean", 0)
                        premium_discount = data.get("premium_discount_pct", 0)
                        percentile = data.get("percentile_rank", 0)
                        
                        valuation_data.append({
                            "Metric": name,
                            "Target Company": f"{target_val:.2f}",
                            "Peer Median": f"{peer_median:.2f}",
                            "Peer Mean": f"{peer_mean:.2f}",
                            "Premium/Discount (%)": f"{premium_discount:+.2f}" if premium_discount else "N/A",
                            "Percentile Rank": f"{percentile:.1f}%"
                        })
                
                if valuation_data:
                    valuation_df = pd.DataFrame(valuation_data)
                    st.dataframe(valuation_df, use_container_width=True, hide_index=True)
                    
                    # Display key valuation metrics cards
                    st.markdown("#### Key Valuation Metrics")
                    key_valuation_metrics = ["ev_ebitda", "pe", "ev_sales"]
                    cols = st.columns(len(key_valuation_metrics))
                    for idx, metric in enumerate(key_valuation_metrics):
                        if metric in valuation_comp:
                            data = valuation_comp[metric]
                            target_val = data.get("target", 0)
                            peer_median = data.get("peer_median", 0)
                            premium_discount = data.get("premium_discount_pct", 0)
                            
                            with cols[idx]:
                                delta_color = "normal"
                                delta_text = ""
                                if premium_discount:
                                    if premium_discount < -10:
                                        delta_color = "normal"  # Discount is good
                                        delta_text = f"Discount {abs(premium_discount):.1f}%"
                                    elif premium_discount > 20:
                                        delta_color = "inverse"  # High premium is bad
                                        delta_text = f"Premium {premium_discount:.1f}%"
                                
                                st.metric(
                                    valuation_metric_names.get(metric, metric),
                                    f"{target_val:.2f}x",
                                    delta=delta_text if delta_text else None,
                                    delta_color=delta_color
                                )
            
            st.markdown("---")
            
            # 4. Financial structure comparison
            structure_comp = peer_comp.get("financial_structure_comparison", {})
            if structure_comp:
                st.subheader("Financial Structure vs Peers")
                
                structure_data = []
                structure_metric_names = {
                    "debt_to_equity": "Debt/Equity",
                    "debt_to_assets": "Debt/Assets",
                    "current_ratio": "Current Ratio",
                    "interest_coverage": "Interest Coverage (x)"
                }
                
                for metric, name in structure_metric_names.items():
                    if metric in structure_comp:
                        data = structure_comp[metric]
                        target_val = data.get("target", 0)
                        peer_median = data.get("peer_median", 0)
                        peer_mean = data.get("peer_mean", 0)
                        premium_discount = data.get("premium_discount_pct", 0)
                        percentile = data.get("percentile_rank", 0)
                        
                        structure_data.append({
                            "Metric": name,
                            "Target Company": f"{target_val:.2f}",
                            "Peer Median": f"{peer_median:.2f}",
                            "Peer Mean": f"{peer_mean:.2f}",
                            "Premium/Discount (%)": f"{premium_discount:+.2f}" if premium_discount else "N/A",
                            "Percentile Rank": f"{percentile:.1f}%"
                        })
                
                if structure_data:
                    structure_df = pd.DataFrame(structure_data)
                    st.dataframe(structure_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            # 5. Comprehensive assessment summary (prioritize LLM analysis results)
            llm_peer_report = peer_comp.get("llm_report", {})
            summary = peer_comp.get("summary", {})
            
            if llm_peer_report and not llm_peer_report.get("error"):
                # Use LLM analysis results
                st.subheader("Comprehensive Assessment Summary (LLM Deep Analysis)")
                
                comp_assessment = llm_peer_report.get("comprehensive_assessment", {})
                investment_insights = llm_peer_report.get("investment_insights", {})
                
                if comp_assessment:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Core Competitive Advantages (LLM Analysis)")
                        advantages = comp_assessment.get("core_competitive_advantages", [])
                        if advantages:
                            for advantage in advantages:
                                st.success(f"• {advantage}")
                        else:
                            st.info("No significant advantages")
                    
                    with col2:
                        st.markdown("#### Key Weaknesses (LLM Analysis)")
                        weaknesses = comp_assessment.get("key_weaknesses", [])
                        if weaknesses:
                            for weakness in weaknesses:
                                st.warning(f"• {weakness}")
                        else:
                            st.info("No significant weaknesses")
                    
                    # Comprehensive competitive assessment
                    competitive_summary = comp_assessment.get('competitive_summary', '')
                    if competitive_summary:
                        st.markdown("#### Comprehensive Competitive Assessment")
                        st.write(competitive_summary)
                    
                    # Overall competitive position
                    overall_position = comp_assessment.get('overall_competitive_position', 'N/A')
                    if overall_position != 'N/A':
                        st.markdown("#### Overall Competitive Position")
                        st.metric("Competitive Position", overall_position)
                
                # Investment insights (if available)
                if investment_insights:
                    st.markdown("---")
                    st.markdown("#### Investment Insights (LLM Analysis)")
                    
                    key_insights = investment_insights.get("key_insights", [])
                    if key_insights:
                        st.markdown("**Key Insights**:")
                        for insight in key_insights:
                            st.info(f"• {insight}")
                    
                    highlights = investment_insights.get("investment_highlights", [])
                    if highlights:
                        st.markdown("**Investment Highlights**:")
                        for highlight in highlights:
                            st.success(f"• {highlight}")
                    
                    risk_points = investment_insights.get("risk_points", [])
                    if risk_points:
                        st.markdown("**Risk Points**:")
                        for risk in risk_points:
                            st.warning(f"• {risk}")
            elif summary:
                # Fallback to rule-based summary
                st.subheader("Comprehensive Assessment Summary")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Relative Strengths")
                    strengths = summary.get("strengths", [])
                    if strengths:
                        for strength in strengths:
                            st.success(f"• {strength}")
                    else:
                        st.info("No significant strengths")
                
                with col2:
                    st.markdown("#### Relative Weaknesses")
                    weaknesses = summary.get("weaknesses", [])
                    if weaknesses:
                        for weakness in weaknesses:
                            st.warning(f"• {weakness}")
                    else:
                        st.info("No significant weaknesses")
                
                # Relative position
                relative_position = summary.get("relative_position", {})
                if relative_position:
                    st.markdown("#### Relative Position")
                    pos_data = []
                    for key, value in relative_position.items():
                        if value is not None:
                            pos_data.append({
                                "Metric": key.replace("_percentile", "").replace("_", " ").title(),
                                "Percentile Rank": f"{value:.1f}%"
                            })
                    if pos_data:
                        pos_df = pd.DataFrame(pos_data)
                        st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No peer comparison performed. Please enter peer company symbols in the sidebar (comma-separated, e.g., AAPL,MSFT,GOOGL)")
    
    # Tab 5: Earnings Quality
    with tabs[4]:
        st.header("Earnings Sustainability Analysis")
        
        earnings_quality = result.get("earnings_quality", {})
        
        # LLM earnings analysis report
        llm_report = earnings_quality.get("llm_report", {})
        if llm_report and not llm_report.get("error"):
            st.subheader("LLM Earnings Analysis Report")
            
            # Executive summary
            exec_summary = llm_report.get("executive_summary", {})
            if exec_summary:
                with st.expander("Executive Summary", expanded=True):
                    if exec_summary.get("core_conclusion"):
                        st.write(exec_summary["core_conclusion"])
                    key_findings = exec_summary.get("key_findings", [])
                    if key_findings:
                        st.write("**Key Findings:**")
                        for finding in key_findings:
                            st.write(f"• {finding}")
            
            # Full report
            if llm_report.get("report"):
                with st.expander("Full Earnings Analysis Report", expanded=False):
                    st.text(llm_report["report"])
                
                # Download button
                report_text = llm_report.get("report", "")
                if report_text:
                    st.download_button(
                        label="Download Earnings Analysis Report",
                        data=report_text,
                        file_name=f"{symbol}_earnings_analysis_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
            
            st.markdown("---")
        
        # Analysis by dimension
        # 1. Cash vs Profit
        cash_profit = earnings_quality.get("cash_vs_profit", {})
        if cash_profit:
            st.subheader("Cash vs Profit")
            cfo_to_ni_dict = cash_profit.get("cfo_to_net_income", {})
            if isinstance(cfo_to_ni_dict, dict):
                cfo_ni = cfo_to_ni_dict.get("latest")
                if cfo_ni is not None:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("CFO/Net Income", f"{cfo_ni:.2f}")
                    with col2:
                        avg = cfo_to_ni_dict.get("average")
                        if avg:
                            st.metric("Historical Average", f"{avg:.2f}")
                    with col3:
                        trend = cash_profit.get("trend_analysis", {}).get("cfo_ni_trend", "N/A")
                        st.metric("Trend", trend)
            st.info(cash_profit.get("assessment", "N/A"))
        
        # 2. Accrual Quality
        accrual = earnings_quality.get("accrual_quality", {})
        if accrual:
            st.subheader("Accrual Quality")
            accruals_dict = accrual.get("accruals", {})
            if isinstance(accruals_dict, dict):
                accruals = accruals_dict.get("latest")
                if accruals is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accruals", f"${accruals/1e9:.2f}B")
                    with col2:
                        trend = accrual.get("trend_analysis", {}).get("accrual_trend", "N/A")
                        st.metric("Trend", trend)
            st.info(accrual.get("assessment", "N/A"))
        
        # 3. Profit Volatility
        volatility = earnings_quality.get("profit_volatility", {})
        if volatility:
            st.subheader("Profit Volatility")
            profit_vol_dict = volatility.get("profit_volatility", {})
            margin_stability = volatility.get("margin_stability", {})
            if isinstance(profit_vol_dict, dict):
                cv = profit_vol_dict.get("coefficient_of_variation")
                if cv is not None:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Margin CV", f"{cv:.1f}%")
                    with col2:
                        stability = profit_vol_dict.get("stability", "N/A")
                        st.metric("Stability", stability)
                    with col3:
                        latest_margin = margin_stability.get("latest")
                        if latest_margin:
                            st.metric("Latest Net Margin", f"{latest_margin:.2f}%")
            st.info(volatility.get("assessment", "N/A"))
        
        # 4. One-time Items
        one_time = earnings_quality.get("one_time_items", {})
        if one_time:
            st.subheader("One-time Items Dependency")
            special_items_dict = one_time.get("special_items_ratio", {})
            if isinstance(special_items_dict, dict):
                one_time_ratio = special_items_dict.get("ratio_to_net_income")
                if one_time_ratio is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("One-time Items Ratio", f"{one_time_ratio:.1f}%")
                    with col2:
                        total_amount = special_items_dict.get("total_one_time_items", 0)
                        st.metric("Total One-time Items", f"${total_amount/1e9:.2f}B")
            st.info(one_time.get("assessment", "N/A"))
        
        # 5. Capital Structure
        capital = earnings_quality.get("capital_structure_support", {})
        if capital:
            st.subheader("Capital Structure Support")
            leverage_trend = capital.get("leverage_trend", {})
            interest_burden = capital.get("interest_burden", {})
            if leverage_trend.get("latest_debt_to_equity"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Debt/Equity", f"{leverage_trend['latest_debt_to_equity']:.2f}")
                with col2:
                    trend = leverage_trend.get("trend", "N/A")
                    st.metric("Trend", trend)
                with col3:
                    interest_cov = interest_burden.get("interest_coverage")
                    if interest_cov:
                        st.metric("Interest Coverage", f"{interest_cov:.2f}x")
            st.info(capital.get("assessment", "N/A"))
        
        # Detailed metrics table
        summary = earnings_quality.get("summary", {})
        if summary.get("detailed_metrics"):
            st.markdown("---")
            st.subheader("Detailed Metrics")
            detailed_metrics = summary["detailed_metrics"]
            
            # Create metrics table
            metrics_data = []
            if detailed_metrics.get("cfo_ni_ratio"):
                cfo_ni = detailed_metrics["cfo_ni_ratio"]
                metrics_data.append({
                    "Metric": "CFO/Net Income",
                    "Latest": f"{cfo_ni.get('latest', 0):.2f}" if cfo_ni.get('latest') else "N/A",
                    "Average": f"{cfo_ni.get('average', 0):.2f}" if cfo_ni.get('average') else "N/A",
                    "Trend": cfo_ni.get('trend', 'N/A')
                })
            
            if detailed_metrics.get("profit_volatility"):
                vol = detailed_metrics["profit_volatility"]
                metrics_data.append({
                    "Metric": "Margin CV",
                    "Latest": f"{vol.get('coefficient_of_variation', 0):.1f}%" if vol.get('coefficient_of_variation') else "N/A",
                    "Stability": vol.get('stability', 'N/A'),
                    "Trend": vol.get('trend', 'N/A')
                })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Tab 6: Catalysts
    with tabs[5]:
        st.header("Catalysts")
        
        qualitative = result.get("qualitative_analysis", {})
        catalysts = qualitative.get("catalyst_analysis", {})
        news_data = qualitative.get("news_data", {})
        
        # News cards display (priority display)
        if news_data and news_data.get("news_count", 0) > 0:
            st.subheader("Related News (for Catalyst Analysis)")
            news_items = news_data.get("news_items", [])
            news_count = news_data.get("news_count", 0)
            
            st.info(f"Retrieved {news_count} news items. Below is the news content used for LLM catalyst analysis")
            st.markdown("---")
            
            # Use grid layout to display news cards (3 per row)
            if news_items:
                # Calculate number of rows needed
                num_cols = 3
                num_rows = (len(news_items) + num_cols - 1) // num_cols
                
                for row in range(num_rows):
                    cols = st.columns(num_cols)
                    for col_idx in range(num_cols):
                        item_idx = row * num_cols + col_idx
                        if item_idx < len(news_items):
                            item = news_items[item_idx]
                            with cols[col_idx]:
                                # Create news card (using expander for scalability)
                                title = item.get('title', 'No Title')
                                time_published = item.get('time_published', 'N/A')
                                summary = item.get('summary', '') or item.get('content', '') or item.get('description', '')
                                url = item.get('url', '') or item.get('link', '')
                                source = item.get('source', 'N/A')
                                sentiment_score = item.get('overall_sentiment_score', None)
                                
                                # Format time
                                time_display = time_published
                                if time_published and time_published != 'N/A':
                                    try:
                                        # Try to parse time format (may be "20240101T120000" format)
                                        if 'T' in time_published:
                                            time_display = time_published.split('T')[0]
                                    except:
                                        pass
                                
                                # Sentiment label
                                sentiment_label = ""
                                sentiment_color = "normal"
                                if sentiment_score is not None:
                                    try:
                                        score = float(sentiment_score)
                                        if score > 0.1:
                                            sentiment_label = "Positive"
                                            sentiment_color = "normal"
                                        elif score < -0.1:
                                            sentiment_label = "Negative"
                                            sentiment_color = "inverse"
                                        else:
                                            sentiment_label = "Neutral"
                                            sentiment_color = "off"
                                    except:
                                        pass
                                
                                # Create scalable news card
                                with st.expander(
                                    f"**{title[:60]}{'...' if len(title) > 60 else ''}**",
                                    expanded=False
                                ):
                                    # News metadata
                                    col_info1, col_info2 = st.columns(2)
                                    with col_info1:
                                        st.caption(f"{time_display}")
                                    with col_info2:
                                        if sentiment_label:
                                            st.caption(f"{sentiment_label}")
                                    
                                    if source and source != 'N/A':
                                        st.caption(f"Source: {source}")
                                    
                                    # News summary
                                    if summary:
                                        st.write("**Summary:**")
                                        st.write(summary[:500] + "..." if len(summary) > 500 else summary)
                                    else:
                                        st.write("*No summary available*")
                                    
                                    # Link
                                    if url:
                                        st.markdown(f"[View Original]({url})")
                                    
                                    # Sentiment score (if available)
                                    if sentiment_score is not None:
                                        try:
                                            score = float(sentiment_score)
                                            st.metric("Sentiment Score", f"{score:.3f}", delta=sentiment_label if sentiment_label else None)
                                        except:
                                            pass
                
                st.markdown("---")
        
        if catalysts:
            # Catalyst summary (LLM analysis)
            cat_summary = catalysts.get("summary", {})
            if cat_summary:
                st.subheader("Catalyst Summary (LLM Analysis)")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Catalysts", cat_summary.get('total_catalysts', 0))
                with col2:
                    st.metric("High Impact", cat_summary.get('high_impact_count', 0))
                with col3:
                    st.metric("Short-term", cat_summary.get('short_term_count', 0))
                with col4:
                    st.metric("Long-term", cat_summary.get('long_term_count', 0))
                
                st.markdown("---")
            
            # Key catalysts (LLM identified high impact, high confidence)
            key_catalysts = cat_summary.get("key_catalysts", []) if cat_summary else []
            if key_catalysts:
                st.subheader("Key Catalysts (LLM Analysis - High Impact, High Confidence)")
                for i, cat in enumerate(key_catalysts[:5], 1):
                    with st.expander(f"{i}. {cat.get('type', 'N/A')} - {cat.get('impact', 'N/A')} Impact | {cat.get('confidence', 'N/A')} Confidence", expanded=True):
                        st.write(f"**Description**: {cat.get('description', 'N/A')}")
                        st.write(f"**Impact Level**: {cat.get('impact', 'N/A')}")
                        st.write(f"**Timeframe**: {cat.get('timeframe', 'N/A')}")
                        if cat.get('confidence'):
                            st.write(f"**Confidence**: {cat.get('confidence', 'N/A')}")
                        if cat.get('expected_timing'):
                            st.write(f"**Expected Timing**: {cat.get('expected_timing', 'N/A')}")
                        if cat.get('evidence'):
                            st.write(f"**Evidence**: {cat.get('evidence', 'N/A')[:300]}...")
                
                st.markdown("---")
            
            # Display all catalysts by category
            st.subheader("All Catalysts Details (LLM Analysis)")
            
            # New products/markets
            product_catalysts = catalysts.get("product_market_catalysts", [])
            if product_catalysts:
                st.markdown("#### New Products/Market Opportunities")
                for i, cat in enumerate(product_catalysts[:8], 1):
                    impact = cat.get('impact', 'N/A')
                    confidence = cat.get('confidence', 'N/A')
                    with st.expander(f"{i}. {cat.get('type', 'N/A')} - {impact} Impact | {confidence} Confidence", expanded=False):
                        st.write(f"**Description**: {cat.get('description', 'N/A')}")
                        st.write(f"**Impact Level**: {impact}")
                        st.write(f"**Timeframe**: {cat.get('timeframe', 'N/A')}")
                        if confidence:
                            st.write(f"**Confidence**: {confidence}")
                        if cat.get('expected_timing'):
                            st.write(f"**Expected Timing**: {cat.get('expected_timing', 'N/A')}")
                        if cat.get('evidence'):
                            st.write(f"**Evidence**: {cat.get('evidence', 'N/A')[:300]}...")
            
            # Cost improvements
            cost_catalysts = catalysts.get("cost_improvement_catalysts", [])
            if cost_catalysts:
                st.markdown("#### Cost Improvement Opportunities")
                for i, cat in enumerate(cost_catalysts[:8], 1):
                    impact = cat.get('impact', 'N/A')
                    confidence = cat.get('confidence', 'N/A')
                    with st.expander(f"{i}. {cat.get('type', 'N/A')} - {impact} Impact | {confidence} Confidence", expanded=False):
                        st.write(f"**Description**: {cat.get('description', 'N/A')}")
                        st.write(f"**Impact Level**: {impact}")
                        st.write(f"**Timeframe**: {cat.get('timeframe', 'N/A')}")
                        if confidence:
                            st.write(f"**Confidence**: {confidence}")
                        if cat.get('expected_timing'):
                            st.write(f"**Expected Timing**: {cat.get('expected_timing', 'N/A')}")
                        if cat.get('evidence'):
                            st.write(f"**Evidence**: {cat.get('evidence', 'N/A')[:300]}...")
            
            # Industry cycle
            cycle_catalysts = catalysts.get("industry_cycle_catalysts", [])
            if cycle_catalysts:
                st.markdown("#### Industry Cycle Changes")
                for i, cat in enumerate(cycle_catalysts[:8], 1):
                    impact = cat.get('impact', 'N/A')
                    confidence = cat.get('confidence', 'N/A')
                    with st.expander(f"{i}. {cat.get('type', 'N/A')} - {impact} Impact | {confidence} Confidence", expanded=False):
                        st.write(f"**Description**: {cat.get('description', 'N/A')}")
                        st.write(f"**Impact Level**: {impact}")
                        st.write(f"**Timeframe**: {cat.get('timeframe', 'N/A')}")
                        if confidence:
                            st.write(f"**Confidence**: {confidence}")
                        if cat.get('expected_timing'):
                            st.write(f"**Expected Timing**: {cat.get('expected_timing', 'N/A')}")
                        if cat.get('evidence'):
                            st.write(f"**Evidence**: {cat.get('evidence', 'N/A')[:300]}...")
            
            # Policy/Interest rates
            policy_catalysts = catalysts.get("policy_rate_catalysts", [])
            if policy_catalysts:
                st.markdown("#### Policy/Interest Rate Changes")
                for i, cat in enumerate(policy_catalysts[:8], 1):
                    impact = cat.get('impact', 'N/A')
                    confidence = cat.get('confidence', 'N/A')
                    with st.expander(f"{i}. {cat.get('type', 'N/A')} - {impact} Impact | {confidence} Confidence", expanded=False):
                        st.write(f"**Description**: {cat.get('description', 'N/A')}")
                        st.write(f"**Impact Level**: {impact}")
                        st.write(f"**Timeframe**: {cat.get('timeframe', 'N/A')}")
                        if confidence:
                            st.write(f"**Confidence**: {confidence}")
                        if cat.get('expected_timing'):
                            st.write(f"**Expected Timing**: {cat.get('expected_timing', 'N/A')}")
                        if cat.get('evidence'):
                            st.write(f"**Evidence**: {cat.get('evidence', 'N/A')[:300]}...")
            
            # Display raw LLM response (optional)
            if catalysts.get("raw_llm_response"):
                with st.expander("View LLM Raw Analysis Response", expanded=False):
                    st.text(catalysts["raw_llm_response"][:2000] + "..." if len(catalysts["raw_llm_response"]) > 2000 else catalysts["raw_llm_response"])
        else:
            st.warning("No catalysts identified. Please check API Key configuration and news data")
    
    # Tab 7: Data Overview
    with tabs[6]:
        st.header("Data Overview")
        
        # Get financial data
        financial_data = result.get("financial_data", {})
        
        # 1. Three major financial statements
        st.subheader("Three Major Financial Statements")
        
        tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow Statement"])
        
        with tab1:
            income_annual = financial_data.get("income_statement", {}).get("annual", pd.DataFrame())
            if not income_annual.empty:
                # Format numeric display
                income_display = income_annual.copy()
                # Convert values to millions of dollars
                numeric_cols = income_display.select_dtypes(include=[float, int]).columns
                for col in numeric_cols:
                    income_display[col] = income_display[col].apply(
                        lambda x: f"${x/1e6:.2f}M" if abs(x) >= 1e6 else (f"${x/1e3:.2f}K" if abs(x) >= 1e3 else f"${x:.2f}")
                    )
                
                st.dataframe(
                    income_display,
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = income_annual.to_csv()
                st.download_button(
                    label="Download Income Statement (CSV)",
                    data=csv,
                    file_name=f"{symbol}_income_statement_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Income statement data not available")
        
        with tab2:
            balance_annual = financial_data.get("balance_sheet", {}).get("annual", pd.DataFrame())
            if not balance_annual.empty:
                # Format numeric display
                balance_display = balance_annual.copy()
                numeric_cols = balance_display.select_dtypes(include=[float, int]).columns
                for col in numeric_cols:
                    balance_display[col] = balance_display[col].apply(
                        lambda x: f"${x/1e6:.2f}M" if abs(x) >= 1e6 else (f"${x/1e3:.2f}K" if abs(x) >= 1e3 else f"${x:.2f}")
                    )
                
                st.dataframe(
                    balance_display,
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = balance_annual.to_csv()
                st.download_button(
                    label="Download Balance Sheet (CSV)",
                    data=csv,
                    file_name=f"{symbol}_balance_sheet_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Balance sheet data not available")
        
        with tab3:
            cashflow_annual = financial_data.get("cash_flow", {}).get("annual", pd.DataFrame())
            if not cashflow_annual.empty:
                # Format numeric display
                cashflow_display = cashflow_annual.copy()
                numeric_cols = cashflow_display.select_dtypes(include=[float, int]).columns
                for col in numeric_cols:
                    cashflow_display[col] = cashflow_display[col].apply(
                        lambda x: f"${x/1e6:.2f}M" if abs(x) >= 1e6 else (f"${x/1e3:.2f}K" if abs(x) >= 1e3 else f"${x:.2f}")
                    )
                
                st.dataframe(
                    cashflow_display,
                    use_container_width=True,
                    height=400
                )
                
                # Download button
                csv = cashflow_annual.to_csv()
                st.download_button(
                    label="Download Cash Flow Statement (CSV)",
                    data=csv,
                    file_name=f"{symbol}_cashflow_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("Cash flow statement data not available")
        
        st.markdown("---")
        
        # 2. Financial ratios summary table
        st.subheader("Financial Ratios Summary")
        
        financial_analysis = result.get("financial_analysis", {})
        
        # Create financial ratios summary table
        ratios_data = []
        
        # Profitability
        profitability = financial_analysis.get("profitability", {})
        if profitability:
            ratios_data.append({
                "Category": "Profitability",
                "Metric": "Gross Margin",
                "Value": f"{profitability.get('gross_margin', 0) or 0:.2f}%",
                "Description": "Gross Margin"
            })
            ratios_data.append({
                "Category": "Profitability",
                "Metric": "Operating Margin",
                "Value": f"{profitability.get('operating_margin', 0) or 0:.2f}%",
                "Description": "Operating Margin"
            })
            ratios_data.append({
                "Category": "Profitability",
                "Metric": "Net Margin",
                "Value": f"{profitability.get('net_margin', 0) or 0:.2f}%",
                "Description": "Net Margin"
            })
            ratios_data.append({
                "Category": "Profitability",
                "Metric": "ROE",
                "Value": f"{profitability.get('roe', 0) or 0:.2f}%",
                "Description": "Return on Equity"
            })
            ratios_data.append({
                "Category": "Profitability",
                "Metric": "ROIC",
                "Value": f"{profitability.get('roic', 0) or 0:.2f}%",
                "Description": "Return on Invested Capital"
            })
        
        # Growth
        growth = financial_analysis.get("growth", {})
        if growth:
            revenue_cagr = growth.get('revenue_cagr_5y', 0) or growth.get('revenue_cagr', 0) or 0
            ebitda_cagr = growth.get('ebitda_cagr_5y', 0) or growth.get('ebitda_cagr', 0) or 0
            eps_cagr = growth.get('eps_cagr_5y', 0) or growth.get('eps_cagr', 0) or 0
            
            ratios_data.append({
                "Category": "Growth",
                "Metric": "Revenue 5Y CAGR",
                "Value": f"{revenue_cagr:.2f}%",
                "Description": "Revenue CAGR (5y)"
            })
            ratios_data.append({
                "Category": "Growth",
                "Metric": "EBITDA 5Y CAGR",
                "Value": f"{ebitda_cagr:.2f}%",
                "Description": "EBITDA CAGR (5y)"
            })
            ratios_data.append({
                "Category": "Growth",
                "Metric": "EPS 5Y CAGR",
                "Value": f"{eps_cagr:.2f}%",
                "Description": "EPS CAGR (5y)"
            })
        
        # Leverage & Solvency
        leverage = financial_analysis.get("leverage", {})
        if leverage:
            ratios_data.append({
                "Category": "Leverage & Solvency",
                "Metric": "Debt/Equity",
                "Value": f"{leverage.get('debt_to_equity', 0) or 0:.2f}",
                "Description": "Debt to Equity"
            })
            ratios_data.append({
                "Category": "Leverage & Solvency",
                "Metric": "Net Debt/EBITDA",
                "Value": f"{leverage.get('net_debt_to_ebitda', 0) or 0:.2f}",
                "Description": "Net Debt/EBITDA"
            })
            ratios_data.append({
                "Category": "Leverage & Solvency",
                "Metric": "Interest Coverage",
                "Value": f"{leverage.get('interest_coverage', 0) or 0:.2f}x",
                "Description": "Interest Coverage"
            })
            ratios_data.append({
                "Category": "Leverage & Solvency",
                "Metric": "Current Ratio",
                "Value": f"{leverage.get('current_ratio', 0) or 0:.2f}",
                "Description": "Current Ratio"
            })
        
        # Operating Efficiency
        efficiency = financial_analysis.get("efficiency", {})
        if efficiency:
            ratios_data.append({
                "Category": "Operating Efficiency",
                "Metric": "Asset Turnover",
                "Value": f"{efficiency.get('asset_turnover', 0) or 0:.2f}",
                "Description": "Asset Turnover"
            })
            ratios_data.append({
                "Category": "Operating Efficiency",
                "Metric": "Inventory Turnover",
                "Value": f"{efficiency.get('inventory_turnover', 0) or 0:.2f}",
                "Description": "Inventory Turnover"
            })
            ratios_data.append({
                "Category": "Operating Efficiency",
                "Metric": "Cash Conversion Cycle",
                "Value": f"{efficiency.get('cash_conversion_cycle', 0) or 0:.2f} days",
                "Description": "Cash Conversion Cycle"
            })
        
        if ratios_data:
            ratios_df = pd.DataFrame(ratios_data)
            st.dataframe(
                ratios_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Download button
            csv = ratios_df.to_csv(index=False)
            st.download_button(
                label="Download Financial Ratios Summary (CSV)",
                data=csv,
                file_name=f"{symbol}_financial_ratios_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # 3. Valuation data summary
        st.subheader("Valuation Data Summary")
        
        valuation = result.get("valuation_result", {})
        valuation_data = []
        
        # DCF valuation
        dcf = valuation.get("dcf_valuation", {})
        if dcf:
            valuation_data.append({
                "Valuation Method": "DCF",
                "Metric": "Target Price",
                "Value": f"${dcf.get('target_price', 0):.2f}",
                "Description": "Target Price per Share"
            })
            valuation_data.append({
                "Valuation Method": "DCF",
                "Metric": "Current Price",
                "Value": f"${dcf.get('current_price', 0):.2f}",
                "Description": "Current Price"
            })
            valuation_data.append({
                "Valuation Method": "DCF",
                "Metric": "Upside Potential",
                "Value": f"{dcf.get('upside_potential', 0) or 0:.2f}%",
                "Description": "Upside Potential"
            })
            valuation_data.append({
                "Valuation Method": "DCF",
                "Metric": "Enterprise Value",
                "Value": f"${dcf.get('enterprise_value', 0)/1e9:.2f}B",
                "Description": "Enterprise Value"
            })
        
        # Multiples valuation
        multiples = valuation.get("multiples_valuation", {}).get("multiples", {})
        if multiples:
            if multiples.get("ev_ebitda"):
                valuation_data.append({
                    "Valuation Method": "Multiples",
                    "Metric": "EV/EBITDA",
                    "Value": f"{multiples.get('ev_ebitda'):.2f}x",
                    "Description": "Enterprise Value / EBITDA"
                })
            if multiples.get("pe") or multiples.get("pe_ratio"):
                pe = multiples.get("pe") or multiples.get("pe_ratio")
                valuation_data.append({
                    "Valuation Method": "Multiples",
                    "Metric": "P/E",
                    "Value": f"{pe:.2f}x",
                    "Description": "Price to Earnings"
                })
            if multiples.get("ev_sales"):
                valuation_data.append({
                    "Valuation Method": "Multiples",
                    "Metric": "EV/Sales",
                    "Value": f"{multiples.get('ev_sales'):.2f}x",
                    "Description": "Enterprise Value / Sales"
                })
        
        if valuation_data:
            valuation_df = pd.DataFrame(valuation_data)
            st.dataframe(
                valuation_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            csv = valuation_df.to_csv(index=False)
            st.download_button(
                label="Download Valuation Data (CSV)",
                data=csv,
                file_name=f"{symbol}_valuation_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        st.markdown("---")
        
        # 4. Company basic information
        st.subheader("Company Basic Information")
        
        overview = financial_data.get("overview", {})
        if overview:
            info_data = []
            info_fields = {
                "Name": "Company Name",
                "Industry": "Industry",
                "Sector": "Sector",
                "MarketCapitalization": "Market Cap",
                "SharesOutstanding": "Shares Outstanding",
                "PERatio": "P/E Ratio",
                "DividendYield": "Dividend Yield",
                "52WeekHigh": "52 Week High",
                "52WeekLow": "52 Week Low",
                "Beta": "Beta",
                "Exchange": "Exchange"
            }
            
            for key, label in info_fields.items():
                value = overview.get(key, "N/A")
                if value and value != "None":
                    if key == "MarketCapitalization" and isinstance(value, (int, float)):
                        value = f"${value/1e9:.2f}B"
                    elif key == "SharesOutstanding" and isinstance(value, (int, float)):
                        value = f"{value/1e9:.2f}B"
                    info_data.append({
                        "Item": label,
                        "Value": str(value),
                        "Field": key
                    })
            
            if info_data:
                info_df = pd.DataFrame(info_data)
                st.dataframe(
                    info_df[["Item", "Value"]],
                    use_container_width=True,
                    hide_index=True
                )
        
        st.markdown("---")
        
        # 5. Raw data view (JSON)
        with st.expander("View Raw Analysis Data (JSON)", expanded=False):
            # Remove non-serializable objects
            result_clean = {}
            for key, value in result.items():
                if key not in ["financial_data"]:  # Skip data containing DataFrames
                    result_clean[key] = value
            st.json(result_clean)
        
        # Download complete results
        result_json = json.dumps(result_clean, ensure_ascii=False, indent=2, default=str)
        st.download_button(
            label="Download Complete Analysis Results (JSON)",
            data=result_json,
            file_name=f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Fundamental Analysis Agent v0.1.0 | AI-Powered Intelligent Investment Analysis</p>
</div>
""", unsafe_allow_html=True)

