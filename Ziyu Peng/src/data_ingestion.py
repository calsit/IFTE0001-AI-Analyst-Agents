"""
Data Ingestion Module for Fundamental Analysis

Features:
1. Fetch financial statement data from Alpha Vantage API (5-7 years)
2. Process three major statements: Income Statement, Balance Sheet, Cash Flow Statement
3. Convert data to structured DataFrames
4. Save data for subsequent analysis

We use a rolling 5-year financial window to ensure trend consistency and cycle-awareness.
"""

import requests
import time
import random
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# ==================== Configuration ====================

# Alpha Vantage API configuration
# 从环境变量读取，如果没有设置则需要在运行时通过模块属性设置
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", None)
BASE_URL = os.getenv("ALPHAVANTAGE_BASE_URL", None)
TIMEOUT = 12
MAX_RETRIES = 5

# Data save path
DEFAULT_OUTPUT_DIR = Path("./fundamental_data")

# ==================== API Request Functions ====================

def request_av(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Send Alpha Vantage API request (with retry mechanism)
    
    Args:
        params: API parameters dictionary
    
    Returns:
        JSON data returned by API, None on failure
    """
    # Dynamically get the latest API_KEY and BASE_URL values
    # This ensures we always use the most recent configuration
    current_api_key = API_KEY
    current_base_url = BASE_URL
    
    # Also check environment variables as fallback
    if current_api_key is None:
        current_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if current_base_url is None:
        current_base_url = os.getenv("ALPHAVANTAGE_BASE_URL")
    
    if current_api_key is None:
        print("⚠️ Error: Alpha Vantage API Key not configured. Please set ALPHAVANTAGE_API_KEY environment variable or configure it in the web interface.")
        return None
    
    if current_base_url is None:
        print("⚠️ Error: Alpha Vantage Base URL not configured. Please set ALPHAVANTAGE_BASE_URL environment variable or configure it in the web interface.")
        return None
    
    params["apikey"] = current_api_key
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(current_base_url, params=params, timeout=TIMEOUT, allow_redirects=False)
            
            if r.status_code in (301, 302, 307, 308):
                print(f"🔀 Redirect occurred, blocked, status: {r.status_code}")
                return None
            
            return r.json()
        except Exception as e:
            print(f"⚠️ Request failed, retry {i+1}/{MAX_RETRIES}: {e}")
            time.sleep(1 + random.random())
    return None

# ==================== Financial Statement Fetching Functions ====================

def get_income_statement(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch Income Statement
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Dictionary containing annual and quarterly income statement data
    """
    return request_av({"function": "INCOME_STATEMENT", "symbol": symbol})

def get_balance_sheet(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch Balance Sheet
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Dictionary containing annual and quarterly balance sheet data
    """
    return request_av({"function": "BALANCE_SHEET", "symbol": symbol})

def get_cash_flow_statement(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch Cash Flow Statement
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Dictionary containing annual and quarterly cash flow statement data
    """
    return request_av({"function": "CASH_FLOW", "symbol": symbol})

def get_overview(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch company overview information
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Company basic information dictionary
    """
    return request_av({"function": "OVERVIEW", "symbol": symbol})

def get_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch stock real-time quote (GLOBAL_QUOTE)
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Real-time quote data dictionary
    """
    return request_av({"function": "GLOBAL_QUOTE", "symbol": symbol})

def get_news(symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
    """
    Fetch stock-related news (News & Sentiment)
    
    Args:
        symbol: Stock symbol
        limit: Maximum number of news items to return (default 50)
    
    Returns:
        Dictionary containing news data
    """
    return request_av({"function": "NEWS_SENTIMENT", "tickers": symbol, "limit": limit})

# ==================== Data Conversion Functions ====================

def financial_reports_to_dataframe(
    reports: List[Dict[str, Any]], 
    report_type: str = "annual"
) -> pd.DataFrame:
    """
    Convert financial statement list to DataFrame
    
    Args:
        reports: Financial statement list (annualReports or quarterlyReports)
        report_type: Report type ("annual" or "quarterly")
    
    Returns:
        Converted DataFrame with report date as index
    """
    if not reports:
        return pd.DataFrame()
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(reports)
        
        # Set date index
        if "fiscalDateEnding" in df.columns:
            df["fiscalDateEnding"] = pd.to_datetime(df["fiscalDateEnding"], errors="coerce")
            df.set_index("fiscalDateEnding", inplace=True)
            df.sort_index(inplace=True)
        
        # Convert numeric columns to numeric type (handle string-formatted numbers)
        for col in df.columns:
            if col != "reportedCurrency":  # Keep currency column
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        return df
    
    except Exception as e:
        print(f"⚠️ Failed to convert financial statement data: {e}")
        return pd.DataFrame()

def filter_reports_by_years(
    reports: List[Dict[str, Any]], 
    years: int = 5
) -> List[Dict[str, Any]]:
    """
    Filter financial statements by years (keep last N years)
    
    Args:
        reports: Financial statement list
        years: Number of years to keep (default 5 years, recommended 5-7 years)
    
    Returns:
        Filtered financial statement list
    """
    if not reports:
        return []
    
    try:
        # Get current date
        current_date = datetime.now()
        cutoff_date = current_date - timedelta(days=years * 365)
        
        # Filter reports
        filtered = []
        for report in reports:
            if "fiscalDateEnding" in report:
                try:
                    report_date = pd.to_datetime(report["fiscalDateEnding"], errors="coerce")
                    if report_date and report_date >= cutoff_date:
                        filtered.append(report)
                except:
                    continue
        
        # Sort by date (newest first)
        filtered.sort(key=lambda x: x.get("fiscalDateEnding", ""), reverse=True)
        
        return filtered
    
    except Exception as e:
        print(f"⚠️ Failed to filter financial statements: {e}")
        return reports  # If failed, return original list

# ==================== Main Data Ingestion Functions ====================

def fetch_financial_statements(
    symbol: str,
    years: int = 5,
    include_quarterly: bool = True
) -> Dict[str, Any]:
    """
    Fetch stock financial statement data (5-7 years)
    
    Args:
        symbol: Stock symbol
        years: Number of years to fetch (default 5 years, recommended 5-7 years)
        include_quarterly: Whether to include quarterly data (default True)
    
    Returns:
        Dictionary containing three major statement data, structure as follows:
        {
            "symbol": "NVDA",
            "overview": {...},
            "income_statement": {
                "annual": DataFrame,
                "quarterly": DataFrame
            },
            "balance_sheet": {
                "annual": DataFrame,
                "quarterly": DataFrame
            },
            "cash_flow": {
                "annual": DataFrame,
                "quarterly": DataFrame
            },
            "raw": {
                "income": {...},
                "balance": {...},
                "cashflow": {...},
                "overview": {...}
            }
        }
    """
    symbol = symbol.upper()
    print(f"\n📊 Starting to fetch financial statement data for {symbol} (last {years} years)...")
    
    result = {
        "symbol": symbol,
        "overview": None,
        "income_statement": {"annual": pd.DataFrame(), "quarterly": pd.DataFrame()},
        "balance_sheet": {"annual": pd.DataFrame(), "quarterly": pd.DataFrame()},
        "cash_flow": {"annual": pd.DataFrame(), "quarterly": pd.DataFrame()},
        "raw": {}
    }
    
    # 1. Fetch company overview
    print(f"  📋 Fetching company overview...")
    overview = get_overview(symbol)
    if overview:
        result["overview"] = overview
        result["raw"]["overview"] = overview
        print(f"  ✅ Company overview fetched successfully")
    else:
        print(f"  ⚠️ Failed to fetch company overview")
    
    # 2. Fetch income statement
    print(f"  💰 Fetching income statement...")
    income_data = get_income_statement(symbol)
    if income_data:
        result["raw"]["income"] = income_data
        
        # Process annual data
        if "annualReports" in income_data and income_data["annualReports"]:
            annual_reports = filter_reports_by_years(income_data["annualReports"], years)
            if annual_reports:
                df_annual = financial_reports_to_dataframe(annual_reports, "annual")
                result["income_statement"]["annual"] = df_annual
                print(f"  ✅ Annual income statement: {len(annual_reports)} reporting periods")
        
        # Process quarterly data
        if include_quarterly and "quarterlyReports" in income_data and income_data["quarterlyReports"]:
            quarterly_reports = filter_reports_by_years(income_data["quarterlyReports"], years)
            if quarterly_reports:
                df_quarterly = financial_reports_to_dataframe(quarterly_reports, "quarterly")
                result["income_statement"]["quarterly"] = df_quarterly
                print(f"  ✅ Quarterly income statement: {len(quarterly_reports)} reporting periods")
    else:
        print(f"  ⚠️ Failed to fetch income statement")
    
    # 3. Fetch balance sheet
    print(f"  📊 Fetching balance sheet...")
    balance_data = get_balance_sheet(symbol)
    if balance_data:
        result["raw"]["balance"] = balance_data
        
        # Process annual data
        if "annualReports" in balance_data and balance_data["annualReports"]:
            annual_reports = filter_reports_by_years(balance_data["annualReports"], years)
            if annual_reports:
                df_annual = financial_reports_to_dataframe(annual_reports, "annual")
                result["balance_sheet"]["annual"] = df_annual
                print(f"  ✅ Annual balance sheet: {len(annual_reports)} reporting periods")
        
        # Process quarterly data
        if include_quarterly and "quarterlyReports" in balance_data and balance_data["quarterlyReports"]:
            quarterly_reports = filter_reports_by_years(balance_data["quarterlyReports"], years)
            if quarterly_reports:
                df_quarterly = financial_reports_to_dataframe(quarterly_reports, "quarterly")
                result["balance_sheet"]["quarterly"] = df_quarterly
                print(f"  ✅ Quarterly balance sheet: {len(quarterly_reports)} reporting periods")
    else:
        print(f"  ⚠️ Failed to fetch balance sheet")
    
    # 4. Fetch cash flow statement
    print(f"  💵 Fetching cash flow statement...")
    cashflow_data = get_cash_flow_statement(symbol)
    if cashflow_data:
        result["raw"]["cashflow"] = cashflow_data
        
        # Process annual data
        if "annualReports" in cashflow_data and cashflow_data["annualReports"]:
            annual_reports = filter_reports_by_years(cashflow_data["annualReports"], years)
            if annual_reports:
                df_annual = financial_reports_to_dataframe(annual_reports, "annual")
                result["cash_flow"]["annual"] = df_annual
                print(f"  ✅ Annual cash flow statement: {len(annual_reports)} reporting periods")
        
        # Process quarterly data
        if include_quarterly and "quarterlyReports" in cashflow_data and cashflow_data["quarterlyReports"]:
            quarterly_reports = filter_reports_by_years(cashflow_data["quarterlyReports"], years)
            if quarterly_reports:
                df_quarterly = financial_reports_to_dataframe(quarterly_reports, "quarterly")
                result["cash_flow"]["quarterly"] = df_quarterly
                print(f"  ✅ Quarterly cash flow statement: {len(quarterly_reports)} reporting periods")
    else:
        print(f"  ⚠️ Failed to fetch cash flow statement")
    
    # 5. Data completeness check
    has_annual_data = (
        not result["income_statement"]["annual"].empty or
        not result["balance_sheet"]["annual"].empty or
        not result["cash_flow"]["annual"].empty
    )
    
    if has_annual_data:
        print(f"\n✅ {symbol} financial statement data fetch completed!")
        print(f"   - Annual data: {'✅' if not result['income_statement']['annual'].empty else '❌'} Income Statement, "
              f"{'✅' if not result['balance_sheet']['annual'].empty else '❌'} Balance Sheet, "
              f"{'✅' if not result['cash_flow']['annual'].empty else '❌'} Cash Flow Statement")
    else:
        print(f"\n⚠️ {symbol} financial statement data fetch incomplete, please check API or stock symbol")
    
    return result

# ==================== Data Saving Functions ====================

def save_financial_data(
    data: Dict[str, Any],
    output_dir: Union[str, Path] = None,
    format: str = "both"  # "csv", "json", "both"
) -> Dict[str, str]:
    """
    Save financial statement data to files
    
    Args:
        data: Data dictionary returned by fetch_financial_statements
        output_dir: Output directory (default uses DEFAULT_OUTPUT_DIR)
        format: Save format ("csv", "json", "both")
    
    Returns:
        Dictionary of saved file paths
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    symbol = data.get("symbol", "UNKNOWN")
    saved_files = {}
    
    # Save annual data
    for statement_type in ["income_statement", "balance_sheet", "cash_flow"]:
        df_annual = data.get(statement_type, {}).get("annual", pd.DataFrame())
        if not df_annual.empty:
            if format in ["csv", "both"]:
                csv_path = output_dir / f"{symbol}_{statement_type}_annual.csv"
                df_annual.to_csv(csv_path, encoding="utf-8-sig")
                saved_files[f"{statement_type}_annual_csv"] = str(csv_path)
                print(f"  💾 Saved: {csv_path}")
        
        df_quarterly = data.get(statement_type, {}).get("quarterly", pd.DataFrame())
        if not df_quarterly.empty:
            if format in ["csv", "both"]:
                csv_path = output_dir / f"{symbol}_{statement_type}_quarterly.csv"
                df_quarterly.to_csv(csv_path, encoding="utf-8-sig")
                saved_files[f"{statement_type}_quarterly_csv"] = str(csv_path)
                print(f"  💾 Saved: {csv_path}")
    
    # Save raw JSON data
    if format in ["json", "both"]:
        json_path = output_dir / f"{symbol}_financial_statements_raw.json"
        # Handle DataFrame and non-serializable objects
        def convert_to_serializable(obj):
            if isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_data = convert_to_serializable(data["raw"])
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2, default=str)
        saved_files["raw_json"] = str(json_path)
        print(f"  💾 Saved: {json_path}")
    
    return saved_files

# ==================== Batch Fetching Functions ====================

def fetch_multiple_symbols(
    symbols: List[str],
    years: int = 5,
    include_quarterly: bool = True,
    save_data: bool = True,
    output_dir: Union[str, Path] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Batch fetch financial statement data for multiple stocks
    
    Args:
        symbols: List of stock symbols
        years: Number of years to fetch (default 5 years)
        include_quarterly: Whether to include quarterly data
        save_data: Whether to automatically save data
        output_dir: Output directory
    
    Returns:
        Dictionary containing all stock data, key is stock symbol
    """
    results = {}
    
    with tqdm(total=len(symbols), desc="📊 Batch fetching financial statements", unit="stock", ncols=100) as pbar:
        for symbol in symbols:
            pbar.set_postfix({"Current": symbol})
            
            try:
                data = fetch_financial_statements(
                    symbol=symbol,
                    years=years,
                    include_quarterly=include_quarterly
                )
                
                if save_data:
                    save_financial_data(data, output_dir=output_dir)
                
                results[symbol] = data
                
            except Exception as e:
                print(f"\n❌ Failed to fetch {symbol} data: {e}")
                results[symbol] = None
            
            pbar.update(1)
    
    return results

# ==================== Main Function (for testing) ====================

if __name__ == "__main__":
    # Test example
    print("=" * 60)
    print("Data Ingestion Module - Financial Statement Data Ingestion Test")
    print("=" * 60)
    
    # Test single stock
    symbol = "NVDA"
    data = fetch_financial_statements(symbol, years=5, include_quarterly=True)
    
    # Display data summary
    print("\n" + "=" * 60)
    print("Data Summary:")
    print("=" * 60)
    
    if not data["income_statement"]["annual"].empty:
        print(f"\n📊 Annual Income Statement ({len(data['income_statement']['annual'])} reporting periods):")
        print(data["income_statement"]["annual"].head())
    
    if not data["balance_sheet"]["annual"].empty:
        print(f"\n📊 Annual Balance Sheet ({len(data['balance_sheet']['annual'])} reporting periods):")
        print(data["balance_sheet"]["annual"].head())
    
    if not data["cash_flow"]["annual"].empty:
        print(f"\n📊 Annual Cash Flow Statement ({len(data['cash_flow']['annual'])} reporting periods):")
        print(data["cash_flow"]["annual"].head())
    
    # Save data
    print("\n" + "=" * 60)
    print("Saving Data:")
    print("=" * 60)
    saved_files = save_financial_data(data, format="both")
    
    print("\n✅ Test completed!")

