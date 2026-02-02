"""
Company Search and Peer Recommendation Module

Features:
1. Search for companies (by stock symbol)
2. Automatically recommend peer companies based on industry
"""

from typing import Dict, Any, List, Optional
try:
    from data_ingestion import get_overview
except ImportError:
    from .data_ingestion import get_overview

# Common industry peer company lists (categorized by industry)
INDUSTRY_PEERS = {
    "Technology": {
        "Semiconductors": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM", "TXN", "MRVL"],
        "Software": ["MSFT", "GOOGL", "META", "ORCL", "CRM", "ADBE", "NOW", "SNOW"],
        "Hardware": ["AAPL", "HPQ", "DELL", "CSCO", "HPE", "NTAP"],
        "Internet": ["AMZN", "GOOGL", "META", "NFLX", "BABA", "JD"],
    },
    "Consumer": {
        "Retail": ["WMT", "TGT", "COST", "HD", "LOW", "TJX", "ROST"],
        "E-commerce": ["AMZN", "BABA", "JD", "PDD", "SE"],
        "Consumer Goods": ["PG", "KO", "PEP", "NKE", "SBUX", "MCD"],
    },
    "Financial Services": {
        "Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
        "Insurance": ["BRK.B", "UNH", "CVS", "CI", "AIG"],
    },
    "Finance": {
        "Banks": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
        "Insurance": ["BRK.B", "UNH", "CVS", "CI", "AIG"],
    },
    "Healthcare": {
        "Pharmaceuticals": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY"],
        "Biotech": ["GILD", "AMGN", "REGN", "VRTX", "BIIB"],
        "Medical Devices": ["TMO", "DHR", "ISRG", "SYK", "BSX"],
    },
    "Energy": {
        "Oil & Gas": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "Renewable": ["NEE", "ENPH", "SEDG", "FSLR"],
    },
    "Industrial": {
        "Aerospace": ["BA", "LMT", "RTX", "NOC", "GD"],
        "Manufacturing": ["CAT", "DE", "GE", "HON", "ETN"],
    },
    "Communication Services": {
        "Telecom": ["T", "VZ", "TMUS", "CMCSA"],
        "Media": ["DIS", "NFLX", "PARA", "WBD"],
    },
    "Communication": {
        "Telecom": ["T", "VZ", "TMUS", "CMCSA"],
        "Media": ["DIS", "NFLX", "PARA", "WBD"],
    },
}

# Common stock peer company mapping (for quick matching)
COMMON_PEER_MAPPING = {
    "NVDA": ["AMD", "INTC", "TSM", "AVGO", "QCOM", "TXN", "MRVL"],
    "AMD": ["NVDA", "INTC", "TSM", "AVGO", "QCOM", "TXN", "MRVL"],
    "INTC": ["NVDA", "AMD", "TSM", "AVGO", "QCOM"],
    "TSM": ["NVDA", "AMD", "INTC", "AVGO", "QCOM"],
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN", "HPQ", "DELL"],
    "MSFT": ["GOOGL", "AAPL", "META", "ORCL", "CRM", "ADBE"],
    "GOOGL": ["MSFT", "META", "AMZN", "AAPL", "NFLX"],
    "META": ["GOOGL", "MSFT", "AMZN", "NFLX", "SNAP"],
    "AMZN": ["GOOGL", "META", "MSFT", "BABA", "WMT"],
    "TSLA": ["F", "GM", "RIVN", "LCID", "NIO"],
    "JPM": ["BAC", "WFC", "C", "GS", "MS"],
    "BAC": ["JPM", "WFC", "C", "GS", "MS"],
    "WMT": ["TGT", "COST", "HD", "LOW", "AMZN"],
    "JNJ": ["PFE", "MRK", "ABBV", "LLY", "BMY"],
    "PFE": ["JNJ", "MRK", "ABBV", "LLY", "BMY"],
    "XOM": ["CVX", "COP", "SLB", "EOG", "BP"],
    "CVX": ["XOM", "COP", "SLB", "EOG", "BP"],
}

def search_company(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Search for company information
    
    Args:
        symbol: Stock symbol
    
    Returns:
        Company information dictionary, containing name, industry, sector, etc.
    """
    try:
        overview = get_overview(symbol.upper())
        if overview and overview.get("Symbol"):
            return {
                "symbol": overview.get("Symbol", ""),
                "name": overview.get("Name", ""),
                "sector": overview.get("Sector", ""),
                "industry": overview.get("Industry", ""),
                "exchange": overview.get("Exchange", ""),
                "market_cap": overview.get("MarketCapitalization", 0),
                "overview": overview
            }
    except Exception as e:
        print(f"⚠️ Failed to search company: {e}")
    
    return None

def get_peer_recommendations(
    target_symbol: str,
    target_sector: str = None,
    target_industry: str = None
) -> List[str]:
    """
    Recommend peer companies based on target company's industry
    
    Args:
        target_symbol: Target company symbol
        target_sector: Target company sector (optional, used directly if provided)
        target_industry: Target company industry (optional, used directly if provided)
    
    Returns:
        List of recommended peer company symbols
    """
    target_symbol = target_symbol.upper()
    recommendations = []
    
    # Method 1: Use common stock mapping (fastest)
    if target_symbol in COMMON_PEER_MAPPING:
        recommendations.extend(COMMON_PEER_MAPPING[target_symbol])
        print(f"  ✅ Used common mapping to recommend {len(recommendations)} companies for {target_symbol}")
    
    # Method 2: If industry information not provided, try to fetch
    if not target_sector or not target_industry:
        try:
            company_info = search_company(target_symbol)
            if company_info:
                target_sector = target_sector or company_info.get("sector", "")
                target_industry = target_industry or company_info.get("industry", "")
                print(f"  📊 {target_symbol} sector: {target_sector}, industry: {target_industry}")
        except Exception as e:
            print(f"  ⚠️ Failed to get {target_symbol} company information: {e}")
    
    # Method 3: Match by sector and industry
    if target_sector:
        # Try multiple possible sector names (more comprehensive matching)
        sector_variants = [target_sector]
        
        # Sector name mapping (supports various names that Alpha Vantage may return)
        # Note: Use if instead of elif, because one sector may match multiple keywords
        if "Financial" in target_sector:
            sector_variants.extend(["Finance", "Financial Services"])
        if "Communication" in target_sector:
            sector_variants.extend(["Communication", "Communication Services"])
        if "Technology" in target_sector or "Tech" in target_sector or "Information Technology" in target_sector:
            if "Technology" not in sector_variants:
                sector_variants.append("Technology")
        if "Consumer" in target_sector:
            sector_variants.extend(["Consumer", "Consumer Cyclical", "Consumer Defensive", "Consumer Staples"])
        if "Health" in target_sector:
            if "Healthcare" not in sector_variants:
                sector_variants.append("Healthcare")
        if "Energy" in target_sector:
            if "Energy" not in sector_variants:
                sector_variants.append("Energy")
        if "Industrial" in target_sector:
            if "Industrial" not in sector_variants:
                sector_variants.append("Industrial")
        
        # Also try partial matching for all known sectors
        for known_sector in INDUSTRY_PEERS.keys():
            if (known_sector.lower() in target_sector.lower() or 
                target_sector.lower() in known_sector.lower()):
                if known_sector not in sector_variants:
                    sector_variants.append(known_sector)
        
        print(f"  🔍 Attempting to match sectors: {sector_variants}")
        
        for sector in sector_variants:
            if sector in INDUSTRY_PEERS:
                sector_peers = INDUSTRY_PEERS[sector]
                print(f"  ✅ Found matching sector: {sector}")
                
                # Try exact industry matching
                industry_matched = False
                if target_industry:
                    for industry_key, peers in sector_peers.items():
                        # More flexible matching logic
                        industry_lower = target_industry.lower()
                        key_lower = industry_key.lower()
                        
                        # Extract keywords (words with length > 3)
                        industry_words = [w for w in industry_lower.split() if len(w) > 3]
                        key_words = [w for w in key_lower.split() if len(w) > 3]
                        
                        # Check keyword matching
                        if (key_lower in industry_lower or 
                            industry_lower in key_lower or
                            any(word in industry_lower for word in key_words) or
                            any(word in key_lower for word in industry_words) or
                            len(set(industry_words) & set(key_words)) > 0):
                            recommendations.extend(peers)
                            industry_matched = True
                            print(f"  ✅ Matched industry: {industry_key}, recommended {len(peers)} companies")
                            break
                
                # If no exact match, use all companies under the sector
                if not industry_matched:
                    print(f"  ⚠️ No exact industry match found, using all companies under sector")
                    for peers in sector_peers.values():
                        recommendations.extend(peers)
                
                # Exit if match found
                if recommendations:
                    break
    
    # Remove target company itself
    recommendations = [p for p in recommendations if p.upper() != target_symbol.upper()]
    
    # Deduplicate and limit quantity
    recommendations = list(dict.fromkeys(recommendations))[:20]  # Maximum 20
    
    print(f"  📊 Finally recommended {len(recommendations)} peer companies for {target_symbol}")
    if recommendations:
        print(f"  📋 Recommendation list: {recommendations[:10]}")
    
    # If still no recommendations, return some common large companies as fallback
    if not recommendations:
        print(f"  ⚠️ No match found, returning common large companies as fallback")
        fallback_peers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "JPM", "JNJ", "V", "WMT"]
        recommendations = [p for p in fallback_peers if p.upper() != target_symbol.upper()][:10]
        print(f"  📋 Fallback recommendations: {recommendations}")
    
    return recommendations

def get_all_peers_by_sector(sector: str) -> List[str]:
    """
    Get all companies under a specific sector
    
    Args:
        sector: Sector name
    
    Returns:
        List of company symbols
    """
    if sector not in INDUSTRY_PEERS:
        return []
    
    all_peers = []
    for peers in INDUSTRY_PEERS[sector].values():
        all_peers.extend(peers)
    
    return list(dict.fromkeys(all_peers))

def get_common_stocks() -> List[Dict[str, str]]:
    """
    Get common stock list (for search)
    
    Returns:
        Stock list, containing symbols and names
    """
    # This can be extended to a more complete stock list
    # Currently returns some common large companies
    common_stocks = [
        {"symbol": "AAPL", "name": "Apple Inc."},
        {"symbol": "MSFT", "name": "Microsoft Corporation"},
        {"symbol": "GOOGL", "name": "Alphabet Inc."},
        {"symbol": "AMZN", "name": "Amazon.com Inc."},
        {"symbol": "NVDA", "name": "NVIDIA Corporation"},
        {"symbol": "META", "name": "Meta Platforms Inc."},
        {"symbol": "TSLA", "name": "Tesla Inc."},
        {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
        {"symbol": "V", "name": "Visa Inc."},
        {"symbol": "JNJ", "name": "Johnson & Johnson"},
        {"symbol": "WMT", "name": "Walmart Inc."},
        {"symbol": "PG", "name": "Procter & Gamble Co."},
        {"symbol": "MA", "name": "Mastercard Inc."},
        {"symbol": "UNH", "name": "UnitedHealth Group Inc."},
        {"symbol": "HD", "name": "The Home Depot Inc."},
        {"symbol": "DIS", "name": "The Walt Disney Company"},
        {"symbol": "BAC", "name": "Bank of America Corp."},
        {"symbol": "ADBE", "name": "Adobe Inc."},
        {"symbol": "NFLX", "name": "Netflix Inc."},
        {"symbol": "CRM", "name": "Salesforce Inc."},
    ]
    
    return common_stocks
