"""
Qualitative Analysis Module for Fundamental Analysis

Features:
1. Fetch news data from Alpha Vantage
2. Use LLM to analyze catalysts (new products, cost improvements, industry cycles, policy changes)
3. Distinguish between short-term vs long-term catalysts

Key Points:
- Distinguish between short-term vs long-term catalysts
- Deep analysis based on news content
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import data ingestion module
try:
    from data_ingestion import get_news, get_overview
except ImportError:
    from .data_ingestion import get_news, get_overview

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
        print("OpenAI module not found, please ensure talk2ai.py is available")
        OpenAIChat = None
        get_config_from_env = None

# ==================== News Data Fetching ====================

def fetch_stock_news(
    symbol: str,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Fetch stock-related news
    
    Args:
        symbol: Stock symbol
        limit: Maximum number of news items to return
    
    Returns:
        Dictionary containing news data
    """
    result = {
        "symbol": symbol.upper(),
        "news_count": 0,
        "news_items": [],
        "raw_data": None
    }
    
    try:
        print(f"  📰 Fetching news data for {symbol}...")
        news_data = get_news(symbol, limit=limit)
        
        if news_data:
            result["raw_data"] = news_data
            
            # Extract news list
            if isinstance(news_data, dict):
                # Alpha Vantage NEWS_SENTIMENT return format
                if "feed" in news_data:
                    news_items = news_data["feed"]
                    result["news_items"] = news_items
                    result["news_count"] = len(news_items)
                    print(f"  ✅ Retrieved {len(news_items)} news items")
                elif "items" in news_data:
                    news_items = news_data["items"]
                    result["news_items"] = news_items
                    result["news_count"] = len(news_items)
                    print(f"  ✅ Retrieved {len(news_items)} news items")
                else:
                    print(f"  ⚠️ Unknown news data format")
            else:
                print(f"  ⚠️ Invalid news data format")
        else:
            print(f"  ⚠️ No news data retrieved")
    
    except Exception as e:
        print(f"⚠️ Failed to fetch news data: {e}")
        import traceback
        traceback.print_exc()
    
    return result

# ==================== LLM Catalyst Analysis ====================

def analyze_catalysts_with_llm(
    symbol: str,
    news_data: Dict[str, Any],
    financial_analysis: Dict[str, Any] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Analyze catalysts using LLM
    
    Analysis dimensions:
    1. New products/new markets
    2. Cost structure improvements
    3. Industry cycle reversals
    4. Interest rate or policy changes
    
    Key: Distinguish between short-term vs long-term catalysts
    
    Args:
        symbol: Stock symbol
        news_data: News data (from fetch_stock_news)
        financial_analysis: Financial analysis results (optional, for additional context)
        api_key: OpenAI API Key
        base_url: OpenAI Base URL
        model: AI model name
    
    Returns:
        Catalyst analysis results
    """
    result = {
        "symbol": symbol.upper(),
        "analysis_timestamp": datetime.now().isoformat(),
        "product_market_catalysts": [],
        "cost_improvement_catalysts": [],
        "industry_cycle_catalysts": [],
        "policy_rate_catalysts": [],
        "summary": {}
    }
    
    if not OPENAI_AVAILABLE:
        return {"error": "OpenAI module not available"}
    
    if not api_key:
        env_api_key, env_base_url = get_config_from_env()
        api_key = env_api_key
        base_url = base_url or env_base_url or "https://api.openai.com/v1"
    
    # Validate API Key
    if not api_key:
        result["error"] = "OpenAI API Key is required"
        return result
    
    try:
        print(f"  🤖 Using LLM to analyze catalysts for {symbol}...")
        
        # Prepare news content
        news_items = news_data.get("news_items", [])
        if not news_items:
            print(f"  ⚠️ No news data, unable to perform catalyst analysis")
            return result
        
        # Format news content
        news_text = format_news_for_llm(news_items)
        
        # Prepare financial analysis summary (if provided)
        financial_summary = ""
        if financial_analysis:
            financial_summary = format_financial_summary_for_llm(financial_analysis)
        
        # Build professional Prompt
        prompt = build_catalyst_analysis_prompt(
            symbol=symbol,
            news_text=news_text,
            financial_summary=financial_summary
        )
        
        # Call LLM
        chat_client = OpenAIChat(api_key=api_key, base_url=base_url)
        messages = [
            {
                "role": "system",
                "content": """You are a senior buy-side research analyst with 20 years of investment research experience.
Your expertise lies in identifying and analyzing catalysts that affect stock value.

Your analysis should:
1. Be based on facts and data, avoid subjective speculation
2. Distinguish between short-term (1-6 months) and long-term (6+ months) catalysts
3. Assess the impact level of each catalyst (High/Medium/Low)
4. Provide clear, actionable investment insights

Please respond in English. Analysis should be deep, professional, and insightful."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        print(f"  ⏳ Calling LLM for analysis...")
        llm_response = chat_client.chat(messages, model=model)
        
        if not llm_response:
            print(f"  ⚠️ LLM response is empty")
            return result
        
        # Parse LLM response
        parsed_result = parse_llm_catalyst_response(llm_response)
        
        # Merge results
        result.update(parsed_result)
        result["raw_llm_response"] = llm_response
        
        # Generate summary
        result["summary"] = generate_catalyst_summary(result)
        
        total_catalysts = (
            len(result.get("product_market_catalysts", [])) +
            len(result.get("cost_improvement_catalysts", [])) +
            len(result.get("industry_cycle_catalysts", [])) +
            len(result.get("policy_rate_catalysts", []))
        )
        print(f"  ✅ LLM analysis completed: Identified {total_catalysts} catalysts")
    
    except Exception as e:
        print(f"⚠️ LLM catalyst analysis failed: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
    
    return result

def format_news_for_llm(news_items: List[Dict[str, Any]], max_items: int = 20) -> str:
    """
    Format news content for LLM analysis
    
    Args:
        news_items: News list
        max_items: Maximum number of news items
    
    Returns:
        Formatted news text
    """
    if not news_items:
        return "No news data"
    
    formatted = f"## Related News (Total {len(news_items)} items, showing latest {min(max_items, len(news_items))} items)\n\n"
    
    for idx, item in enumerate(news_items[:max_items], 1):
        title = item.get("title", "No Title")
        summary = item.get("summary", "") or item.get("content", "") or item.get("text", "")
        url = item.get("url", "")
        published_date = item.get("time_published", "") or item.get("published_date", "")
        sentiment = item.get("overall_sentiment_score", "")
        
        formatted += f"### News {idx}\n"
        formatted += f"**Title**: {title}\n"
        if published_date:
            formatted += f"**Published Date**: {published_date}\n"
        if sentiment:
            formatted += f"**Sentiment Score**: {sentiment}\n"
        if summary:
            # Limit summary length
            summary_text = summary[:500] + "..." if len(summary) > 500 else summary
            formatted += f"**Summary**: {summary_text}\n"
        if url:
            formatted += f"**Link**: {url}\n"
        formatted += "\n"
    
    return formatted

def format_financial_summary_for_llm(financial_analysis: Dict[str, Any]) -> str:
    """
    Format financial analysis summary for LLM reference
    
    Args:
        financial_analysis: Financial analysis results
    
    Returns:
        Formatted financial summary text
    """
    if not financial_analysis:
        return ""
    
    summary_parts = ["## Financial Analysis Summary (For Reference)\n\n"]
    
    # Profitability
    profitability = financial_analysis.get("profitability", {})
    if profitability:
        summary_parts.append("### Profitability\n")
        if profitability.get("roic"):
            summary_parts.append(f"- ROIC: {profitability['roic']:.2f}%\n")
        if profitability.get("net_margin"):
            summary_parts.append(f"- Net Margin: {profitability['net_margin']:.2f}%\n")
        summary_parts.append("\n")
    
    # Growth
    growth = financial_analysis.get("growth", {})
    if growth:
        summary_parts.append("### Growth\n")
        if growth.get("revenue_cagr"):
            summary_parts.append(f"- Revenue CAGR (5Y): {growth['revenue_cagr']:.2f}%\n")
        if growth.get("revenue_growth_yoy"):
            summary_parts.append(f"- Revenue YoY Growth: {growth['revenue_growth_yoy']:.2f}%\n")
        summary_parts.append("\n")
    
    # Cost structure (for cost improvement analysis)
    if profitability.get("gross_margin"):
        summary_parts.append("### Cost Structure\n")
        summary_parts.append(f"- Gross Margin: {profitability['gross_margin']:.2f}%\n")
        summary_parts.append("\n")
    
    return "".join(summary_parts)

def build_catalyst_analysis_prompt(
    symbol: str,
    news_text: str,
    financial_summary: str = ""
) -> str:
    """
    Build professional catalyst analysis prompt
    
    Args:
        symbol: Stock symbol
        news_text: News text
        financial_summary: Financial summary (optional)
    
    Returns:
        Complete prompt text
    """
    prompt = f"""Please perform professional catalyst analysis for {symbol} based on the following news and financial data.

{news_text}

{financial_summary if financial_summary else ""}

## Analysis Requirements

Please identify and analyze catalysts from the following 4 dimensions, and **clearly distinguish between short-term (1-6 months) and long-term (6+ months) impacts**:

### 1. New Products/New Markets (Product/Market Catalysts)
- New product launches, technological breakthroughs
- New market entry, geographic expansion
- Major contracts, order wins
- Partnerships, strategic alliances
- **Timeframe**: Distinguish short-term (announced, launching soon) vs long-term (in R&D, planned)

### 2. Cost Structure Improvements (Cost Improvement Catalysts)
- Operational efficiency improvements
- Cost reduction plans
- Supply chain optimization
- Automation, digital transformation
- Scale effects emerging
- **Timeframe**: Distinguish short-term (implemented, showing results) vs long-term (ongoing optimization)

### 3. Industry Cycle Reversals (Industry Cycle Catalysts)
- Industry demand recovery
- Supply-side contraction
- Price trend reversals
- Inventory cycle changes
- Industry consolidation, concentration increase
- **Timeframe**: Distinguish short-term (quarterly changes) vs long-term (structural changes)

### 4. Interest Rate or Policy Changes (Policy/Rate Catalysts)
- Monetary policy changes (interest rates, QE, etc.)
- Fiscal policy (taxes, subsidies, etc.)
- Industry regulatory policies
- Trade policies
- Geopolitical impacts
- **Timeframe**: Distinguish short-term (announced, implementing soon) vs long-term (policy trends)

## Output Format Requirements

Please return in **JSON format**, containing the following structure:

```json
{{
    "product_market_catalysts": [
        {{
            "type": "New Product Launch",
            "description": "Detailed description of catalyst",
            "evidence": "Supporting evidence (from news or data)",
            "impact": "High/Medium/Low",
            "timeframe": "Short-term/Long-term",
            "expected_timing": "Expected timing (e.g., Q2 2024)",
            "confidence": "High/Medium/Low"
        }}
    ],
    "cost_improvement_catalysts": [
        {{
            "type": "Operational Efficiency Improvement",
            "description": "Detailed description",
            "evidence": "Supporting evidence",
            "impact": "High/Medium/Low",
            "timeframe": "Short-term/Long-term",
            "expected_timing": "Expected timing",
            "confidence": "High/Medium/Low"
        }}
    ],
    "industry_cycle_catalysts": [
        {{
            "type": "Industry Demand Recovery",
            "description": "Detailed description",
            "evidence": "Supporting evidence",
            "impact": "High/Medium/Low",
            "timeframe": "Short-term/Long-term",
            "expected_timing": "Expected timing",
            "confidence": "High/Medium/Low"
        }}
    ],
    "policy_rate_catalysts": [
        {{
            "type": "Interest Rate Change",
            "description": "Detailed description",
            "evidence": "Supporting evidence",
            "impact": "High/Medium/Low",
            "timeframe": "Short-term/Long-term",
            "expected_timing": "Expected timing",
            "confidence": "High/Medium/Low"
        }}
    ],
    "summary": {{
        "total_catalysts": 0,
        "high_impact_count": 0,
        "short_term_count": 0,
        "long_term_count": 0,
        "key_insights": [
            "Key insight 1",
            "Key insight 2"
        ],
        "investment_implications": "Investment implications summary"
    }}
}}
```

## Analysis Principles

1. **Fact-based**: All catalysts must have clear news or data support
2. **Distinguish timeframe**: Must clearly label short-term vs long-term
3. **Assess impact level**: High impact = may significantly change company value; Medium impact = may bring moderate gains; Low impact = marginal improvement
4. **Assess confidence**: High confidence = information clear, high certainty; Medium confidence = information relatively clear; Low confidence = needs further verification
5. **Avoid duplication**: List each catalyst only once, choose the most relevant category
6. **Prioritize**: High impact, high confidence catalysts should be listed first

Please begin analysis, ensure returning valid JSON format."""

    return prompt

def parse_llm_catalyst_response(llm_response: str) -> Dict[str, Any]:
    """
    Parse LLM catalyst analysis response
    
    Args:
        llm_response: LLM response text
    
    Returns:
        Parsed catalyst dictionary
    """
    result = {
        "product_market_catalysts": [],
        "cost_improvement_catalysts": [],
        "industry_cycle_catalysts": [],
        "policy_rate_catalysts": [],
        "parse_error": None
    }
    
    try:
        # Try to extract JSON
        json_str = None
        
        # Method 1: Find ```json code block
        if "```json" in llm_response:
            json_start = llm_response.find("```json") + 7
            json_end = llm_response.find("```", json_start)
            if json_end > json_start:
                json_str = llm_response[json_start:json_end].strip()
        
        # Method 2: Find ``` code block
        elif "```" in llm_response:
            json_start = llm_response.find("```") + 3
            json_end = llm_response.find("```", json_start)
            if json_end > json_start:
                json_str = llm_response[json_start:json_end].strip()
        
        # Method 3: Find first { to last }
        if not json_str:
            json_start = llm_response.find("{")
            json_end = llm_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = llm_response[json_start:json_end]
        
        if json_str:
            parsed = json.loads(json_str)
            
            # Extract catalysts by category
            result["product_market_catalysts"] = parsed.get("product_market_catalysts", [])
            result["cost_improvement_catalysts"] = parsed.get("cost_improvement_catalysts", [])
            result["industry_cycle_catalysts"] = parsed.get("industry_cycle_catalysts", [])
            result["policy_rate_catalysts"] = parsed.get("policy_rate_catalysts", [])
            
            # Save summary (if exists)
            if "summary" in parsed:
                result["llm_summary"] = parsed["summary"]
        
        else:
            result["parse_error"] = "Unable to extract JSON from response"
            print("⚠️ Unable to parse LLM response as JSON, attempting text extraction")
            # Try to extract key information as text
            result["raw_text_analysis"] = llm_response[:1000]  # Save first 1000 characters
    
    except json.JSONDecodeError as e:
        result["parse_error"] = f"JSON parse error: {str(e)}"
        print(f"⚠️ JSON parsing failed: {e}")
        result["raw_text_analysis"] = llm_response[:1000]
    
    except Exception as e:
        result["parse_error"] = f"Parse error: {str(e)}"
        print(f"⚠️ Failed to parse LLM response: {e}")
        result["raw_text_analysis"] = llm_response[:1000]
    
    return result

def generate_catalyst_summary(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate catalyst summary
    
    Args:
        analysis_result: Analysis results
    
    Returns:
        Summary dictionary
    """
    summary = {
        "total_catalysts": 0,
        "high_impact_count": 0,
        "short_term_count": 0,
        "long_term_count": 0,
        "key_catalysts": []
    }
    
    try:
        all_catalysts = (
            analysis_result.get("product_market_catalysts", []) +
            analysis_result.get("cost_improvement_catalysts", []) +
            analysis_result.get("industry_cycle_catalysts", []) +
            analysis_result.get("policy_rate_catalysts", [])
        )
        
        summary["total_catalysts"] = len(all_catalysts)
        summary["high_impact_count"] = sum(1 for c in all_catalysts if c.get("impact") == "High")
        summary["short_term_count"] = sum(1 for c in all_catalysts if c.get("timeframe") == "Short-term")
        summary["long_term_count"] = sum(1 for c in all_catalysts if c.get("timeframe") == "Long-term")
        
        # Extract key catalysts (high impact, high confidence)
        key_catalysts = [
            c for c in all_catalysts
            if c.get("impact") == "High" and c.get("confidence") in ["High", "Medium"]
        ]
        # Sort by impact level and confidence
        key_catalysts.sort(
            key=lambda x: (
                3 if x.get("impact") == "High" else 2 if x.get("impact") == "Medium" else 1,
                3 if x.get("confidence") == "High" else 2 if x.get("confidence") == "Medium" else 1
            ),
            reverse=True
        )
        summary["key_catalysts"] = key_catalysts[:5]  # Maximum 5
    
    except Exception as e:
        print(f"⚠️ Error generating catalyst summary: {e}")
    
    return summary

# ==================== Comprehensive Qualitative Analysis ====================

def comprehensive_qualitative_analysis(
    symbol: str,
    financial_analysis: Dict[str, Any] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
    news_limit: int = 50
) -> Dict[str, Any]:
    """
    Comprehensive qualitative analysis (news collection + LLM catalyst analysis)
    
    Args:
        symbol: Stock symbol
        financial_analysis: Financial analysis results (optional)
        api_key: OpenAI API Key
        base_url: OpenAI Base URL
        model: AI model name
        news_limit: Maximum number of news items
    
    Returns:
        Comprehensive qualitative analysis results
    """
    result = {
        "symbol": symbol.upper(),
        "news_data": {},
        "catalyst_analysis": {},
        "summary": {}
    }
    
    print(f"\n📊 Starting qualitative analysis: {symbol}")
    
    try:
        # 1. Fetch news
        print("  📰 Step 1/2: Collecting news data...")
        news_data = fetch_stock_news(symbol, limit=news_limit)
        result["news_data"] = news_data
        
        # 2. LLM catalyst analysis
        if news_data.get("news_count", 0) > 0:
            print("  🤖 Step 2/2: LLM catalyst analysis...")
            catalyst_analysis = analyze_catalysts_with_llm(
                symbol=symbol,
                news_data=news_data,
                financial_analysis=financial_analysis,
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            result["catalyst_analysis"] = catalyst_analysis
        else:
            print("  ⚠️ No news data, skipping LLM analysis")
            result["catalyst_analysis"] = {"error": "No news data"}
        
        # 3. Generate comprehensive summary
        result["summary"] = generate_comprehensive_qualitative_summary(result)
        
        print(f"\n✅ {symbol} qualitative analysis completed")
    
    except Exception as e:
        print(f"⚠️ Qualitative analysis failed: {e}")
        import traceback
        traceback.print_exc()
        result["error"] = str(e)
    
    return result

def generate_comprehensive_qualitative_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive qualitative analysis summary"""
    summary = {
        "news_count": result.get("news_data", {}).get("news_count", 0),
        "catalysts_found": 0,
        "key_insights": [],
        "investment_implications": ""
    }
    
    try:
        catalyst_analysis = result.get("catalyst_analysis", {})
        cat_summary = catalyst_analysis.get("summary", {})
        
        summary["catalysts_found"] = cat_summary.get("total_catalysts", 0)
        summary["high_impact_catalysts"] = cat_summary.get("high_impact_count", 0)
        summary["short_term_catalysts"] = cat_summary.get("short_term_count", 0)
        summary["long_term_catalysts"] = cat_summary.get("long_term_count", 0)
        
        # Extract key insights
        if cat_summary.get("key_catalysts"):
            for catalyst in cat_summary["key_catalysts"][:3]:
                summary["key_insights"].append(
                    f"{catalyst.get('type', 'N/A')}: {catalyst.get('description', 'N/A')[:100]}"
                )
        
        # Investment implications
        if catalyst_analysis.get("llm_summary", {}).get("investment_implications"):
            summary["investment_implications"] = catalyst_analysis["llm_summary"]["investment_implications"]
    
    except Exception as e:
        print(f"⚠️ Error generating comprehensive summary: {e}")
    
    return summary

# ==================== Main Function (for testing) ====================

if __name__ == "__main__":
    # Test example
    print("=" * 60)
    print("Qualitative Analysis Module - Qualitative Analysis Test")
    print("=" * 60)
    
    symbol = "NVDA"
    print(f"\nPerforming qualitative analysis for {symbol}...")
    
    # Optional: provide financial analysis results
    financial_analysis = None
    # from financial_analysis import analyze_financial_statements
    # from data_ingestion import fetch_financial_statements
    # financial_data = fetch_financial_statements(symbol, years=5)
    # financial_analysis = analyze_financial_statements(financial_data)
    
    # Comprehensive qualitative analysis
    result = comprehensive_qualitative_analysis(
        symbol=symbol,
        financial_analysis=financial_analysis,
        news_limit=30
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("Analysis Results:")
    print("=" * 60)
    
    print(f"\n📰 News Data:")
    print(f"  News Count: {result['news_data'].get('news_count', 0)}")
    
    print(f"\n🚀 Catalyst Analysis:")
    catalyst_analysis = result.get("catalyst_analysis", {})
    if not catalyst_analysis.get("error"):
        cat_summary = catalyst_analysis.get("summary", {})
        print(f"  Total Catalysts: {cat_summary.get('total_catalysts', 0)}")
        print(f"  High Impact Catalysts: {cat_summary.get('high_impact_count', 0)}")
        print(f"  Short-term Catalysts: {cat_summary.get('short_term_count', 0)}")
        print(f"  Long-term Catalysts: {cat_summary.get('long_term_count', 0)}")
        
        if cat_summary.get("key_catalysts"):
            print(f"\n  Key Catalysts:")
            for cat in cat_summary["key_catalysts"][:3]:
                print(f"    • {cat.get('type', 'N/A')}: {cat.get('description', 'N/A')[:80]}...")
                print(f"      Impact: {cat.get('impact', 'N/A')}, Timeframe: {cat.get('timeframe', 'N/A')}")
    else:
        print(f"  ⚠️ {catalyst_analysis.get('error', 'Unknown error')}")
    
    print("\n✅ Test completed!")

