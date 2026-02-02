# Fundamental Analysis Agent

An AI-powered fundamental analysis tool for comprehensive stock investment research. This application provides automated financial analysis, valuation, peer comparison, and investment memo generation.

## Features

- **Financial Data Collection**: Automatically fetches 5-7 years of financial statements (Income Statement, Balance Sheet, Cash Flow Statement) from Alpha Vantage API
- **Financial Analysis**: Calculates and analyzes profitability, growth, leverage, and efficiency ratios
- **Valuation Analysis**: DCF (Discounted Cash Flow) and multiples-based valuation models
- **Peer Comparison**: Compares target company against peer companies across multiple dimensions
- **Earnings Quality Analysis**: Evaluates earnings sustainability through cash flow, accruals, and volatility analysis
- **Qualitative Analysis**: Identifies catalysts and investment opportunities using LLM
- **Investment Memo Generation**: Automatically generates professional investment research reports

## Requirements

- Python 3.8+
- OpenAI API Key (for LLM analysis and memo generation)
- Alpha Vantage API Key (for financial data)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Fundamental-Analyst-Agent
```

2. Create and activate a virtual environment (recommended):

**Using Conda:**
```bash
conda create -n fun_agent python=3.12
conda activate fun_agent
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### API Keys

You need to configure two API keys:

1. **OpenAI API Key**: For LLM-powered analysis and investment memo generation
2. **Alpha Vantage API Key**: For fetching financial statement data

You can set them via:
- configure them in the web interface when running the app

## Usage

### Web Interface (Recommended)

Start the Streamlit web application:

```bash
# On Linux/Mac
./start_app.sh

# On Windows
start_app.bat

# Or directly
streamlit run src/app.py
```

### Demo Runing

```
python run_demo.py
```


Then:
1. Configure your API keys in the web interface
2. Select a target company (by symbol or search)
3. Optionally select peer companies for comparison
4. Click "Start Analysis" to run the complete analysis

## Output

The analysis generates:
- Investment recommendation (Buy/Hold/Sell)
- Target price and upside potential
- Comprehensive financial analysis
- Peer comparison report
- Earnings quality assessment
- Catalyst analysis
- Full investment memo (downloadable)

### Report Storage

All analysis reports are automatically saved to `report/{stock_code}/` directory, including:
- **Investment Memo** (`investment_memo_YYYYMMDD_HHMMSS.txt`)
- **Peer Comparison Report** (`peer_comparison_report_YYYYMMDD_HHMMSS.txt`)
- **Earnings Quality Report** (`earnings_quality_report_YYYYMMDD_HHMMSS.txt`)
- **Financial Statements** (CSV format: income statement, balance sheet, cash flow - annual and quarterly)
- **Financial Ratios** (`financial_ratios_YYYYMMDD_HHMMSS.csv`)
- **Valuation Data** (`valuation_YYYYMMDD_HHMMSS.csv`)
- **Complete Analysis Results** (`complete_analysis_YYYYMMDD_HHMMSS.json`)

Results can also be downloaded directly from the web interface as:
- JSON format (complete analysis data)
- Text format (investment memo)
- CSV format (financial statements and ratios)

## Notes

- Ensure you have valid API keys with sufficient quotas
- Analysis may take several minutes depending on the number of peer companies
- Financial data is fetched from Alpha Vantage API (free tier has rate limits)

## License

This project is provided as-is for educational and research purposes.

