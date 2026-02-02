# msft_agent_project
<a id="top"></a>

LLM-assisted trading agent.

This repo implements a lightweight end-to-end pipeline:
data download → technical signals → LLM decision → backtest → report


<details>
<summary>Table of contents (click to expand)</summary>

- [What this project does](#what-this-project-does)
- [Quickstart](#quickstart)
  - [Option A: Streamlit Web App (UI input, no API.env required)](#option-a-streamlit-web-app-ui-input-no-apienv-required)
  - [Option B: CLI run_demo.py (env file / environment variables)](#option-b-cli-rundemopy-env-file--environment-variables)
- [Web App (Streamlit)](#web-app-streamlit)
- [Configuration](#configuration)
  - [Two ways to provide API key](#two-ways-to-provide-api-key-recommended)
  - [Precedence](#precedence-which-key-is-used)
- [Usage](#usage)
  - [Modes](#modes)
  - [Rebalance options](#rebalance-options)
  - [Examples](#examples)
- [Outputs and reproducibility](#outputs-and-reproducibility)
- [Repository structure](#repository-structure)
- [Notes & limitations](#notes-and-limitations)
- [Troubleshooting](#troubleshooting)

</details>

---

## What this project does

- Market data download via `yfinance`
- Technical indicators (MA / RSI / MACD, etc.)
- LLM decision: the agent asks an LLM to return a strict JSON decision
- Backtest engine: rebalancing + transaction cost model
- Report generation: exports both `*.md` and `*.pdf`
- Optional Streamlit web app: enter API key + ticker + parameters in a UI 

[Back to top](#top)

---

## Quickstart

Prerequisites:
- Python 3.9+
- An OpenAI API key

### Option A: Streamlit Web App (UI input, no API.env required)

1) Create and activate an environment

Example (conda):
```bash
conda create -n msft-agent python=3.10 -y
conda activate msft-agent
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Start the app (run in a terminal, not inside a normal Python cell)
```bash
python -m streamlit run app.py
```

4) Open the Local URL shown in the terminal, then:
- paste your OpenAI API key in the UI
- choose a ticker (e.g., MSFT)
- choose mode / years / rebalance / cost rate
- click Run

[Back to top](#top)

### Option B: CLI run_demo.py (env file / environment variables)

1) Create and activate an environment (same as above)

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Add your API key 
- Put `OPENAI_API_KEY=...` in `API.env`.

4) Run the demo

Full pipeline (data → signals → agent → backtest → report):
```bash
python run_demo.py --mode agent --ticker MSFT --years 10 --rebalance weekly --outdir outputs --prefix msft
```

Backtest only (no LLM decision step):
```bash
python run_demo.py --mode backtest --ticker MSFT --years 10 --rebalance weekly --outdir outputs --prefix msft
```

[Back to top](#top)

---

## Web App (Streamlit)
<a id="web-app-streamlit"></a>

Start:
```bash
python -m streamlit run app.py
```

[Back to top](#top)

---

## Configuration

Configuration is loaded from environment variables (see `settings.py`).

Required:
- `OPENAI_API_KEY`

Optional:
- `OPENAI_MODEL` (default: `gpt-4.1-mini`)
- `OPENAI_MODEL_STRONG` (default: `gpt-4.1`)
- `OPENAI_TEMPERATURE` (default: `0.2`)
- `OPENAI_TIMEOUT_S` (default: `60`)
- `LOG_LEVEL` (default: `INFO`)

### Two ways to provide API key

1) Web UI input (Streamlit)
- Paste API key into the app.
- The app injects the key into the current session only.
- You do not need `API.env` for this path.

2) Env file / environment variables (CLI)
- Put `OPENAI_API_KEY=...` into `API.env`

### Precedence

- If you run the Streamlit app and provide a key in the UI, that key is used for that session.
- If the UI key is empty, the code falls back to environment variables / `API.env` / `.env`.

[Back to top](#top)

---

## Usage

### Modes

- `--mode agent`
  - uses LLM decision step (JSON decision) + backtest + report
- `--mode backtest`
  - no LLM; runs signals + backtest + report (deterministic given same data)

### Rebalance options

- `daily`: rebalance every trading day (more turnover, cost matters more)
- `weekly`: rebalance once per week (default)
- `monthly`: rebalance once per month (lower turnover)

### Examples

Agent (LLM decision):
```bash
python run_demo.py --mode agent --ticker MSFT --years 10 --rebalance weekly --outdir outputs --prefix msft_agent
```

Backtest only:
```bash
python run_demo.py --mode backtest --ticker MSFT --years 10 --rebalance monthly --outdir outputs --prefix msft_monthly
```

Daily rebalance:
```bash
python run_demo.py --mode backtest --ticker MSFT --years 10 --rebalance daily --outdir outputs --prefix msft_daily
```

[Back to top](#top)

---

## Outputs and reproducibility

By default, results are written to the folder specified by `--outdir` (default: `outputs/`).
If the folder does not exist, the pipeline creates it automatically.

Typical artifacts:
- `{prefix}_report.pdf`
- `{prefix}_report.md`
- `{prefix}_metrics.csv`
- `{prefix}_run_manifest.json`
- `{prefix}_run_summary.json`
- `{prefix}_agent_decision.json` (only for mode=agent)
- optional: charts/images used in the report

Reproducibility notes:
- Data source: yfinance daily OHLCV
- Transaction cost model is simplified (fixed cost rate per turnover)
- LLM decisions can vary across runs unless determinism is enforced

[Back to top](#top)

---

## Repository structure

A minimal map of the codebase:

- `run_demo.py`
  - project entrypoint
- `app.py`
  - Streamlit UI (enter API key + ticker + parameters)
- `agent.py`
  - generic agent loop and prompt/schema wiring
- `llm_client.py`
  - OpenAI client wrapper (JSON output + retries)
- `tools_market_data.py`
  - data download and preprocessing
- `tools_backtest.py`
  - backtest logic, portfolio accounting, metrics
- `agent_trader.py`
  - trading-agent glue code (signals → decision → backtest)
- `report_writer.py`
  - Markdown + PDF report generation (reportlab)
- `settings.py`
  - environment variable loading and validation
- `requirements.txt`
  - minimal runnable dependency set
- `starter.ipynb`
  - optional notebook for interactive runs
- `outputs/`
  - generated artifacts

[Back to top](#top)

---

## Notes and limitations

- Single-asset backtest (MSFT example). Multi-asset support is not implemented.
- Slippage and market impact are not modeled.
- LLM decisions can be stochastic; use backtest-only mode for purely deterministic runs.

[Back to top](#top)

---

## Troubleshooting

1) API key not found
- Streamlit: paste a key into the UI
- CLI: ensure `.env` or `API.env` exists in project root and contains `OPENAI_API_KEY`

2) PDF generation fails
- Ensure `reportlab` is installed
- Ensure output directory is writable

3) yfinance download issues
- Check network access
- Retry later (Yahoo rate limits can happen)

[Back to top](#top)
