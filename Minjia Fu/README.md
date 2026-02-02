# Fundamental Analyst Agent — Investment Thesis & Risk Synthesis (MSFT)

This project implements a **Fundamental Analyst Agent** for Microsoft Corporation (MSFT), designed to support **investment thesis construction and risk synthesis** using publicly available financial data.

The agent follows a transparent and end-to-end pipeline that transforms raw financial statements into:
- cleaned tables,
- ratio analysis,
- multiples-based valuation (primary decision signal),
- simplified DCF (secondary cross-check),
- interpretable risk synthesis (constrains conviction),
- and an LLM-generated investment memo.

The emphasis is on **reproducibility, interpretability, and disciplined judgement under uncertainty** rather than model complexity.

---

## Project objective

Demonstrate a realistic buy-side research workflow:
- ingest and standardize financial statement data,
- evaluate firm performance using ratios,
- apply interpretable valuation techniques (multiples + simplified DCF anchor),
- formalize a lightweight risk synthesis layer,
- generate a coherent and auditable memo.

Valuation results are treated as **decision inputs**, not precise price targets.

---

## LLM usage

In this project:
- the LLM is used only for narrative synthesis and structured reporting,
- all numbers must come strictly from the pipeline payload,
- strict formatting rules reduce hallucinations (e.g., % vs x for ratios),
- recommendation is driven primarily by the multiples signal.

Default backend (free):
- **Local LLM via Ollama** (no paid API key required).

If the local LLM is unavailable, the agent generates a deterministic fallback memo so the pipeline remains runnable end-to-end.

---

## Project structure

```text
msft_fundamental_agent/
│
├── agent/
│   ├── __init__.py
│   ├── data_ingestion.py        # Fetches & standardizes financial statements (yfinance)
│   ├── processing.py            # Cleans/validates financial data and surfaces warnings
│   ├── ratios.py                # Computes profitability, leverage, growth & return ratios
│   ├── valuation_multiples.py   # Relative valuation (primary decision signal)
│   ├── valuation_dcf.py         # Simplified DCF cross-check (single point)
│   ├── figures.py               # Diagnostic charts (revenue/FCF, margins)
│   ├── risk_synthesis.py        # Risk score/level + interpretable drivers (config-based)
│   └── memo_generator.py        # Memo generation via local LLM (Ollama) + fallback
│
├── data/
│   └── config.json              # Assumptions, thresholds, LLM config, and output paths
│
├── framework/
│   ├── investment_thesis_risk_synthesis_framework.md
│   ├── agent_execution_flow.md
│   └── agent_limitations_and_reflection.md
│
├── outputs/
│   ├── tables/                  # Reproducible numerical outputs
│   ├── figures/                 # Diagnostic figures
│   └── memo.md                  # Final investment memo
│
├── run_demo.py                  # One-command end-to-end execution
├── requirements.txt
└── README.md
```

---

## Requirements

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Ollama (local LLM) and pull a model:

```bash
ollama pull llama3.2
```

Ollama runs a local server by default at:

```text
http://localhost:11434
```

---

## Running the demo

Run the end-to-end pipeline:

```bash
python run_demo.py
```

Output paths (important)

All outputs are written to disk according to the paths specified in:

```text
data/config.json
```

The `output` section of the configuration file defines:

- where tabular outputs are saved,
- where figures are saved,
- where the final investment memo is written.

This design ensures that **output locations are configurable and reproducible**, and that no paths are hard-coded in the analytical logic.

### Typical outputs (default configuration)

Under the default configuration, the pipeline produces:

**Tables**

- Financial statements (cleaned, multi-year)
- Financial ratios and ratio summary
- Valuation outputs (multiples and DCF)
- Risk synthesis summary

**Figures**

- Revenue and free cash flow trends
- Margin profile over time

**Memo**

- A Markdown investment memo synthesizing valuation and risk outputs

> **Note:** The exact filenames and directories are determined by `config.json`.  
> If output paths are modified in the configuration file, the pipeline and memo will automatically reflect those changes.

---

## Configuration

All assumptions and thresholds are defined externally in:

```text
data/config.json
```

This includes:

- ticker and look-back window,
- valuation reference multiple and decision hurdle,
- DCF parameters (projection years, WACC, terminal growth),
- risk synthesis weights and thresholds,
- output paths.

The agent can be adapted to other tickers by editing `config.json` without changing core logic.

---

## Limitations (summary)

- Data quality depends on third-party statement availability via Yahoo Finance.
- Models are simplified and sensitive to assumptions (reference multiple, WACC, terminal growth).
- Risk synthesis is proxy-based and does not replace qualitative judgement.
- Local LLM outputs can vary; a deterministic fallback memo is provided for robustness.

See `framework/agent_limitations_and_reflection.md` for details.