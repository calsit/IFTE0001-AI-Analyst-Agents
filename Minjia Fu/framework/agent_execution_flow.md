# Agent Execution Flow


The purpose of this document is to explain how the agent operates step by step, from data ingestion to investment memo generation, and how each component contributes to the final investment decision.

This document focuses on operational logic and data flow rather than theoretical motivation, which is covered separately in the framework document.

---

## 1. Entry point and configuration loading

The agent is executed through a single entry-point script:

```text
run_demo.py
```

At runtime, the agent first loads all configuration parameters from:

```text
data/config.json
```

This configuration file defines:
- the target ticker and look-back window,
- valuation reference parameters and the BUY/HOLD/SELL hurdle,
- DCF assumptions (single-point cross-check),
- risk synthesis weights and thresholds,
- local LLM settings (Ollama host/model),
- output paths.

All modeling assumptions are externalized before any financial data are accessed, ensuring transparency and reproducibility.

---

## 2. Financial data ingestion

The first analytical step is the ingestion of historical financial statement data, implemented in:

```text
agent/data_ingestion.py
```

Using the `yfinance` library, the agent retrieves annual financial statements from Yahoo Finance, including:
- income statements,
- balance sheets,
- cash flow statements.

Because financial line item labels may vary across reporting periods and tickers, field mappings are applied.
Missing fields are allowed and are explicitly recorded as warnings rather than being imputed.

Output:

A multi-year financial statements table written to the tables directory specified in `config.json`.

---

## 3. Financial data processing and validation

The ingested data are passed to a processing stage implemented in:

```text
agent/processing.py
```

This step:
- enforces numeric data types,
- sorts observations by fiscal year,
- drops fully empty fiscal years,
- flags year-level data gaps.

No imputation, smoothing, or extrapolation is performed.  
All issues are propagated as warnings to later stages of the pipeline.

---

## 4. Financial ratio computation

Once the processed financial table is available, the agent computes a focused set of financial ratios using:

```text
agent/ratios.py
```

The ratio framework covers:
- profitability (margins),
- leverage and capital structure,
- returns and efficiency,
- growth indicators.

Ratios are computed on a year-by-year basis and summarized using the most recent available fiscal year.

Outputs:

- A detailed year-by-year ratio table
- A latest-year ratio summary table

Both outputs are written to the tables directory specified in `config.json`.

---

## 5. Valuation analysis

The agent applies two complementary valuation approaches.

### 5.1 Multiples-based valuation

Relative valuation is implemented in:

```text
agent/valuation_multiples.py
```

Key steps include:
- computing trailing valuation multiples,
- comparing them to a reference benchmark defined in the configuration file,
- translating the comparison into an implied upside or downside signal.

This signal is evaluated against a predefined decision hurdle.

Output:

A valuation output table written to the tables directory specified in `config.json`.

---

### 5.2 Simplified DCF valuation

A simplified discounted cash flow valuation is computed in:

```text
agent/valuation_dcf.py
```

The DCF model:
- projects free cash flow over a finite horizon,
- applies conservative growth assumptions,
- uses explicit discount and terminal growth rates.

The DCF output serves as a robustness check rather than a definitive price estimate.

Output:

A valuation output table written to the tables directory specified in `config.json`.

---

## 6. Visual diagnostics

To support interpretation, the agent generates a limited set of diagnostic figures using:

```text
agent/figures.py
```

These figures illustrate:
- revenue and free cash flow trends,
- margin evolution over time.

DCF outputs are reported numerically rather than visualized, as the model
produces a single-point intrinsic value rather than a full sensitivity grid.

Outputs:

A small set of diagnostic figures written to the figures directory specified in `config.json`.

Visualizations are intentionally restrained and are used to support analytical reasoning rather than to drive decisions mechanically.

---

## 7. Risk synthesis

After valuation outputs are produced, the agent performs a dedicated risk synthesis stage.

This stage:
- aggregates selected quantitative risk proxies,
- applies configurable weights and thresholds from the configuration file,
- produces an interpretable risk signal that constrains conviction.

Risk synthesis is designed to contextualise valuation outputs rather than to override them.

Output:

A risk synthesis summary table written to the tables directory specified in `config.json`.

---

## 8. Investment memo generation (LLM synthesis)

The final stage is investment memo generation, implemented in:

```text
agent/memo_generator.py
```

Role of the LLM:
- The LLM is used only for narrative synthesis and structured reporting.
- All numbers come strictly from the pipeline payload (tables + summaries).
- The memo is generated under strict formatting rules to reduce hallucinations
(e.g., correct % vs x formatting for ratios).

Default LLM backend (free):
- A local model served via Ollama (no paid API key required).

Configuration:
- `OLLAMA_HOST (default: http://localhost:11434)`
- `cfg["llm"]["model"] (default: llama3.2)`

If the local LLM is unavailable, a deterministic fallback memo is generated so the pipeline remains runnable end-to-end.

Output:

A Markdown investment memo written to the memo path specified in `config.json`.

---

## 9. End-to-end execution and reproducibility

Running the following command executes the full pipeline:

```bash
python run_demo.py
```

All intermediate and final outputs are written to disk according to the output paths defined in `data/config.json`, enabling independent inspection, auditing, and extension.

The modular design allows the agent to be reused for other securities by modifying configuration parameters without changing core analytical logic.