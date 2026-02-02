# Investment Thesis & Risk Synthesis Framework

## 1. Purpose and positioning

This framework defines the analytical logic of an **Investment Thesis & Risk Synthesis Agent** applied to Microsoft Corporation (MSFT).

The agent is designed to replicate the workflow of a buy-side fundamental analyst.
It does not attempt to forecast prices mechanically, nor to optimize a numerical objective.
Instead, it synthesizes financial performance, valuation signals, and structured risk considerations into a disciplined investment view.

The core objective is to support judgement under uncertainty via:
- explicit assumptions,
- transparent decision rules,
- and auditable outputs.

---

## 2. Agent scope

The agent operates within a deliberately constrained scope:

- publicly available historical financial statement data (annual),
- core profitability, leverage, efficiency, return, and growth metrics,
- simple and interpretable valuation techniques,
- explicit decision thresholds defined ex ante,
- a lightweight and transparent risk synthesis layer,
- LLM-driven memo generation for narrative synthesis only.

The agent explicitly does **not**:
- forecast short-term prices,
- optimize portfolios,
- apply black-box machine learning models.

This constrained scope prioritizes transparency, interpretability, and reproducibility.

---

## 3. Data foundation

Using the `yfinance` library, the agent retrieves annual financial statements from Yahoo Finance, including:
- income statement,
- balance sheet,
- cash flow statement.

Because reporting labels may vary across years, the ingestion layer applies flexible field mappings.
Missing values are retained rather than imputed, and all data gaps are surfaced explicitly as warnings.

This design choice reflects standard research practice: incomplete information is treated as a risk input rather than hidden.

---

## 4. Processing and standardization

Raw financial data are passed through a processing stage that:
- enforces numeric consistency,
- aligns observations by fiscal year,
- drops fully empty fiscal years,
- validates the presence of core analytical fields.

No smoothing, extrapolation, or interpolation is performed.
The processed dataset is therefore a clean but faithful representation of the underlying disclosures.

---

## 5. Financial ratio analysis

From the processed statements, the agent computes a focused set of ratios capturing:

- profitability: EBIT margin, net margin, free cash flow margin,
- leverage: debt-to-equity (as a multiple),
- returns/efficiency: ROE, asset turnover (as a multiple),
- growth dynamics: revenue CAGR, free cash flow CAGR.

Ratios inform business quality and sustainability but do not mechanically drive the final recommendation.
They are used to contextualize valuation and risk.

---

## 6. Valuation logic

Valuation is implemented as a two-layer decision support system.

### 6.1 Multiples-based valuation (primary decision signal)

The primary signal comes from a multiples-based screen (e.g., trailing P/E, P/FCF).
A reference multiple and decision hurdle are defined ex ante in `config.json`.

Decision rule (core):
- compute implied upside vs the reference multiple,
- recommend BUY if upside clears the configured hurdle,
- otherwise HOLD/SELL depending on the configured logic.

This mirrors common practice where relative valuation provides a practical signal under uncertainty.

### 6.2 Simplified DCF (secondary cross-check)

The DCF model is implemented as a single-point valuation anchor.
Its role is to:
- cross-check the direction implied by the multiples signal,
- highlight sensitivity to key assumptions (WACC, terminal growth),
- provide a conservative anchor for discussion.

The DCF does not override the multiples-driven recommendation unless explicitly justified.

---

## 7. Risk synthesis logic

Risk is treated as an explicit analytical stage rather than an afterthought.

The agent implements a lightweight, interpretable risk synthesis step that:
- aggregates selected quantitative proxies,
- applies configurable weights and thresholds,
- produces:
  - risk_score (0–1),
  - risk_level (low/medium/high),
  - and interpretable risk drivers.

Risk synthesis constrains conviction:
- higher risk generally reduces conviction,
- warnings and data gaps are treated as additional uncertainty.

All parameters are externalized in `config.json`, supporting reproducibility.

---

## 8. LLM memo generation and governance

The final output is an LLM-generated investment memo.
The LLM’s role is narrowly defined:

- narrative synthesis and structured reporting only,
- strict grounding in the pipeline payload (tables + summaries),
- strict formatting rules (e.g., % vs x) to reduce hallucinations,
- recommendation primarily driven by the multiples signal,
- conviction adjusted by risk_score / risk_level and warnings.

Default backend:
- local LLM via Ollama (free), so assessors can run end-to-end without paid API keys.

A deterministic fallback memo is provided if the local LLM is unavailable.

---

## 9. Reproducibility and audibility

All assumptions, thresholds, and parameters are defined externally in `data/config.json`.
All intermediate outputs are saved as tables and figures, enabling traceability from raw inputs to final recommendation.

This ensures the agent’s conclusions can be replicated, audited, and critically evaluated.

---

## 10. Summary

This framework formalizes a disciplined approach to investment thesis construction using an agent-based architecture.

By prioritizing transparency, interpretability, and explicit risk acknowledgement, the agent reflects how fundamental analysis is conducted in real-world research workflows.