# Limitations and Reflection

This project implements an Investment Thesis & Risk Synthesis Agent using publicly available financial data and simplified valuation techniques.

While the agent produces coherent and reproducible outputs, several limitations should be acknowledged to contextualize the results and clarify the scope of interpretation.

---

## 1. Data limitations

The agent relies on annual financial statement data retrieved from Yahoo Finance via the `yfinance` library.

As a result:
- financial statement line items may be missing or inconsistently labelled,
- some fiscal years may be partially reported or unavailable,
- restatements and accounting adjustments are not explicitly handled.

No data imputation is performed.
Missing values are retained and surfaced explicitly as warnings.
This prioritizes transparency over completeness.

---

## 2. Modelling simplifications

The valuation framework is intentionally simplified to remain interpretable and coursework-friendly.

Key simplifications include:
- reliance on trailing multiples rather than forward-looking estimates,
- a single reference multiple defined ex ante in `config.json`,
- a simplified DCF model implemented as a single-point cross-check rather than a full scenario grid.

These choices support audibility but limit precision.
Valuation outputs should therefore be interpreted as **directional decision signals**, not point estimates.

---

## 3. Sensitivity to assumptions

Valuation outputs are sensitive to:
- the reference multiple and BUY hurdle (multiples screen),
- WACC and terminal growth (DCF),
- the assumed medium-term free cash flow growth rate.

The agent surfaces these assumptions explicitly in configuration and memo text but does not run full scenario or stress-testing grids.

---

## 4. Limitations of risk synthesis

The risk synthesis component is intentionally lightweight and interpretable.

Limitations include:
- reliance on proxy variables rather than direct measurement of qualitative risks,
- fixed weights and thresholds defined ex ante,
- absence of probabilistic modeling or scenario-based stress testing.

These limitations are deliberate design choices.
The objective is not to optimize a risk score, but to formalize how selected risks constrain valuation-driven conclusions and conviction.

---

## 5. LLM constraints and output governance

The large language model (LLM) is used exclusively for structured narrative synthesis in the final investment memo.
It does not perform any numerical computation, valuation, or risk scoring.

The LLM is explicitly constrained to:
- use only numbers provided in the structured pipeline payload,
- follow strict formatting rules (e.g., percentages vs. x-multiples) to reduce hallucinations,
- generate a recommendation driven primarily by the multiples-based valuation signal,
  with the DCF used strictly as a secondary cross-check.

Despite these constraints, local LLM outputs may still exhibit minor variability or formatting artifacts, as the model is a probabilistic text generator rather than a rule-based executor.

To preserve reproducibility and analytical integrity:
- all quantitative results are generated deterministically upstream and saved as CSV outputs,
- these tabular outputs constitute the single source of truth for the analysis,
- pipeline warnings are explicitly recorded and surfaced in the memo,
- and a deterministic fallback memo is produced if the local LLM is unavailable.

Accordingly, the investment memo should be interpreted as a narrative synthesis layer built on top of reproducible analytical outputs, rather than as an independent analytical artifact.

---

## 6. Scope of analysis

The agent focuses primarily on quantitative signals derived from historical financial performance and simple valuation mechanics.

Non-financial risks — such as competitive dynamics, technological disruption, regulatory developments, governance issues, and strategic execution — are not explicitly modeled.

The risk synthesis is therefore partial by design and should be complemented by qualitative judgement.

---

## 7. Reflection

This project demonstrates how a modular agent architecture can replicate key elements of a fundamental research workflow:
- data ingestion and validation,
- ratio-based performance assessment,
- valuation cross-checking (multiples + DCF anchor),
- structured risk synthesis that constrains conviction,
- synthesis into a coherent investment narrative.

A key insight is that the value of the agent is not numerical “accuracy” alone, but disciplined transparency:
- assumptions are explicit,
- warnings are surfaced,
- and narrative conclusions are grounded in auditable tables.