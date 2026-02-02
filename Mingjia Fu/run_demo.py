# run_demo.py
"""
Pipeline:
1) ingestion
2) processing
3) ratios
4) valuation (multiples + DCF)
5) risk synthesis
6) figures
7) memo (LLM-generated via local Ollama)

Output paths are controlled by data/config.json -> output.
Defaults (if missing):
- tables_dir:  outputs/tables
- figures_dir: outputs/figures
- memo_path:   outputs/memo.md
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_config(path: str = "data/config.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unpack_ratios(obj: Any) -> Tuple[Any, Any, List[str]]:
    """
    Support:
    - RatiosResult with ratios_df / summary_df / warnings
    - RatiosResult with ratios / summary / warnings
    - tuple: (ratios_df, summary_df, warnings)
    """
    if isinstance(obj, tuple) and len(obj) == 3:
        return obj[0], obj[1], list(obj[2] or [])

    if hasattr(obj, "ratios_df"):
        return (
            getattr(obj, "ratios_df"),
            getattr(obj, "summary_df", None),
            list(getattr(obj, "warnings", []) or []),
        )

    if hasattr(obj, "ratios"):
        return (
            getattr(obj, "ratios"),
            getattr(obj, "summary", None),
            list(getattr(obj, "warnings", []) or []),
        )

    raise TypeError("compute_ratios() returned an unexpected shape.")


def _unpack_multiples(obj: Any) -> Tuple[Any, Dict[str, Any], List[str]]:
    """
    Support:
    - tuple: (valuation_df, summary_dict, warnings)
    - dataclass-ish: df/valuation_df/table + summary + warnings
    """
    if isinstance(obj, tuple) and len(obj) == 3:
        df, summary, warnings = obj
        return df, dict(summary or {}), list(warnings or [])

    for df_attr in ("df", "valuation_df", "table"):
        if hasattr(obj, df_attr):
            valuation_df = getattr(obj, df_attr)
            summary = getattr(obj, "summary", {}) or {}
            warnings = getattr(obj, "warnings", []) or []
            return valuation_df, dict(summary), list(warnings)

    raise TypeError("run_multiples_valuation() returned an unexpected shape.")


def _unpack_dcf(obj: Any) -> Tuple[Any, Dict[str, Any], List[str]]:
    """
    Support:
    - DCFResult with dcf_table + summary + warnings
    - tuple: (dcf_table, summary_dict, warnings)
    """
    if isinstance(obj, tuple) and len(obj) == 3:
        table, summary, warnings = obj
        return table, dict(summary or {}), list(warnings or [])

    if hasattr(obj, "dcf_table"):
        table = getattr(obj, "dcf_table")
        summary = getattr(obj, "summary", {}) or {}
        warnings = getattr(obj, "warnings", []) or []
        return table, dict(summary), list(warnings)

    raise TypeError("run_simple_dcf() returned an unexpected shape.")


def main() -> None:
    print("=== Running Fundamental Analyst Agent Demo ===")

    cfg = load_config()

    ticker = cfg.get("ticker", "MSFT")
    years = int(cfg.get("years", 5))

    # ---- output paths: CONFIG IS SOURCE OF TRUTH ----
    out_cfg = cfg.get("output", {}) or {}
    tables_dir = out_cfg.get("tables_dir", "outputs/tables")
    figures_dir = out_cfg.get("figures_dir", "outputs/figures")
    memo_path = out_cfg.get("memo_path", "outputs/memo.md")

    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    Path(memo_path).parent.mkdir(parents=True, exist_ok=True)

    print("\nOutput paths (from config):")
    print(f"- tables_dir:  {tables_dir}")
    print(f"- figures_dir: {figures_dir}")
    print(f"- memo_path:   {memo_path}")

    # --- Core imports ---
    from agent.data_ingestion import fetch_financials_5y, save_financials_table
    from agent.processing import process_financials
    from agent.ratios import compute_ratios, save_ratios_table, save_ratios_summary
    from agent.valuation_multiples import run_multiples_valuation, save_valuation_table

    # DCF (expected)
    from agent.valuation_dcf import run_simple_dcf, save_dcf_table

    # Risk synthesis (expected)
    from agent.risk_synthesis import compute_risk_synthesis, save_risk_table

    # Figures (expected)
    from agent.figures import generate_all_figures

    # Memo generator (LOCAL LLM via Ollama)
    from agent.memo_generator import generate_investment_memo

    # 1) Ingestion
    ing = fetch_financials_5y(ticker, years)

    # 2) Processing
    cleaned, proc_warnings = process_financials(ing.financials)

    # Save financials
    financials_path = save_financials_table(cleaned, tables_dir)

    # 3) Ratios
    ratios_obj = compute_ratios(cleaned)
    ratios_df, ratios_summary_df, ratios_warnings = _unpack_ratios(ratios_obj)

    ratios_path = save_ratios_table(ratios_df, tables_dir)
    ratios_summary_path = save_ratios_summary(ratios_summary_df, tables_dir)

    # 4A) Multiples valuation
    multiples_obj = run_multiples_valuation(
        ticker=ticker,
        config=cfg,
        financials=cleaned,
    )
    valuation_df, valuation_summary, valuation_warnings = _unpack_multiples(multiples_obj)
    valuation_path = save_valuation_table(valuation_df, tables_dir)

    # 4B) DCF valuation
    dcf_obj = run_simple_dcf(
        financials=cleaned,
        config=cfg,
        shares_outstanding=ing.metadata.get("shares_outstanding"),
    )
    dcf_table, dcf_summary, dcf_warnings = _unpack_dcf(dcf_obj)
    dcf_path = save_dcf_table(dcf_table, tables_dir)

    # 5) Risk synthesis
    pipeline_warnings: List[str] = []
    pipeline_warnings.extend(list(ing.warnings or []))
    pipeline_warnings.extend(list(proc_warnings or []))
    pipeline_warnings.extend(list(ratios_warnings or []))
    pipeline_warnings.extend(list(valuation_warnings or []))
    pipeline_warnings.extend(list(dcf_warnings or []))

    risk_res = compute_risk_synthesis(
        cfg=cfg,
        financials_df=cleaned,
        ratios_df=ratios_df,
        valuation_df=valuation_df,
        pipeline_warnings=pipeline_warnings,
    )
    risk_path = save_risk_table(risk_res.risk_table, tables_dir)

    # 6) Figures
    fig_res = generate_all_figures(
        financials_df=cleaned,
        ratios_df=ratios_df,
        dcf_df=None,  # project intentionally does NOT generate DCF grid
        out_dir=figures_dir,
    )
    figures_saved = list(fig_res.paths or [])
    figures_warnings = list(fig_res.warnings or [])

    # 7) Memo (LOCAL LLM)
    memo_text = generate_investment_memo(
        cfg=cfg,
        metadata=ing.metadata,
        financials_df=cleaned,
        ratios_df=ratios_df,
        ratios_summary_df=ratios_summary_df,
        valuation_df=valuation_df,
        valuation_summary=valuation_summary,
        dcf_df=dcf_table,
        dcf_summary=dcf_summary,
        risk_df=risk_res.risk_table,
        risk_summary=risk_res.summary,
        ingestion_warnings=ing.warnings,
        processing_warnings=proc_warnings,
        ratios_warnings=ratios_warnings,
        valuation_warnings=valuation_warnings,
        dcf_warnings=dcf_warnings,
        risk_warnings=risk_res.warnings,
        figures_warnings=figures_warnings,
    )
    out = Path(memo_path)
    out.write_text(memo_text, encoding="utf-8")
    memo_saved = str(out)

    # --- Console summary ---
    print("\n=== Demo run complete ===")
    print(f"Ticker: {ing.metadata.get('ticker')}")
    print(f"Company: {ing.metadata.get('company_name')}")
    print(f"Years (available): {ing.metadata.get('years')}")

    print("\nSaved tables:")
    print(f"- Financials: {financials_path}")
    print(f"- Ratios: {ratios_path}")
    print(f"- Ratios summary: {ratios_summary_path}")
    print(f"- Valuation (multiples): {valuation_path}")
    print(f"- Valuation (DCF): {dcf_path}")
    print(f"- Risk summary: {risk_path}")

    print("\nSaved figures:")
    if figures_saved:
        for p in figures_saved:
            print(f"- {p}")
    else:
        print("- (none)")

    print("\nSaved memo:")
    print(f"- {memo_saved}")

    if valuation_summary:
        print("\nKey multiples output:")
        for k, v in valuation_summary.items():
            print(f"- {k}: {v}")

    if dcf_summary:
        print("\nKey DCF output:")
        for k, v in dcf_summary.items():
            print(f"- {k}: {v}")

    if risk_res.summary:
        print("\nKey risk output:")
        for k, v in risk_res.summary.items():
            print(f"- {k}: {v}")

    all_warnings: List[str] = []
    all_warnings.extend(list(ing.warnings or []))
    all_warnings.extend(list(proc_warnings or []))
    all_warnings.extend(list(ratios_warnings or []))
    all_warnings.extend(list(valuation_warnings or []))
    all_warnings.extend(list(dcf_warnings or []))
    all_warnings.extend(list(risk_res.warnings or []))
    all_warnings.extend(list(figures_warnings or []))

    if all_warnings:
        print("\nWARNINGS:")
        for w in all_warnings:
            print("-", w)

    # Demo hint: ensure Ollama is running
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    print(f"\nLLM backend: local Ollama at {host}")
    print(
        "If the memo looks like a fallback memo, make sure Ollama is running and the model is pulled "
        "(e.g., `ollama pull llama3.2`)."
    )


if __name__ == "__main__":
    main()
