# agent/memo_generator.py
"""

This module represents the final Investment Thesis & Risk Synthesis stage.
It uses a LOCAL LLM (Ollama) to translate structured quantitative outputs into a concise,
professional 1–2 page memo, while keeping inputs auditable and reproducible.

Key points:
- LLM is used ONLY for narrative synthesis and structured reporting.
- All numbers come from the pipeline tables/summary (no invented data).
- Memo includes explicit assumptions, risk signal, and pipeline warnings.
- Default is FREE local execution via Ollama (no paid API key required).

Requirements:
- Install Ollama and pull a model, e.g.: ollama pull llama3.2
- Ensure Ollama is running (default server: http://localhost:11434)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


def _is_missing(x: Any) -> bool:
    return x is None or (isinstance(x, float) and pd.isna(x))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if _is_missing(x):
            return None
        return float(x)
    except Exception:
        return None


def _fmt_money_compact(x: Optional[float], currency: str = "USD") -> str:
    if x is None:
        return "N/A"
    try:
        v = float(x)
    except Exception:
        return "N/A"

    av = abs(v)
    if av >= 1e12:
        return f"{currency} {v/1e12:.2f}T"
    if av >= 1e9:
        return f"{currency} {v/1e9:.2f}B"
    if av >= 1e6:
        return f"{currency} {v/1e6:.2f}M"
    return f"{currency} {v:,.2f}"


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "N/A"


def _latest_year(df: pd.DataFrame) -> Optional[int]:
    if df is None or df.empty:
        return None
    try:
        return int(df.index.max())
    except Exception:
        return None


def _latest_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    return df.iloc[-1]


def _df_tail(df: Optional[pd.DataFrame], n: int = 5) -> Optional[List[Dict[str, Any]]]:
    """
    Convert last n rows to JSON-serializable list[dict] for auditability.
    Keeps memo prompt compact.
    """
    if df is None or df.empty:
        return None
    out = df.tail(n).copy()
    # make index a column for readability
    if out.index.name:
        out = out.reset_index()
    else:
        out = out.reset_index().rename(columns={"index": "index"})
    # coerce NaN -> None
    return json.loads(out.where(pd.notna(out), None).to_json(orient="records"))


def _table_to_records(df: Optional[pd.DataFrame], max_rows: int = 30) -> Optional[List[Dict[str, Any]]]:
    if df is None or df.empty:
        return None
    out = df.head(max_rows).copy()
    return json.loads(out.where(pd.notna(out), None).to_json(orient="records"))


def _ollama_available(host: str, timeout: float = 2.0) -> bool:
    """
    Quick health check: /api/tags should return JSON if Ollama server is up.
    """
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _call_ollama_chat(
    *,
    host: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> str:
    """
    Call Ollama /api/chat (non-streaming).
    """
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": float(temperature)},
    }

    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    msg = (data or {}).get("message") or {}
    text = msg.get("content") or ""
    return text.strip()


def _output_paths_from_cfg(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Source of truth for output paths.
    Aligns with run_demo.py:
      cfg["output"]["tables_dir"]
      cfg["output"]["figures_dir"]
      cfg["output"]["memo_path"]
    """
    out_cfg = cfg.get("output", {}) or {}
    tables_dir = str(out_cfg.get("tables_dir", "outputs/tables"))
    figures_dir = str(out_cfg.get("figures_dir", "outputs/figures"))
    memo_path = str(out_cfg.get("memo_path", "outputs/memo.md"))
    return {"tables_dir": tables_dir, "figures_dir": figures_dir, "memo_path": memo_path}


def _output_files_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate consistent 'output_files' block for memo auditability.
    List the standard filenames produced by the pipeline,
    but do NOT hardcode directories.
    """
    p = _output_paths_from_cfg(cfg)

    tables_dir = Path(p["tables_dir"])
    figures_dir = Path(p["figures_dir"])
    memo_path = Path(p["memo_path"])

    # Standard filenames (match save_* in pipeline)
    table_files = [
        "financials_5y.csv",
        "ratios.csv",
        "ratios_summary.csv",
        "valuation_multiples.csv",
        "dcf_projection.csv",
        "risk_summary.csv",
    ]
    figure_files = [
        "fig1_revenue_fcf.png",
        "fig2_margin_profile.png",
    ]

    tables = [str(tables_dir / fn) for fn in table_files]
    figures = [str(figures_dir / fn) for fn in figure_files]

    return {"tables": tables, "figures": figures, "memo": str(memo_path)}


def _fallback_memo(
    *,
    company: str,
    ticker: str,
    asof: str,
    payload: Dict[str, Any],
    reason: str,
) -> str:
    """
    Deterministic memo if LLM is unavailable. Keeps demo runnable.
    Uses only payload numbers; does NOT invent data.
    """
    snap = payload.get("snapshot", {}) or {}
    ratios = payload.get("ratios_latest", {}) or {}
    val = (payload.get("valuation", {}) or {}).get("valuation_summary", {}) or {}
    risk = (payload.get("risk", {}) or {}).get("risk_summary", {}) or {}
    warnings = payload.get("pipeline_warnings", []) or []

    rec = "HOLD"
    conv = "Low"

    pe_upside = val.get("pe_upside_pct")
    buy_hurdle = val.get("min_upside_pct_for_buy")
    meets = val.get("meets_buy_threshold")

    try:
        if meets is True:
            rec = "BUY"
        elif meets is False:
            rec = "HOLD"
    except Exception:
        pass

    try:
        risk_level = str(risk.get("risk_level", "")).lower()
        w_count = len(warnings)

        # transparent heuristic
        if rec == "BUY" and risk_level in ("low", "medium") and w_count <= 2:
            conv = "High"
        elif rec == "BUY" and risk_level in ("medium", "high"):
            conv = "Moderate"
        else:
            conv = "Low"

        val["recommendation"] = rec
        val["conviction"] = conv
    except Exception:
        pass

    lines: List[str] = []
    lines.append(f"# Investment Memo — {company} ({ticker})")
    lines.append("")
    lines.append(f"**As of:** {asof}")
    lines.append("")
    lines.append(
        f"**Note:** Local LLM memo generation was unavailable ({reason}). "
        "This memo is generated deterministically from pipeline outputs only."
    )
    lines.append("")

    lines.append("## 1) Snapshot (latest available fiscal year)")
    lines.append(f"- Latest fiscal year: {snap.get('latest_fiscal_year', 'N/A')}")
    lines.append(f"- Revenue: {snap.get('revenue', 'N/A')}")
    lines.append(f"- EBIT: {snap.get('ebit', 'N/A')}")
    lines.append(f"- Net income: {snap.get('net_income', 'N/A')}")
    lines.append(f"- Free cash flow: {snap.get('free_cash_flow', 'N/A')}")
    lines.append("")

    lines.append("## 2) Financial profile (selected ratios)")
    lines.append(f"- EBIT margin: {ratios.get('ebit_margin', 'N/A')}")
    lines.append(f"- Net margin: {ratios.get('net_margin', 'N/A')}")
    lines.append(f"- FCF margin: {ratios.get('fcf_margin', 'N/A')}")
    lines.append(f"- Debt-to-equity: {ratios.get('debt_to_equity', 'N/A')}")
    lines.append(f"- ROE: {ratios.get('roe', 'N/A')}")
    lines.append("")

    lines.append("## 3) Valuation (multiples + DCF cross-check)")
    if pe_upside is not None:
        try:
            lines.append(
                f"- P/E implied upside vs reference: {float(pe_upside):.1f}% "
                f"(buy hurdle: {float(buy_hurdle):.1f}%)"
            )
        except Exception:
            lines.append(f"- P/E implied upside vs reference: {pe_upside}")
    lines.append("")

    lines.append("## 4) Risk synthesis (score/level + drivers)")
    lines.append(f"- Risk score: {risk.get('risk_score', 'N/A')}")
    lines.append(f"- Risk level: {risk.get('risk_level', 'N/A')}")
    drivers = risk.get("risk_drivers") or []
    if drivers:
        lines.append("- Drivers:")
        for d in drivers[:3]:
            lines.append(f"  - {d}")
    else:
        lines.append("- Drivers: N/A")
    lines.append("")

    lines.append("## 5) Key sensitivities and what could break the thesis")
    lines.append(
        "- Key sensitivities are driven by assumptions (reference multiple, buy hurdle, WACC, terminal growth) "
        "and by data completeness."
    )
    lines.append("")

    lines.append("## 6) Data & reproducibility (data source + pipeline outputs)")
    lines.append(f"- Data source: {payload.get('data_source', 'N/A')}")
    outs = payload.get("output_files", {}) or {}
    lines.append("- Key outputs:")
    for pth in (outs.get("tables") or [])[:8]:
        lines.append(f"  - {pth}")
    for pth in (outs.get("figures") or [])[:4]:
        lines.append(f"  - {pth}")
    lines.append("")

    lines.append("## 7) Warnings / data gaps (for transparency)")
    if warnings:
        for w in warnings[:10]:
            lines.append(f"- {w}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## 8) Investment recommendation")
    lines.append(f"**{rec} ({conv} Conviction)**")
    return "\n".join(lines).strip()


def generate_investment_memo(
    *,
    cfg: Dict[str, Any],
    metadata: Dict[str, Any],
    financials_df: pd.DataFrame,
    ratios_df: pd.DataFrame,
    ratios_summary_df: Optional[pd.DataFrame],
    valuation_df: pd.DataFrame,
    valuation_summary: Dict[str, Any],
    dcf_df: Optional[pd.DataFrame] = None,
    dcf_summary: Optional[Dict[str, Any]] = None,
    risk_df: Optional[pd.DataFrame] = None,
    risk_summary: Optional[Dict[str, Any]] = None,
    ingestion_warnings: Optional[List[str]] = None,
    processing_warnings: Optional[List[str]] = None,
    ratios_warnings: Optional[List[str]] = None,
    valuation_warnings: Optional[List[str]] = None,
    dcf_warnings: Optional[List[str]] = None,
    risk_warnings: Optional[List[str]] = None,
    figures_warnings: Optional[List[str]] = None,
    model: Optional[str] = None,
) -> str:
    """
    Generate a concise 1–2 page Markdown memo via LOCAL LLM (Ollama), grounded in pipeline outputs.

    Configuration:
    - host: env OLLAMA_HOST or default http://localhost:11434
    - model: cfg["llm"]["model"] or function arg `model` or default "llama3.2"
    """

    dcf_summary = dcf_summary or {}
    risk_summary = risk_summary or {}

    ticker = metadata.get("ticker", cfg.get("ticker", "N/A"))
    company = metadata.get("company_name", cfg.get("company_name", ticker))
    currency = metadata.get("currency", cfg.get("currency", "USD"))
    asof = datetime.utcnow().strftime("%Y-%m-%d")

    latest_year = _latest_year(financials_df)
    fin_latest = _latest_row(financials_df)
    rat_latest = _latest_row(ratios_df)

    # --- build compact, auditable structured inputs ---
    snapshot = {
        "latest_fiscal_year": latest_year,
        "revenue": _fmt_money_compact(_safe_float(None if fin_latest is None else fin_latest.get("revenue")), currency),
        "ebit": _fmt_money_compact(_safe_float(None if fin_latest is None else fin_latest.get("ebit")), currency),
        "net_income": _fmt_money_compact(_safe_float(None if fin_latest is None else fin_latest.get("net_income")), currency),
        "free_cash_flow": _fmt_money_compact(_safe_float(None if fin_latest is None else fin_latest.get("free_cash_flow")), currency),
        "currency": currency,
    }

    ratios_latest = {
        "percent_metrics": {
            "ebit_margin": _safe_float(...),
            "net_margin": _safe_float(...),
            "fcf_margin": _safe_float(...),
            "roe": _safe_float(...),
            "revenue_cagr": _safe_float(...),
            "fcf_cagr": _safe_float(...),
        },
        "multiple_metrics": {
            "debt_to_equity": _safe_float(...),
            "asset_turnover": _safe_float(...),
        },
    }
    
    valuation_inputs = {
        "valuation_summary": valuation_summary,
        "valuation_table_head": _table_to_records(valuation_df, max_rows=20),
        "dcf_summary": dcf_summary,
        "dcf_projection_tail": _df_tail(dcf_df, n=5),
        "dcf_assumptions_from_config": cfg.get("dcf", {}),
        "valuation_assumptions_from_config": cfg.get("valuation", {}),
    }

    risk_inputs = {
        "risk_summary": risk_summary,
        "risk_table": _table_to_records(risk_df, max_rows=20),
        "risk_config": cfg.get("risk", {}),
    }

    warnings_all: List[str] = []
    for block in (
        ingestion_warnings,
        processing_warnings,
        ratios_warnings,
        valuation_warnings,
        dcf_warnings,
        risk_warnings,
        figures_warnings,
    ):
        if block:
            warnings_all.extend([w for w in block if isinstance(w, str) and w.strip()])

    warnings_shown = warnings_all[:10]

    payload = {
        "as_of": asof,
        "ticker": ticker,
        "company": company,
        "data_source": metadata.get("source", "yfinance (Yahoo Finance)"),
        "snapshot": snapshot,
        "ratios_latest": ratios_latest,
        "ratios_summary_table": _table_to_records(ratios_summary_df, max_rows=5),
        "financials_tail": _df_tail(financials_df, n=5),
        "ratios_tail": _df_tail(ratios_df, n=5),
        "valuation": valuation_inputs,
        "risk": risk_inputs,
        "pipeline_warnings": warnings_shown,
        # ✅ FIX: derive from cfg output paths instead of hardcoding
        "output_files": _output_files_manifest(cfg),
    }

    system = (
        "You are a buy-side fundamental analyst assistant. "
        "Write concise, professional investment memos grounded strictly in provided data. "
        "Do not invent numbers. If a field is missing, write N/A. "
        "Keep the memo ~1–2 pages in Markdown."
    )

    user = f"""
Generate an investment memo in Markdown with the following sections (use these exact headings):

# Investment Memo — {company} ({ticker})

## 1) Snapshot (latest available fiscal year)
## 2) Financial profile (selected ratios)
## 3) Valuation (multiples + DCF cross-check)
## 4) Risk synthesis (score/level + drivers)
## 5) Key sensitivities and what could break the thesis
## 6) Data & reproducibility (data source + pipeline outputs)
## 7) Warnings / data gaps (for transparency)
## 8) Investment recommendation

Rules:

General principles:
- The memo must be concise, professional, and decision-oriented.
- Use ONLY numbers provided in the structured payload.
- Do NOT invent, infer, or extrapolate data.
- If a required value is missing, explicitly write "N/A".

Snapshot rules:
- The Snapshot must include only scale, profitability, and cash generation metrics.
- Use ONLY fields from payload.snapshot.
- Do NOT include balance sheet totals unless they are explicitly present in the snapshot payload.
- Do NOT write calendar dates (e.g., "As of February 2026") inside the Snapshot.

Formatting requirements (strict):

Section 1 — Snapshot:
- Start the section with the exact phrase:
  "Latest fiscal year: <YEAR>"
  where <YEAR> = payload.snapshot.latest_fiscal_year.
- Use fiscal-year framing only (not calendar time).

Section 2 — Financial profile (ratios):
- Format margins, ROE, and CAGR as percentages (multiply by 100).
- Format debt-to-equity and asset turnover as "x" multiples.
- IMPORTANT: asset turnover values are already in "x" units and must NOT be multiplied by 100.
- Typical asset turnover values are below 2.0x; values like 20x or 40x are invalid.
- Present ratios clearly; do not mix units.
- Percentages must be rounded to ONE decimal place.
- Multiples (x) must be rounded to TWO decimal places.
- The ratios_summary_table is for internal reference only.
- Do NOT reproduce it as a table in the memo.

Section 3 — Valuation:
- The primary decision signal MUST come from the multiples-based valuation:
  valuation.valuation_summary.pe_upside_pct
  vs
  valuation.valuation_summary.min_upside_pct_for_buy.
- The DCF is a secondary cross-check only.
- If multiples and DCF imply different conclusions, explain this explicitly as
  assumption sensitivity and treat the DCF as a conservative anchor.
- Do NOT allow the DCF to override the multiples-driven recommendation
  unless you clearly justify why.

Risk writing rules:
- Risk drivers MUST be written as bullet points.
- Each bullet must include exactly one driver and a one-line explanation.
- Do NOT compress risk drivers into a single sentence or a single table cell.
- Do NOT imply that higher risk automatically implies higher upside.

Recommendation & conviction rules (strict):

Section 8 — Investment recommendation:
- You MUST include a section titled exactly:
  "## 8) Investment recommendation"
- The recommendation MUST be one of:
  BUY, HOLD, SELL.
- You MUST explicitly state conviction as one of:
  Low, Moderate, High.
- Use the exact one-line format:
  **BUY (Moderate Conviction)**
  (Replace BUY / Moderate accordingly.)
- Do NOT use brackets, angle brackets, or alternative formatting.
- The recommendation line must appear immediately after the section header.
- Do NOT add a subheading like "### Recommendation".

Conviction logic:
- Conviction must be explained directly below the recommendation using:
  (i) valuation strength (distance vs buy hurdle),
  (ii) risk_score and risk_level,
  (iii) key risk drivers,
  (iv) data quality (number of pipeline warnings).
- If risk_level is "medium" or "high", conviction should generally be "Moderate"
  unless you provide a very strong and explicit justification.
- Conviction should be reduced if risk is elevated or data warnings exist,
  even when the valuation signal is positive.
- Risk score should be shown with two decimal places.

Here is the structured input payload (JSON). Use it as the only source of numbers:
{json.dumps(payload, ensure_ascii=False)}
""".strip()

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    cfg_llm = cfg.get("llm", {}) or {}
    model_name = model or cfg_llm.get("model") or "llama3.2"

    if not _ollama_available(host):
        return _fallback_memo(
            company=company,
            ticker=ticker,
            asof=asof,
            payload=payload,
            reason=f"Ollama server not reachable at {host}",
        )

    try:
        text = _call_ollama_chat(
            host=host,
            model=str(model_name),
            system=system,
            user=user,
            temperature=0.2,
            timeout=float(cfg_llm.get("timeout_seconds", 120.0)),
        )
        if not text:
            return _fallback_memo(
                company=company,
                ticker=ticker,
                asof=asof,
                payload=payload,
                reason="Empty response from local LLM",
            )
        return text.strip()
    except Exception as e:
        return _fallback_memo(
            company=company,
            ticker=ticker,
            asof=asof,
            payload=payload,
            reason=f"Local LLM call failed: {type(e).__name__}: {e}",
        )
