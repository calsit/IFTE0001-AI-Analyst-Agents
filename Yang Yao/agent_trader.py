# agent_trader.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError, ConfigDict

from llm_client import LLMClient
from tools_market_data import fetch_ohlcv, add_indicators, latest_snapshot
from tools_backtest import StrategyConfig, end_to_end_backtest


# -----------------------------
# 1) Schemas
# -----------------------------
Action = Literal["BUY", "HOLD", "SELL"]


class AgentDecision(BaseModel):
    """
    Strict decision object (final validated object).
    """
    model_config = ConfigDict(extra="forbid")

    as_of: str = Field(..., description="YYYY-MM-DD of the latest data point used")
    ticker: str = Field(..., description="Ticker symbol, e.g. MSFT")
    action: Action = Field(..., description="BUY/HOLD/SELL")
    recommended_position: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Target position weight in [0,1]"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in [0,1]"
    )

    thesis: str = Field(..., description="1-paragraph rationale")
    key_signals: list[str] = Field(..., description="Bullet-like list of key signals")
    risks: list[str] = Field(..., description="Main risks / failure modes")
    constraints: list[str] = Field(..., description="Costs, slippage, data limits, etc.")


class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    decision: AgentDecision
    latest_snapshot: Dict[str, Any]
    metrics: Dict[str, Any]
    benchmark_metrics: Dict[str, Any]
    outdir: str
    note_path: str
    decision_path: str


# -----------------------------
# 2) Helpers
# -----------------------------
def _clip01(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    return v


def _coerce_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        # try parse json from string
        try:
            return json.loads(x)
        except Exception:
            # best-effort: find first {...}
            s = x.strip()
            lb = s.find("{")
            rb = s.rfind("}")
            if lb != -1 and rb != -1 and rb > lb:
                try:
                    return json.loads(s[lb : rb + 1])
                except Exception:
                    pass
    return {}


def _safe_llm_json(llm: LLMClient, prompt: str) -> Dict[str, Any]:
    """
    Call llm.json(prompt) but tolerate different signatures.
    """
    try:
        obj = llm.json(prompt)
    except TypeError:
        # some wrappers might require named args
        obj = llm.json(prompt=prompt)
    return _coerce_dict(obj)


def _safe_llm_text(llm: LLMClient, prompt: str) -> str:
    try:
        return llm.text(prompt)
    except TypeError:
        return llm.text(prompt=prompt)


def _build_decision_prompt(
    ticker: str,
    snap: Dict[str, Any],
    metrics: Dict[str, Any],
    bh_metrics: Dict[str, Any],
    rebal_mode: str,
    cost_rate: float,
    target_vol: float,
) -> str:
    """
    Strongly constrain the JSON keys so the model stops drifting.
    """
    schema_hint = {
        "as_of": "YYYY-MM-DD",
        "ticker": ticker,
        "action": "BUY | HOLD | SELL",
        "recommended_position": "number in [0,1]",
        "confidence": "number in [0,1]",
        "thesis": "string",
        "key_signals": ["string", "..."],
        "risks": ["string", "..."],
        "constraints": ["string", "..."],
    }

    return f"""
You are a trading agent. You MUST return ONLY valid JSON (no markdown, no commentary).
Your JSON MUST have EXACTLY these keys and types (no extra keys): 
{json.dumps(schema_hint, indent=2)}

Rules:
- Use ONLY the provided numbers. Do NOT invent metrics or prices.
- 'as_of' must equal the latest snapshot date.
- recommended_position must be in [0,1]. If action=SELL -> position close to 0.
- confidence in [0,1].
- Keep thesis concise (4-6 sentences).
- key_signals / risks / constraints: 3-7 items each.

Context:
Ticker: {ticker}
Rebalance mode: {rebal_mode}
Cost rate (single-side per turnover): {cost_rate}
Target vol: {target_vol}

Latest snapshot (already computed indicators):
{json.dumps(snap, indent=2)}

Strategy backtest metrics (net of costs):
{json.dumps(metrics, indent=2)}

Benchmark (buy & hold) metrics:
{json.dumps(bh_metrics, indent=2)}
""".strip()


def _repair_decision_with_defaults(
    raw: Dict[str, Any],
    *,
    ticker: str,
    snap: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Fill missing required fields + normalize common mistakes.
    """
    d = dict(raw) if isinstance(raw, dict) else {}

    # normalize key variants
    if "position" in d and "recommended_position" not in d:
        d["recommended_position"] = d["position"]
    if "weight" in d and "recommended_position" not in d:
        d["recommended_position"] = d["weight"]
    if "conf" in d and "confidence" not in d:
        d["confidence"] = d["conf"]

    # fill required fields
    d.setdefault("ticker", ticker)
    d.setdefault("as_of", str(snap.get("as_of", "")))

    # defaults if missing
    d.setdefault("action", "HOLD")
    d.setdefault("recommended_position", 0.0 if d.get("action") == "SELL" else 0.5)
    d.setdefault("confidence", 0.55)

    d.setdefault("thesis", "Decision generated with limited context; see signals and risks.")
    d.setdefault("key_signals", [])
    d.setdefault("risks", [])
    d.setdefault("constraints", [])

    # final clipping
    d["recommended_position"] = _clip01(d.get("recommended_position"), default=0.5)
    d["confidence"] = _clip01(d.get("confidence"), default=0.55)

    # enforce action semantics a bit
    act = str(d.get("action", "HOLD")).upper()
    if act not in {"BUY", "HOLD", "SELL"}:
        act = "HOLD"
    d["action"] = act
    if act == "SELL":
        d["recommended_position"] = min(d["recommended_position"], 0.1)

    # ensure lists
    for k in ["key_signals", "risks", "constraints"]:
        if not isinstance(d.get(k), list):
            d[k] = [str(d.get(k))] if d.get(k) is not None else []
        d[k] = [str(x) for x in d[k] if str(x).strip()]

    return d


def _build_trade_note_prompt(
    ticker: str,
    snap: Dict[str, Any],
    metrics: Dict[str, Any],
    bh_metrics: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    return f"""
Write a 1–2 page markdown trade note for {ticker}. Use only the provided context.
No fabricated numbers. If something is unknown, say "unknown".

Required sections (use these headings):
1) Overview
2) Latest Signals (technical)
3) Backtest Summary (net of costs) + Benchmark Comparison
4) Risks & Limitations
5) Recommendation (must match the decision JSON)

Context:
Latest snapshot:
{json.dumps(snap, indent=2)}

Strategy metrics:
{json.dumps(metrics, indent=2)}

Benchmark metrics:
{json.dumps(bh_metrics, indent=2)}

Decision JSON (must match):
{json.dumps(decision, indent=2)}
""".strip()


# -----------------------------
# 3) Main Agent
# -----------------------------
def run_agent(
  
    ticker: str = "MSFT",
    *,
    years: int = 10,
    rebalance_mode: str = "weekly",
    outdir: str = "outputs",
    prefix: str = "msft",
    cost_rate: float = 0.001,
    target_vol: float = 0.15,
) -> Dict[str, Any]:
    """
    End-to-end agent run:
    - fetch data -> add indicators -> backtest
    - LLM decision (JSON) with repair + strict validation
    - LLM trade note (markdown)
    - save artifacts
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()

    # 1) Observe: data + indicators
    df = fetch_ohlcv(ticker, years)
    df = add_indicators(df)
    snap = latest_snapshot(df)

    # 2) Act/Simulate: backtest + metrics
    cfg = StrategyConfig(
        rebalance_mode=rebalance_mode,
        cost_rate=cost_rate,
        target_vol=target_vol,
    )
    bt = end_to_end_backtest(df, cfg, outdir=outdir, prefix=prefix)

    # Support both naming conventions (in case your tools_backtest differs)
    metrics = bt.get("metrics") or bt.get("strategy_metrics") or {}
    bh_metrics = bt.get("benchmark_metrics") or bt.get("bh_metrics") or {}

    # 3) Reason: LLM decision (raw dict first)
    decision_prompt = _build_decision_prompt(
        ticker=ticker,
        snap=snap,
        metrics=metrics,
        bh_metrics=bh_metrics,
        rebal_mode=rebalance_mode,
        cost_rate=cost_rate,
        target_vol=target_vol,
    )

    raw1 = _safe_llm_json(llm, decision_prompt)
    repaired1 = _repair_decision_with_defaults(raw1, ticker=ticker, snap=snap)

    # validate; if fail -> ask LLM to fix once
    try:
        decision_obj = AgentDecision.model_validate(repaired1)
    except ValidationError as e:
        fix_prompt = f"""
You returned JSON that failed validation. Fix it and return ONLY corrected JSON (no extra keys).
Validation error:
{str(e)}

Bad JSON:
{json.dumps(repaired1, indent=2)}

Remember: required keys are: as_of, ticker, action, recommended_position, confidence, thesis, key_signals, risks, constraints.
"""
        raw2 = _safe_llm_json(llm, fix_prompt)
        repaired2 = _repair_decision_with_defaults(raw2, ticker=ticker, snap=snap)
        # final attempt
        decision_obj = AgentDecision.model_validate(repaired2)

    decision_dict = decision_obj.model_dump()

    # 4) Write: LLM trade note
    note_prompt = _build_trade_note_prompt(ticker, snap, metrics, bh_metrics, decision_dict)
    note_md = _safe_llm_text(llm, note_prompt)

    # 5) Save outputs
    decision_path = out_path / f"{prefix}_agent_decision.json"
    decision_path.write_text(json.dumps(decision_dict, indent=2), encoding="utf-8")

    note_path = out_path / f"{prefix}_trade_note_ai.md"
    note_path.write_text(note_md, encoding="utf-8")

    res = AgentRunResult(
        decision=decision_obj,
        latest_snapshot=snap,
        metrics=metrics,
        benchmark_metrics=bh_metrics,
        outdir=str(out_path),
        note_path=str(note_path),
        decision_path=str(decision_path),
    )

    # return dict for notebook convenience
    return res.model_dump()


if __name__ == "__main__":
    r = run_agent("MSFT", years=10, rebalance_mode="weekly", outdir="outputs", prefix="msft")
    print("Decision:", r["decision"])
    print("Saved:", r["decision_path"], r["note_path"])
