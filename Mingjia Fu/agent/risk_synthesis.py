# agent/risk_synthesis.py
"""
Purpose:
- Turn a small set of observable signals into:
  - risk_score in [0, 1]
  - risk_level: low / medium / high
  - risk_drivers: interpretable reasons

Design principles:
- Deterministic, reproducible
- Minimal, coursework-friendly proxies (no external data dependencies)
- Assumptions externalised in config.json (weights, thresholds, normalization)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class RiskResult:
    risk_table: pd.DataFrame
    summary: Dict[str, Any]
    warnings: List[str]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _latest_year(df: pd.DataFrame) -> Optional[int]:
    if df is None or df.empty:
        return None
    try:
        return int(df.index.max())
    except Exception:
        return None


def _get_market_cap_from_valuation_table(valuation_df: Optional[pd.DataFrame]) -> Optional[float]:
    """
    valuation_multiples.py outputs a table with rows:
      metric = "Market Cap (USD)" and column "current"
    """
    if valuation_df is None or valuation_df.empty:
        return None

    if "metric" in valuation_df.columns and "current" in valuation_df.columns:
        hit = valuation_df.loc[valuation_df["metric"] == "Market Cap (USD)"]
        if not hit.empty:
            return _safe_float(hit.iloc[-1]["current"])

    return None


def _normalize_linear_0_1(x: Optional[float], low: float, high: float) -> Optional[float]:
    """
    Generic linear normalization into [0,1] with clamping.
    - If x is None -> None.
    - If high <= low -> None (invalid range).
    """
    if x is None:
        return None
    if high <= low:
        return None

    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)


def compute_risk_synthesis(
    *,
    cfg: Dict[str, Any],
    financials_df: pd.DataFrame,
    ratios_df: pd.DataFrame,  # kept for interface consistency; not used in this proxy set
    valuation_df: Optional[pd.DataFrame] = None,
    pipeline_warnings: Optional[List[str]] = None,
) -> RiskResult:
    """
    Compute a lightweight risk synthesis using config weights.

    Components (aligned with cfg['risk']['weights']):
    1) investment_intensity:
       uses |capex| / revenue (latest year), normalized into [0,1]
    2) regulatory:
       uses market cap proxy (USD), normalized into [0,1]

    Output:
    - risk_score in [0,1] (weighted average of available components)
    - risk_level based on cfg['risk']['thresholds']
    - risk_drivers: interpretable reasons + data caveat
    """
    warnings: List[str] = []

    risk_cfg = cfg.get("risk", {}) or {}
    weights = (risk_cfg.get("weights", {}) or {})
    thresholds = (risk_cfg.get("thresholds", {}) or {})
    norm_cfg = (risk_cfg.get("normalization", {}) or {})

    # ---- weights ----
    w_invest = _safe_float(weights.get("investment_intensity"))
    w_reg = _safe_float(weights.get("regulatory"))

    if w_invest is None:
        w_invest = 0.6
        warnings.append("risk.weights.investment_intensity missing; defaulted to 0.6.")
    if w_reg is None:
        w_reg = 0.4
        warnings.append("risk.weights.regulatory missing; defaulted to 0.4.")

    # Normalize weights if they don't sum to 1
    w_sum = w_invest + w_reg
    if w_sum <= 0:
        w_invest, w_reg = 0.6, 0.4
        w_sum = 1.0
        warnings.append("risk weights invalid; defaulted to 0.6/0.4.")
    else:
        w_invest = w_invest / w_sum
        w_reg = w_reg / w_sum

    # ---- thresholds (must align with config defaults) ----
    th_medium = _safe_float(thresholds.get("medium"))
    th_high = _safe_float(thresholds.get("high"))
    if th_medium is None:
        th_medium = 0.50
        warnings.append("risk.thresholds.medium missing; defaulted to 0.50.")
    if th_high is None:
        th_high = 0.75
        warnings.append("risk.thresholds.high missing; defaulted to 0.75.")

    # ---- normalization ranges (externalised) ----
    inv_norm = (norm_cfg.get("investment_intensity", {}) or {})
    inv_low = _safe_float(inv_norm.get("low"))
    inv_high = _safe_float(inv_norm.get("high"))

    reg_norm = (norm_cfg.get("regulatory_size_usd", {}) or {})
    reg_low = _safe_float(reg_norm.get("low"))
    reg_high = _safe_float(reg_norm.get("high"))

    # Fallbacks (should match your documented defaults)
    if inv_low is None:
        inv_low = 0.02
        warnings.append("risk.normalization.investment_intensity.low missing; defaulted to 0.02.")
    if inv_high is None:
        inv_high = 0.30
        warnings.append("risk.normalization.investment_intensity.high missing; defaulted to 0.30.")

    if reg_low is None:
        reg_low = 5e11
        warnings.append("risk.normalization.regulatory_size_usd.low missing; defaulted to 5e11.")
    if reg_high is None:
        reg_high = 5e12
        warnings.append("risk.normalization.regulatory_size_usd.high missing; defaulted to 5e12.")

    # ---- latest year inputs ----
    latest = _latest_year(financials_df)
    if latest is None:
        warnings.append("Cannot determine latest fiscal year; risk synthesis limited.")

    capex_intensity: Optional[float] = None
    inv_component: Optional[float] = None

    if latest is not None and financials_df is not None and (not financials_df.empty):
        rev = _safe_float(financials_df.loc[latest].get("revenue")) if "revenue" in financials_df.columns else None
        capex = _safe_float(financials_df.loc[latest].get("capex")) if "capex" in financials_df.columns else None

        if rev is not None and rev != 0 and capex is not None:
            capex_intensity = abs(capex) / abs(rev)
            inv_component = _normalize_linear_0_1(capex_intensity, inv_low, inv_high)
            if inv_component is None:
                warnings.append("Investment intensity normalization range invalid; component skipped.")
        else:
            warnings.append("Investment intensity unavailable (need revenue and capex).")

    # Regulatory proxy from market cap
    market_cap = _get_market_cap_from_valuation_table(valuation_df)
    reg_component = _normalize_linear_0_1(market_cap, reg_low, reg_high)
    if market_cap is None:
        warnings.append("Regulatory proxy unavailable (market cap missing).")
    elif reg_component is None:
        warnings.append("Regulatory size normalization range invalid; component skipped.")

    # ---- weighted score using available components only ----
    components: List[Tuple[str, float, float]] = []
    if inv_component is not None:
        components.append(("investment_intensity", float(inv_component), float(w_invest)))
    if reg_component is not None:
        components.append(("regulatory", float(reg_component), float(w_reg)))

    if not components:
        risk_score = 0.50
        warnings.append("Risk score defaulted to 0.50 because no components were computable.")
    else:
        w_avail = sum(w for _, _, w in components)
        risk_score = sum(v * (w / w_avail) for _, v, w in components)
        risk_score = _clamp01(float(risk_score))

    # ---- risk level ----
    if risk_score >= float(th_high):
        risk_level = "high"
    elif risk_score >= float(th_medium):
        risk_level = "medium"
    else:
        risk_level = "low"

    # ---- drivers (interpretable, memo-friendly) ----
    drivers: List[str] = []
    if capex_intensity is not None:
        drivers.append(
            f"Investment intensity proxy (|capex|/revenue) ≈ {capex_intensity*100:.1f}% "
            f"(higher implies execution/reinvestment risk)."
        )
    else:
        drivers.append("Investment intensity proxy unavailable (capex/revenue missing).")

    if market_cap is not None:
        drivers.append(
            f"Regulatory proxy uses size (market cap ≈ {market_cap/1e12:.2f}T USD); "
            f"larger platforms face structurally higher scrutiny."
        )
    else:
        drivers.append("Regulatory proxy unavailable (market cap missing).")

    pw = list(pipeline_warnings or [])
    if pw:
        drivers.append(f"Data caveat: {len(pw)} pipeline warning(s) surfaced (see memo disclosure).")

    # ---- output table ----
    table_rows: List[Dict[str, Any]] = [
        {
            "component": "investment_intensity",
            "raw_proxy": (None if capex_intensity is None else capex_intensity),
            "normalized_0_1": inv_component,
            "weight_used": (w_invest if inv_component is not None else None),
        },
        {
            "component": "regulatory",
            "raw_proxy": (None if market_cap is None else market_cap),
            "normalized_0_1": reg_component,
            "weight_used": (w_reg if reg_component is not None else None),
        },
        {
            "component": "overall",
            "raw_proxy": None,
            "normalized_0_1": float(risk_score),
            "weight_used": 1.0,
        },
    ]

    risk_table = pd.DataFrame(table_rows, columns=["component", "raw_proxy", "normalized_0_1", "weight_used"])

    summary: Dict[str, Any] = {
        "risk_score": float(risk_score),
        "risk_level": risk_level,
        "risk_drivers": drivers[:3],  # keep concise
        "threshold_medium": float(th_medium),
        "threshold_high": float(th_high),
    }

    return RiskResult(risk_table=risk_table, summary=summary, warnings=warnings)


def save_risk_table(risk_table: pd.DataFrame, out_dir: str) -> str:
    out_path = Path(out_dir) / "risk_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    risk_table.to_csv(out_path, index=False)
    return str(out_path)
