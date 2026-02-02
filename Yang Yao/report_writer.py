# -*- coding: utf-8 -*-
"""
report_writer.py

Generate a human-readable Markdown + PDF report from artifacts produced by the
backtest / trading agent pipeline.

Designed to be:
- callable from code: generate_report(outdir, prefix, ticker, pdf=True)
- callable from CLI: python report_writer.py --outdir outputs --prefix msft --ticker MSFT --pdf

This file is intentionally self-contained and Python 3.9 compatible.
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ----------------------------
# Optional dependency: reportlab
# ----------------------------
try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        PageBreak,
        Table,
        TableStyle,
        Image as RLImage,
        Preformatted,
    )
    from reportlab.lib import colors

    _HAS_REPORTLAB = True
except Exception:
    _HAS_REPORTLAB = False


# ----------------------------
# Data model
# ----------------------------
@dataclass
class Artifacts:
    outdir: Path
    prefix: str
    ticker: str

    run_manifest: Optional[Path] = None
    run_summary: Optional[Path] = None
    agent_decision: Optional[Path] = None
    latest_signal: Optional[Path] = None

    metrics: Optional[Path] = None
    benchmark_metrics: Optional[Path] = None

    equity_curve_png: Optional[Path] = None
    equity_curve_csv: Optional[Path] = None

    trade_note_md: Optional[Path] = None

    def as_table_rows(self) -> List[Tuple[str, str, bool]]:
        rows: List[Tuple[str, str, bool]] = []
        items = [
            ("Run manifest", self.run_manifest),
            ("Run summary", self.run_summary),
            ("Agent decision", self.agent_decision),
            ("Latest signal", self.latest_signal),
            ("Strategy metrics", self.metrics),
            ("Benchmark metrics", self.benchmark_metrics),
            ("Equity curve (png)", self.equity_curve_png),
            ("Equity curve (csv)", self.equity_curve_csv),
            ("Trade note", self.trade_note_md),
        ]
        for name, p in items:
            if p is None:
                rows.append((name, "", False))
            else:
                rows.append((name, p.name, p.exists()))
        return rows


# ----------------------------
# Small utilities
# ----------------------------
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _safe_read_json(path: Optional[Path]) -> Optional[Any]:
    if not path or not path.exists():
        return None
    try:
        return _read_json(path)
    except Exception:
        return None


def _safe_read_text(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    try:
        return _read_text(path)
    except Exception:
        return None


def _coerce_path(outdir: Union[str, Path]) -> Path:
    return outdir if isinstance(outdir, Path) else Path(outdir)


def _fmt_pct(x: Any, decimals: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    return f"{v*100:.{decimals}f}%"


def _fmt_float(x: Any, decimals: int = 3) -> str:
    try:
        v = float(x)
    except Exception:
        return "-"
    return f"{v:.{decimals}f}"


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")


def _wrap(s: str, width: int = 92) -> str:
    return "\n".join(textwrap.wrap(s, width=width))


# ----------------------------
# Artifact discovery
# ----------------------------
def load_artifacts(outdir: Union[str, Path], prefix: str, ticker: str) -> Artifacts:
    """
    Locate known artifacts in outdir for a given run prefix.
    This function never raises; it simply records paths if they exist.
    """
    outdir_p = _coerce_path(outdir)
    prefix = str(prefix)
    ticker = str(ticker)

    def p(name: str) -> Path:
        return outdir_p / name

    art = Artifacts(outdir=outdir_p, prefix=prefix, ticker=ticker)

    # canonical names (preferred)
    art.run_manifest = p(f"{prefix}_run_manifest.json")
    art.run_summary = p(f"{prefix}_run_summary.json")
    art.agent_decision = p(f"{prefix}_agent_decision.json")
    art.latest_signal = p(f"{prefix}_latest_signal.json")
    art.metrics = p(f"{prefix}_metrics.csv")
    art.benchmark_metrics = p(f"{prefix}_metrics_bh.csv")
    art.equity_curve_png = p(f"{prefix}_equity_curve.png")
    art.equity_curve_csv = p(f"{prefix}_equity_curve.csv")
    art.trade_note_md = p(f"{prefix}_trade_note.md")

    # some older code may write without prefix
    if not art.latest_signal.exists():
        cand = p("latest_signal.json")
        if cand.exists():
            art.latest_signal = cand

    return art


# ----------------------------
# Parsing helpers
# ----------------------------
def _parse_metrics_csv(path: Optional[Path]) -> Dict[str, Any]:
    """
    Try to parse a metrics CSV to a dict.
    Supports:
      1) single-row wide table: columns are metric names
      2) two-column long table: metric,value
    """
    if not path or not path.exists():
        return {}
    try:
        import pandas as pd  # optional but commonly available
    except Exception:
        return {}

    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    if df.empty:
        return {}

    # case 1: single row, wide format
    if len(df) == 1 and df.shape[1] >= 2:
        row = df.iloc[0].to_dict()
        for k in list(row.keys()):
            if str(k).lower() in ("date", "time", "timestamp"):
                row.pop(k, None)
        return row

    # case 2: long format metric/value
    low_cols = [c.lower() for c in df.columns]
    if len(df.columns) >= 2 and ("metric" in low_cols[0] or "name" in low_cols[0]):
        mcol = df.columns[0]
        vcol = df.columns[1]
        out: Dict[str, Any] = {}
        for _, r in df.iterrows():
            k = str(r.get(mcol, "")).strip()
            if not k:
                continue
            out[k] = r.get(vcol)
        return out

    # fallback: take first two columns as key/value
    out2: Dict[str, Any] = {}
    try:
        kcol = df.columns[0]
        vcol = df.columns[1]
        for _, r in df.iterrows():
            k = str(r.get(kcol, "")).strip()
            if not k:
                continue
            out2[k] = r.get(vcol)
    except Exception:
        pass
    return out2


def _pick(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
        for kk in d.keys():
            if kk.lower() == k.lower() and d[kk] is not None:
                return d[kk]
    return None


def _normalize_metrics(d: Dict[str, Any]) -> Dict[str, Any]:
    if not d:
        return {}
    out: Dict[str, Any] = {}
    out["CAGR"] = _pick(d, ["CAGR", "cagr"])
    out["Sharpe"] = _pick(d, ["Sharpe", "sharpe", "SharpeRatio", "sharpe_ratio"])
    out["MaxDrawdown"] = _pick(d, ["MaxDrawdown", "maxdrawdown", "MaxDD", "max_dd"])
    out["HitRate"] = _pick(d, ["HitRate", "hitrate", "Hit Rate"])
    out["AvgExposure"] = _pick(d, ["AvgExposure", "avgexposure", "Exposure"])
    out["AnnualTurnover"] = _pick(d, ["AnnualTurnover", "annualturnover", "Turnover"])
    out["CostRate"] = _pick(d, ["CostRate", "costrate", "Cost"])
    out["TargetVol"] = _pick(
        d, ["TargetVol", "targetvol", "TargetVolatility", "Target volatility"]
    )
    return out


# ----------------------------
# Trade note parsing (Notes / Limitations)
# ----------------------------
_SECTION_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.M)


def parse_trade_note(md_text: Optional[str]) -> Dict[str, List[str]]:
    """
    Extract only relevant narrative sections from trade_note.md.

    Returns dict with keys:
      - assumptions
      - limitations

    If the note doesn't have clear headings, will heuristically pull bullet lines
    containing keywords.
    """
    out = {"assumptions": [], "limitations": []}
    if not md_text:
        return out

    text = md_text.replace("\r\n", "\n")

    matches = list(_SECTION_RE.finditer(text))
    if matches:
        blocks: List[Tuple[str, str]] = []
        for i, m in enumerate(matches):
            title = m.group(2).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            blocks.append((title, body))

        for title, body in blocks:
            t = title.lower()
            lines = [ln.strip() for ln in body.split("\n") if ln.strip()]

            bullets: List[str] = []
            for ln in lines:
                if ln.startswith(("-", "*", "+")):
                    bullets.append(ln.lstrip("-*+ ").strip())
                elif len(ln) <= 160:
                    bullets.append(ln.strip())

            if "assumption" in t:
                out["assumptions"].extend(bullets)

            if any(k in t for k in ["limitation", "caveat"]):
                out["limitations"].extend(bullets)

            if t in ("notes", "note"):
                for b in bullets:
                    if re.search(
                        r"(limit|assum|simplif|slippage|cost|data|lag|single)",
                        b,
                        re.I,
                    ):
                        out["limitations"].append(b)

    if not out["assumptions"] and not out["limitations"]:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        for ln in lines:
            if not ln.startswith(("-", "*", "+")):
                continue
            b = ln.lstrip("-*+ ").strip()
            if re.search(r"(assum|data source|execution|lag)", b, re.I):
                out["assumptions"].append(b)
            elif re.search(r"(limit|slippage|simplif|risk|not guaranteed)", b, re.I):
                out["limitations"].append(b)

    for k in list(out.keys()):
        seen = set()
        cleaned: List[str] = []
        for item in out[k]:
            s = " ".join(item.split())
            if not s or s in seen:
                continue
            seen.add(s)
            cleaned.append(s)
        out[k] = cleaned

    return out


# ----------------------------
# Markdown report builder
# ----------------------------
def build_markdown(
    ticker: str,
    artifacts: Artifacts,
    run_summary: Optional[Dict[str, Any]],
    agent_decision: Optional[Dict[str, Any]],
    latest_signal: Optional[Dict[str, Any]],
    strat_metrics: Dict[str, Any],
    bench_metrics: Dict[str, Any],
    notes: Dict[str, List[str]],
) -> str:
    lines: List[str] = []
    lines.append(f"# {ticker} Technical Agent Report")
    lines.append("")
    lines.append(f"- Generated: {_now_iso()}")
    lines.append(f"- Prefix: `{artifacts.prefix}`")
    lines.append("")

    lines.append("## 1. Run summary")
    if run_summary:
        start = run_summary.get("start")
        end = run_summary.get("end")
        mode = run_summary.get("rebalance_mode") or run_summary.get("rebalance") or run_summary.get("rebalanceMode")
        years = run_summary.get("years")
        lines.append("")
        lines.append(f"- Ticker: `{run_summary.get('ticker', ticker)}`")
        if start or end:
            lines.append(f"- Period: {start} → {end}")
        if years is not None:
            lines.append(f"- Years: {years}")
        if mode:
            lines.append(f"- Rebalance mode: {mode}")
    else:
        lines.append("")
        lines.append("_Run summary not found._")

    lines.append("")
    lines.append("## 2. Agent decision (formatted)")
    if agent_decision:
        lines.append("")
        lines.append(f"- as_of: {agent_decision.get('as_of', '-')}")
        lines.append(f"- action: {agent_decision.get('action', '-')}")
        lines.append(f"- recommended_position: {agent_decision.get('recommended_position', '-')}")
        lines.append(f"- confidence: {agent_decision.get('confidence', '-')}")
        thesis = agent_decision.get("thesis")
        if thesis:
            lines.append("")
            lines.append("Thesis:")
            lines.append("")
            lines.append(_wrap(str(thesis)))
        for k, title in [("key_signals", "Key signals"), ("risks", "Risks"), ("constraints", "Constraints")]:
            arr = agent_decision.get(k)
            if isinstance(arr, list) and arr:
                lines.append("")
                lines.append(f"{title}:")
                lines.append("")
                for item in arr:
                    lines.append(f"- {item}")
    else:
        lines.append("")
        lines.append("_Agent decision not found._")

    lines.append("")
    lines.append("## 3. Latest signal")
    if latest_signal:
        keys_keep = ["as_of", "close", "adj_close", "ma", "MA", "MA200", "RSI", "rsi", "macd_hist", "vol", "recommended_position"]
        lines.append("")
        for key in keys_keep:
            if key in latest_signal:
                lines.append(f"- {key}: {latest_signal.get(key)}")
    else:
        lines.append("")
        lines.append("_Latest signal not found._")

    lines.append("")
    lines.append("## 4. Backtest metrics")
    def metric_line(name: str, v: Any) -> str:
        if name in ("CAGR", "MaxDrawdown", "HitRate", "AvgExposure"):
            return _fmt_pct(v, 2)
        if name in ("Sharpe",):
            return _fmt_float(v, 3)
        if name in ("AnnualTurnover",):
            return _fmt_float(v, 2)
        if name in ("CostRate", "TargetVol"):
            return _fmt_float(v, 4)
        return str(v)

    rows = ["| Metric | Strategy | Buy & Hold |", "|---|---:|---:|"]
    metric_order = ["CAGR", "Sharpe", "MaxDrawdown", "HitRate", "AvgExposure", "AnnualTurnover", "CostRate", "TargetVol"]
    for m in metric_order:
        rows.append(f"| {m} | {metric_line(m, strat_metrics.get(m))} | {metric_line(m, bench_metrics.get(m))} |")
    lines.append("")
    lines.extend(rows)

    lines.append("")
    lines.append("## 5. Equity curve")
    if artifacts.equity_curve_png and artifacts.equity_curve_png.exists():
        rel = artifacts.equity_curve_png.name
        lines.append("")
        lines.append(f"![Equity curve]({rel})")
    else:
        lines.append("")
        lines.append("_Equity curve image not found._")

    lines.append("")
    lines.append("## 6. Notes / limitations")
    lines.append("")
    if notes.get("assumptions"):
        lines.append("Assumptions:")
        lines.append("")
        for x in notes["assumptions"]:
            lines.append(f"- {x}")
        lines.append("")
    if notes.get("limitations"):
        lines.append("Limitations:")
        lines.append("")
        for x in notes["limitations"]:
            lines.append(f"- {x}")
        lines.append("")
    if not notes.get("assumptions") and not notes.get("limitations"):
        lines.append("_No assumptions/limitations extracted from trade_note.md._")

    lines.append("")
    lines.append("## 7. Reproducibility / artifacts")
    lines.append("")
    lines.append("| Artifact | File | Exists |")
    lines.append("|---|---|---:|")
    for name, fname, ok in artifacts.as_table_rows():
        lines.append(f"| {name} | {fname} | {str(ok)} |")

    def add_raw(title: str, obj: Any):
        if obj is None:
            return
        lines.append("")
        lines.append(f"## Appendix: {title} (raw JSON)")
        lines.append("")
        raw = json.dumps(obj, ensure_ascii=False, indent=2)
        lines.append("```json")
        raw_lines = raw.splitlines()
        cap = 4000
        if len(raw_lines) > cap:
            raw_lines = raw_lines[:cap] + ["...", f"... truncated, total lines = {len(raw.splitlines())}"]
        lines.extend(raw_lines)
        lines.append("```")

    add_raw("Run summary", run_summary)
    add_raw("Agent decision", agent_decision)
    add_raw("Latest signal", latest_signal)

    lines.append("")
    return "\n".join(lines)


# ----------------------------
# PDF builder
# ----------------------------
def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceAfter=12))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceAfter=10))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], leading=14, spaceAfter=6))
    styles.add(ParagraphStyle(name="Mono", parent=styles["Code"], fontName="Courier", fontSize=8, leading=9))
    return styles


def build_pdf(
    ticker: str,
    artifacts: Artifacts,
    run_summary: Optional[Dict[str, Any]],
    agent_decision: Optional[Dict[str, Any]],
    latest_signal: Optional[Dict[str, Any]],
    strat_metrics: Dict[str, Any],
    bench_metrics: Dict[str, Any],
    notes: Dict[str, List[str]],
    pdf_path: Union[str, Path],
    debug: bool = False,
    page_size: str = "A4",
) -> Tuple[bool, Dict[str, Any]]:
    info: Dict[str, Any] = {"pdf_written": str(pdf_path), "reportlab": _HAS_REPORTLAB}
    if not _HAS_REPORTLAB:
        info["error"] = "reportlab not installed"
        return False, info

    pdf_p = _coerce_path(pdf_path)
    pdf_p.parent.mkdir(parents=True, exist_ok=True)

    ps = A4 if page_size.upper() == "A4" else letter
    styles = _pdf_styles()
    doc = SimpleDocTemplate(
        str(pdf_p),
        pagesize=ps,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title=f"{ticker} Technical Agent Report",
        author="msft_agent_project",
    )

    story: List[Any] = []
    story.append(Paragraph(f"{ticker} Technical Agent Report", styles["H1"]))
    story.append(Paragraph(f"Generated: {_now_iso()}", styles["Body"]))
    story.append(Paragraph(f"Prefix: {artifacts.prefix}", styles["Body"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1. Run summary", styles["H2"]))
    if run_summary:
        items = []
        items.append(("Ticker", run_summary.get("ticker", ticker)))
        if run_summary.get("start") or run_summary.get("end"):
            items.append(("Period", f"{run_summary.get('start', '-') } → {run_summary.get('end', '-') }"))
        if run_summary.get("years") is not None:
            items.append(("Years", str(run_summary.get("years"))))
        mode = run_summary.get("rebalance_mode") or run_summary.get("rebalance") or run_summary.get("rebalanceMode")
        if mode:
            items.append(("Rebalance mode", str(mode)))
        data = [["Field", "Value"]] + [[k, v] for k, v in items]
        t = Table(data, hAlign="LEFT", colWidths=[1.8 * inch, 4.6 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
    else:
        story.append(Paragraph("Run summary not found.", styles["Body"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Agent decision (formatted)", styles["H2"]))
    if agent_decision:
        kv = [
            ("as_of", agent_decision.get("as_of", "-")),
            ("action", agent_decision.get("action", "-")),
            ("recommended_position", str(agent_decision.get("recommended_position", "-"))),
            ("confidence", str(agent_decision.get("confidence", "-"))),
        ]
        data = [["Field", "Value"]] + [[k, v] for k, v in kv]
        t = Table(data, hAlign="LEFT", colWidths=[1.8 * inch, 4.6 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
        thesis = agent_decision.get("thesis")
        if thesis:
            story.append(Spacer(1, 8))
            story.append(Paragraph("Thesis:", styles["Body"]))
            story.append(Paragraph(str(thesis), styles["Body"]))

        def add_list(title: str, arr: Any):
            if isinstance(arr, list) and arr:
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"{title}:", styles["Body"]))
                bullets = "<br/>".join([f"• {str(x)}" for x in arr])
                story.append(Paragraph(bullets, styles["Body"]))

        add_list("Key signals", agent_decision.get("key_signals"))
        add_list("Risks", agent_decision.get("risks"))
        add_list("Constraints", agent_decision.get("constraints"))
    else:
        story.append(Paragraph("Agent decision not found.", styles["Body"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Latest signal", styles["H2"]))
    if latest_signal:
        keys_keep = ["as_of", "close", "adj_close", "ma", "MA200", "RSI14", "rsi", "macd_hist", "vol", "recommended_position"]
        rows = [["Field", "Value"]]
        for k in keys_keep:
            if k in latest_signal:
                rows.append([k, str(latest_signal.get(k))])
        if len(rows) == 1:
            story.append(Paragraph("No known latest-signal fields found.", styles["Body"]))
        else:
            t = Table(rows, hAlign="LEFT", colWidths=[1.8 * inch, 4.6 * inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(t)
    else:
        story.append(Paragraph("Latest signal not found.", styles["Body"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("4. Backtest metrics", styles["H2"]))

    def metric_cell(name: str, v: Any) -> str:
        if name in ("CAGR", "MaxDrawdown", "HitRate", "AvgExposure"):
            return _fmt_pct(v, 2)
        if name in ("Sharpe",):
            return _fmt_float(v, 3)
        if name in ("AnnualTurnover",):
            return _fmt_float(v, 2)
        if name in ("CostRate", "TargetVol"):
            return _fmt_float(v, 4)
        return "-" if v is None else str(v)

    metric_order = ["CAGR", "Sharpe", "MaxDrawdown", "HitRate", "AvgExposure", "AnnualTurnover", "CostRate", "TargetVol"]
    data = [["Metric", "Strategy", "Buy & Hold"]]
    for m in metric_order:
        data.append([m, metric_cell(m, strat_metrics.get(m)), metric_cell(m, bench_metrics.get(m))])

    t = Table(data, hAlign="LEFT", colWidths=[2.0 * inch, 2.2 * inch, 2.2 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("5. Equity curve", styles["H2"]))
    if artifacts.equity_curve_png and artifacts.equity_curve_png.exists():
        available_w = doc.width
        try:
            img = RLImage(str(artifacts.equity_curve_png))
            iw, ih = img.imageWidth, img.imageHeight
            if iw and ih:
                scale = min(1.0, float(available_w) / float(iw))
                img.drawWidth = iw * scale
                img.drawHeight = ih * scale
            story.append(img)
        except Exception as e:
            story.append(Paragraph(f"Failed to embed equity curve image: {repr(e)}", styles["Body"]))
    else:
        story.append(Paragraph("Equity curve image not found.", styles["Body"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("6. Notes / limitations", styles["H2"]))
    if notes.get("assumptions"):
        story.append(Paragraph("Assumptions:", styles["Body"]))
        story.append(Paragraph("<br/>".join([f"• {x}" for x in notes["assumptions"]]), styles["Body"]))
        story.append(Spacer(1, 6))
    if notes.get("limitations"):
        story.append(Paragraph("Limitations:", styles["Body"]))
        story.append(Paragraph("<br/>".join([f"• {x}" for x in notes["limitations"]]), styles["Body"]))
        story.append(Spacer(1, 6))
    if not notes.get("assumptions") and not notes.get("limitations"):
        story.append(Paragraph("No assumptions/limitations extracted from trade_note.md.", styles["Body"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("7. Reproducibility / artifacts", styles["H2"]))
    rows = [["Artifact", "File", "Exists"]]
    for name, fname, ok in artifacts.as_table_rows():
        rows.append([name, fname, str(ok)])
    t = Table(rows, hAlign="LEFT", colWidths=[2.3 * inch, 3.3 * inch, 0.8 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(t)

    def add_raw(title: str, obj: Any, max_chars: int = 12000):
        if obj is None:
            return
        story.append(PageBreak())
        story.append(Paragraph(f"Appendix: {title} (raw JSON)", styles["H2"]))
        raw = json.dumps(obj, ensure_ascii=False, indent=2)
        if len(raw) > max_chars:
            raw = raw[:max_chars] + "\n... truncated ..."
        story.append(Preformatted(raw, styles["Mono"]))

    add_raw("Run summary", run_summary)
    add_raw("Agent decision", agent_decision)
    add_raw("Latest signal", latest_signal)

    try:
        doc.build(story)
        info["ok"] = True
        return True, info
    except Exception as e:
        info["error"] = repr(e)
        return False, info


# ----------------------------
# Main entrypoint: generate_report
# ----------------------------
def generate_report(
    outdir: Union[str, Path],
    prefix: str,
    ticker: Optional[str] = None,
    pdf: bool = True,
    debug: bool = False,
) -> Dict[str, Any]:
    outdir_p = _coerce_path(outdir)
    outdir_p.mkdir(parents=True, exist_ok=True)

    tick = ticker or prefix.upper()
    artifacts = load_artifacts(outdir_p, prefix, tick)

    run_summary = _safe_read_json(artifacts.run_summary)
    agent_decision = _safe_read_json(artifacts.agent_decision)
    latest_signal = _safe_read_json(artifacts.latest_signal)

    strat_metrics = _normalize_metrics(_parse_metrics_csv(artifacts.metrics))
    bench_metrics = _normalize_metrics(_parse_metrics_csv(artifacts.benchmark_metrics))

    if not strat_metrics and isinstance(run_summary, dict):
        strat_metrics = _normalize_metrics(run_summary.get("metrics", {}) or {})
    if not bench_metrics and isinstance(run_summary, dict):
        bench_metrics = _normalize_metrics(
            run_summary.get("bh_metrics", {})
            or run_summary.get("benchmark_metrics", {})
            or {}
        )

    trade_note = _safe_read_text(artifacts.trade_note_md)
    notes = parse_trade_note(trade_note)

    # If the note file is missing or has no extractable structure, use a minimal default set
    if (not notes.get("assumptions")) and (not notes.get("limitations")):
        notes["assumptions"] = [
            "Data: daily OHLCV via yfinance",
            "Execution: signal computed on day t, applied from day t+1 (1-day lag)",
        ]
        notes["limitations"] = [
            "Single-asset backtest; simplified transaction cost model; no slippage modelling",
            "Indicators computed on Close / Adj Close; results depend on data quality and corporate actions handling",
            "Backtests do not guarantee future performance",
        ]

    md_path = outdir_p / f"{prefix}_report.md"
    md_text = build_markdown(
        ticker=tick,
        artifacts=artifacts,
        run_summary=run_summary if isinstance(run_summary, dict) else None,
        agent_decision=agent_decision if isinstance(agent_decision, dict) else None,
        latest_signal=latest_signal if isinstance(latest_signal, dict) else None,
        strat_metrics=strat_metrics,
        bench_metrics=bench_metrics,
        notes=notes,
    )
    md_path.write_text(md_text, encoding="utf-8")

    pdf_path = outdir_p / f"{prefix}_report.pdf"
    pdf_ok = False
    pdf_info: Dict[str, Any] = {}
    if pdf:
        pdf_ok, pdf_info = build_pdf(
            ticker=tick,
            artifacts=artifacts,
            run_summary=run_summary if isinstance(run_summary, dict) else None,
            agent_decision=agent_decision if isinstance(agent_decision, dict) else None,
            latest_signal=latest_signal if isinstance(latest_signal, dict) else None,
            strat_metrics=strat_metrics,
            bench_metrics=bench_metrics,
            notes=notes,
            pdf_path=pdf_path,
            debug=debug,
        )

    return {
        "md_path": str(md_path),
        "pdf_path": str(pdf_path),
        "pdf_ok": bool(pdf_ok),
        "info": pdf_info,
        "artifacts": {
            name: (p.name if p else None)
            for name, p in [
                ("run_manifest", artifacts.run_manifest),
                ("run_summary", artifacts.run_summary),
                ("agent_decision", artifacts.agent_decision),
                ("latest_signal", artifacts.latest_signal),
                ("metrics", artifacts.metrics),
                ("benchmark_metrics", artifacts.benchmark_metrics),
                ("equity_curve_png", artifacts.equity_curve_png),
                ("equity_curve_csv", artifacts.equity_curve_csv),
                ("trade_note_md", artifacts.trade_note_md),
            ]
        },
    }


# ----------------------------
# CLI
# ----------------------------
def _cli() -> int:
    ap = argparse.ArgumentParser(description="Generate Markdown/PDF report from run artifacts.")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory containing artifacts")
    ap.add_argument("--prefix", type=str, required=True, help="Run prefix, e.g. msft")
    ap.add_argument("--ticker", type=str, default=None, help="Ticker override, e.g. MSFT")
    ap.add_argument("--pdf", action="store_true", help="Generate PDF (requires reportlab)")
    ap.add_argument("--no-pdf", action="store_true", help="Do not generate PDF")
    ap.add_argument("--debug", action="store_true", help="Verbose debug info")
    args = ap.parse_args()

    want_pdf = args.pdf and not args.no_pdf
    rep = generate_report(
        outdir=args.outdir,
        prefix=args.prefix,
        ticker=args.ticker,
        pdf=want_pdf,
        debug=args.debug,
    )
    print(f"Report written: {rep['md_path']}")
    if want_pdf:
        if rep["pdf_ok"]:
            print(f"PDF written: {rep['pdf_path']}")
        else:
            print(f"PDF skipped/failed: {rep.get('info')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
