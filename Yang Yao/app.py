from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_DEMO = PROJECT_ROOT / "run_demo.py"


@dataclass
class RunResult:
    ok: bool
    returncode: int
    outdir: Path
    prefix: str
    stdout: str
    stderr: str
    files: Dict[str, Path]
    bytes_map: Dict[str, bytes]


def _read_bytes(p: Path) -> Optional[bytes]:
    try:
        if p.exists() and p.is_file():
            return p.read_bytes()
    except Exception:
        return None
    return None


def _collect_artifacts(outdir: Path, prefix: str, mode: str) -> Dict[str, Path]:
    candidates: Dict[str, Path] = {
        "PDF report": outdir / f"{prefix}_report.pdf",
        "Markdown report": outdir / f"{prefix}_report.md",
        "metrics.csv": outdir / f"{prefix}_metrics.csv",
        "run_manifest.json": outdir / f"{prefix}_run_manifest.json",
        "run_summary.json": outdir / f"{prefix}_run_summary.json",
    }
    if mode == "agent":
        candidates["agent_decision.json"] = outdir / f"{prefix}_agent_decision.json"

    # Keep only existing files
    return {k: v for k, v in candidates.items() if v.exists() and v.is_file()}


def run_pipeline(
    *,
    mode: str,
    ticker: str,
    years: int,
    rebalance: str,
    cost_rate: float,
    outdir: str,
    prefix: str,
    use_temp_outdir: bool,
    keep_outputs: bool,
    api_key: str,
    openai_model: str,
    openai_model_strong: str,
    temperature: float,
) -> RunResult:
    if not RUN_DEMO.exists():
        return RunResult(
            ok=False,
            returncode=2,
            outdir=PROJECT_ROOT,
            prefix=prefix,
            stdout="",
            stderr=f"run_demo.py not found at: {RUN_DEMO}",
            files={},
            bytes_map={},
        )

    temp_dir: Optional[Path] = None
    run_outdir = Path(outdir)

    if use_temp_outdir:
        temp_dir = Path(tempfile.mkdtemp(prefix="msft_agent_outputs_"))
        run_outdir = temp_dir
    else:
        run_outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(RUN_DEMO),
        "--mode",
        mode,
        "--ticker",
        ticker,
        "--years",
        str(int(years)),
        "--rebalance",
        rebalance,
        "--cost-rate",
        f"{float(cost_rate):.10f}",
        "--outdir",
        str(run_outdir),
        "--prefix",
        prefix,
    ]

    env = os.environ.copy()
    if mode == "agent":
        if api_key.strip():
            env["OPENAI_API_KEY"] = api_key.strip()
        if openai_model.strip():
            env["OPENAI_MODEL"] = openai_model.strip()
        if openai_model_strong.strip():
            env["OPENAI_MODEL_STRONG"] = openai_model_strong.strip()
        env["OPENAI_TEMPERATURE"] = str(float(temperature))

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        if temp_dir and (not keep_outputs):
            shutil.rmtree(temp_dir, ignore_errors=True)
        return RunResult(
            ok=False,
            returncode=1,
            outdir=run_outdir,
            prefix=prefix,
            stdout="",
            stderr=f"Failed to start subprocess: {e}",
            files={},
            bytes_map={},
        )

    files = _collect_artifacts(run_outdir, prefix, mode)
    bytes_map: Dict[str, bytes] = {}
    for label, path in files.items():
        b = _read_bytes(path)
        if b is not None:
            bytes_map[label] = b

    ok = proc.returncode == 0

    # If using temp outdir and not keeping outputs, delete after reading into memory.
    if temp_dir and (not keep_outputs):
        shutil.rmtree(temp_dir, ignore_errors=True)

    return RunResult(
        ok=ok,
        returncode=proc.returncode,
        outdir=run_outdir,
        prefix=prefix,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        files=files,
        bytes_map=bytes_map,
    )


def _render_metrics_preview(bytes_map: Dict[str, bytes]) -> None:
    b = bytes_map.get("metrics.csv")
    if not b:
        return
    try:
        df = pd.read_csv(io.BytesIO(b))
        st.subheader("Metrics preview")
        st.dataframe(df, use_container_width=True)
    except Exception:
        pass


# ===== Intro / Overview =====
with st.expander("What does this app do? (Overview)", expanded=True):
    st.markdown(
        """
This web app demonstrates a lightweight **AI trading agent** workflow for a single stock (e.g., **MSFT**):

1) Download historical market data (Yahoo Finance via `yfinance`)  
2) Compute technical indicators (e.g., Moving Average, RSI, MACD)  
3) (Optional) Ask an LLM to produce a **JSON decision** (e.g., target position / confidence / rationale)  
4) Run a backtest with a chosen **rebalance frequency** (daily / weekly / monthly) and transaction cost  
5) Export results as **Markdown + PDF report**, plus metrics and run manifests for reproducibility

**Goal:** Provide a clear, end-to-end pipeline that is easy to run and easy to grade/review.
"""
    )

with st.expander("Modes: Backtest vs Agent (LLM)", expanded=False):
    st.markdown(
        """
- **Backtest mode**  
  Uses only the rule-based signal pipeline (indicators -> position), then evaluates performance in a backtest.  
  No API key required.

- **Agent mode (LLM-assisted)**  
  Adds an LLM step that outputs a structured **JSON decision** (e.g., position/confidence/rationale).  
  Requires `OPENAI_API_KEY`.  
  This mode demonstrates how an agent can sit on top of a tool pipeline and make decisions in a constrained schema.
"""
    )

with st.expander("Outputs (saved to your output folder)", expanded=False):
    st.markdown(
        """
After each run, the app writes artifacts to `--outdir` (default: `outputs/`). Typical files include:

- `{prefix}_report.pdf` (final report)
- `{prefix}_report.md` (markdown version)
- `{prefix}_metrics.csv` (performance metrics)
- `{prefix}_run_manifest.json` (inputs, config, runtime info)
- `{prefix}_agent_decision.json` (agent mode only)

You can download the PDF/MD/CSV directly from the UI after a successful run.
"""
    )

with st.expander("Notes & limitations", expanded=False):
    st.markdown(
        """
- Backtests are simplified and may not generalize out-of-sample.
- Market impact / slippage are not modeled (only a simple transaction cost).
- LLM outputs can vary across runs unless strict determinism is enforced.
"""
    )
# ===== End Intro / Overview =====

with st.sidebar:
    st.header("Run settings")

    mode_ui = st.selectbox(
        "Mode",
        options=["backtest", "agent"],
        index=0,
        help="backtest: no LLM. agent: includes LLM decision step (requires API key).",
    )

    ticker = st.text_input("Ticker", value="MSFT").strip().upper()
    years = st.slider("Years of history", min_value=1, max_value=20, value=10, step=1)

    rebalance = st.selectbox(
        "Rebalance",
        options=["daily", "weekly", "monthly"],
        index=2,  # weekly
        help="How often the strategy is allowed to change its position.",
    )

    cost_rate = st.number_input(
        "Transaction cost rate (per turnover, e.g. 0.001 = 0.1%)",
        min_value=0.0,
        max_value=0.05,
        value=0.0010,
        step=0.0005,
        format="%.4f",
    )

    st.divider()
    st.subheader("Outputs")

    use_temp_outdir = st.checkbox(
        "Use temporary output folder (recommended)",
        value=True,
        help="If enabled, artifacts are generated into a temp folder and can be downloaded from the page.",
    )

    keep_outputs = st.checkbox(
        "Keep outputs on disk",
        value=False,
        help="If disabled while using a temp folder, the temp folder will be deleted after the app loads artifacts into memory.",
    )

    outdir = st.text_input(
        "Output folder (used only when NOT using temporary folder)",
        value="outputs",
        disabled=use_temp_outdir,
    )

    default_prefix = f"{ticker.lower()}_{rebalance}"
    prefix = st.text_input("Filename prefix", value=default_prefix)

    # ---- OpenAI inputs: ONLY show for agent mode ----
    st.divider()
    st.subheader("OpenAI (agent mode only)")

    api_key = ""
    openai_model = "gpt-4.1-mini"
    openai_model_strong = "gpt-4.1"
    temperature = 0.2

    if mode_ui == "agent":
        api_key = st.text_input("OPENAI_API_KEY", value="", type="password")
        openai_model = st.text_input("OPENAI_MODEL", value="gpt-4.1-mini")
        openai_model_strong = st.text_input("OPENAI_MODEL_STRONG", value="gpt-4.1")
        temperature = st.slider("OPENAI_TEMPERATURE", 0.0, 1.0, 0.2, 0.05)
    else:
        st.caption("Not needed for backtest mode.")

    run_btn = st.button("Run", type="primary", use_container_width=True)

# Persist last run across reruns
if "last_run" not in st.session_state:
    st.session_state["last_run"] = None

if run_btn:
    if not ticker:
        st.error("Ticker is required.")
        st.stop()

    if mode_ui == "agent" and not api_key.strip():
        st.error("Agent mode requires an OPENAI_API_KEY.")
        st.stop()

    with st.spinner("Running..."):
        rr = run_pipeline(
            mode=mode_ui,
            ticker=ticker,
            years=int(years),
            rebalance=rebalance,
            cost_rate=float(cost_rate),
            outdir=outdir,
            prefix=prefix,
            use_temp_outdir=use_temp_outdir,
            keep_outputs=keep_outputs,
            api_key=api_key,
            openai_model=openai_model,
            openai_model_strong=openai_model_strong,
            temperature=float(temperature),
        )
    st.session_state["last_run"] = rr

rr: Optional[RunResult] = st.session_state.get("last_run")

col1, col2 = st.columns([1.35, 1.0], gap="large")

with col1:
    st.subheader("Logs")
    if rr is None:
        st.info("Run the pipeline from the sidebar.")
    else:
        if rr.ok:
            st.success("Run OK")
        else:
            st.error(f"Run failed (exit code {rr.returncode}).")

        st.code(
            f"{sys.executable} {RUN_DEMO.name} --mode {mode_ui} --ticker {ticker} --years {years} "
            f"--rebalance {rebalance} --cost-rate {cost_rate:.4f} --outdir {str(rr.outdir)} --prefix {prefix}",
            language="bash",
        )

        st.caption("stdout")
        st.code(rr.stdout.strip() or "(empty)", language="text")

        if rr.stderr.strip():
            st.caption("stderr")
            st.code(rr.stderr.strip(), language="text")

        _render_metrics_preview(rr.bytes_map)

with col2:
    st.subheader("Artifacts")
    if rr is None:
        st.write("No artifacts yet.")
    else:
        if use_temp_outdir and (not keep_outputs):
            st.caption("Temp outputs were loaded into memory and then cleaned up automatically.")
        else:
            st.caption(f"Output folder: `{rr.outdir}`")

        if not rr.files:
            st.warning("No artifacts found. Check logs above.")
        else:
            for label, path in rr.files.items():
                data = rr.bytes_map.get(label)
                if data is None:
                    continue
                mime = "application/octet-stream"
                if path.suffix.lower() == ".pdf":
                    mime = "application/pdf"
                elif path.suffix.lower() in (".md", ".markdown"):
                    mime = "text/markdown"
                elif path.suffix.lower() == ".csv":
                    mime = "text/csv"
                elif path.suffix.lower() == ".json":
                    mime = "application/json"

                st.download_button(
                    label=f"Download {label}",
                    data=data,
                    file_name=path.name,
                    mime=mime,
                    use_container_width=True,
                )

        st.divider()
        if st.button("Clear results", use_container_width=True):
            st.session_state["last_run"] = None
            st.rerun()
