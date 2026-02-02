#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_demo.py
============
Project entrypoint for:
1) LLM Trading Agent (agent_trader.run_agent)
2) Pure backtest pipeline (tools_market_data + tools_backtest)

Usage examples
--------------
# 1) Run the full LLM agent pipeline (recommended)
python run_demo.py --mode agent --ticker MSFT --years 10 --rebalance weekly --outdir outputs --prefix msft

# 2) Run pure backtest only (no LLM decision)
python run_demo.py --mode backtest --ticker MSFT --years 10 --rebalance weekly --outdir outputs --prefix msft

Notes
-----
- This script attempts to load env vars from BOTH ".env" and "API.env".
- Ensure these environment variables exist in one of them:
  OPENAI_API_KEY=...
  OPENAI_MODEL=gpt-4.1-mini
  OPENAI_MODEL_STRONG=gpt-4.1
  OPENAI_TEMPERATURE=0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# -----------------------
# Helpers
# -----------------------
def _try_load_dotenv() -> None:
    """Load environment variables from .env / API.env if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        return

    # prefer user local env already set; do not override by default
    load_dotenv(".env", override=False)
    load_dotenv("API.env", override=False)

def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        raise RuntimeError(
            f"Missing env var: {name}. Put it in API.env or .env (KEY=VALUE), then rerun."
        )
    return v

def _safe_json_dump(obj: Any) -> str:
    def _default(x: Any):
        try:
            return x.__dict__
        except Exception:
            return str(x)

    return json.dumps(obj, ensure_ascii=False, indent=2, default=_default)

@dataclass
class RunManifest:
    mode: str
    ticker: str
    years: int
    rebalance: str
    outdir: str
    prefix: str
    started_at_utc: float
    ended_at_utc: float
    elapsed_s: float
    python: str
    cwd: str
    env_has_key: bool
    model: Optional[str] = None
    strong_model: Optional[str] = None
    temperature: Optional[float] = None

def _write_manifest(outdir: Path, prefix: str, manifest: RunManifest) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / f"{prefix}_run_manifest.json"
    p.write_text(_safe_json_dump(asdict(manifest)), encoding="utf-8")
    return p

# -----------------------
# CLI
# -----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MSFT trading agent project runner")

    parser.add_argument(
        "--mode",
        choices=["agent", "backtest"],
        default="agent",
        help="agent: run LLM trading agent; backtest: strategy backtest only",
    )
    parser.add_argument("--ticker", default="MSFT", help="Ticker symbol, e.g. MSFT")
    parser.add_argument("--years", type=int, default=10, help="Years of daily data")
    parser.add_argument(
    "--rebalance",
    choices=["daily", "weekly", "monthly"],
    default="weekly",
    help='Rebalance frequency: daily | weekly | monthly',
    )

    parser.add_argument(
        "--cost-rate",
        type=float,
        default=0.001,
        help="Transaction cost rate per unit turnover (e.g., 0.001 = 0.1%%).",
    )
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--prefix", default="msft", help="Output filename prefix")

    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v or -vv)",
    )

    return parser.parse_args()

# -----------------------
# Main runners
# -----------------------
def run_mode_agent(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Full agent pipeline: fetch data -> indicators -> backtest -> LLM decision -> save outputs.
    Delegates the core logic to agent_trader.run_agent.
    """
    from agent_trader import run_agent  # your existing module

    res = run_agent(
        args.ticker,
        years=args.years,
        rebalance_mode=args.rebalance,
        outdir=args.outdir,
        prefix=args.prefix,
    )
    return res

def run_mode_backtest(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Pure strategy pipeline: fetch -> indicators -> end-to-end backtest.
    No LLM decision. Useful for debugging strategy + data issues.
    """
    from tools_market_data import fetch_ohlcv, add_indicators
    from tools_backtest import StrategyConfig, end_to_end_backtest

    df = fetch_ohlcv(args.ticker, args.years)
    df = add_indicators(df)
    cfg = StrategyConfig(rebalance_mode=args.rebalance, cost_rate=args.cost_rate)
    res = end_to_end_backtest(df, cfg, outdir=args.outdir, prefix=args.prefix)
    return res

def main() -> int:
    args = parse_args()
    _setup_logging(args.verbose)
    _try_load_dotenv()

    started = time.time()

    # record env snapshot (but never print key)
    env_has_key = bool(os.getenv("OPENAI_API_KEY", "").strip())
    model = os.getenv("OPENAI_MODEL")
    strong_model = os.getenv("OPENAI_MODEL_STRONG")
    temp_raw = os.getenv("OPENAI_TEMPERATURE")
    temperature = float(temp_raw) if temp_raw else None

    logging.info("mode=%s ticker=%s years=%s rebalance=%s", args.mode, args.ticker, args.years, args.rebalance)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "agent":
            if not env_has_key:
                raise RuntimeError("OPENAI_API_KEY is missing. Put it in API.env or .env then rerun.")
            res = run_mode_agent(args)
        elif args.mode == "backtest":
            res = run_mode_backtest(args)
        else:
            res = run_mode_llm(args)

        ended = time.time()

        manifest = RunManifest(
            mode=args.mode,
            ticker=args.ticker,
            years=args.years,
            rebalance=args.rebalance,
            outdir=str(outdir),
            prefix=args.prefix,
            started_at_utc=started,
            ended_at_utc=ended,
            elapsed_s=ended - started,
            python=sys.executable,
            cwd=str(Path.cwd()),
            env_has_key=env_has_key,
            model=model,
            strong_model=strong_model,
            temperature=temperature,
        )
        manifest_path = _write_manifest(outdir, args.prefix, manifest)

        # Console summary
        print("\n=== RUN OK ===")
        print(f"mode: {args.mode}")
        print(f"outputs: {outdir.resolve()}")
        print(f"manifest: {manifest_path.resolve()}")

        if isinstance(res, dict) and "decision" in res:
            print("\n--- Agent decision ---")
            print(_safe_json_dump(res["decision"]))

        # also print compact metrics if present
        if isinstance(res, dict) and "metrics" in res:
            print("\n--- Strategy metrics ---")
            print(_safe_json_dump(res["metrics"]))

                # -----------------------
        # Auto-generate report (MD + PDF) by default
        # -----------------------

        if args.mode in ("agent", "backtest"):
            try:
                from report_writer import generate_report
                rep = generate_report(
                    outdir=Path(args.outdir),
                    prefix=args.prefix,
                    ticker=args.ticker,
                    pdf=True,                       
                    debug=(args.verbose >= 2),
                )
                print("\n--- Report generated ---")
                print("md:", Path(rep["md_path"]).resolve())
                if rep.get("pdf_ok"):
                    print("pdf:", Path(rep["pdf_path"]).resolve())
                else:
                    print("pdf: failed:", rep.get("pdf_error"))
            except Exception as e:
                print("[WARN] Report generation failed:", repr(e))

        
        return 0

    except Exception as e:
        ended = time.time()
        logging.exception("Run failed: %s", e)

        # try write a failure manifest to help debugging
        try:
            manifest = RunManifest(
                mode=args.mode,
                ticker=args.ticker,
                years=args.years,
                rebalance=args.rebalance,
                outdir=str(outdir),
                prefix=args.prefix,
                started_at_utc=started,
                ended_at_utc=ended,
                elapsed_s=ended - started,
                python=sys.executable,
                cwd=str(Path.cwd()),
                env_has_key=env_has_key,
                model=model,
                strong_model=strong_model,
                temperature=temperature,
            )
            _write_manifest(outdir, args.prefix + "_FAILED", manifest)
        except Exception:
            pass

        print("\n=== RUN FAILED ===")
        print(str(e))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
