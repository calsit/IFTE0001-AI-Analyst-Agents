# tools_backtest.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rebalance import make_rebalance_flags


@dataclass
class StrategyConfig:
    # --- Column names the backtest expects (we'll rename common variants into these) ---
    date_col: str = "Date"
    close_col: str = "Close"
    adj_close_col: str = "Adj Close"

    ma_col: str = "ma"              # MA200 normalized name
    rsi_col: str = "rsi"            # RSI14 normalized name
    macd_hist_col: str = "macd_hist"
    vol_col: str = "vol"            # rolling vol (annualized)

    # --- Strategy logic ---
    ma_window: int = 200
    rsi_window: int = 14

    rsi_entry_low: float = 30.0
    rsi_entry_high: float = 60.0
    rsi_overheat: float = 70.0

    # --- Position sizing (vol targeting) ---
    target_vol: float = 0.15        # annualized target vol
    w_max: float = 1.0              # max leverage (long-only so just cap weight)

    # --- Trading / costs ---
    cost_rate: float = 0.001        # cost per unit turnover (single-side)
    rebalace_print: bool = True     # keep your debug prints

    # --- Rebalance mode: "daily" | "weekly" | "monthly"
    # daily: can change position every day
    # weekly: only change position on Monday
    # monthly: only change position on first trading day of month
    rebalance_mode: str = "weekly"

    # --- Output ---
    plot: bool = True


# ----------------------------
# Helpers: data + column mapping
# ----------------------------
def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _rename_if_missing(df: pd.DataFrame, target: str, candidates: Sequence[str]) -> pd.DataFrame:
    """
    If `target` column does not exist, try renaming from any candidate.
    Candidate matching is case-insensitive.
    """
    if target in df.columns:
        return df

    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return df.rename(columns={cand: target})
        cl = cand.lower()
        if cl in cols_lower:
            return df.rename(columns={cols_lower[cl]: target})

    return df


def _ensure_core_cols(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Ensure Date/Close/Adj Close exist, sort by date, drop duplicates.
    """
    out = df.copy()

    # common rename variants for OHLCV
    out = _rename_if_missing(out, cfg.date_col, ["date", "Date", "datetime", "Datetime", "timestamp"])
    out = _rename_if_missing(out, cfg.close_col, ["close", "Close"])
    out = _rename_if_missing(out, cfg.adj_close_col, ["adj close", "Adj Close", "adj_close", "Adj_Close", "AdjClose"])

    if cfg.date_col not in out.columns:
        raise ValueError(f"Missing {cfg.date_col}. Found columns: {list(out.columns)}")
    if cfg.close_col not in out.columns:
        raise ValueError(f"Missing {cfg.close_col}. Found columns: {list(out.columns)}")
    if cfg.adj_close_col not in out.columns:
        
        out[cfg.adj_close_col] = out[cfg.close_col].astype(float)

    out[cfg.date_col] = _to_datetime(out[cfg.date_col])
    out = out.dropna(subset=[cfg.date_col]).sort_values(cfg.date_col)
    out = out.drop_duplicates(subset=[cfg.date_col], keep="last").reset_index(drop=True)
    return out


def _normalize_indicator_cols(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Map common indicator column names produced by tools_market_data.add_indicators()
    into cfg.ma_col/cfg.rsi_col/cfg.macd_hist_col/cfg.vol_col.
    """
    out = df.copy()

    # MA200 typical names: MA200 / ma200 / SMA200 ...
    out = _rename_if_missing(
        out,
        cfg.ma_col,
        ["ma", "MA200", "ma200", "sma200", "SMA200", "MA_200", "MA 200"],
    )
    # RSI14 typical names: RSI14 / rsi14 / RSI_14 ...
    out = _rename_if_missing(
        out,
        cfg.rsi_col,
        ["rsi", "RSI", "rsi_14", "RSI_14", "rsi14", "RSI14"],
    )
    # MACD hist typical names
    out = _rename_if_missing(
        out,
        cfg.macd_hist_col,
        ["macd_hist", "MACD_hist", "macd_histogram", "MACD_histogram", "MACDHistogram"],
    )
    # vol typical names (your tools_market_data already outputs vol)
    out = _rename_if_missing(
        out,
        cfg.vol_col,
        ["vol", "Vol", "VOL", "realized_vol", "realised_vol", "ann_vol"],
    )

    missing = [c for c in [cfg.ma_col, cfg.rsi_col, cfg.macd_hist_col, cfg.vol_col] if c not in out.columns]
    if missing:
        raise ValueError(
            f"Missing indicator columns: {missing}. "
            f"Found columns: {list(out.columns)}. "
            f"Make sure tools_market_data.add_indicators() produced them."
        )
    return out


def _rebalance_mask(dates: pd.Series, mode: str) -> pd.Series:
    """
    Returns boolean mask: True means "allowed to rebalance today".
    """
    mode = (mode or "daily").lower()
    dts = pd.to_datetime(dates)

    if mode == "daily":
        return pd.Series(True, index=dates.index)

    if mode == "weekly":
        # Monday == 0
        m = (dts.dt.weekday == 0)
        # also allow first day
        if len(m) > 0:
            m.iloc[0] = True
        return m

    if mode == "monthly":
        # first trading day of each month
        m = dts.dt.to_period("M").ne(dts.dt.to_period("M").shift(1))
        if len(m) > 0:
            m.iloc[0] = True
        return m

    raise ValueError(f"Unknown rebalance_mode={mode}. Use daily|weekly|monthly.")


# ----------------------------
# Strategy + Backtest
# ----------------------------
def generate_signal_and_weight(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """
    Create:
      - signal (0/1) based on MA regime + RSI band + MACD histogram
      - weight via vol targeting (target_vol / vol) clipped to [0, w_max]
      - position_raw = signal * weight  (this is "decision on day t")
      - position = position_raw shifted by 1 day to avoid lookahead (applied on day t+1)
      - optional rebalance constraint
    """
    out = df.copy()

    close = out[cfg.close_col].astype(float)
    ma = out[cfg.ma_col].astype(float)
    rsi = out[cfg.rsi_col].astype(float)
    macd_hist = out[cfg.macd_hist_col].astype(float)
    vol = out[cfg.vol_col].astype(float)

    regime = close > ma
    entry = regime & (macd_hist > 0) & (rsi >= cfg.rsi_entry_low) & (rsi <= cfg.rsi_entry_high)
    exit_ = (~regime) | (macd_hist < 0) | (rsi > cfg.rsi_overheat)

    signal = np.zeros(len(out), dtype=float)
    in_pos = False
    for i in range(len(out)):
        if (not in_pos) and bool(entry.iloc[i]):
            in_pos = True
        elif in_pos and bool(exit_.iloc[i]):
            in_pos = False
        signal[i] = 1.0 if in_pos else 0.0

    out["signal"] = signal

    raw_w = cfg.target_vol / (vol + 1e-12)
    raw_w = np.clip(raw_w, 0.0, cfg.w_max)
    out["weight"] = raw_w

    out["position_raw"] = out["signal"] * out["weight"]

    # rebalance constraint: only allow changing position on rebalance days
    reb_flags = make_rebalance_flags(pd.to_datetime(out[cfg.date_col]).values, cfg.rebalance_mode)
    out["rebalance_ok"] = reb_flags.values.astype(bool)

    # hold last position_raw between rebalance days
    out["position_raw"] = out["position_raw"].where(out["rebalance_ok"]).ffill().fillna(0.0)


    # apply from next day to avoid lookahead
    out["position"] = out["position_raw"].shift(1).fillna(0.0)

    # debug prints (optional)
    if cfg.rebalace_print:
        changes = (out["position_raw"].diff().abs() > 1e-12).sum()
        print(f"rebalance_mode = {cfg.rebalance_mode} | rebalance_count = {int(out['rebalance_ok'].sum())}")
        print(f"position_raw changes = {int(changes)}")

    return out


def backtest(df_sig: pd.DataFrame, cfg: StrategyConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Backtest on close-to-close returns using Adj Close.
    Position is applied with 1-day lag already in df_sig["position"].
    """
    out = df_sig.copy()
    px = out[cfg.adj_close_col].astype(float)

    ret = px.pct_change().fillna(0.0)
    out["ret"] = ret

    pos = out["position"].astype(float)
    gross = pos * ret

    turnover = (pos - pos.shift(1).fillna(0.0)).abs()
    cost = turnover * cfg.cost_rate

    net = gross - cost

    out["turnover"] = turnover
    out["cost"] = cost
    out["net_ret"] = net
    out["equity"] = (1.0 + net).cumprod()

    metrics = compute_metrics(out, cfg)
    return out, metrics


def compute_metrics(bt: pd.DataFrame, cfg: StrategyConfig) -> Dict[str, float]:
    """
    Standard metrics for grading: CAGR/Sharpe/MaxDD/HitRate/AvgExposure/AnnualTurnover/CostRate
    """
    equity = bt["equity"].astype(float)
    net = bt["net_ret"].astype(float)
    pos = bt["position"].astype(float)

    n = len(bt)
    if n < 2:
        return {
            "CAGR": 0.0,
            "Sharpe": 0.0,
            "MaxDrawdown": 0.0,
            "HitRate": 0.0,
            "AvgExposure": float(pos.mean()) if n else 0.0,
            "AnnualTurnover": 0.0,
            "CostRate": cfg.cost_rate,
            "TargetVol": cfg.target_vol,
        }

    years = n / 252.0
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    mu = float(net.mean() * 252.0)
    sd = float(net.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(mu / (sd + 1e-12))

    peak = equity.cummax()
    dd = equity / peak - 1.0
    maxdd = float(dd.min())

    held = pos > 1e-12
    hit = float((net[held] > 0).mean()) if held.any() else 0.0

    avg_exp = float(pos.mean())
    annual_turnover = float(bt["turnover"].mean() * 252.0)

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDrawdown": maxdd,
        "HitRate": hit,
        "AvgExposure": avg_exp,
        "AnnualTurnover": annual_turnover,
        "CostRate": cfg.cost_rate,
        "TargetVol": cfg.target_vol,
    }


def buy_and_hold_backtest(df: pd.DataFrame, cfg: StrategyConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Benchmark: always long 1.0 (no costs).
    """
    out = df.copy()
    px = out[cfg.adj_close_col].astype(float)
    ret = px.pct_change().fillna(0.0)

    out["position"] = 1.0
    out["net_ret"] = ret
    out["equity"] = (1.0 + ret).cumprod()
    out["turnover"] = 0.0
    out["cost"] = 0.0

    metrics = compute_metrics(out, cfg)
    return out, metrics


# ----------------------------
# Reporting / Saving
# ----------------------------
def latest_snapshot(df_sig: pd.DataFrame, cfg: StrategyConfig) -> Dict[str, Any]:
    last = df_sig.iloc[-1]
    return {
        "as_of": str(pd.to_datetime(last[cfg.date_col]).date()),
        "close": float(last[cfg.close_col]),
        "adj_close": float(last[cfg.adj_close_col]),
        "ma": float(last[cfg.ma_col]),
        "rsi": float(last[cfg.rsi_col]),
        "macd_hist": float(last[cfg.macd_hist_col]),
        "vol": float(last[cfg.vol_col]),
        "recommended_position": float(last["position_raw"]),
        "rebalance_mode": cfg.rebalance_mode,
        "target_vol": cfg.target_vol,
        "cost_rate": cfg.cost_rate,
    }


def render_trade_note(
    ticker: str,
    snap: Dict[str, Any],
    metrics: Dict[str, float],
    bh_metrics: Dict[str, float],
) -> str:
    lines: List[str] = []
    lines.append(f"{ticker} Technical Agent Trade Note (as of {snap['as_of']})\n")
    lines.append("Assumptions")
    lines.append("- Data: daily OHLCV (10y) via yfinance")
    lines.append("- Execution: decisions computed on day t, applied from day t+1 (1-day lag)")
    lines.append(f"- Transaction cost: {snap['cost_rate']} per unit turnover (single-side)")
    lines.append(f"- Position sizing: volatility targeting (target vol {snap['target_vol']}, max weight 1.0)")
    lines.append(f"- Rebalance mode: {snap['rebalance_mode']}\n")

    lines.append("Latest signals")
    lines.append(f"- Close: {snap['close']:.2f}")
    lines.append(f"- MA(200): {snap['ma']:.2f}")
    lines.append(f"- RSI(14): {snap['rsi']:.2f}")
    lines.append(f"- MACD histogram: {snap['macd_hist']:.4f}")
    lines.append(f"- Vol (ann.): {snap['vol']:.4f}")
    lines.append(f"- Recommended position (0-1): {snap['recommended_position']:.2f}\n")

    lines.append("Backtest performance (net of costs)")
    lines.append(f"- CAGR: {metrics['CAGR']:.2%}")
    lines.append(f"- Sharpe: {metrics['Sharpe']:.2f}")
    lines.append(f"- Max Drawdown: {metrics['MaxDrawdown']:.2%}")
    lines.append(f"- Hit Rate (held days): {metrics['HitRate']:.2f}")
    lines.append(f"- Avg Exposure: {metrics['AvgExposure']:.2f}")
    lines.append(f"- Annual Turnover: {metrics['AnnualTurnover']:.2f}\n")

    lines.append("Benchmark comparison (Buy & Hold)")
    lines.append(f"- Buy&Hold CAGR: {bh_metrics['CAGR']:.2%}, Sharpe: {bh_metrics['Sharpe']:.2f}, MaxDD: {bh_metrics['MaxDrawdown']:.2%}")
    lines.append(f"- Strategy  CAGR: {metrics['CAGR']:.2%}, Sharpe: {metrics['Sharpe']:.2f}, MaxDD: {metrics['MaxDrawdown']:.2%}")
    lines.append(f"- Turnover: {metrics['AnnualTurnover']:.2f}, Avg exposure: {metrics['AvgExposure']:.2f}\n")

    lines.append("Notes / Limitations")
    lines.append("- Single-asset backtest; simplified transaction cost model; no slippage modelling.")
    lines.append("- Indicators are computed on Close; returns are based on Adj Close.")
    return "\n".join(lines).strip() + "\n"


def save_outputs(
    bt: pd.DataFrame,
    bh: pd.DataFrame,
    ticker: str,
    snap: Dict[str, Any],
    metrics: Dict[str, float],
    bh_metrics: Dict[str, float],
    outdir: Path,
    prefix: str,
    cfg: StrategyConfig,
) -> Dict[str, Any]:
    outdir.mkdir(parents=True, exist_ok=True)

    # equity curve
    eq = pd.DataFrame(
        {
            "Date": bt[cfg.date_col],
            "Close": bt[cfg.close_col],
            "Adj Close": bt[cfg.adj_close_col],
            "position": bt["position"],
            "net_ret": bt["net_ret"],
            "equity": bt["equity"],
            "bh_equity": bh["equity"],
        }
    )
    eq.to_csv(outdir / f"{prefix}_equity_curve.csv", index=False)

    # metrics
    m = dict(metrics)
    m.update({"Ticker": ticker, "Start": str(pd.to_datetime(bt[cfg.date_col].iloc[0]).date()), "End": str(pd.to_datetime(bt[cfg.date_col].iloc[-1]).date())})
    pd.DataFrame([m]).to_csv(outdir / f"{prefix}_metrics.csv", index=False)

    mb = dict(bh_metrics)
    mb.update({"Ticker": ticker, "Start": str(pd.to_datetime(bt[cfg.date_col].iloc[0]).date()), "End": str(pd.to_datetime(bt[cfg.date_col].iloc[-1]).date())})
    pd.DataFrame([mb]).to_csv(outdir / f"{prefix}_metrics_bh.csv", index=False)

    # json snapshots
    import json
    (outdir / f"{prefix}_latest_signal.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")

    run_summary = {
        "ticker": ticker,
        "start": m["Start"],
        "end": m["End"],
        "rebalance_mode": cfg.rebalance_mode,
        "metrics": metrics,
        "benchmark_metrics": bh_metrics,
        "latest_snapshot": snap,
        "files": {
            "equity_curve": str((outdir / f"{prefix}_equity_curve.csv").as_posix()),
            "metrics": str((outdir / f"{prefix}_metrics.csv").as_posix()),
            "metrics_bh": str((outdir / f"{prefix}_metrics_bh.csv").as_posix()),
            "latest_signal": str((outdir / f"{prefix}_latest_signal.json").as_posix()),
        },
    }
    (outdir / f"{prefix}_run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    # trade note
    note = render_trade_note(ticker, snap, metrics, bh_metrics)
    (outdir / f"{prefix}_trade_note.md").write_text(note, encoding="utf-8")

    # plot (optional)
    if cfg.plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(pd.to_datetime(eq["Date"]), eq["equity"], label="Strategy (net)")
        plt.plot(pd.to_datetime(eq["Date"]), eq["bh_equity"], label="Buy & Hold")
        plt.title(f"{ticker} Equity Curve: Strategy vs Buy&Hold")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"{prefix}_equity_curve.png", dpi=150)
        plt.close()

        run_summary["files"]["equity_curve_png"] = str((outdir / f"{prefix}_equity_curve.png").as_posix())

    return run_summary


def end_to_end_backtest(
    df_with_indicators: pd.DataFrame,
    cfg: Optional[StrategyConfig] = None,
    outdir: str = "outputs",
    prefix: str = "msft",
    ticker: str = "MSFT",
) -> Dict[str, Any]:
    cfg = cfg or StrategyConfig()

    df = _ensure_core_cols(df_with_indicators, cfg)
    df = _normalize_indicator_cols(df, cfg)

    df_sig = generate_signal_and_weight(df, cfg)
    bt, metrics = backtest(df_sig, cfg)

    bh, bh_metrics = buy_and_hold_backtest(df, cfg)

    snap = latest_snapshot(df_sig, cfg)
    summary = save_outputs(bt, bh, ticker, snap, metrics, bh_metrics, Path(outdir), prefix, cfg)

    return summary
