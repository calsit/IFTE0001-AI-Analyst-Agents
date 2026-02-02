from __future__ import annotations

from typing import Iterable
import pandas as pd


def make_rebalance_flags(index: Iterable, mode: str) -> pd.Series:
    """
    Create a boolean Series indexed by trading dates.
    True means "rebalance allowed on this date".

    Supported:
    - daily: every trading day
    - weekly: last trading day of each calendar week
    - monthly: last trading day of each calendar month
    - none: rebalance only on the first day (then hold)
    """
    idx = pd.DatetimeIndex(index)
    m = (mode or "weekly").strip().lower()

    if len(idx) == 0:
        return pd.Series(dtype=bool)

    if m == "daily":
        return pd.Series(True, index=idx)

    if m == "weekly":
        last_by_week = pd.Series(idx).groupby(idx.to_period("W")).max().values
        return pd.Series(idx.isin(last_by_week), index=idx)

    if m == "monthly":
        last_by_month = pd.Series(idx).groupby(idx.to_period("M")).max().values
        return pd.Series(idx.isin(last_by_month), index=idx)

    if m == "none":
        flags = pd.Series(False, index=idx)
        flags.iloc[0] = True
        return flags

    raise ValueError(f"Unknown rebalance mode: {mode}. Use daily|weekly|monthly|none")
