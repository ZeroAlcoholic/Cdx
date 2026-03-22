"""基準策略集合。"""

from __future__ import annotations

import pandas as pd


def buy_and_hold_policy(_: pd.Series) -> int:
    """買入持有：永遠維持多單。"""
    return 1


def ma_trend_policy(row: pd.Series) -> int:
    """均線趨勢：ma_gap_5_20 > 0 做多，否則空手。"""
    return 1 if row.get("ma_gap_5_20", 0) > 0 else 0


def price_rule_policy(row: pd.Series) -> int:
    """僅價格規則：動能正且回撤不大則做多，反向則做空，否則空手。"""
    if row.get("return_20d", 0) > 0 and row.get("drawdown_20d", 0) > -0.08:
        return 1
    if row.get("return_20d", 0) < 0 and row.get("vol_20d", 0) > 0.02:
        return -1
    return 0
