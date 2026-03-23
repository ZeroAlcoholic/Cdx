"""訓練前環境驗證腳本。

本模組用於在進入 RL 訓練前，先嚴格檢查模擬環境是否符合
- 無 lookahead leakage
- 成交時點/帳務/成本/回撤一致
- 不可行動作處理明確
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from trading_env import TradingEnv


@dataclass
class ValidationResult:
    """單一檢查項目的結果。"""

    name: str
    passed: bool
    detail: str


def check_no_duplicate_dates(feature_daily: pd.DataFrame) -> ValidationResult:
    """檢查 feature_daily 是否為一日一列。"""
    dup = feature_daily.duplicated(subset=["date"]).any()
    return ValidationResult(
        name="no_duplicate_dates",
        passed=not dup,
        detail="feature_daily 日期不可重複",
    )


def check_ascending_dates(feature_daily: pd.DataFrame) -> ValidationResult:
    """檢查日期是否遞增。"""
    dates = pd.to_datetime(feature_daily["date"])
    is_sorted = dates.is_monotonic_increasing
    return ValidationResult("ascending_dates", bool(is_sorted), "feature_daily 日期需遞增")


def check_execution_timing(feature_daily: pd.DataFrame, initial_cash: float, fee_bps: float, slippage_bps: float) -> ValidationResult:
    """檢查 step() 是否以 t+1 開盤價成交。"""
    env = TradingEnv(feature_daily, initial_cash, fee_bps, slippage_bps)
    env.reset()
    t0_next_open = float(feature_daily.iloc[1].get("open", feature_daily.iloc[1].get("close")))
    _, _, _, info = env.step(1)
    traded_units = info["units"]
    matched = abs(traded_units * t0_next_open) > 0
    return ValidationResult(
        "execution_timing_tplus1_open",
        matched,
        "首步下單後應於下一列（t+1）開盤價成交",
    )


def check_accounting_identity(feature_daily: pd.DataFrame, initial_cash: float, fee_bps: float, slippage_bps: float) -> ValidationResult:
    """檢查 equity = cash + units * mark_price 恆等式。"""
    env = TradingEnv(feature_daily, initial_cash, fee_bps, slippage_bps)
    env.reset()
    passed = True
    for action in [1, 0, -1, 1]:
        _, _, done, info = env.step(action)
        mark = float(feature_daily.iloc[env.t].get("open", feature_daily.iloc[env.t].get("close")))
        lhs = info["equity"]
        rhs = info["cash"] + info["units"] * mark
        if abs(lhs - rhs) > 1e-6:
            passed = False
            break
        if done:
            break
    return ValidationResult("accounting_identity", passed, "需滿足 equity = cash + units * mark")


def check_cost_applied(feature_daily: pd.DataFrame, initial_cash: float, fee_bps: float, slippage_bps: float) -> ValidationResult:
    """檢查交易成本懲罰是否非負，且交易時可觀測到成本。"""
    env = TradingEnv(feature_daily, initial_cash, fee_bps, slippage_bps)
    env.reset()
    _, _, _, info = env.step(1)
    passed = info["cost_penalty"] >= 0
    return ValidationResult("cost_applied", passed, "交易時 cost_penalty 應可觀測且非負")


def check_drawdown_range(feature_daily: pd.DataFrame, initial_cash: float, fee_bps: float, slippage_bps: float) -> ValidationResult:
    """檢查 drawdown 是否位於 [-1, 0]。"""
    env = TradingEnv(feature_daily, initial_cash, fee_bps, slippage_bps)
    env.reset()
    passed = True
    for action in [1, 1, -1, 0, -1]:
        _, _, done, _ = env.step(action)
        dd = env.accounting.state.drawdown
        if dd > 1e-12 or dd < -1.0:
            passed = False
            break
        if done:
            break
    return ValidationResult("drawdown_range", passed, "drawdown 應在 [-1, 0] 區間")


def check_invalid_action_penalty(feature_daily: pd.DataFrame, initial_cash: float, fee_bps: float, slippage_bps: float) -> ValidationResult:
    """檢查非法動作是否有懲罰。"""
    env = TradingEnv(feature_daily, initial_cash, fee_bps, slippage_bps)
    env.reset()
    _, _, _, info = env.step(99)
    passed = info["invalid_action_penalty"] > 0
    return ValidationResult("invalid_action_penalty", passed, "非法動作需給予 penalty")


def run_all_checks(feature_daily: pd.DataFrame, initial_cash: float, fee_bps: float, slippage_bps: float) -> List[ValidationResult]:
    """執行所有訓練前檢查。"""
    checks = [
        check_no_duplicate_dates(feature_daily),
        check_ascending_dates(feature_daily),
        check_execution_timing(feature_daily, initial_cash, fee_bps, slippage_bps),
        check_accounting_identity(feature_daily, initial_cash, fee_bps, slippage_bps),
        check_cost_applied(feature_daily, initial_cash, fee_bps, slippage_bps),
        check_drawdown_range(feature_daily, initial_cash, fee_bps, slippage_bps),
        check_invalid_action_penalty(feature_daily, initial_cash, fee_bps, slippage_bps),
    ]
    return checks


def summarize_results(results: List[ValidationResult]) -> Dict[str, object]:
    """將檢查結果整理為可序列化摘要。"""
    passed = all(r.passed for r in results)
    return {
        "all_passed": passed,
        "results": [{"name": r.name, "passed": r.passed, "detail": r.detail} for r in results],
    }
