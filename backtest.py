"""統一回測器：benchmark / RL policy 共用成交與成本假設。"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from risk_overlay import apply_risk_overlay
from trading_env import TradingEnv


def run_backtest(feature_daily: pd.DataFrame, policy_fn: Callable[[pd.Series], int], initial_cash: float, fee_bps: float, slippage_bps: float, overlay_cfg: dict | None = None) -> pd.DataFrame:
    """執行回測並輸出 backtest_log。"""
    cfg = overlay_cfg or {}
    env = TradingEnv(feature_daily=feature_daily, initial_cash=initial_cash, fee_bps=fee_bps, slippage_bps=slippage_bps)
    env.reset()
    logs = []
    done = False
    t = 0
    while not done:
        row = feature_daily.iloc[t]
        raw_action = int(policy_fn(row))
        overlay = apply_risk_overlay(raw_action=raw_action, **cfg)
        _, reward, done, info = env.step(overlay["final_action"])
        logs.append(
            {
                "date": feature_daily.iloc[t + 1]["date"],
                "raw_action": raw_action,
                "final_action": overlay["final_action"],
                "risk_flag": overlay["risk_flag"],
                "stop_trade_flag": overlay["stop_trade_flag"],
                "fill_price": feature_daily.iloc[t + 1].get("open", feature_daily.iloc[t + 1].get("close")),
                "cash": info["cash"],
                "units": info["units"],
                "equity": info["equity"],
                "turnover": env.accounting.state.turnover,
                "drawdown": env.accounting.state.drawdown,
                "reward": reward,
            }
        )
        t += 1
    return pd.DataFrame(logs)
