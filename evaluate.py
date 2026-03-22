"""績效評估模組。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_backtest(backtest_log: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """計算 cumulative return、CAGR、Sharpe、MDD、turnover 等指標。"""
    eq = backtest_log["equity"].astype(float)
    ret = eq.pct_change().fillna(0.0)
    cum_return = eq.iloc[-1] / eq.iloc[0] - 1 if len(eq) > 1 else 0.0
    years = max(len(ret) / periods_per_year, 1 / periods_per_year)
    cagr = (1 + cum_return) ** (1 / years) - 1
    ann_vol = ret.std() * np.sqrt(periods_per_year)
    sharpe = (ret.mean() * periods_per_year) / ann_vol if ann_vol else 0.0
    downside = ret[ret < 0].std() * np.sqrt(periods_per_year)
    sortino = (ret.mean() * periods_per_year) / downside if downside else 0.0
    running_max = eq.cummax()
    dd = eq / running_max - 1
    mdd = dd.min() if len(dd) else 0.0
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0

    actions = backtest_log["final_action"].astype(int)
    trades = int((actions.diff().fillna(0) != 0).sum())
    turnover = float(backtest_log["turnover"].mean()) if "turnover" in backtest_log else 0.0
    win_rate = float((ret > 0).mean())
    holding = _avg_holding_period(actions)

    return {
        "cumulative_return": float(cum_return),
        "CAGR": float(cagr),
        "Sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "turnover": float(turnover),
        "trades": trades,
        "win_rate": win_rate,
        "average_holding_period": holding,
        "Sortino": float(sortino),
        "Calmar": float(calmar),
        "annualized_volatility": float(ann_vol),
    }


def _avg_holding_period(actions: pd.Series) -> float:
    """計算平均持有天數。"""
    runs = []
    current = 0
    for a in actions:
        if a != 0:
            current += 1
        elif current > 0:
            runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return float(np.mean(runs)) if runs else 0.0
