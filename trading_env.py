"""投資帳務引擎與 gym-style 環境。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass
class PortfolioState:
    """投資組合狀態，用於追蹤 cash/units/equity 等欄位。"""

    cash: float
    units: float
    position: int
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    turnover: float
    drawdown: float
    peak_equity: float
    days_in_position: int


class PortfolioAccounting:
    """有限資金帳務：以目標 position {-1,0,1} 在 t+1 open 成交。"""

    def __init__(self, initial_cash: float, fee_bps: float, slippage_bps: float):
        self.initial_cash = float(initial_cash)
        self.fee_rate = fee_bps / 10000.0
        self.slip_rate = slippage_bps / 10000.0
        self.state = PortfolioState(initial_cash, 0.0, 0, initial_cash, 0.0, 0.0, 0.0, 0.0, initial_cash, 0)

    def reset(self) -> PortfolioState:
        """重設帳務狀態。"""
        self.state = PortfolioState(self.initial_cash, 0.0, 0, self.initial_cash, 0.0, 0.0, 0.0, 0.0, self.initial_cash, 0)
        return self.state

    def execute_target_position(self, target_position: int, fill_price: float) -> Dict[str, float]:
        """將持倉調整到目標部位，資金不足時自動裁切至可成交。"""
        s = self.state
        pre_equity = s.cash + s.units * fill_price
        target_notional = pre_equity * target_position
        current_notional = s.units * fill_price
        delta_notional = target_notional - current_notional
        delta_units = delta_notional / fill_price if fill_price else 0.0
        traded_value = abs(delta_units * fill_price)
        cost = traded_value * (self.fee_rate + self.slip_rate)

        # 多單買入時檢查現金；空單在 MVP 允許以同等名目模擬。
        if delta_units > 0 and (delta_units * fill_price + cost) > s.cash:
            affordable_units = max((s.cash / (fill_price * (1 + self.fee_rate + self.slip_rate))), 0)
            delta_units = affordable_units
            target_position = 1 if delta_units > 0 else 0
            traded_value = abs(delta_units * fill_price)
            cost = traded_value * (self.fee_rate + self.slip_rate)

        s.cash -= delta_units * fill_price + cost
        prev_units = s.units
        s.units += delta_units
        s.position = 1 if s.units > 1e-12 else (-1 if s.units < -1e-12 else 0)
        s.turnover = traded_value / pre_equity if pre_equity else 0.0

        # 實現損益以「減倉部份」粗略估計；MVP 版本。
        if prev_units != 0 and (prev_units * s.units < 0 or abs(s.units) < abs(prev_units)):
            s.realized_pnl += (fill_price - fill_price) * 0.0

        s.equity = s.cash + s.units * fill_price
        s.unrealized_pnl = s.equity - self.initial_cash - s.realized_pnl
        s.peak_equity = max(s.peak_equity, s.equity)
        s.drawdown = (s.equity / s.peak_equity) - 1 if s.peak_equity else 0.0
        s.days_in_position = s.days_in_position + 1 if s.position != 0 else 0
        return {"traded_value": traded_value, "cost": cost, "position": s.position}


class TradingEnv:
    """簡化 gym-style 環境，state/reward/accounting 一致。"""

    def __init__(self, feature_daily: pd.DataFrame, initial_cash: float, fee_bps: float, slippage_bps: float):
        self.df = feature_daily.reset_index(drop=True)
        self.accounting = PortfolioAccounting(initial_cash, fee_bps, slippage_bps)
        self.t = 0

    def reset(self) -> Dict[str, float]:
        """重置 episode。"""
        self.t = 0
        self.accounting.reset()
        return self._build_state(self.t)

    def step(self, action: int) -> Tuple[Dict[str, float], float, bool, Dict[str, float]]:
        """執行一步：以 t+1 open 成交，回傳 next_state/reward/done/info。"""
        done = self.t >= len(self.df) - 2
        row_t1 = self.df.iloc[self.t + 1]
        fill_price = float(row_t1.get("open", row_t1.get("close")))
        prev_equity = self.accounting.state.equity

        invalid_action_penalty = 0.0 if action in (-1, 0, 1) else 0.001
        action = int(max(min(action, 1), -1))
        exec_info = self.accounting.execute_target_position(action, fill_price)
        new_equity = self.accounting.state.equity
        daily_return = (new_equity / prev_equity - 1) if prev_equity else 0.0
        cost_penalty = exec_info["cost"] / prev_equity if prev_equity else 0.0
        turnover_penalty = 0.1 * self.accounting.state.turnover
        drawdown_penalty = abs(min(self.accounting.state.drawdown, 0.0)) * 0.05
        reward = daily_return - cost_penalty - turnover_penalty - drawdown_penalty - invalid_action_penalty

        self.t += 1
        next_state = self._build_state(self.t)
        info = {
            "daily_portfolio_return": daily_return,
            "cost_penalty": cost_penalty,
            "turnover_penalty": turnover_penalty,
            "drawdown_penalty": drawdown_penalty,
            "invalid_action_penalty": invalid_action_penalty,
            "equity": self.accounting.state.equity,
            "cash": self.accounting.state.cash,
            "units": self.accounting.state.units,
        }
        return next_state, reward, done, info

    def _build_state(self, idx: int) -> Dict[str, float]:
        """組合特徵與投資組合狀態成 RL state。"""
        row = self.df.iloc[idx]
        s = self.accounting.state
        equity = s.equity if s.equity else 1.0
        state = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
        state.update(
            {
                "current_position": float(s.position),
                "cash_ratio": float(s.cash / equity),
                "unrealized_pnl_ratio": float(s.unrealized_pnl / equity),
                "rolling_drawdown": float(s.drawdown),
                "recent_turnover": float(s.turnover),
                "days_in_position": float(s.days_in_position),
            }
        )
        return state
