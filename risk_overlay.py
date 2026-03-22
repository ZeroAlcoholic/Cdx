"""風控覆蓋層：裁切曝險、可選停交易。"""

from __future__ import annotations


def apply_risk_overlay(raw_action: int, max_long: int = 1, max_short: int = -1, high_vol: bool = False, high_vol_scale: float = 0.5, stop_trade: bool = False) -> dict:
    """套用可重現規則，回傳 raw/final action 與風險旗標。"""
    if stop_trade:
        return {"raw_action": raw_action, "final_action": 0, "risk_flag": "stop_trade", "stop_trade_flag": True}

    action = max(min(raw_action, max_long), max_short)
    risk_flag = "none"
    if high_vol:
        action = int(round(action * high_vol_scale))
        risk_flag = "high_vol_scaled"
    if action != raw_action and risk_flag == "none":
        risk_flag = "clipped"
    return {"raw_action": raw_action, "final_action": action, "risk_flag": risk_flag, "stop_trade_flag": False}
