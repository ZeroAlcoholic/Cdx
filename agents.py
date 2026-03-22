"""結構化 deterministic agents。"""

from __future__ import annotations

import pandas as pd


class MarketAgent:
    """根據市場特徵輸出趨勢/動能/波動/回撤風險分數。"""

    def transform(self, feature_daily: pd.DataFrame) -> pd.DataFrame:
        out = feature_daily.copy()
        out["trend_score"] = out[["ma_gap_5_20", "ma_gap_20_60"]].mean(axis=1).clip(-1, 1)
        out["momentum_score"] = out[["return_5d", "return_20d", "macd_hist"]].mean(axis=1).clip(-1, 1)
        out["volatility_score"] = (1 - (out["vol_20d"].fillna(0) / 0.05)).clip(-1, 1)
        out["drawdown_risk_score"] = (-out["drawdown_20d"].fillna(0)).clip(0, 1)
        return out


class MacroNewsAgent:
    """將新聞聚合因子映射成總經與風險分數。"""

    def transform(self, feature_daily: pd.DataFrame) -> pd.DataFrame:
        out = feature_daily.copy()
        out["inflation_score"] = out["avg_inflation_signal"].clip(-1, 1)
        out["growth_score"] = out["avg_growth_signal"].clip(-1, 1)
        out["policy_score"] = out["avg_policy_signal"].clip(-1, 1)
        out["risk_on_off_score"] = out["avg_risk_on_off"].clip(-1, 1)
        out["news_confidence"] = out["weighted_confidence"].clip(0, 1)
        return out


class RegimeAgent:
    """輸出 regime 標籤與信心。"""

    def transform(self, feature_daily: pd.DataFrame) -> pd.DataFrame:
        out = feature_daily.copy()
        bull = (out["trend_score"] > 0) & (out["vol_20d"].fillna(0) < 0.03)
        bear = (out["trend_score"] < 0) & (out["drawdown_20d"].fillna(0) < -0.05)
        out["regime_label"] = "neutral"
        out.loc[bull, "regime_label"] = "bull"
        out.loc[bear, "regime_label"] = "bear"
        out["regime_confidence"] = (out["trend_score"].abs() * 0.6 + (1 - out["vol_20d"].fillna(0) / 0.05).clip(0, 1) * 0.4).clip(0, 1)
        return out


class RiskAgent:
    """根據風控結果輸出最終動作與旗標。"""

    def apply(self, raw_action: int, overlay_result: dict) -> dict:
        return {
            "final_action": int(overlay_result["final_action"]),
            "risk_flag": overlay_result.get("risk_flag", "none"),
            "stop_trade_flag": bool(overlay_result.get("stop_trade_flag", False)),
            "raw_action": int(raw_action),
        }
