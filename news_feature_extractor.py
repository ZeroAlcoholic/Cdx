"""將新聞文字轉為結構化訊號並做每日聚合。"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict

import pandas as pd


# 簡易關鍵字規則，用於 MVP。正式版可替換成受控模型。
KEYWORDS = {
    "inflation": ["inflation", "cpi", "pce", "price pressure"],
    "growth": ["growth", "gdp", "recovery", "expansion", "slowdown"],
    "policy": ["fed", "fomc", "rate", "policy", "central bank"],
    "risk_on": ["risk-on", "rally", "bullish", "optimism"],
    "risk_off": ["risk-off", "selloff", "recession", "geopolitical"],
}


def _score_text(text: str, words: list[str]) -> float:
    t = text.lower()
    cnt = sum(1 for w in words if w in t)
    return min(cnt / max(len(words), 1), 1.0)


def extract_news_features(news_raw: pd.DataFrame) -> pd.DataFrame:
    """逐篇新聞輸出固定欄位，禁止 raw text 直接進 RL。"""
    df = news_raw.copy()
    text = (df["title"].fillna("") + " " + df["summary"].fillna(""))
    df["sentiment_for_equity"] = _score_text_series(text, KEYWORDS["risk_on"]) - _score_text_series(text, KEYWORDS["risk_off"])
    df["inflation_signal"] = _score_text_series(text, KEYWORDS["inflation"])
    df["growth_signal"] = _score_text_series(text, KEYWORDS["growth"])
    df["policy_signal"] = _score_text_series(text, KEYWORDS["policy"])
    df["risk_on_off"] = _score_text_series(text, KEYWORDS["risk_on"]) - _score_text_series(text, KEYWORDS["risk_off"])
    df["confidence"] = (df[["inflation_signal", "growth_signal", "policy_signal"]].abs().sum(axis=1) / 3.0).clip(0, 1)
    return df


def _score_text_series(text_series: pd.Series, words: list[str]) -> pd.Series:
    """批次關鍵字分數計算。"""
    return text_series.apply(lambda s: _score_text(s, words))


def aggregate_daily_news_features(news_features: pd.DataFrame, market_close_hour_utc: int = 20) -> pd.DataFrame:
    """每日聚合；收盤後新聞延後到次一交易日生效。"""
    df = news_features.copy()
    ts = pd.to_datetime(df["published_at"], utc=True)
    asof_date = ts.dt.date
    after_close = ts.dt.hour >= market_close_hour_utc
    asof_date = pd.Series(asof_date)
    asof_date[after_close] = asof_date[after_close] + timedelta(days=1)
    df["date"] = asof_date

    grouped = df.groupby("date", as_index=False).agg(
        news_count=("news_id", "count"),
        avg_sentiment_for_equity=("sentiment_for_equity", "mean"),
        avg_inflation_signal=("inflation_signal", "mean"),
        avg_growth_signal=("growth_signal", "mean"),
        avg_policy_signal=("policy_signal", "mean"),
        avg_risk_on_off=("risk_on_off", "mean"),
        weighted_confidence=("confidence", "mean"),
    )
    return grouped
