"""市場/風險特徵建構。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_market_features(price_daily: pd.DataFrame, indicator_daily: pd.DataFrame) -> pd.DataFrame:
    """建立最小市場特徵集合，並確保僅使用歷史資料。"""
    df = price_daily.copy().sort_values("date").reset_index(drop=True)
    close = df["close"]
    ret_1d = close.pct_change(1)
    df["return_1d"] = ret_1d
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["vol_5d"] = ret_1d.rolling(5).std()
    df["vol_20d"] = ret_1d.rolling(20).std()

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    df["ma_gap_5_20"] = (ma5 - ma20) / ma20
    df["ma_gap_20_60"] = (ma20 - ma60) / ma60

    df["rsi"] = _compute_rsi(close, 14)
    macd, signal, hist = _compute_macd(close)
    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    roll_max = close.rolling(20).max()
    df["drawdown_20d"] = (close / roll_max) - 1.0
    df["volume_change"] = df["volume"].pct_change(5)

    out = df.merge(indicator_daily, on="date", how="left")
    if "vix" in out.columns:
        out["vix_level"] = out["vix"]
        out["vix_change"] = out["vix"].pct_change(1)
    if "us10y_yield" in out.columns:
        out["yield_change"] = out["us10y_yield"].diff(1)
    if "dxy" in out.columns:
        out["dxy_change"] = out["dxy"].pct_change(1)
    return out


def merge_feature_daily(market_features: pd.DataFrame, news_daily: pd.DataFrame) -> pd.DataFrame:
    """整合價格/指標/新聞聚合成一日一列特徵表。"""
    out = market_features.merge(news_daily, on="date", how="left")
    fill_zero_cols = [
        "news_count",
        "avg_sentiment_for_equity",
        "avg_inflation_signal",
        "avg_growth_signal",
        "avg_policy_signal",
        "avg_risk_on_off",
        "weighted_confidence",
    ]
    for c in fill_zero_cols:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
    return out.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)


def _compute_rsi(close: pd.Series, window: int) -> pd.Series:
    """計算 RSI 指標。"""
    delta = close.diff(1)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(window).mean()
    avg_loss = pd.Series(loss).rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """計算 MACD / Signal / Histogram。"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist
