"""資料載入與 schema 驗證模組。"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import yaml


# 以表名描述必要欄位與型別規則，供驗證函式使用。
TABLE_SCHEMAS: Dict[str, Dict[str, object]] = {
    "price_daily": {
        "required": ["symbol", "date", "open", "high", "low", "close", "volume"],
        "primary_key": ["symbol", "date"],
    },
    "indicator_daily": {
        "required": ["date", "vix", "us10y_yield"],
        "primary_key": ["date"],
    },
    "news_raw": {
        "required": ["news_id", "source", "title", "published_at", "fetched_at", "content_hash"],
        "primary_key": ["news_id"],
    },
    "news_features_daily": {
        "required": [
            "date",
            "news_count",
            "avg_sentiment_for_equity",
            "avg_inflation_signal",
            "avg_growth_signal",
            "avg_policy_signal",
            "avg_risk_on_off",
            "weighted_confidence",
        ],
        "primary_key": ["date"],
    },
    "feature_daily": {"required": ["date"], "primary_key": ["date"]},
    "backtest_log": {"required": ["date", "raw_action", "final_action", "equity"], "primary_key": ["date"]},
}


def load_config(path: str) -> Dict[str, object]:
    """讀取 YAML 設定檔並回傳字典。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_schema(df: pd.DataFrame, table_name: str) -> List[str]:
    """驗證資料表是否符合既定 schema，回傳錯誤清單（空代表通過）。"""
    errors: List[str] = []
    schema = TABLE_SCHEMAS[table_name]
    for col in schema["required"]:
        if col not in df.columns:
            errors.append(f"缺少欄位: {col}")
    for col in schema["primary_key"]:
        if col in df.columns and df[col].isna().any():
            errors.append(f"主鍵欄位不可為空: {col}")
    if set(schema["primary_key"]).issubset(df.columns):
        duplicated = df.duplicated(subset=schema["primary_key"], keep=False)
        if duplicated.any():
            errors.append("主鍵重複")
    return errors


def load_price_daily(csv_path: str, symbol: str = "SPY") -> pd.DataFrame:
    """讀取並清理 SPY 日線價格為標準 `price_daily`。"""
    df = pd.read_csv(csv_path)
    rename_map = {"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
    df = df.rename(columns=rename_map)
    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").drop_duplicates(subset=["symbol", "date"], keep="last")
    required = ["symbol", "date", "open", "high", "low", "close", "volume"]
    df = df.dropna(subset=required).reset_index(drop=True)
    return df[[c for c in ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]]


def align_indicator_daily(indicator_df: pd.DataFrame, trading_dates: pd.Series) -> pd.DataFrame:
    """將外生指標與交易日對齊，只允許前值補值避免未來資訊。"""
    out = indicator_df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    aligned = pd.DataFrame({"date": pd.to_datetime(trading_dates).dt.date})
    aligned = aligned.merge(out, on="date", how="left").sort_values("date")
    fill_cols = [c for c in ["vix", "us10y_yield", "dxy"] if c in aligned.columns]
    aligned[fill_cols] = aligned[fill_cols].ffill()
    return aligned
