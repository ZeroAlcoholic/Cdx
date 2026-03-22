# 資料 Schema（繁中）

## 1) `price_daily`
- 主鍵：`symbol`, `date`
- 欄位：
  - `symbol` (str, not null)
  - `date` (date, not null)
  - `open/high/low/close` (float, not null)
  - `adj_close` (float, 可空)
  - `volume` (float, not null)
- 規則：日期遞增、不可重複、OHLCV 不可缺。

## 2) `indicator_daily`
- 主鍵：`date`
- 欄位：`vix`, `us10y_yield`, `dxy`（可選）
- 規則：與交易日對齊；僅允許以前值補值（ffill）。

## 3) `news_raw`
- 主鍵：`news_id`
- 欄位：`source`, `title`, `summary`, `url`, `published_at`, `fetched_at`, `content_hash`
- 規則：以 `content_hash` + `published_at` 去重，保留原始資料。

## 4) `news_features_daily`
- 主鍵：`date`
- 欄位：`news_count`, `avg_sentiment_for_equity`, `avg_inflation_signal`, `avg_growth_signal`, `avg_policy_signal`, `avg_risk_on_off`, `weighted_confidence`
- 規則：收盤後新聞僅影響次日。

## 5) `feature_daily`
- 主鍵：`date`
- 欄位：市場特徵 + 指標 + 新聞聚合 + agent 輸出
- 規則：一日一列、無重複日期、缺值須有明確策略。

## 6) `backtest_log`
- 主鍵：`date`
- 欄位：`raw_action`, `final_action`, `fill_price`, `cash`, `units`, `equity`, `turnover`, `drawdown`, `reward`
- 規則：可由 log 重算績效。
