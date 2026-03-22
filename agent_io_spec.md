# Agent I/O 規格

## MarketAgent
輸入：市場特徵。
輸出：`trend_score`, `momentum_score`, `volatility_score`, `drawdown_risk_score`。

## MacroNewsAgent
輸入：新聞每日聚合。
輸出：`inflation_score`, `growth_score`, `policy_score`, `risk_on_off_score`, `news_confidence`。

## RegimeAgent
輸入：市場 + 波動訊號。
輸出：`regime_label`, `regime_confidence`。

## RiskAgent
輸入：raw action + overlay 訊號。
輸出：`final_action`, `risk_flag`, `stop_trade_flag`。

所有 agent 皆為 deterministic：同輸入必得同輸出。
