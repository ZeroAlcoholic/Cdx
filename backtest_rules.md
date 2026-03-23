# 回測規則

1. 所有策略使用相同 execution、fee、slippage。
2. 下單在 t close 決策，t+1 open 成交。
3. 先套用 risk overlay，再成交。
4. 成交失敗（資金不足）需 clip/reject 並記錄。
5. 報酬需同時輸出 gross/net。
6. `backtest_log` 必須可重算核心績效。
