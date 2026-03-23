# RL 訓練完整落地流程（機器學習專家版）

> 目標：讓 RL 在金融模擬環境中「完整、可訓練、可重現、可審計」。

## 1. 先決條件（不通過不得訓練）

1. **資料完整性**
   - `feature_daily` 一日一列、日期遞增、無重複主鍵。
   - `price_daily`、`indicator_daily`、`news_features_daily` 的對齊規則固定。
2. **時間一致性**
   - 決策：t 收盤；成交：t+1 開盤。
   - 收盤後新聞僅影響次日特徵。
3. **帳務一致性**
   - 恆等式：`equity = cash + units * mark_price`。
   - 交易成本（fee/slippage）必須體現在 reward 與績效（net）。
4. **動作可執行性**
   - 不可行動作需 clip/reject，且有 log 與 penalty。
5. **風控可重現**
   - raw action → final action 規則 deterministic。

## 2. 訓練前驗證（建議每次資料更新都跑）

使用 `validate_env.py` 檢查：
- 無重複日期、日期遞增
- t+1 開盤成交
- 帳務恆等式
- 成本是否被正確扣除
- drawdown 範圍是否合理
- illegal action penalty

## 3. Reward 設計建議（避免學到錯誤行為）

建議 reward 拆成可追蹤分量：
- `daily_portfolio_return`
- `cost_penalty`
- `turnover_penalty`
- `drawdown_penalty`
- `invalid_action_penalty`

實務原則：
- turnover penalty 不可過強（避免策略永遠不交易）。
- drawdown penalty 不可過弱（避免高槓桿短期衝績效）。
- reward 需與最終 KPI（net Sharpe / MDD / turnover）一致。

## 4. 演算法與訓練穩定性建議（PPO）

1. **資料切分**：time-based split（train/valid/test）不可打亂。
2. **觀測標準化**：僅用 train 統計量（避免 leakage）。
3. **隨機種子**：固定 seed 並記錄。
4. **Early stopping**：以 valid 的 net Sharpe / drawdown monitor。
5. **Checkpoint**：固定週期存檔 + best-valid 存檔。
6. **多次重跑**：不同 seed 比較均值與信賴區間，避免單次偶然。

## 5. 診斷儀表板（訓練中必看）

- 策略動作分佈（-1/0/+1 比例）
- 每日 turnover 分佈
- 成本占報酬比例
- drawdown 軌跡
- invalid action 次數
- raw/final action 差距（風控介入率）

## 6. 常見失敗模式與對策

1. **學不到交易（全 0 動作）**
   - 降低 turnover penalty 或增加探索。
2. **過度交易**
   - 提高成本/turnover 懲罰，並檢查 reward 尺度。
3. **回測好、測試差**
   - 檢查 leakage、做 walk-forward 驗證。
4. **高報酬但高回撤**
   - 提升 drawdown penalty、增加風控停交易條件。

## 7. 最小可接受訓練完成定義（DoD）

- 驗證腳本全部通過。
- PPO 可完成訓練，且有 checkpoint。
- 在 test 集可推論並輸出 `backtest_log`。
- 報告同時含 gross/net 指標與 ablation 比較。
