# Quant RL MVP（US Equity Index）

本專案實作 **SPY 日線**為核心的可投資、可回測、可重現模擬系統骨架，並依照 SPEC 以「先資料與模擬環境，再模型」為原則。

## 開發順序（已落地）
1. 資料與模擬環境（`data_loader.py`, `trading_env.py`, `risk_overlay.py`）
2. 特徵與 benchmark（`feature_builder.py`, `baselines.py`, `backtest.py`, `evaluate.py`）
3. 新聞因子與 agents（`crawler.py`, `news_feature_extractor.py`, `agents.py`）
4. RL 訓練骨架（`train_rl.py`）

## 快速開始

```bash
python -c "from data_loader import load_config; print(load_config('configs/default.yaml')['project']['symbol'])"
python -c "import data_loader,crawler,feature_builder,news_feature_extractor,agents,trading_env,risk_overlay,baselines,train_rl,backtest,evaluate; print('imports ok')"
```

## 模組總覽
- `data_loader.py`：價格/指標資料清理、schema 驗證
- `crawler.py`：新聞擷取流程（來源、時間切點、去重規則）
- `news_feature_extractor.py`：新聞結構化與每日聚合
- `feature_builder.py`：市場與風險特徵
- `agents.py`：結構化 agent（deterministic）
- `trading_env.py`：帳務引擎 + gym-style 環境
- `risk_overlay.py`：風控覆蓋層
- `baselines.py`：基準策略
- `backtest.py`：統一回測器
- `evaluate.py`：績效指標與比較表
- `train_rl.py`：PPO 訓練流程骨架（可替換成 Stable-Baselines3）

## 重點約束
- 決策：t 日收盤後
- 成交：t+1 日開盤
- 動作：{-1, 0, +1}
- 成本：所有績效皆 net of fee/slippage
- 嚴禁資料洩漏：特徵與新聞僅能使用決策時點前可見資訊

## 其他文件
- `data_schema.md`
- `feature_spec.md`
- `agent_io_spec.md`
- `env_spec.md`
- `backtest_rules.md`


## 視覺化架構圖
- `architecture_flow.html`：完整系統流程架構與方法論（可直接用瀏覽器開啟）。

## 訓練前驗證與方法論
- `training_protocol.md`：RL 訓練完整流程、風險與失敗模式對策。
- `validate_env.py`：訓練前環境驗證檢查（leakage/execution/accounting/cost/drawdown/invalid action）。
