# 市場特徵規格

- `return_1d/5d/20d`：以 close 報酬率計算。
- `vol_5d/20d`：1d 報酬 rolling std。
- `ma_gap_5_20`, `ma_gap_20_60`：短長均線差除以長均線。
- `rsi`：14 日 RSI。
- `macd`, `macd_signal`, `macd_hist`：EMA(12,26,9)。
- `drawdown_20d`：近 20 日相對高點回撤。
- `volume_change`：volume 的 5 日變化率。
- `vix_level`, `vix_change`, `yield_change`, `dxy_change`：外生因子。

所有特徵皆使用 t 日收盤前可得資料，不得 lookahead。
