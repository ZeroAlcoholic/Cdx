# Trading Environment 規格

- 介面：`reset()`, `step(action)`
- 決策：t close
- 成交：t+1 open
- 動作：{-1,0,+1}

## state
- 市場/外生/新聞特徵
- agent scores
- regime features
- portfolio state：
  - `current_position`
  - `cash_ratio`
  - `unrealized_pnl_ratio`
  - `rolling_drawdown`
  - `recent_turnover`
  - `days_in_position`

## reward
- `daily_portfolio_return`
- `cost_penalty`
- `turnover_penalty`
- `drawdown_penalty`
- `invalid_action_penalty`
