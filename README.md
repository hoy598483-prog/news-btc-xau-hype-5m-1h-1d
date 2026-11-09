# news_btc/XAU/Hype_5m/1h/1d

Truth Social 文字新闻对 BTC、XAU 和 HYPE 的多时间尺度事件研究项目。

项目只保留 Donald Trump 原创的纯文字帖子，使用 RSS `pubDate` 作为发布时间，不下载正文附件、图片、视频、引用帖或转帖。市场数据来自 Binance：

- BTCUSDT：Spot
- HYPEUSDT、XAUUSDT：USDⓈ-M Futures
- 5 分钟影响：1 分钟 Kline
- 1 小时影响：5 分钟 Kline
- 1 天影响：1 小时 Kline

## 有效突破规则

对同一资产和同一新闻事件：

```text
effective = volume_ratio_5m > 3
         OR volume_ratio_1h > 3
         OR volume_ratio_1d > 3
```

一个标题 token 被保留需要同时满足：

1. 出现次数不低于 `max(3, ceil(有效新闻数 × 1%))`；
2. 至少 60% 的相关事件在任意一个时间阶段出现有效突破。

每个 token 在同一标题中只计一次。旧的“每个时间阶段单独判定”结果仍会输出，便于比较。

## 安装

```powershell
python -m pip install -e .
```

## 运行完整事件研究

下面的时间范围采用左闭右开区间，即包含 2026-01-01，不包含 2026-08-01：

```powershell
python run_event_study.py `
  --start 2026-01-01T00:00:00Z `
  --end 2026-08-01T00:00:00Z `
  --output results/binance_truthsocial_2026H1
```

然后清洗失败请求、计算统计量并筛选 token：

```powershell
python process_event_metrics.py `
  --input results/binance_truthsocial_2026H1/event_metrics.csv `
  --output-dir results/binance_truthsocial_2026H1/processed `
  --retry-failures
```

主要输出：

- `event_metrics.csv`：新闻 × 资产 × 时间阶段的价格、波动率和成交量指标；
- `asset_horizon_summary.csv`：资产与时间阶段汇总；
- `token_any_horizon_summary.csv`：所有达到最低次数的 token；
- `token_ranking_top100.csv`：通过任一阶段突破规则的 token；
- `token_ranking_old_rule.csv`：旧规则结果，仅作对照；
- `processing_summary.json`：运行与筛选摘要。

## 已验证样本

`examples/verified_run/` 保存 2026-01-01 至 2026-07-31 的小型汇总结果，不包含原始帖子全文或完整行情缓存。该次运行包括 2,110 个事件、18,990 个有效的资产/时间阶段观测，任一阶段突破规则保留 4 个 XAUUSDT token。

`artifacts/truthsocial_binance_2026H1/` 保存同一次已验证运行的可复核成品：

- `event_metrics_cleaned.csv`：18,990 行完整清洗事件指标；
- `asset_horizon_summary.csv`：资产与时间阶段统计；
- `token_any_horizon_summary.csv`：达到最低出现次数的 token；
- `token_ranking_top100.csv`：通过任一阶段突破规则的 token；
- `processing_summary.json`：处理摘要；
- `news_event_analysis.xlsx`：可直接阅读的 Excel 工作簿。

## 测试

```powershell
python -m unittest discover -s tests -v
```

## 限制

- RSS 发布时间是平台声明时间，不等于第三方系统真正首次观测到帖子的时间。
- 事件研究结果不代表可交易收益；手续费、滑点、延迟和流动性尚未纳入。
- 同一新闻在多个时间阶段突破时只算一个有效事件，不能重复累计。
- token 筛选存在多重检验和样本选择偏差，必须使用独立时间段验证。
