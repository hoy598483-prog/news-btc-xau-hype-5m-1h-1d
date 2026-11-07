# News events for multi-day prediction

This dataset supports a prediction model for BTC returns over the next **7 days** and **30 days**. It does not support high-frequency event trading.

One row represents one real news item. Required timestamps use ISO-8601 with an explicit UTC offset:

- `published_at`: the source's stated publication time.
- `available_at`: the earliest time the pipeline could have read the item. It must not be earlier than `published_at`.
- `retrieved_at`: the time this project fetched or imported the item.

Required fields include source URL, time-zone evidence, use permission, extracted token list and a versioned sentiment score. Store `tokens` as a JSON list, for example `["BTC", "ETF"]`. Keep `sentiment_score` in `[-1, 1]` and record the rule or model in `sentiment_model`.

At a daily decision time `t`, the factor table includes only rows where `available_at <= t`. It produces four prediction-layer features:

- `news_attention_24h`: number of events available in the last 24 hours.
- `news_sentiment_24h`: mean sentiment of those events.
- `news_event_intensity_7d`: number of events available in the last seven days.
- `news_btc_token_share_7d`: share of seven-day events tagged `BTC`.

Future 7-day and 30-day returns are labels, not inputs. Do not add post-publication price movements, a post-event best entry price, or a later article revision to the feature table.

The prior `news-btc-token-research` project may supply titles, URLs, publication times, extracted tokens and market windows. Its publication timestamp must be retained with time evidence. If it cannot establish when the item was observable, leave it out of the prediction dataset rather than substituting the scrape time.
