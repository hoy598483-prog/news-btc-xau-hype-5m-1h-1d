"""Point-in-time news features for multi-day BTC prediction research."""
from __future__ import annotations

import json

import pandas as pd


REQUIRED_COLUMNS = {
    "event_id", "published_at", "available_at", "source", "source_url", "title",
    "event_type", "tokens", "sentiment_score", "sentiment_model", "timezone_evidence",
    "license_or_terms", "retrieved_at",
}


def validate_news_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate research events and return them ordered by availability time."""
    missing = sorted(REQUIRED_COLUMNS - set(events.columns))
    if missing:
        raise ValueError(f"News events missing required columns: {missing}")
    result = events.copy()
    if result["event_id"].isna().any() or result["event_id"].duplicated().any():
        raise ValueError("event_id must be present and unique")
    for field in ("published_at", "available_at"):
        result[field] = pd.to_datetime(result[field], utc=True, errors="raise")
    if (result["available_at"] < result["published_at"]).any():
        raise ValueError("available_at must not precede published_at")
    if result[["source", "source_url", "timezone_evidence", "license_or_terms"]].isna().any().any():
        raise ValueError("source, source_url, timezone evidence and license are required")
    result["sentiment_score"] = pd.to_numeric(result["sentiment_score"], errors="raise")
    if (~result["sentiment_score"].between(-1, 1)).any():
        raise ValueError("sentiment_score must be in [-1, 1]")
    return result.sort_values("available_at").reset_index(drop=True)


def _has_btc_token(tokens: object) -> bool:
    try:
        values = tokens if isinstance(tokens, list) else json.loads(str(tokens))
    except json.JSONDecodeError as error:
        raise ValueError("tokens must be a JSON list of strings") from error
    return "BTC" in {str(value).upper() for value in values}


def make_prediction_features(events: pd.DataFrame, decision_times: pd.DatetimeIndex) -> pd.DataFrame:
    """Create features known at each decision time; 7d/30d returns remain labels."""
    verified = validate_news_events(events)
    times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True, errors="raise"))
    rows = []
    for decision_time in times:
        known = verified.loc[verified["available_at"] <= decision_time]
        last_day = known.loc[known["available_at"] > decision_time - pd.Timedelta(days=1)]
        last_week = known.loc[known["available_at"] > decision_time - pd.Timedelta(days=7)]
        btc_share = last_week["tokens"].map(_has_btc_token).mean() if len(last_week) else 0.0
        rows.append({"timestamp": decision_time, "news_attention_24h": float(len(last_day)),
                     "news_sentiment_24h": float(last_day["sentiment_score"].mean()) if len(last_day) else 0.0,
                     "news_event_intensity_7d": float(len(last_week)), "news_btc_token_share_7d": float(btc_share)})
    return pd.DataFrame(rows).set_index("timestamp")
