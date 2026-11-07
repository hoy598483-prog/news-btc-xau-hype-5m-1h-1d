import unittest
import pandas as pd
from news_btc_research.availability import make_prediction_features, validate_news_events


def _events():
    return pd.DataFrame([
        {"event_id": "e1", "published_at": "2026-01-01T09:00:00Z", "available_at": "2026-01-01T10:00:00Z", "source": "example", "source_url": "https://example.test/1", "title": "BTC update", "event_type": "market", "tokens": '["BTC"]', "sentiment_score": 0.5, "sentiment_model": "manual-v1", "timezone_evidence": "RSS RFC822", "license_or_terms": "example terms", "retrieved_at": "2026-01-01T10:01:00Z"},
        {"event_id": "e2", "published_at": "2026-01-02T09:00:00Z", "available_at": "2026-01-02T09:05:00Z", "source": "example", "source_url": "https://example.test/2", "title": "Macro update", "event_type": "macro", "tokens": '[]', "sentiment_score": -0.5, "sentiment_model": "manual-v1", "timezone_evidence": "RSS RFC822", "license_or_terms": "example terms", "retrieved_at": "2026-01-02T09:06:00Z"},
    ])


class NewsFactorTests(unittest.TestCase):
    def test_only_available_events_enter_feature_table(self):
        decisions = pd.DatetimeIndex(["2026-01-01T09:30:00Z", "2026-01-02T10:00:00Z"])
        features = make_prediction_features(_events(), decisions)
        self.assertEqual(features.loc[decisions[0], "news_attention_24h"], 0.0)
        self.assertEqual(features.loc[decisions[1], "news_attention_24h"], 1.0)
        self.assertEqual(features.loc[decisions[1], "news_event_intensity_7d"], 2.0)
        self.assertEqual(features.loc[decisions[1], "news_btc_token_share_7d"], 0.5)

    def test_availability_before_publication_is_rejected(self):
        events = _events(); events.loc[0, "available_at"] = "2026-01-01T08:59:00Z"
        with self.assertRaises(ValueError): validate_news_events(events)

