from __future__ import annotations

import math
import unittest
from datetime import datetime, timezone

import numpy as np

from news_btc_research.analysis import fourier_rank, rank_tokens, tokenize
from news_btc_research.config import AnalysisConfig
from news_btc_research.models import EventMarket, Kline, NewsEvent


ANALYSIS = AnalysisConfig(
    shock_threshold=3.0,
    min_title_fraction=0.2,
    min_shock_frequency=0.6,
    high_frequency_fraction_of_nyquist=0.2,
)


def market(identifier: int, title: str, shock: float, closes: np.ndarray) -> EventMarket:
    event = NewsEvent(
        event_id=str(identifier),
        url=f"https://example.com/{identifier}",
        title=title,
        published_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        timestamp_source="test",
    )
    bars = [
        Kline(i * 1000, value, value, value, value, 1.0, 1.0, 1, 0.5)
        for i, value in enumerate(closes)
    ]
    midpoint = len(bars) // 2
    relative_bars = list(range(-midpoint, len(bars) - midpoint))
    return EventMarket(event, "1s", 1, relative_bars, bars, shock, bars[midpoint - 1].close)


def closes_from_returns(returns: np.ndarray) -> np.ndarray:
    return 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(returns))))


class AnalysisTests(unittest.TestCase):
    def test_tokenizer_is_case_insensitive_and_unicode_aware(self) -> None:
        self.assertEqual(tokenize("BTC's Rally：政策 Update"), ["btc's", "rally", "政策", "update"])

    def test_token_filters_apply_all_three_thresholds(self) -> None:
        closes = np.full(16, 100.0)
        markets = [
            market(1, "alpha common", 4.0, closes),
            market(2, "alpha common", 5.0, closes),
            market(3, "alpha common", 2.0, closes),
            market(4, "beta common", 4.0, closes),
            market(5, "other common", 1.0, closes),
        ]
        scores, minimum_count = rank_tokens(markets, ANALYSIS)
        self.assertEqual(minimum_count, 1)
        # Tokens exactly on a threshold are retained. Rare/low-shock tokens are not.
        self.assertEqual([score.token for score in scores], ["beta", "alpha", "common"])
        alpha = next(score for score in scores if score.token == "alpha")
        self.assertAlmostEqual(alpha.shock_frequency, 2 / 3)
        self.assertAlmostEqual(alpha.mean_volume_shock, 11 / 3)

    def test_fourier_ranks_high_frequency_price_pattern_first(self) -> None:
        pre_x = np.arange(29)
        post_x = np.arange(30)
        pre_low = 0.001 * np.sin(2 * math.pi * 2 * pre_x / pre_x.size)
        post_high = 0.001 * np.sin(2 * math.pi * 10 * post_x / post_x.size)
        post_low = 0.001 * np.sin(2 * math.pi * 2 * post_x / post_x.size)
        fast = closes_from_returns(np.concatenate((pre_low, post_high)))
        slow = closes_from_returns(np.concatenate((pre_low, post_low)))
        markets = [
            market(1, "fast", 4.0, fast),
            market(2, "fast", 4.0, fast),
            market(3, "slow", 4.0, slow),
            market(4, "slow", 4.0, slow),
        ]
        scores, _ = rank_tokens(markets, ANALYSIS)
        ranked, pre_freq, post_freq, pre_spectra, post_spectra, tokens = fourier_rank(
            markets, scores, ANALYSIS
        )
        self.assertEqual(ranked[0].token, "fast")
        self.assertGreater(
            ranked[0].high_frequency_power_ratio_change,
            ranked[1].high_frequency_power_ratio_change,
        )
        self.assertEqual(pre_spectra.shape, (2, pre_freq.size))
        self.assertEqual(post_spectra.shape, (2, post_freq.size))
        self.assertEqual(tokens, [row.token for row in ranked])


if __name__ == "__main__":
    unittest.main()
