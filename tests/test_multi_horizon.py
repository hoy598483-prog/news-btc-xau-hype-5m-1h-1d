from __future__ import annotations

import unittest

import pandas as pd

from process_event_metrics import token_any_horizon_summary


class AnyHorizonBreakthroughTests(unittest.TestCase):
    def test_different_horizons_are_combined_as_event_union(self) -> None:
        rows = []
        ratios = {
            "event-1": {"5m": 4.0, "1h": 1.0, "1d": 1.0},
            "event-2": {"5m": 1.0, "1h": 5.0, "1d": 1.0},
            "event-3": {"5m": 1.0, "1h": 1.0, "1d": 1.0},
        }
        for event_id, horizons in ratios.items():
            for horizon, volume_ratio in horizons.items():
                rows.append({
                    "asset": "BTCUSDT",
                    "event_id": event_id,
                    "title": "shared token",
                    "horizon": horizon,
                    "volume_ratio": volume_ratio,
                })

        result = token_any_horizon_summary(pd.DataFrame(rows))
        shared = result[result["token"] == "shared"].iloc[0]

        self.assertEqual(shared["effective_breakthrough_count"], 2)
        self.assertAlmostEqual(shared["effective_breakthrough_frequency"], 2 / 3)
        self.assertTrue(shared["passes_any_horizon_filter"])


if __name__ == "__main__":
    unittest.main()
