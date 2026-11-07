from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from news_btc_research.strategy import make_example, train_strategy


def synthetic_payload(index: int, title: str, best_second: int) -> dict:
    relative = list(range(-30, 30))
    bars = []
    for second in relative:
        close = 100.0
        if second >= 5:
            close = 102.0 - abs(second - best_second) * 0.05
        high = close
        if 0 <= second < 5:
            high = 101.0 if second == 2 else 100.5
        bars.append({
            "open_time_ms": (index * 100 + second) * 1000,
            "open": close,
            "high": high,
            "low": close - 0.1,
            "close": close,
            "volume": 10.0 + max(second, 0),
            "quote_volume": (10.0 + max(second, 0)) * close,
            "trades": 5 + max(second, 0),
            "taker_buy_volume": 5.0,
        })
    return {
        "event_id": f"event-{index:03d}",
        "published_at": f"2026-01-{index + 1:02d}T00:00:00Z",
        "title": title,
        "url": f"https://example.com/{index}",
        "interval": "1s",
        "interval_seconds": 1,
        "relative_bars": relative,
        "klines": bars,
        "volume_shock": 1.0,
        "baseline_close": 100.0,
    }


class StrategyTests(unittest.TestCase):
    def test_requested_oracle_entry_and_best_exit_label(self) -> None:
        example = make_example(synthetic_payload(0, "fast policy", 12))
        self.assertEqual(example.entry_second, 2)
        self.assertEqual(example.entry_price, 101.0)
        self.assertEqual(example.best_exit_second, 12)
        self.assertAlmostEqual(example.best_exit_price, 102.0)

    def test_training_writes_auditable_outputs(self) -> None:
        payloads = [
            synthetic_payload(
                index,
                "fast policy" if index % 2 == 0 else "slow policy",
                10 if index % 2 == 0 else 20,
            )
            for index in range(28)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "btc_windows.json"
            input_path.write_text(json.dumps(payloads), encoding="utf-8")
            output = root / "strategy"
            metrics = train_strategy(
                input_path,
                output,
                ridge_alpha=1.0,
                min_token_count=2,
            )
            with (output / "strategy_training_events.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue((output / "strategy_model.npz").exists())
            self.assertTrue((output / "strategy_model_metadata.json").exists())
        self.assertEqual(metrics["counts"]["events"], 28)
        self.assertEqual(len(rows), 28)
        self.assertEqual(sum(row["split"] == "test" for row in rows), 6)
        self.assertGreater(metrics["holdout"]["mean_model_return"], 0.0)


if __name__ == "__main__":
    unittest.main()
