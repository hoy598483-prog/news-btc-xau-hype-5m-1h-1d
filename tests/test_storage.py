from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from news_btc_research.storage import format_fourier_table, write_csv, write_fourier_table


ROWS = [{
    "token": "bitcoin",
    "event_count": 12,
    "pre_spectral_centroid_hz": 0.08,
    "post_spectral_centroid_hz": 0.14,
    "spectral_centroid_change_hz": 0.06,
    "high_frequency_cutoff_hz": 0.10,
    "pre_high_frequency_power_ratio": 0.42,
    "post_high_frequency_power_ratio": 0.7345,
    "high_frequency_power_ratio_change": 0.3145,
}]


class StorageTests(unittest.TestCase):
    def test_readable_table_excludes_dominant_frequency(self) -> None:
        table = format_fourier_table(ROWS)
        self.assertIn("bitcoin", table)
        self.assertIn("73.45%", table)
        self.assertIn("+31.45%", table)
        self.assertNotIn("主频", table)
        self.assertNotIn("0.200000", table)

    def test_fourier_csv_and_markdown_outputs(self) -> None:
        fields = [
            "token", "event_count", "pre_spectral_centroid_hz",
            "post_spectral_centroid_hz", "spectral_centroid_change_hz",
            "high_frequency_cutoff_hz", "pre_high_frequency_power_ratio",
            "post_high_frequency_power_ratio", "high_frequency_power_ratio_change",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_csv(root / "ranking.csv", ROWS, fields)
            write_fourier_table(root / "ranking.md", ROWS)
            csv_text = (root / "ranking.csv").read_text(encoding="utf-8-sig")
            md_text = (root / "ranking.md").read_text(encoding="utf-8")
        self.assertNotIn("dominant_frequency_hz", csv_text)
        self.assertNotIn("dominant_period_seconds", csv_text)
        self.assertIn("Fourier 高频 Token 排名", md_text)


if __name__ == "__main__":
    unittest.main()
