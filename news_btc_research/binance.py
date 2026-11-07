from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime

from .config import BinanceConfig, WindowConfig
from .models import Kline


class BinanceClient:
    def __init__(self, config: BinanceConfig) -> None:
        self.config = config
        self._last_request = 0.0

    def _get_json(self, path: str, params: dict[str, str | int]) -> object:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.config.request_delay_seconds:
            time.sleep(self.config.request_delay_seconds - elapsed)
        base = self.config.base_url.rstrip("/")
        url = f"{base}{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "news-btc-token-research/0.1"},
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self._last_request = time.monotonic()
        return payload

    def klines_for_event(
        self,
        symbol: str,
        published_at: datetime,
        window: WindowConfig,
    ) -> list[Kline]:
        interval_ms = window.interval_seconds * 1000
        event_ms = int(published_at.timestamp() * 1000)
        anchor_ms = event_ms - (event_ms % interval_ms)
        start_ms = anchor_ms - window.before * interval_ms
        count = window.before + window.after
        end_ms = start_ms + count * interval_ms - 1
        payload = self._get_json(
            "/api/v3/klines",
            {
                "symbol": symbol,
                "interval": window.interval,
                "startTime": start_ms,
                "endTime": end_ms,
                "limit": count,
            },
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected Binance response: {payload!r}")
        bars = [
            Kline(
                open_time_ms=int(row[0]),
                open=float(row[1]),
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                volume=float(row[5]),
                quote_volume=float(row[7]),
                trades=int(row[8]),
                taker_buy_volume=float(row[9]),
            )
            for row in payload
        ]
        expected_times = [start_ms + index * interval_ms for index in range(count)]
        actual_times = [bar.open_time_ms for bar in bars]
        if actual_times != expected_times:
            raise RuntimeError(
                f"Incomplete or misaligned Binance window at {published_at.isoformat()}: "
                f"expected {count} bars, received {len(bars)}"
            )
        return bars

