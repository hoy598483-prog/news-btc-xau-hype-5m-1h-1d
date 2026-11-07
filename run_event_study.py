"""Truth Social -> Binance multi-horizon event study.

Uses 1m/5m/1h candles for 5-minute/1-hour/1-day horizons.  It deliberately
does not infer sub-minute prices from candle data.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from news_btc_research.config import CrawlConfig, utc_datetime
from news_btc_research.web import NewsCrawler


HORIZONS = {
    "5m": ("1m", 5 * 60),
    "1h": ("5m", 60 * 60),
    "1d": ("1h", 24 * 60 * 60),
}
ASSETS = {"BTCUSDT": ("spot", "/api/v3/klines"), "HYPEUSDT": ("futures", "/fapi/v1/klines"), "XAUUSDT": ("futures", "/fapi/v1/klines")}


class BinanceClient:
    def __init__(self, delay: float = 0.1, timeout: float = 30.0) -> None:
        self.delay = delay
        self.timeout = timeout
        self.last_request = 0.0

    def candles(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
        elapsed = time.monotonic() - self.last_request
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        venue, path = ASSETS[symbol]
        base = "https://data-api.binance.vision" if venue == "spot" else "https://fapi.binance.com"
        query = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "startTime": start_ms, "endTime": end_ms, "limit": 1000})
        request = urllib.request.Request(
            f"{base}{path}?{query}", headers={"User-Agent": "news-btc-xau-hype-multihorizon/0.1"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        self.last_request = time.monotonic()
        if not isinstance(result, list):
            raise RuntimeError(f"Unexpected Binance response for {symbol}: {result!r}")
        return result


def floor_epoch(timestamp: datetime, seconds: int) -> int:
    return int(timestamp.timestamp()) // seconds * seconds


def event_metrics(client: BinanceClient, symbol: str, published: datetime, interval: str, horizon_s: int) -> dict:
    interval_s = {"1m": 60, "5m": 300, "1h": 3600}[interval]
    anchor = floor_epoch(published, interval_s)
    # Use equal-length pre and post windows for volatility comparison.
    start_s = anchor - horizon_s
    end_s = anchor + horizon_s + interval_s * 2
    rows = client.candles(symbol, interval, start_s * 1000, end_s * 1000)
    bars = sorted((int(row[0] // 1000), float(row[4]), float(row[5])) for row in rows)
    pre = [bar for bar in bars if bar[0] < anchor]
    post = [bar for bar in bars if bar[0] >= anchor and bar[0] < anchor + horizon_s]
    if not pre or not post:
        raise ValueError(f"Incomplete {symbol} {interval} window at {published.isoformat()}")
    baseline = pre[-1][1]
    terminal = post[-1][1]
    pre_closes = [bar[1] for bar in pre]
    post_closes = [bar[1] for bar in post]
    pre_returns = [math.log(b / a) for a, b in zip(pre_closes, pre_closes[1:]) if a > 0 and b > 0]
    post_returns = [math.log(b / a) for a, b in zip(post_closes, post_closes[1:]) if a > 0 and b > 0]
    pre_vol = math.sqrt(sum((x - sum(pre_returns) / len(pre_returns)) ** 2 for x in pre_returns) / max(1, len(pre_returns) - 1)) if len(pre_returns) > 1 else 0.0
    post_vol = math.sqrt(sum((x - sum(post_returns) / len(post_returns)) ** 2 for x in post_returns) / max(1, len(post_returns) - 1)) if len(post_returns) > 1 else 0.0
    pre_volume = sum(bar[2] for bar in pre)
    post_volume = sum(bar[2] for bar in post)
    return {
        "return": terminal / baseline - 1.0,
        "baseline_close": baseline,
        "terminal_close": terminal,
        "pre_volatility": pre_vol,
        "post_volatility": post_vol,
        "volatility_ratio": post_vol / pre_vol if pre_vol else None,
        "pre_volume": pre_volume,
        "post_volume": post_volume,
        "volume_ratio": post_volume / pre_volume if pre_volume else None,
        "bar_interval": interval,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-01T00:00:00Z")
    parser.add_argument("--output", default="results/binance_truthsocial_2026H1")
    parser.add_argument("--assets", nargs="+", default=list(ASSETS))
    args = parser.parse_args()
    start, end = utc_datetime(args.start), utc_datetime(args.end)
    crawl = CrawlConfig(220, 0.25, 30.0, True, True)
    events = NewsCrawler("https://www.trumpstruth.org/feed", crawl).crawl(start, end)
    client = BinanceClient()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index, event in enumerate(events, 1):
        for asset in args.assets:
            for horizon, (interval, seconds) in HORIZONS.items():
                try:
                    metrics = event_metrics(client, asset, event.published_at, interval, seconds)
                    rows.append({"event_id": event.event_id, "published_at": event.published_at.isoformat(), "author": event.author, "title": event.title, "source_url": event.url, "asset": asset, "horizon": horizon, **metrics})
                except Exception as error:
                    rows.append({"event_id": event.event_id, "published_at": event.published_at.isoformat(), "author": event.author, "title": event.title, "source_url": event.url, "asset": asset, "horizon": horizon, "error": f"{type(error).__name__}: {error}"})
        if index % 25 == 0 or index == len(events):
            print(f"{index}/{len(events)} events")
    fields = ["event_id", "published_at", "author", "title", "source_url", "asset", "horizon", "bar_interval", "return", "baseline_close", "terminal_close", "pre_volatility", "post_volatility", "volatility_ratio", "pre_volume", "post_volume", "volume_ratio", "error"]
    with (output / "event_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output / "manifest.json").write_text(json.dumps({"source": "Trump's Truth RSS", "market_data": "Binance Spot BTCUSDT + USDⓈ-M Futures HYPEUSDT/XAUUSDT", "start": args.start, "end": args.end, "assets": args.assets, "horizons": HORIZONS, "events": len(events), "note": "Binance candle API; no sub-minute inference"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
