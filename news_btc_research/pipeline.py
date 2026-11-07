from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .analysis import build_event_market, fourier_rank, rank_tokens
from .binance import BinanceClient
from .config import Config
from .models import EventMarket, Kline, NewsEvent
from .storage import format_fourier_table, write_csv, write_fourier_table, write_json
from .web import NewsCrawler


def load_or_fetch_klines(
    event: NewsEvent,
    config: Config,
    client: BinanceClient,
    cache_dir: Path,
) -> list[Kline]:
    cache_path = cache_dir / f"{event.event_id}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            payload.get("symbol") == config.symbol
            and payload.get("interval") == config.window.interval
            and payload.get("before") == config.window.before
            and payload.get("after") == config.window.after
        ):
            return [Kline(**row) for row in payload["klines"]]
    klines = client.klines_for_event(config.symbol, event.published_at, config.window)
    write_json(cache_path, {
        "event_id": event.event_id,
        "symbol": config.symbol,
        "interval": config.window.interval,
        "before": config.window.before,
        "after": config.window.after,
        "klines": [bar.to_dict() for bar in klines],
    })
    return klines


def run_pipeline(config: Config) -> dict:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "btc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering news metadata...")
    events = NewsCrawler(config.source_url, config.crawl).crawl(config.start, config.end)
    write_csv(
        output_dir / "news_events.csv",
        [event.to_dict() for event in events],
        ["event_id", "published_at", "title", "author", "url", "timestamp_source"],
    )

    print(f"Fetching BTC windows for {len(events)} events...")
    binance = BinanceClient(config.binance)
    markets: list[EventMarket] = []
    failures: list[dict] = []
    for index, event in enumerate(events, start=1):
        try:
            bars = load_or_fetch_klines(event, config, binance, cache_dir)
            markets.append(build_event_market(event, bars, config.window))
        except Exception as error:  # Preserve the rest of a long batch.
            failures.append({
                "event_id": event.event_id,
                "url": event.url,
                "error": f"{type(error).__name__}: {error}",
            })
        if index % 50 == 0 or index == len(events):
            print(f"  {index}/{len(events)} events processed")

    event_rows = [{
        **market.event.to_dict(),
        "interval": market.interval,
        "interval_seconds": market.interval_seconds,
        "bar_count": len(market.klines),
        "baseline_close": market.baseline_close,
        "volume_shock": market.volume_shock,
    } for market in markets]
    write_csv(
        output_dir / "event_market.csv",
        event_rows,
        [
            "event_id", "published_at", "title", "author", "url", "timestamp_source",
            "interval", "interval_seconds", "bar_count", "baseline_close", "volume_shock",
        ],
    )
    write_json(output_dir / "btc_windows.json", [market.to_dict() for market in markets])
    write_json(output_dir / "failures.json", failures)

    token_scores, minimum_title_count = rank_tokens(markets, config.analysis)
    write_csv(
        output_dir / "token_ranking.csv",
        [score.to_dict() for score in token_scores],
        [
            "token", "title_count", "shock_count", "shock_frequency",
            "mean_volume_shock", "median_volume_shock",
        ],
    )

    (
        fourier_scores,
        pre_frequencies,
        post_frequencies,
        pre_spectra,
        post_spectra,
        spectrum_tokens,
    ) = fourier_rank(
        markets, token_scores, config.analysis
    )
    fourier_rows = [score.to_dict() for score in fourier_scores]
    write_csv(
        output_dir / "fourier_token_ranking.csv",
        fourier_rows,
        [
            "token", "event_count", "pre_spectral_centroid_hz",
            "post_spectral_centroid_hz", "spectral_centroid_change_hz",
            "high_frequency_cutoff_hz", "pre_high_frequency_power_ratio",
            "post_high_frequency_power_ratio", "high_frequency_power_ratio_change",
        ],
    )
    write_fourier_table(output_dir / "fourier_token_table.md", fourier_rows)
    print("\nFourier high-frequency token ranking (top 20):")
    print(format_fourier_table(fourier_rows, limit=20))
    np.savez_compressed(
        output_dir / "fourier_spectra.npz",
        tokens=np.asarray(spectrum_tokens),
        pre_frequencies_hz=pre_frequencies,
        post_frequencies_hz=post_frequencies,
        pre_mean_power=pre_spectra,
        post_mean_power=post_spectra,
    )

    result = {
        "configuration": {
            "source_url": config.source_url,
            "start": config.start.isoformat(),
            "end": config.end.isoformat(),
            "symbol": config.symbol,
            "interval": config.window.interval,
            "interval_seconds": config.window.interval_seconds,
            "before": config.window.before,
            "after": config.window.after,
            "shock_threshold": config.analysis.shock_threshold,
            "min_title_fraction": config.analysis.min_title_fraction,
            "min_shock_frequency": config.analysis.min_shock_frequency,
            "high_frequency_fraction_of_nyquist": config.analysis.high_frequency_fraction_of_nyquist,
        },
        "counts": {
            "news_events": len(events),
            "complete_market_events": len(markets),
            "failed_market_events": len(failures),
            "minimum_token_title_count": minimum_title_count,
            "retained_tokens": len(token_scores),
        },
        "token_ranking": [score.to_dict() for score in token_scores],
        "fourier_token_ranking": fourier_rows,
    }
    write_json(output_dir / "results.json", result)
    return result
