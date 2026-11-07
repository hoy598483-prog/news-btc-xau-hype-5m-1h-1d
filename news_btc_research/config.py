from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dateutil.parser import isoparse


UNIT_MAP = {
    "seconds": ("1s", 1),
    "minutes": ("1m", 60),
    "hours": ("1h", 3600),
}


@dataclass(frozen=True)
class WindowConfig:
    before: int
    after: int
    unit: str

    @property
    def interval(self) -> str:
        return UNIT_MAP[self.unit][0]

    @property
    def interval_seconds(self) -> int:
        return UNIT_MAP[self.unit][1]


@dataclass(frozen=True)
class CrawlConfig:
    max_pages: int
    request_delay_seconds: float
    timeout_seconds: float
    respect_robots: bool
    same_host_only: bool


@dataclass(frozen=True)
class AnalysisConfig:
    shock_threshold: float
    min_title_fraction: float
    min_shock_frequency: float
    high_frequency_fraction_of_nyquist: float


@dataclass(frozen=True)
class BinanceConfig:
    base_url: str
    request_delay_seconds: float
    timeout_seconds: float


@dataclass(frozen=True)
class Config:
    source_url: str
    start: datetime
    end: datetime
    symbol: str
    window: WindowConfig
    crawl: CrawlConfig
    analysis: AnalysisConfig
    binance: BinanceConfig
    output_dir: Path


def utc_datetime(value: str) -> datetime:
    parsed = isoparse(value)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp must include a timezone: {value}")
    return parsed.astimezone(timezone.utc)


def load_config(path: Path) -> Config:
    raw = json.loads(path.read_text(encoding="utf-8"))
    window = WindowConfig(**raw["window"])
    crawl = CrawlConfig(**raw["crawl"])
    analysis = AnalysisConfig(**raw["analysis"])
    binance = BinanceConfig(**raw["binance"])
    start, end = utc_datetime(raw["start"]), utc_datetime(raw["end"])

    if start >= end:
        raise ValueError("start must be earlier than end")
    if window.unit not in UNIT_MAP:
        raise ValueError(f"window.unit must be one of {sorted(UNIT_MAP)}")
    if window.before < 2 or window.after < 2:
        raise ValueError("window.before and window.after must each be at least 2")
    if window.before + window.after > 1000:
        raise ValueError("before + after must not exceed 1000 Binance klines")
    if crawl.max_pages < 1:
        raise ValueError("crawl.max_pages must be positive")
    if analysis.shock_threshold <= 0:
        raise ValueError("shock_threshold must be positive")
    for name, value in {
        "min_title_fraction": analysis.min_title_fraction,
        "min_shock_frequency": analysis.min_shock_frequency,
        "high_frequency_fraction_of_nyquist": analysis.high_frequency_fraction_of_nyquist,
    }.items():
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must be in (0, 1]")

    return Config(
        source_url=raw["source_url"],
        start=start,
        end=end,
        symbol=raw.get("symbol", "BTCUSDT").upper(),
        window=window,
        crawl=crawl,
        analysis=analysis,
        binance=binance,
        output_dir=Path(raw.get("output_dir", "results")),
    )

