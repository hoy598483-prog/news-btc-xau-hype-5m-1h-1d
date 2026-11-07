from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass(frozen=True)
class NewsEvent:
    event_id: str
    url: str
    title: str
    published_at: datetime
    timestamp_source: str
    author: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["published_at"] = self.published_at.isoformat().replace("+00:00", "Z")
        return data


@dataclass(frozen=True)
class Kline:
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float
    trades: int
    taker_buy_volume: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EventMarket:
    event: NewsEvent
    interval: str
    interval_seconds: int
    relative_bars: list[int]
    klines: list[Kline]
    volume_shock: float
    baseline_close: float

    def to_dict(self) -> dict:
        return {
            **self.event.to_dict(),
            "interval": self.interval,
            "interval_seconds": self.interval_seconds,
            "relative_bars": self.relative_bars,
            "klines": [bar.to_dict() for bar in self.klines],
            "volume_shock": self.volume_shock,
            "baseline_close": self.baseline_close,
        }


@dataclass(frozen=True)
class TokenScore:
    token: str
    title_count: int
    shock_count: int
    shock_frequency: float
    mean_volume_shock: float
    median_volume_shock: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FourierScore:
    token: str
    event_count: int
    pre_spectral_centroid_hz: float
    post_spectral_centroid_hz: float
    spectral_centroid_change_hz: float
    high_frequency_cutoff_hz: float
    pre_high_frequency_power_ratio: float
    post_high_frequency_power_ratio: float
    high_frequency_power_ratio_change: float

    def to_dict(self) -> dict:
        return asdict(self)
