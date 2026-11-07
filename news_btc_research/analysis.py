from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict

import numpy as np

from .config import AnalysisConfig, WindowConfig
from .models import EventMarket, FourierScore, NewsEvent, TokenScore


TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def tokenize(title: str) -> list[str]:
    return [token.replace("’", "'").lower() for token in TOKEN_RE.findall(title or "")]


def build_event_market(
    event: NewsEvent,
    klines: list,
    window: WindowConfig,
) -> EventMarket:
    if len(klines) != window.before + window.after:
        raise ValueError("Kline window length does not match configuration")
    pre = klines[: window.before]
    post = klines[window.before :]
    pre_mean = statistics.fmean(bar.volume for bar in pre)
    post_mean = statistics.fmean(bar.volume for bar in post)
    volume_shock = math.inf if pre_mean == 0 and post_mean > 0 else (
        1.0 if pre_mean == 0 and post_mean == 0 else post_mean / pre_mean
    )
    return EventMarket(
        event=event,
        interval=window.interval,
        interval_seconds=window.interval_seconds,
        relative_bars=list(range(-window.before, window.after)),
        klines=klines,
        volume_shock=volume_shock,
        baseline_close=pre[-1].close,
    )


def rank_tokens(
    markets: list[EventMarket],
    config: AnalysisConfig,
) -> tuple[list[TokenScore], int]:
    if not markets:
        return [], 0
    minimum_count = max(1, math.ceil(len(markets) * config.min_title_fraction))
    shocks_by_token: dict[str, list[float]] = defaultdict(list)
    for market in markets:
        for token in set(tokenize(market.event.title)):
            shocks_by_token[token].append(market.volume_shock)

    retained: list[TokenScore] = []
    for token, shocks in shocks_by_token.items():
        finite_shocks = [value for value in shocks if math.isfinite(value)]
        if len(finite_shocks) != len(shocks) or not finite_shocks:
            continue
        shock_count = sum(value > config.shock_threshold for value in finite_shocks)
        frequency = shock_count / len(finite_shocks)
        mean_shock = statistics.fmean(finite_shocks)
        if (
            len(finite_shocks) >= minimum_count
            and mean_shock >= config.shock_threshold
            and frequency >= config.min_shock_frequency
        ):
            retained.append(TokenScore(
                token=token,
                title_count=len(finite_shocks),
                shock_count=shock_count,
                shock_frequency=frequency,
                mean_volume_shock=mean_shock,
                median_volume_shock=statistics.median(finite_shocks),
            ))
    retained.sort(key=lambda row: (
        -row.shock_frequency,
        -row.mean_volume_shock,
        -row.title_count,
        row.token,
    ))
    return retained, minimum_count


def hann_log_returns(market: EventMarket) -> tuple[np.ndarray, np.ndarray]:
    closes = np.asarray([bar.close for bar in market.klines], dtype=np.float64)
    if np.any(closes <= 0) or not np.all(np.isfinite(closes)):
        raise ValueError(f"Invalid close price for event {market.event.event_id}")
    try:
        event_index = market.relative_bars.index(0)
    except ValueError as error:
        raise ValueError("Event window must contain relative bar zero") from error
    if event_index < 3 or closes.size - event_index < 3:
        raise ValueError("Fourier analysis needs at least three bars on each side")

    # Pre returns stay entirely before publication. Post returns include the
    # publication-bar move from the final pre-event close to the first post close.
    pre_returns = np.diff(np.log(closes[:event_index]))
    post_returns = np.diff(np.log(closes[event_index - 1 :]))
    pre_windowed = (pre_returns - pre_returns.mean()) * np.hanning(pre_returns.size)
    post_windowed = (post_returns - post_returns.mean()) * np.hanning(post_returns.size)
    return pre_windowed, post_windowed


def spectral_metrics(
    power: np.ndarray,
    frequencies: np.ndarray,
    cutoff: float,
) -> tuple[float, float]:
    usable = power.copy()
    usable[0] = 0.0
    total_power = float(usable.sum())
    if not total_power:
        return 0.0, 0.0
    centroid = float(np.sum(frequencies * usable) / total_power)
    high_ratio = float(usable[frequencies >= cutoff].sum() / total_power)
    return centroid, high_ratio


def fourier_rank(
    markets: list[EventMarket],
    tokens: list[TokenScore],
    config: AnalysisConfig,
) -> tuple[
    list[FourierScore],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
]:
    if not markets or not tokens:
        empty = np.asarray([])
        return [], empty, empty, np.empty((0, 0)), np.empty((0, 0)), []
    interval_seconds = markets[0].interval_seconds
    sample_pre, sample_post = hann_log_returns(markets[0])
    pre_frequencies = np.fft.rfftfreq(sample_pre.size, d=float(interval_seconds))
    post_frequencies = np.fft.rfftfreq(sample_post.size, d=float(interval_seconds))
    nyquist = min(float(pre_frequencies[-1]), float(post_frequencies[-1]))
    cutoff = nyquist * config.high_frequency_fraction_of_nyquist

    market_tokens = {
        market.event.event_id: set(tokenize(market.event.title)) for market in markets
    }
    pre_spectra: list[np.ndarray] = []
    post_spectra: list[np.ndarray] = []
    scores: list[FourierScore] = []
    token_order: list[str] = []
    for token_score in tokens:
        relevant = [
            market for market in markets
            if token_score.token in market_tokens[market.event.event_id]
        ]
        event_pre_power = []
        event_post_power = []
        for market in relevant:
            pre_returns, post_returns = hann_log_returns(market)
            event_pre_power.append(np.abs(np.fft.rfft(pre_returns)) ** 2)
            event_post_power.append(np.abs(np.fft.rfft(post_returns)) ** 2)
        mean_pre_power = np.mean(np.stack(event_pre_power), axis=0)
        mean_post_power = np.mean(np.stack(event_post_power), axis=0)
        pre_centroid, pre_high_ratio = spectral_metrics(
            mean_pre_power, pre_frequencies, cutoff
        )
        post_centroid, post_high_ratio = spectral_metrics(
            mean_post_power, post_frequencies, cutoff
        )
        scores.append(FourierScore(
            token=token_score.token,
            event_count=len(relevant),
            pre_spectral_centroid_hz=pre_centroid,
            post_spectral_centroid_hz=post_centroid,
            spectral_centroid_change_hz=post_centroid - pre_centroid,
            high_frequency_cutoff_hz=cutoff,
            pre_high_frequency_power_ratio=pre_high_ratio,
            post_high_frequency_power_ratio=post_high_ratio,
            high_frequency_power_ratio_change=post_high_ratio - pre_high_ratio,
        ))
        token_order.append(token_score.token)
        pre_spectra.append(mean_pre_power)
        post_spectra.append(mean_post_power)

    order = sorted(range(len(scores)), key=lambda index: (
        -scores[index].high_frequency_power_ratio_change,
        -scores[index].post_high_frequency_power_ratio,
        -scores[index].event_count,
        scores[index].token,
    ))
    sorted_scores = [scores[index] for index in order]
    sorted_pre_spectra = np.stack([pre_spectra[index] for index in order])
    sorted_post_spectra = np.stack([post_spectra[index] for index in order])
    sorted_tokens = [token_order[index] for index in order]
    return (
        sorted_scores,
        pre_frequencies,
        post_frequencies,
        sorted_pre_spectra,
        sorted_post_spectra,
        sorted_tokens,
    )
