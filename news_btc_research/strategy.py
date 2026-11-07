from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analysis import tokenize
from .storage import write_csv, write_json


CONTINUOUS_FEATURES = [
    "pre_return",
    "pre_volatility",
    "log_pre_mean_volume",
    "pre_volume_cv",
    "early_return",
    "early_volatility",
    "early_volume_ratio",
    "early_quote_volume_ratio",
    "early_trade_ratio",
    "entry_to_pre_return",
    "entry_second",
]


@dataclass
class StrategyExample:
    event_id: str
    published_at: str
    title: str
    url: str
    entry_price: float
    entry_second: int
    candidate_seconds: np.ndarray
    candidate_prices: np.ndarray
    candidate_returns: np.ndarray
    best_exit_second: int
    best_exit_price: float
    oracle_return: float
    continuous_features: np.ndarray
    tokens: set[str]


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 1.0
    return numerator / denominator


def make_example(
    payload: dict,
    entry_seconds: int = 5,
    min_exit_second: int = 5,
    max_exit_second: int = 29,
) -> StrategyExample:
    if int(payload.get("interval_seconds", 0)) != 1:
        raise ValueError("Strategy training requires 1-second klines")
    relative = [int(value) for value in payload["relative_bars"]]
    klines = payload["klines"]
    if len(relative) != len(klines):
        raise ValueError("relative_bars and klines have different lengths")
    by_second = {second: bar for second, bar in zip(relative, klines)}
    pre_seconds = [second for second in relative if second < 0]
    entry_range = list(range(0, entry_seconds))
    exit_range = list(range(min_exit_second, max_exit_second + 1))
    if len(pre_seconds) < 3:
        raise ValueError("At least three pre-event bars are required")
    missing = [second for second in entry_range + exit_range if second not in by_second]
    if missing:
        raise ValueError(f"Missing required relative seconds: {missing}")

    pre = [by_second[second] for second in pre_seconds]
    early = [by_second[second] for second in entry_range]
    exits = [by_second[second] for second in exit_range]

    # Requested experimental convention. This is an oracle entry because the
    # five-second maximum is only known after the five-second window ends.
    entry_index = int(np.argmax([float(bar["high"]) for bar in early]))
    entry_price = float(early[entry_index]["high"])
    entry_second = entry_range[entry_index]
    candidate_prices = np.asarray([float(bar["close"]) for bar in exits])
    candidate_seconds = np.asarray(exit_range, dtype=np.int64)
    candidate_returns = candidate_prices / entry_price - 1.0
    best_index = int(np.argmax(candidate_returns))

    pre_closes = np.asarray([float(bar["close"]) for bar in pre])
    pre_returns = np.diff(np.log(pre_closes))
    pre_volumes = np.asarray([float(bar["volume"]) for bar in pre])
    pre_quote = np.asarray([float(bar["quote_volume"]) for bar in pre])
    pre_trades = np.asarray([float(bar["trades"]) for bar in pre])
    early_closes = np.asarray([pre_closes[-1]] + [float(bar["close"]) for bar in early])
    early_returns = np.diff(np.log(early_closes))
    early_volumes = np.asarray([float(bar["volume"]) for bar in early])
    early_quote = np.asarray([float(bar["quote_volume"]) for bar in early])
    early_trades = np.asarray([float(bar["trades"]) for bar in early])
    pre_volume_mean = float(pre_volumes.mean())

    continuous = np.asarray([
        math.log(pre_closes[-1] / pre_closes[0]),
        float(pre_returns.std(ddof=0)),
        math.log1p(pre_volume_mean),
        float(pre_volumes.std(ddof=0) / pre_volume_mean) if pre_volume_mean else 0.0,
        math.log(early_closes[-1] / pre_closes[-1]),
        float(early_returns.std(ddof=0)),
        safe_ratio(float(early_volumes.mean()), pre_volume_mean),
        safe_ratio(float(early_quote.mean()), float(pre_quote.mean())),
        safe_ratio(float(early_trades.mean()), float(pre_trades.mean())),
        math.log(entry_price / pre_closes[-1]),
        float(entry_second),
    ], dtype=np.float64)

    return StrategyExample(
        event_id=str(payload["event_id"]),
        published_at=str(payload["published_at"]),
        title=str(payload["title"]),
        url=str(payload["url"]),
        entry_price=entry_price,
        entry_second=entry_second,
        candidate_seconds=candidate_seconds,
        candidate_prices=candidate_prices,
        candidate_returns=candidate_returns,
        best_exit_second=int(candidate_seconds[best_index]),
        best_exit_price=float(candidate_prices[best_index]),
        oracle_return=float(candidate_returns[best_index]),
        continuous_features=continuous,
        tokens=set(tokenize(str(payload["title"]))),
    )


def build_vocabulary(
    examples: list[StrategyExample],
    min_token_count: int,
    max_token_features: int,
) -> list[str]:
    counts: Counter[str] = Counter()
    for example in examples:
        counts.update(example.tokens)
    eligible = [
        (token, count) for token, count in counts.items()
        if count >= min_token_count
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [token for token, _ in eligible[:max_token_features]]


def feature_matrix(
    examples: list[StrategyExample],
    vocabulary: list[str],
) -> np.ndarray:
    token_index = {token: index for index, token in enumerate(vocabulary)}
    matrix = np.zeros(
        (len(examples), len(CONTINUOUS_FEATURES) + len(vocabulary)),
        dtype=np.float64,
    )
    for row, example in enumerate(examples):
        matrix[row, : len(CONTINUOUS_FEATURES)] = example.continuous_features
        for token in example.tokens:
            index = token_index.get(token)
            if index is not None:
                matrix[row, len(CONTINUOUS_FEATURES) + index] = 1.0
    return matrix


def standardize_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0, ddof=0)
    scales[scales < 1e-12] = 1.0
    return (matrix - means) / scales, means, scales


def standardize_apply(matrix: np.ndarray, means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (matrix - means) / scales


def fit_multioutput_ridge(matrix: np.ndarray, targets: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack((np.ones(matrix.shape[0]), matrix))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(design.T @ design + penalty) @ design.T @ targets


def predict_return_curves(matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(matrix.shape[0]), matrix))
    return design @ coefficients


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def train_strategy(
    input_path: Path,
    output_dir: Path,
    entry_seconds: int = 5,
    min_exit_second: int = 5,
    max_exit_second: int = 29,
    test_fraction: float = 0.20,
    ridge_alpha: float = 10.0,
    min_token_count: int = 2,
    max_token_features: int = 200,
) -> dict:
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be in (0, 0.5)")
    if entry_seconds < 1 or min_exit_second < entry_seconds:
        raise ValueError("min_exit_second must be at least entry_seconds")
    if max_exit_second < min_exit_second:
        raise ValueError("max_exit_second must be >= min_exit_second")

    payloads = json.loads(input_path.read_text(encoding="utf-8"))
    examples = sorted(
        [
            make_example(item, entry_seconds, min_exit_second, max_exit_second)
            for item in payloads
        ],
        key=lambda item: (item.published_at, item.event_id),
    )
    if len(examples) < 10:
        raise ValueError("At least 10 complete events are required")
    candidate_seconds = examples[0].candidate_seconds
    if any(not np.array_equal(item.candidate_seconds, candidate_seconds) for item in examples):
        raise ValueError("All events must have the same candidate exit seconds")

    test_count = max(1, math.ceil(len(examples) * test_fraction))
    train_examples = examples[:-test_count]
    test_examples = examples[-test_count:]
    vocabulary = build_vocabulary(train_examples, min_token_count, max_token_features)
    train_raw = feature_matrix(train_examples, vocabulary)
    test_raw = feature_matrix(test_examples, vocabulary)
    train_x, means, scales = standardize_fit(train_raw)
    test_x = standardize_apply(test_raw, means, scales)
    train_y = np.stack([item.candidate_returns for item in train_examples])
    coefficients = fit_multioutput_ridge(train_x, train_y, ridge_alpha)
    test_curves = predict_return_curves(test_x, coefficients)
    predicted_indices = np.argmax(test_curves, axis=1)

    predicted_returns: list[float] = []
    oracle_returns: list[float] = []
    exit_errors: list[float] = []
    prediction_by_id: dict[str, dict] = {}
    for example, predicted_index in zip(test_examples, predicted_indices):
        predicted_index = int(predicted_index)
        predicted_second = int(candidate_seconds[predicted_index])
        predicted_return = float(example.candidate_returns[predicted_index])
        predicted_price = float(example.candidate_prices[predicted_index])
        predicted_returns.append(predicted_return)
        oracle_returns.append(example.oracle_return)
        exit_errors.append(abs(predicted_second - example.best_exit_second))
        prediction_by_id[example.event_id] = {
            "predicted_exit_second": predicted_second,
            "predicted_exit_price": predicted_price,
            "predicted_return": predicted_return,
        }

    baseline_seconds = [
        second for second in (5, 10, 15, 20, max_exit_second)
        if second in set(candidate_seconds.tolist())
    ]
    baseline_returns = {}
    for second in baseline_seconds:
        index = int(np.where(candidate_seconds == second)[0][0])
        baseline_returns[str(second)] = float(np.mean([
            example.candidate_returns[index] for example in test_examples
        ]))

    rows = []
    test_ids = {item.event_id for item in test_examples}
    for example in examples:
        prediction = prediction_by_id.get(example.event_id, {})
        rows.append({
            "event_id": example.event_id,
            "published_at": example.published_at,
            "title": example.title,
            "url": example.url,
            "split": "test" if example.event_id in test_ids else "train",
            "entry_rule": f"max_high_seconds_0_to_{entry_seconds - 1}",
            "entry_second": example.entry_second,
            "entry_price": example.entry_price,
            "best_exit_second": example.best_exit_second,
            "best_exit_price": example.best_exit_price,
            "oracle_return": example.oracle_return,
            "predicted_exit_second": prediction.get("predicted_exit_second", ""),
            "predicted_exit_price": prediction.get("predicted_exit_price", ""),
            "predicted_return": prediction.get("predicted_return", ""),
        })

    metrics = {
        "warning": (
            "The max-high entry in the first five seconds is an oracle rule and "
            "cannot be executed without look-ahead. Results are research-only."
        ),
        "assumptions": {
            "fees": 0.0,
            "slippage": 0.0,
            "entry_rule": f"maximum high over relative seconds 0..{entry_seconds - 1}",
            "candidate_exit_seconds": candidate_seconds.tolist(),
            "split": "chronological holdout; last events are test data",
        },
        "counts": {
            "events": len(examples),
            "train_events": len(train_examples),
            "test_events": len(test_examples),
            "continuous_features": len(CONTINUOUS_FEATURES),
            "title_token_features": len(vocabulary),
        },
        "holdout": {
            "mean_model_return": mean_or_none(predicted_returns),
            "median_model_return": float(np.median(predicted_returns)),
            "model_win_rate": float(np.mean(np.asarray(predicted_returns) > 0)),
            "mean_oracle_return": mean_or_none(oracle_returns),
            "mean_exit_second_absolute_error": mean_or_none(exit_errors),
            "fixed_exit_mean_returns": baseline_returns,
        },
        "training": {
            "ridge_alpha": ridge_alpha,
            "min_token_count": min_token_count,
            "max_token_features": max_token_features,
            "vocabulary": vocabulary,
        },
    }

    # Refit the deployable research model on every known event after measuring
    # holdout performance with the earlier-only model.
    final_vocabulary = build_vocabulary(examples, min_token_count, max_token_features)
    final_raw = feature_matrix(examples, final_vocabulary)
    final_x, final_means, final_scales = standardize_fit(final_raw)
    final_y = np.stack([item.candidate_returns for item in examples])
    final_coefficients = fit_multioutput_ridge(final_x, final_y, ridge_alpha)
    feature_names = CONTINUOUS_FEATURES + [f"token:{token}" for token in final_vocabulary]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "strategy_training_events.csv",
        rows,
        [
            "event_id", "published_at", "title", "url", "split", "entry_rule",
            "entry_second", "entry_price", "best_exit_second", "best_exit_price",
            "oracle_return", "predicted_exit_second", "predicted_exit_price",
            "predicted_return",
        ],
    )
    write_json(output_dir / "strategy_metrics.json", metrics)
    write_json(output_dir / "strategy_model_metadata.json", {
        "model_type": "multi-output ridge regression",
        "feature_names": feature_names,
        "candidate_exit_seconds": candidate_seconds.tolist(),
        "ridge_alpha": ridge_alpha,
        "entry_seconds": entry_seconds,
        "min_exit_second": min_exit_second,
        "max_exit_second": max_exit_second,
        "training_events": len(examples),
    })
    np.savez_compressed(
        output_dir / "strategy_model.npz",
        coefficients=final_coefficients,
        feature_means=final_means,
        feature_scales=final_scales,
        feature_names=np.asarray(feature_names),
        candidate_exit_seconds=candidate_seconds,
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a news-event BTC exit-time model from btc_windows.json"
    )
    parser.add_argument("--input", type=Path, default=Path("results/btc_windows.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/strategy"))
    parser.add_argument("--entry-seconds", type=int, default=5)
    parser.add_argument("--min-exit-second", type=int, default=5)
    parser.add_argument("--max-exit-second", type=int, default=29)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--min-token-count", type=int, default=2)
    parser.add_argument("--max-token-features", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = train_strategy(
        input_path=args.input,
        output_dir=args.output_dir,
        entry_seconds=args.entry_seconds,
        min_exit_second=args.min_exit_second,
        max_exit_second=args.max_exit_second,
        test_fraction=args.test_fraction,
        ridge_alpha=args.ridge_alpha,
        min_token_count=args.min_token_count,
        max_token_features=args.max_token_features,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
