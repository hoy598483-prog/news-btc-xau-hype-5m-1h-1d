"""Clean and summarize the Truth Social multi-asset event study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_event_study import BinanceClient, HORIZONS, event_metrics
from news_btc_research.analysis import tokenize


NUMERIC_COLUMNS = [
    "return", "baseline_close", "terminal_close", "pre_volatility",
    "post_volatility", "volatility_ratio", "pre_volume", "post_volume",
    "volume_ratio",
]


def finite(values: pd.Series) -> np.ndarray:
    result = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return result[np.isfinite(result)]


def one_sample_p(values: pd.Series, baseline: float = 0.0) -> float:
    data = finite(values)
    if data.size < 2 or np.allclose(data, data[0]):
        return float("nan")
    return float(stats.ttest_1samp(data, baseline, nan_policy="omit").pvalue)


def paired_p(left: pd.Series, right: pd.Series) -> float:
    frame = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(frame) < 2 or np.allclose(frame["right"] - frame["left"], 0):
        return float("nan")
    return float(stats.ttest_rel(frame["right"], frame["left"], nan_policy="omit").pvalue)


def retry_failures(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    client = BinanceClient(delay=0.1, timeout=30.0)
    failed = result.index[result["error"].fillna("").ne("")].tolist()
    for number, index in enumerate(failed, 1):
        row = result.loc[index]
        interval, seconds = HORIZONS[row["horizon"]]
        try:
            metrics = event_metrics(client, row["asset"], pd.Timestamp(row["published_at"]).to_pydatetime(), interval, seconds)
            for key, value in metrics.items():
                result.at[index, key] = value
            result.at[index, "error"] = ""
        except Exception as error:
            result.at[index, "error"] = f"{type(error).__name__}: {error}"
        if number % 25 == 0 or number == len(failed):
            print(f"retried {number}/{len(failed)} failures")
    return result


def asset_summary(valid: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (asset, horizon), group in valid.groupby(["asset", "horizon"], sort=True):
        rows.append({
            "asset": asset, "horizon": horizon, "event_count": group["event_id"].nunique(),
            "mean_return": group["return"].mean(), "median_return": group["return"].median(),
            "return_p_value": one_sample_p(group["return"]),
            "mean_pre_volatility": group["pre_volatility"].mean(),
            "mean_post_volatility": group["post_volatility"].mean(),
            "mean_volatility_change": (group["post_volatility"] - group["pre_volatility"]).mean(),
            "volatility_change_p_value": paired_p(group["pre_volatility"], group["post_volatility"]),
            "mean_volume_ratio": group["volume_ratio"].mean(),
            "median_volume_ratio": group["volume_ratio"].median(),
            "volume_shock_gt3_frequency": (group["volume_ratio"] > 3).mean(),
        })
    return pd.DataFrame(rows)


def token_summary(valid: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in valid.to_dict("records"):
        for token in set(tokenize(row["title"])):
            records.append({"asset": row["asset"], "horizon": row["horizon"], "event_id": row["event_id"], "token": token,
                            "return": row["return"], "pre_volatility": row["pre_volatility"],
                            "post_volatility": row["post_volatility"], "volume_ratio": row["volume_ratio"]})
    expanded = pd.DataFrame(records)
    output = []
    group_counts = valid.groupby(["asset", "horizon"])["event_id"].nunique().to_dict()
    for (asset, horizon, token), group in expanded.groupby(["asset", "horizon", "token"], sort=False):
        count = group["event_id"].nunique()
        minimum_count = max(3, math.ceil(group_counts[(asset, horizon)] * 0.01))
        if count < minimum_count:
            continue
        mean_volume = group["volume_ratio"].mean()
        shock_frequency = (group["volume_ratio"] > 3).mean()
        output.append({
            "asset": asset, "horizon": horizon, "token": token, "event_count": count,
            "minimum_event_count": minimum_count,
            "mean_return": group["return"].mean(), "median_return": group["return"].median(),
            "return_p_value": one_sample_p(group["return"]),
            "mean_pre_volatility": group["pre_volatility"].mean(),
            "mean_post_volatility": group["post_volatility"].mean(),
            "mean_volatility_change": (group["post_volatility"] - group["pre_volatility"]).mean(),
            "volatility_change_p_value": paired_p(group["pre_volatility"], group["post_volatility"]),
            "mean_volume_ratio": mean_volume, "median_volume_ratio": group["volume_ratio"].median(),
            "volume_shock_gt3_frequency": shock_frequency,
            "passes_volume_shock_filter": count >= minimum_count and mean_volume >= 3 and shock_frequency >= 0.60,
        })
    result = pd.DataFrame(output)
    return result.sort_values(["asset", "horizon", "passes_volume_shock_filter", "volume_shock_gt3_frequency", "mean_volume_ratio", "event_count"], ascending=[True, True, False, False, False, False])


def token_any_horizon_summary(valid: pd.DataFrame) -> pd.DataFrame:
    """Count one event as a breakthrough when any requested horizon exceeds 3x."""
    event_rows = []
    for (asset, event_id), group in valid.groupby(["asset", "event_id"], sort=False):
        by_horizon = group.set_index("horizon")
        title = str(group.iloc[0]["title"])
        ratios = {horizon: float(by_horizon.at[horizon, "volume_ratio"]) if horizon in by_horizon.index else float("nan") for horizon in HORIZONS}
        finite_ratios = [value for value in ratios.values() if math.isfinite(value)]
        event_rows.append({
            "asset": asset, "event_id": event_id, "title": title,
            "max_volume_ratio": max(finite_ratios) if finite_ratios else float("nan"),
            "effective_breakthrough": any(value > 3 for value in finite_ratios),
            **{f"breakthrough_{horizon}": bool(math.isfinite(value) and value > 3) for horizon, value in ratios.items()},
        })
    events = pd.DataFrame(event_rows)
    expanded = []
    for row in events.to_dict("records"):
        for token in set(tokenize(row["title"])):
            expanded.append({**row, "token": token})
    token_events = pd.DataFrame(expanded)
    output = []
    asset_counts = events.groupby("asset")["event_id"].nunique().to_dict()
    for (asset, token), group in token_events.groupby(["asset", "token"], sort=False):
        count = group["event_id"].nunique()
        minimum_count = max(3, math.ceil(asset_counts[asset] * 0.01))
        if count < minimum_count:
            continue
        frequency = float(group["effective_breakthrough"].mean())
        output.append({
            "asset": asset, "token": token, "event_count": count,
            "minimum_event_count": minimum_count,
            "effective_breakthrough_count": int(group["effective_breakthrough"].sum()),
            "effective_breakthrough_frequency": frequency,
            "mean_max_volume_ratio": float(group["max_volume_ratio"].mean()),
            "median_max_volume_ratio": float(group["max_volume_ratio"].median()),
            "breakthrough_5m_frequency": float(group["breakthrough_5m"].mean()),
            "breakthrough_1h_frequency": float(group["breakthrough_1h"].mean()),
            "breakthrough_1d_frequency": float(group["breakthrough_1d"].mean()),
            "passes_any_horizon_filter": frequency >= 0.60,
        })
    result = pd.DataFrame(output)
    return result.sort_values(
        ["passes_any_horizon_filter", "effective_breakthrough_frequency", "mean_max_volume_ratio", "event_count"],
        ascending=[False, False, False, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/binance_truthsocial_2026H1/event_metrics.csv")
    parser.add_argument("--output-dir", default="results/binance_truthsocial_2026H1/processed")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.input)
    frame["error"] = frame["error"].fillna("")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if args.retry_failures:
        frame = retry_failures(frame)
    frame.to_csv(output / "event_metrics_cleaned.csv", index=False, encoding="utf-8-sig")
    failures = frame[frame["error"].fillna("").ne("")].copy()
    failures.to_csv(output / "failed_rows.csv", index=False, encoding="utf-8-sig")
    valid = frame[frame["error"].fillna("").eq("")].copy()
    assets = asset_summary(valid)
    assets.to_csv(output / "asset_horizon_summary.csv", index=False, encoding="utf-8-sig")
    tokens = token_summary(valid)
    tokens.to_csv(output / "token_summary.csv", index=False, encoding="utf-8-sig")
    old_ranked = tokens[tokens["passes_volume_shock_filter"]].groupby(["asset", "horizon"], group_keys=False).head(100)
    old_ranked.to_csv(output / "token_ranking_old_rule.csv", index=False, encoding="utf-8-sig")
    any_horizon = token_any_horizon_summary(valid)
    any_horizon.to_csv(output / "token_any_horizon_summary.csv", index=False, encoding="utf-8-sig")
    ranked = any_horizon[any_horizon["passes_any_horizon_filter"]].groupby("asset", group_keys=False).head(100)
    ranked.to_csv(output / "token_ranking_top100.csv", index=False, encoding="utf-8-sig")
    summary = {"input_rows": int(len(frame)), "valid_rows": int(len(valid)), "failed_rows": int(len(failures)),
               "unique_events": int(frame["event_id"].nunique()), "token_rows": int(len(tokens)),
               "retained_token_rows": int(len(ranked)), "method": {"minimum_count": "max(3, ceil(valid events * 1%))", "volume_filter": "an event breaks through when any of 5m/1h/1d volume_ratio > 3; token passes when effective breakthrough frequency >= 60%", "p_values": "two-sided one-sample t-test for return; paired t-test for post versus pre volatility"}}
    (output / "processing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
