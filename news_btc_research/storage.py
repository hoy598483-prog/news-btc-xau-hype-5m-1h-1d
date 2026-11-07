from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def write_csv(path: Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def format_fourier_table(rows: Iterable[dict], limit: int | None = None) -> str:
    selected = list(rows)
    if limit is not None:
        selected = selected[:limit]
    lines = [
        "| 排名 | Token | 事件数 | 发布前高频占比 | 发布后高频占比 | 变化 | 前频谱重心 (Hz) | 后频谱重心 (Hz) | 高频起点 (Hz) |",
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(selected, start=1):
        token = str(row["token"]).replace("|", "\\|")
        lines.append(
            f"| {rank} | {token} | {int(row['event_count'])} | "
            f"{float(row['pre_high_frequency_power_ratio']):.2%} | "
            f"{float(row['post_high_frequency_power_ratio']):.2%} | "
            f"{float(row['high_frequency_power_ratio_change']):+.2%} | "
            f"{float(row['pre_spectral_centroid_hz']):.6f} | "
            f"{float(row['post_spectral_centroid_hz']):.6f} | "
            f"{float(row['high_frequency_cutoff_hz']):.6f} |"
        )
    if not selected:
        lines.append("| — | 没有 token 通过筛选 | — | — | — | — | — | — | — |")
    return "\n".join(lines) + "\n"


def write_fourier_table(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Fourier 高频 Token 排名\n\n"
        "使用1秒对数收益率，按发布后减发布前的高频功率占比变化排序；主频和主周期不输出。\n\n"
        + format_fourier_table(rows),
        encoding="utf-8",
    )
