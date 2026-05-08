#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _load_dataset_results(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # New structure from ocr-defense.py eval --dataset-mode.
    if isinstance(payload, dict) and isinstance(payload.get("items"), list):
        return payload["items"]
    # Backward compatible with old evaluate-dataset.py list output.
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported dataset results format. Expected {'items': [...]} or plain list.")


def _collect_engine_series(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    series: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
    for idx, item in enumerate(items):
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for engine, entry in metrics.items():
            e = series.setdefault(
                engine,
                {
                    "orig_cer": [],
                    "orig_wer": [],
                    "att_cer": [],
                    "att_wer": [],
                    "skipped": [],
                },
            )
            if entry.get("skipped"):
                e["skipped"].append((idx, 1.0))
                continue
            original = entry.get("original", {})
            attacked = entry.get("attacked", {})
            if "cer" in original:
                e["orig_cer"].append(float(original["cer"]))
            if "wer" in original:
                e["orig_wer"].append(float(original["wer"]))
            if "cer" in attacked:
                e["att_cer"].append(float(attacked["cer"]))
            if "wer" in attacked:
                e["att_wer"].append(float(attacked["wer"]))
    return series


def _plot_engine(engine: str, s: Dict[str, List[Tuple[int, float]]], out_dir: Path) -> Path:
    # print(s)
    indices = list(range(len(s["orig_cer"])))
    records = s

    
    x = np.arange(len(indices))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    
    fig, ax = plt.subplots(layout='tight')
    
    for attribute, measurement in s.items():
        if attribute == 'skipped':
            continue
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        # ax.bar_label(rects, padding=3)
        multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Metrics')
    ax.set_title(f"OCR metrics for engine: {engine}")
    ax.set_xticks(x + width, indices)
    ax.legend(loc='upper left', ncols=3)
    # ax.set_ylim(0, 250)


#     print(s)
#     fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
#     fig.suptitle(f"OCR metrics for engine: {engine}")
# 
#     def draw(ax, key_a: str, key_b: str, title: str):
#         xa = [x for x, _ in s[key_a]]
#         ya = [y for _, y in s[key_a]]
#         xb = [x for x, _ in s[key_b]]
#         yb = [y for _, y in s[key_b]]
#         if xa:
#             ax.bar(xa, ya, label=f"{key_a}") #  , marker="o", linewidth=1.5
#         if xb:
#             ax.bar(xb, yb, label=f"{key_b}") #  , marker="o", linewidth=1.5
#         for x, _ in s["skipped"]:
#             ax.axvline(x, color="gray", alpha=0.15, linewidth=1)
#         ax.set_ylabel(title)
#         # ax.set_ylim(0.0, 1.0)
#         ax.grid(True, alpha=0.3)
#         ax.legend()
# 
#     draw(axes[0], "orig_cer", "att_cer", "CER")
#     draw(axes[1], "orig_wer", "att_wer", "WER")
#     axes[1].set_xlabel("Dataset item index")

    out_path = out_dir / f"{engine}_metrics.png"
#     fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build per-engine CER/WER charts from dataset evaluation JSON.",
    )
    parser.add_argument("-i", "--input", required=True, help="Path to dataset evaluation JSON.")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="plots",
        help="Directory for generated PNG charts (default: ./plots).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = _load_dataset_results(in_path)
    series = _collect_engine_series(items)
    if not series:
        raise RuntimeError("No engine metrics found in input JSON.")

    generated: List[Path] = []
    for engine, s in sorted(series.items()):
        generated.append(_plot_engine(engine, s, out_dir))

    print(f"Built {len(generated)} charts in {out_dir}")
    for p in generated:
        print(f"- {p}")


if __name__ == "__main__":
    main()

