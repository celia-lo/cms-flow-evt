#!/usr/bin/env python3
"""
Overlay every numeric branch (1-D) found in two postprocessed ROOT files and save per-variable
histograms with matching names.

Example:
    python compare_two_samples.py \
        --first-path ZtoQQB_postprocess_9k_test_eta30Eval.root --first-label "Parnassus" \
        --second-path delphes_postprocess_9k_test_eta30Eval.root --second-label "Delphes" \
        --output-dir comparison_overlays/ZtoQQB_vs_Delphes
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import uproot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two postprocessed ROOT files.")
    parser.add_argument("--first-path", required=True, type=Path)
    parser.add_argument("--second-path", required=True, type=Path)
    parser.add_argument("--first-label", default="Sample A")
    parser.add_argument("--second-label", default="Sample B")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("comparison_overlays_simple")
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional number of events to read from each tree.",
    )
    parser.add_argument("--font-size", type=int, default=18)
    parser.add_argument(
        "--bins",
        type=int,
        default=60,
        help="Number of bins for automatic histograms.",
    )
    return parser.parse_args()


def _flatten(value: np.ndarray) -> np.ndarray:
    if value.dtype == np.object_:
        entries = [np.asarray(v, dtype=float).ravel() for v in value if len(v)]
        if not entries:
            return np.array([], dtype=float)
        return np.concatenate(entries)
    arr = np.asarray(value, dtype=float)
    return arr.ravel()


def load_branches(path: Path, max_events: int | None) -> Mapping[str, np.ndarray]:
    with uproot.open(path) as root_file:
        tree = root_file["evt_tree"]
        keys = list(tree.keys())
        arrays = tree.arrays(keys, entry_stop=max_events, library="np")
    flattened = {}
    for key, value in arrays.items():
        try:
            flattened[key] = _flatten(value)
        except Exception:
            # Skip branches that cannot be converted to floats (e.g., strings or nested records).
            continue
    return flattened


def plot_single(
    variable: str,
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    labels: tuple[str, str],
    out_path: Path,
    bins: int,
    font_size: int,
) -> None:
    combined = np.concatenate([sample_a, sample_b]) if sample_a.size or sample_b.size else np.array([0])
    vmin, vmax = np.min(combined), np.max(combined)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = -1, 1
    hist_bins = np.linspace(vmin, vmax, bins + 1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(sample_a, bins=hist_bins, histtype="step", linewidth=2, label=labels[0])
    ax.hist(sample_b, bins=hist_bins, histtype="step", linewidth=2, label=labels[1])
    ax.set_title(variable, fontsize=font_size)
    ax.set_ylabel("Entries", fontsize=font_size)
    ax.tick_params(labelsize=font_size - 2)
    ax.legend(fontsize=font_size - 2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sample_a = load_branches(args.first_path, args.max_events)
    sample_b = load_branches(args.second_path, args.max_events)

    common_keys = sorted(set(sample_a) & set(sample_b))
    for key in common_keys:
        vec_a = sample_a[key]
        vec_b = sample_b[key]
        if vec_a.size == 0 and vec_b.size == 0:
            continue
        safe_name = key.replace("/", "_")
        out_path = args.output_dir / f"{safe_name}.png"
        plot_single(
            key,
            vec_a,
            vec_b,
            (args.first_label, args.second_label),
            out_path,
            args.bins,
            args.font_size,
        )


if __name__ == "__main__":
    main()
