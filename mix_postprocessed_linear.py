#!/usr/bin/env python3
"""Mix postprocessed HS + MB ntuples using linear-fraction thinning.

Unlike ``mix_postprocessed_pf.py`` (which concatenates entire MB events onto
HS events), this script enforces user-specified fractions at the **particle
level**. After concatenating the selected HS event with a random set of MB
overlays, it randomly thins the HS and MB particles so that the expected share
of survivors matches ``w_hs / (w_hs + w_mb)`` and
``w_mb / (w_hs + w_mb)`` respectively. The result is a new postprocessed file
whose particle multiplicities follow the desired linear combination, avoiding
the bias toward the higher-multiplicity sample.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import awkward as ak
import numpy as np
import uproot

PARTICLE_PREFIXES = ("fastsim", "pflow")
EXCLUDE_SUFFIX_PREFIXES = ("jet", "jets", "hung_cost", "tr_idx", "pf_idx")


def load_arrays(path: Path, max_events: int | None = None) -> ak.Array:
    with uproot.open(path) as file:
        tree = file["evt_tree"]
        arrays = tree.arrays(entry_stop=max_events, library="ak")
    return arrays


def select_particle_branches(arrays: ak.Array, prefix: str) -> list[str]:
    branches: list[str] = []
    for field in arrays.fields:
        if not field.startswith(f"{prefix}_"):
            continue
        suffix = field.split("_", 1)[1]
        if any(suffix.startswith(ex) for ex in EXCLUDE_SUFFIX_PREFIXES):
            continue
        branches.append(field)
    return branches


def intersection_branches(hs_arrays: ak.Array, mb_arrays: ak.Array) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for prefix in PARTICLE_PREFIXES:
        hs_fields = set(select_particle_branches(hs_arrays, prefix))
        mb_fields = set(select_particle_branches(mb_arrays, prefix))
        common = sorted(hs_fields & mb_fields)
        if not common:
            raise RuntimeError(f"No common {prefix}_* particle branches found between inputs.")
        missing = sorted((hs_fields ^ mb_fields) - (hs_fields & mb_fields))
        if missing:
            print(
                f"[mix_postprocessed_linear] Skipping branches not shared by both files: {missing}"
            )
        result[prefix] = common
    return result


def build_mask(
    base_len: int,
    mb_lens: Sequence[int],
    target_frac_hs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    total_len = base_len + sum(mb_lens)
    if total_len == 0:
        return np.zeros(0, dtype=bool)
    actual_frac_hs = base_len / total_len if total_len else 0.0
    actual_frac_mb = 1.0 - actual_frac_hs

    keep_prob_hs = 1.0
    keep_prob_mb = 1.0
    if base_len > 0 and target_frac_hs < actual_frac_hs:
        keep_prob_hs = target_frac_hs / actual_frac_hs
    if sum(mb_lens) > 0:
        target_frac_mb = 1.0 - target_frac_hs
        if target_frac_mb < actual_frac_mb and actual_frac_mb > 0:
            keep_prob_mb = target_frac_mb / actual_frac_mb

    hs_mask = rng.random(base_len) < keep_prob_hs if base_len else np.zeros(0, dtype=bool)
    mb_masks = [rng.random(mb_len) < keep_prob_mb for mb_len in mb_lens]
    if not mb_masks:
        return hs_mask
    return np.concatenate([hs_mask, *mb_masks])


def concatenate_with_mask(
    hs_arrays: ak.Array,
    mb_arrays: ak.Array,
    field: str,
    hs_event_idx: int,
    mb_indices: list[int],
    mask: np.ndarray,
) -> ak.Array:
    segments = [np.asarray(hs_arrays[field][hs_event_idx])]
    segments.extend(np.asarray(mb_arrays[field][j]) for j in mb_indices)
    if len(segments) == 1:
        combined = segments[0]
    else:
        combined = np.concatenate(segments, axis=0)
    if combined.size != mask.size:
        raise RuntimeError(
            f"Mask length {mask.size} does not match concatenated array length {combined.size} for {field}."
        )
    if combined.size == 0:
        return ak.Array([])
    return ak.Array(combined[mask])


def mix_events_linear(
    hs_arrays: ak.Array,
    mb_arrays: ak.Array,
    ratio: float,
    target_frac_hs: float,
    rng: np.random.Generator,
    branches: dict[str, list[str]],
) -> dict[str, list[ak.Array]]:
    n_hs = len(hs_arrays["eventNumber"])
    n_mb = len(mb_arrays["eventNumber"])

    integer = int(np.floor(ratio))
    frac = ratio - integer

    mixed: dict[str, list[ak.Array]] = {field: [] for fields in branches.values() for field in fields}
    npflow: list[int] = []
    nfastsim: list[int] = []

    for i in range(n_hs):
        overlays = integer + (1 if rng.random() < frac else 0)
        mb_indices = (
            rng.integers(0, n_mb, size=overlays, endpoint=False).tolist()
            if overlays > 0
            else []
        )

        for prefix in PARTICLE_PREFIXES:
            ref_field = branches[prefix][0]
            base_len = len(np.asarray(hs_arrays[ref_field][i]))
            mb_lens = [len(np.asarray(mb_arrays[ref_field][idx])) for idx in mb_indices]
            mask = build_mask(base_len, mb_lens, target_frac_hs, rng)
            for field in branches[prefix]:
                mixed[field].append(
                    concatenate_with_mask(hs_arrays, mb_arrays, field, i, mb_indices, mask)
                )

        npflow.append(len(mixed[branches["pflow"][0]][-1]))
        nfastsim.append(len(mixed[branches["fastsim"][0]][-1]))

    mixed["npflow"] = npflow
    mixed["nfastsim"] = nfastsim
    return mixed


def assemble_output(
    hs_arrays: ak.Array,
    mixed_particles: dict[str, list[ak.Array] | list[int]],
) -> dict[str, ak.Array]:
    n_events = len(mixed_particles["npflow"])
    output: dict[str, ak.Array] = {}
    for field, values in mixed_particles.items():
        output[field] = ak.Array(values)

    truth_fields = [field for field in hs_arrays.fields if field.startswith("truth_") or field in {"eventNumber", "ntruth"}]
    for field in truth_fields:
        output[field] = hs_arrays[field][:n_events]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mix postprocessed HS and MB ntuples with linear fractions.")
    parser.add_argument("--hs", required=True, type=Path, help="Postprocessed HS ROOT file")
    parser.add_argument("--mb", required=True, type=Path, help="Postprocessed MB ROOT file")
    parser.add_argument("--output", required=True, type=Path, help="Output ROOT path")
    parser.add_argument("--w-hs", required=True, type=float, help="HS weight (e.g. 0.3452)")
    parser.add_argument("--w-mb", required=True, type=float, help="MB weight (e.g. 0.6548)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--max-events", type=int, default=None, help="Optional HS event cap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.w_hs <= 0 or args.w_mb <= 0:
        raise ValueError("Weights must be positive.")
    total_w = args.w_hs + args.w_mb
    target_frac_hs = args.w_hs / total_w
    ratio = args.w_mb / args.w_hs
    rng = np.random.default_rng(args.seed)

    hs_arrays = load_arrays(args.hs, args.max_events)
    mb_arrays = load_arrays(args.mb)
    branches = intersection_branches(hs_arrays, mb_arrays)

    mixed_particles = mix_events_linear(
        hs_arrays,
        mb_arrays,
        ratio,
        target_frac_hs,
        rng,
        branches,
    )
    output = assemble_output(hs_arrays, mixed_particles)

    with uproot.recreate(args.output) as fout:
        fout["evt_tree"] = output

    print(f"Linear mix written to {args.output}")
    print(f"HS events processed: {len(output['npflow'])}")
    print(f"Target fractions (HS, MB): ({target_frac_hs:.4f}, {1-target_frac_hs:.4f})")


if __name__ == "__main__":
    main()
