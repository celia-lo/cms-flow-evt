#!/usr/bin/env python3
"""Pairwise mixing of postprocessed HS and MB ROOT files (1:1 concatenation).

This utility is a simplified variant of ``mix_postprocessed_pf.py``: for each HS
event it selects exactly one MB event (cycling through or sampling if lengths
are different) and concatenates all particle-level branches from both events
without any thinning. The resulting ROOT file keeps the same schema as the
inputs (fastsim_*, pflow_*, truth_*), with truth copied from the HS file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

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


def particle_fields(arrays: ak.Array, prefix: str) -> list[str]:
    fields = []
    for field in arrays.fields:
        if not field.startswith(f"{prefix}_"):
            continue
        suffix = field.split("_", 1)[1]
        if any(suffix.startswith(ex) for ex in EXCLUDE_SUFFIX_PREFIXES):
            continue
        fields.append(field)
    return fields


def intersect_fields(hs_arrays: ak.Array, mb_arrays: ak.Array) -> dict[str, list[str]]:
    result = {}
    for prefix in PARTICLE_PREFIXES:
        hs_fields = set(particle_fields(hs_arrays, prefix))
        mb_fields = set(particle_fields(mb_arrays, prefix))
        common = sorted(hs_fields & mb_fields)
        if not common:
            raise RuntimeError(f"No common {prefix}_* particle branches between inputs.")
        missing = sorted((hs_fields ^ mb_fields) - (hs_fields & mb_fields))
        if missing:
            print(
                f"[mix_postprocessed_pair] Skipping branches not shared by both files: {missing}"
            )
        result[prefix] = common
    return result


def pairwise_mix(
    hs_arrays: ak.Array,
    mb_arrays: ak.Array,
    branches: dict[str, list[str]],
) -> dict[str, list[ak.Array]]:
    n_hs = len(hs_arrays["eventNumber"])
    n_mb = len(mb_arrays["eventNumber"])
    if n_hs == 0 or n_mb == 0:
        raise ValueError("Both inputs must contain at least one event.")

    mixed: dict[str, list[ak.Array]] = {field: [] for fields in branches.values() for field in fields}
    npflow: list[int] = []
    nfastsim: list[int] = []

    for i in range(n_hs):
        j = i % n_mb
        for prefix in PARTICLE_PREFIXES:
            for field in branches[prefix]:
                hs_vals = np.asarray(hs_arrays[field][i])
                mb_vals = np.asarray(mb_arrays[field][j])
                if hs_vals.size == 0 and mb_vals.size == 0:
                    combined = ak.Array([])
                elif hs_vals.size == 0:
                    combined = ak.Array(mb_vals)
                elif mb_vals.size == 0:
                    combined = ak.Array(hs_vals)
                else:
                    combined = ak.Array(np.concatenate([hs_vals, mb_vals], axis=0))
                mixed[field].append(combined)
        npflow.append(len(mixed[branches["pflow"][0]][-1]))
        nfastsim.append(len(mixed[branches["fastsim"][0]][-1]))

    mixed["npflow"] = npflow
    mixed["nfastsim"] = nfastsim
    return mixed


def assemble_output(hs_arrays: ak.Array, mixed_particles: dict[str, list[ak.Array] | list[int]]) -> dict[str, ak.Array]:
    n_events = len(mixed_particles["npflow"])
    output: dict[str, ak.Array] = {}
    for field, values in mixed_particles.items():
        output[field] = ak.Array(values)
    truth_fields = [field for field in hs_arrays.fields if field.startswith("truth_") or field in {"eventNumber", "ntruth"}]
    for field in truth_fields:
        output[field] = hs_arrays[field][:n_events]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise mix postprocessed HS + MB ntuples (1 HS + 1 MB).")
    parser.add_argument("--hs", required=True, type=Path, help="Postprocessed HS ROOT file")
    parser.add_argument("--mb", required=True, type=Path, help="Postprocessed MB ROOT file")
    parser.add_argument("--output", required=True, type=Path, help="Output ROOT file path")
    parser.add_argument("--max-events", type=int, default=None, help="Optional number of HS events to mix")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hs_arrays = load_arrays(args.hs, args.max_events)
    mb_arrays = load_arrays(args.mb)
    branches = intersect_fields(hs_arrays, mb_arrays)

    mixed_particles = pairwise_mix(hs_arrays, mb_arrays, branches)
    output = assemble_output(hs_arrays, mixed_particles)

    with uproot.recreate(args.output) as fout:
        fout["evt_tree"] = output

    print(f"Pairwise mix written to {args.output}")
    print(f"HS events processed: {len(output['npflow'])}")


if __name__ == "__main__":
    main()
