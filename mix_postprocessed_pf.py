#!/usr/bin/env python3
"""Mix two postprocessed ROOT ntuples (HS + MB) across all particle branches.

The script assumes its inputs are the outputs of ``postprocess_eval.py`` so that
both ``fastsim_*`` (model prediction) and ``pflow_*`` (PF reference) branches
are present. For every HS event we randomly overlay a configurable number of MB
events and concatenate the per-particle arrays for both collections. Truth-level
information is copied from the HS sample only.

Use the resulting ROOT file as the ``--input`` to a subsequent
``postprocess_eval.py`` call (typically with ``--df`` so that the script reads
``fastsim_*`` branches from the mixed file).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


PARTICLE_PREFIXES = ("fastsim", "pflow")
EXCLUDE_SUFFIX_PREFIXES = (
    "jet",
    "jets",
    "hung_cost",
    "tr_idx",
    "pf_idx",
)


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
                f"[mix_postprocessed_pf] Skipping branches not shared by both files:"
                f" {missing}"
            )
        result[prefix] = common
    return result


def mix_events(
    hs_arrays: ak.Array,
    mb_arrays: ak.Array,
    ratio: float,
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
            rng.integers(0, n_mb, size=overlays, endpoint=False)
            if overlays > 0
            else []
        )

        for prefix in PARTICLE_PREFIXES:
            for field in branches[prefix]:
                base = hs_arrays[field][i]
                extras = [mb_arrays[field][j] for j in mb_indices]
                combined = ak.concatenate([base, *extras], axis=0) if extras else base
                mixed[field].append(combined)

        npflow.append(int(ak.num(mixed[branches["pflow"][0]][-1], axis=0)))
        nfastsim.append(int(ak.num(mixed[branches["fastsim"][0]][-1], axis=0)))

    mixed["npflow"] = npflow
    mixed["nfastsim"] = nfastsim
    return mixed


def assemble_output(
    hs_arrays: ak.Array,
    mixed_particles: dict[str, list[ak.Array] | list[int]],
    max_events: int | None,
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
    parser = argparse.ArgumentParser(description="Mix postprocessed HS and MB ntuples (particle branches).")
    parser.add_argument("--hs", required=True, type=Path, help="Postprocessed HS ROOT file")
    parser.add_argument("--mb", required=True, type=Path, help="Postprocessed MB ROOT file")
    parser.add_argument("--output", required=True, type=Path, help="Output ROOT path")
    parser.add_argument("--w-hs", required=True, type=float, help="HS weight")
    parser.add_argument("--w-mb", required=True, type=float, help="MB weight")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    parser.add_argument("--max-events", type=int, default=None, help="Optional number of HS events to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.w_hs <= 0 or args.w_mb <= 0:
        raise ValueError("Weights must be positive.")

    ratio = args.w_mb / args.w_hs
    rng = np.random.default_rng(args.seed)

    hs_arrays = load_arrays(args.hs, args.max_events)
    mb_arrays = load_arrays(args.mb)

    branches = intersection_branches(hs_arrays, mb_arrays)

    mixed_particles = mix_events(hs_arrays, mb_arrays, ratio, rng, branches)
    output = assemble_output(hs_arrays, mixed_particles, args.max_events)

    with uproot.recreate(args.output) as fout:
        fout["evt_tree"] = output

    print(f"Mixed file written to {args.output}")
    print(f"HS events processed: {len(output['npflow'])}")
    print(f"Effective MB overlays per HS event: {ratio:.3f}")


if __name__ == "__main__":
    main()
