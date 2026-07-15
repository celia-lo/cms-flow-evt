#!/usr/bin/env python3
"""
Mix postprocessed HS and MB ROOT files (HS + pileup MB overlay).

Main use case:
- HS = no-PU QCD with correct fastsim_* branches.
- MB = minbias with *bad* fastsim_* but *good* pflow_* (real PF).

In `poisson` mode, the script builds:
    fastsim_* (output) = HS(fastsim_*) + sum over N MB(pflow_* with matching suffix)
    pflow_*  (output) = HS(pflow_*)   + sum over N MB(pflow_*)

where N is drawn from a Poisson distribution with mean `mu`.

In `pair` mode, it behaves like the original script:
    prefix_* (output) = HS(prefix_*) + MB(prefix_*), for prefix in ("fastsim", "pflow")
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
    """Return all fields starting with '<prefix>_' that are not excluded."""
    fields = []
    for field in arrays.fields:
        if not field.startswith(f"{prefix}_"):
            continue
        suffix = field.split("_", 1)[1]
        if any(suffix.startswith(ex) for ex in EXCLUDE_SUFFIX_PREFIXES):
            continue
        fields.append(field)
    return fields


def field_suffix(field: str) -> str:
    """Return the part after the first underscore, e.g. 'fastsim_pt' -> 'pt'."""
    return field.split("_", 1)[1]


def intersect_fields(hs_arrays: ak.Array, mb_arrays: ak.Array) -> dict[str, list[str]]:
    """
    Find common particle fields for each prefix between HS and MB.

    Returns:
        dict[prefix] -> sorted list of common fields (e.g. 'fastsim_pt', 'pflow_eta', ...)

    Raises:
        RuntimeError if *no* common particle branches exist for any prefix.
    """
    result: dict[str, list[str]] = {}
    for prefix in PARTICLE_PREFIXES:
        hs_fields = set(particle_fields(hs_arrays, prefix))
        mb_fields = set(particle_fields(mb_arrays, prefix))
        common = sorted(hs_fields & mb_fields)

        if not common:
            if hs_fields or mb_fields:
                print(
                    f"[mix_postprocessed] No common {prefix}_* branches between HS "
                    f"({len(hs_fields)}) and MB ({len(mb_fields)}); skipping {prefix}."
                )
            continue

        missing = sorted((hs_fields ^ mb_fields) - (hs_fields & mb_fields))
        if missing:
            print(
                f"[mix_postprocessed] Skipping branches not shared by both files "
                f"for prefix '{prefix}': {missing}"
            )
        result[prefix] = common

    if not result:
        raise RuntimeError("No common particle branches between inputs.")

    return result


def pairwise_mix(
    hs_arrays: ak.Array,
    mb_arrays: ak.Array,
    branches: dict[str, list[str]],
) -> dict[str, list[ak.Array]]:
    """
    Original behavior: 1 HS + 1 MB per event (cycling MB if needed).

    For each prefix in branches (e.g. fastsim, pflow):
        prefix_* (out) = HS(prefix_*) ⊕ MB(prefix_*)
    """
    n_hs = len(hs_arrays["eventNumber"])
    n_mb = len(mb_arrays["eventNumber"])
    if n_hs == 0 or n_mb == 0:
        raise ValueError("Both inputs must contain at least one event.")

    mixed: dict[str, list[ak.Array]] = {
        field: [] for fields in branches.values() for field in fields
    }
    npflow: list[int] = []
    nfastsim: list[int] = []

    for i in range(n_hs):
        j = i % n_mb

        for prefix, fields in branches.items():
            for field in fields:
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

        # multiplicities
        if "pflow" in branches:
            npflow.append(len(mixed[branches["pflow"][0]][-1]))
        else:
            npflow.append(0)

        if "fastsim" in branches:
            nfastsim.append(len(mixed[branches["fastsim"][0]][-1]))
        else:
            nfastsim.append(0)

    mixed["npflow"] = npflow
    mixed["nfastsim"] = nfastsim
    return mixed


def poisson_mix(
    hs_arrays: ak.Array,
    mb_arrays: ak.Array,
    branches: dict[str, list[str]],
    mu: float,
    seed: int = 12345,
) -> dict[str, list[ak.Array]]:
    """
    1 HS + N MB events with N ~ Poisson(mu) for each HS event.

    Behavior by prefix:

      - fastsim_* (output):
          HS(fastsim_*)  +  MB(pflow_* with matching suffix)
        i.e. MB fastsim_* is ignored; MB pflow_* is mapped by suffix:
             fastsim_pt <- pflow_pt, fastsim_eta <- pflow_eta, ...

      - pflow_* (output):
          HS(pflow_*) + MB(pflow_*), as usual.

    Assumes that for every fastsim_<suffix> in HS/MB there exists a
    corresponding pflow_<suffix> branch in MB.
    """
    n_hs = len(hs_arrays["eventNumber"])
    n_mb = len(mb_arrays["eventNumber"])
    if n_hs == 0 or n_mb == 0:
        raise ValueError("Both inputs must contain at least one event.")
    if mu < 0:
        raise ValueError("Pileup mean mu must be non-negative.")

    rng = np.random.default_rng(seed)

    mixed: dict[str, list[ak.Array]] = {
        field: [] for fields in branches.values() for field in fields
    }
    npflow: list[int] = []
    nfastsim: list[int] = []

    # Walk through MB events cyclically as we consume them
    mb_cursor = 0

    for i in range(n_hs):
        # Number of pileup interactions to overlay
        n_pu = rng.poisson(mu)

        if n_pu == 0:
            # No MB overlay: just copy HS particles for all prefixes
            for prefix, fields in branches.items():
                for field in fields:
                    hs_vals = np.asarray(hs_arrays[field][i])
                    combined = ak.Array(hs_vals) if hs_vals.size > 0 else ak.Array([])
                    mixed[field].append(combined)
        else:
            mb_indices = [(mb_cursor + k) % n_mb for k in range(n_pu)]
            mb_cursor = (mb_cursor + n_pu) % n_mb

            for prefix, fields in branches.items():
                for field in fields:
                    hs_vals = np.asarray(hs_arrays[field][i])

                    # Decide which MB branch to use
                    if prefix == "fastsim":
                        # For fastsim_*: ignore MB fastsim_*, use MB pflow_* with same suffix
                        suffix = field_suffix(field)            # e.g. fastsim_pt -> 'pt'
                        mb_field_name = f"pflow_{suffix}"       # MB branch name
                    else:
                        # For pflow_*: use MB pflow_* as usual
                        mb_field_name = field

                    if mb_field_name not in mb_arrays.fields:
                        raise KeyError(
                            f"Expected MB branch '{mb_field_name}' for overlay but it is missing. "
                            f"Check that MB has matching pflow_* branches for fastsim_* suffix '{suffix}'."
                        )

                    mb_chunks = [np.asarray(mb_arrays[mb_field_name][j]) for j in mb_indices]

                    # Concatenate MB chunks (some may be empty)
                    if mb_chunks:
                        non_empty = [c for c in mb_chunks if c.size > 0]
                        if non_empty:
                            mb_all = np.concatenate(non_empty, axis=0)
                        else:
                            mb_all = np.asarray([], dtype=hs_vals.dtype)
                    else:
                        mb_all = np.asarray([], dtype=hs_vals.dtype)

                    # Combine HS + MB
                    if hs_vals.size == 0 and mb_all.size == 0:
                        combined = ak.Array([])
                    elif hs_vals.size == 0:
                        combined = ak.Array(mb_all)
                    elif mb_all.size == 0:
                        combined = ak.Array(hs_vals)
                    else:
                        combined = ak.Array(np.concatenate([hs_vals, mb_all], axis=0))

                    mixed[field].append(combined)

        # multiplicities (total number of particles after mixing)
        if "pflow" in branches:
            npflow.append(len(mixed[branches["pflow"][0]][-1]))
        else:
            npflow.append(0)

        if "fastsim" in branches:
            nfastsim.append(len(mixed[branches["fastsim"][0]][-1]))
        else:
            nfastsim.append(0)

    mixed["npflow"] = npflow
    mixed["nfastsim"] = nfastsim
    return mixed


def assemble_output(
    hs_arrays: ak.Array, mixed_particles: dict[str, list[ak.Array] | list[int]]
) -> dict[str, ak.Array]:
    """Assemble the final output dict of awkward Arrays for writing."""
    n_events = len(mixed_particles["npflow"])
    output: dict[str, ak.Array] = {}

    # Mixed particle branches + npflow/nfastsim
    for field, values in mixed_particles.items():
        output[field] = ak.Array(values)

    # Copy truth + eventNumber + ntruth from HS
    truth_fields = [
        field
        for field in hs_arrays.fields
        if field.startswith("truth_") or field in {"eventNumber", "ntruth"}
    ]
    for field in truth_fields:
        output[field] = hs_arrays[field][:n_events]

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mix postprocessed HS + MB ntuples. "
            "Use 'poisson' mode for HS(fastsim) + N×MB(pflow) pileup overlay."
        )
    )
    parser.add_argument("--hs", required=True, type=Path, help="Postprocessed HS ROOT file")
    parser.add_argument("--mb", required=True, type=Path, help="Postprocessed MB ROOT file")
    parser.add_argument("--output", required=True, type=Path, help="Output ROOT file path")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Optional number of HS events to mix (entry_stop for HS).",
    )

    parser.add_argument(
        "--mode",
        choices=["pair", "poisson"],
        default="poisson",
        help="pair: 1 HS + 1 MB; poisson: 1 HS + N MB with N ~ Poisson(mu).",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=30.0,
        help="Mean pileup for poisson mode (ignored in pair mode).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for poisson mode.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hs_arrays = load_arrays(args.hs, args.max_events)
    mb_arrays = load_arrays(args.mb)
    branches = intersect_fields(hs_arrays, mb_arrays)

    if args.mode == "pair":
        mixed_particles = pairwise_mix(hs_arrays, mb_arrays, branches)
    else:
        mixed_particles = poisson_mix(
            hs_arrays, mb_arrays, branches, mu=args.mu, seed=args.seed
        )

    output = assemble_output(hs_arrays, mixed_particles)

    with uproot.recreate(args.output) as fout:
        fout["evt_tree"] = output

    print(f"Mix written to {args.output}")
    print(f"HS events processed: {len(output['npflow'])}")
    print(f"Mode: {args.mode}")
    if args.mode == "poisson":
        print(f"Poisson pileup mean mu = {args.mu}, seed = {args.seed}")


if __name__ == "__main__":
    main()
