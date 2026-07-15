
#!/usr/bin/env python3
"""Mix PF-level HS and MB RCFM samples prior to post-processing.

This script merges the PF collections from two ROOT ntuples (HS and MB)
according to user-supplied weights. It preserves the HS truth branches
and writes an output ntuple compatible with postprocess_eval.py.
"""

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


PF_BRANCHES = [
    "pflow_pt",
    "pflow_eta",
    "pflow_phi",
    "pflow_vx",
    "pflow_vy",
    "pflow_vz",
    "pflow_class",
    "pflow_ind",  # optional; some reference ntuples omit this mask
]


def load_arrays(path: Path, max_events: int | None = None):
    with uproot.open(path) as file:
        tree = file["evt_tree"]
        arrays = tree.arrays(library="ak", entry_stop=max_events)
    return arrays


def pick_common_branches(arr_hs, arr_mb):
    """Return PF branches present in both inputs, preserving PF_BRANCHES order."""
    available = []
    missing = []
    hs_fields = set(arr_hs.fields)
    mb_fields = set(arr_mb.fields)
    for branch in PF_BRANCHES:
        if branch in hs_fields and branch in mb_fields:
            available.append(branch)
        else:
            missing.append(branch)
    if missing:
        print(
            f"[mix_rcfm_pf] Skipping absent branches: {', '.join(missing)}; "
            "they do not exist in both inputs."
        )
    return available


def mix_pf_collections(arr_hs, arr_mb, ratio, rng, branches):
    n_hs = len(arr_hs["pflow_pt"])
    n_mb = len(arr_mb["pflow_pt"])

    base_count = int(np.floor(ratio))
    frac_part = ratio - base_count

    mixed = {branch: [] for branch in branches}
    npflow = []

    for i in range(n_hs):
        overlays = base_count
        if rng.random() < frac_part:
            overlays += 1

        mb_indices = (
            rng.integers(0, n_mb, size=overlays, endpoint=False)
            if overlays > 0
            else []
        )

        event_count = None
        for branch in branches:
            base = arr_hs[branch][i]
            extras = [arr_mb[branch][j] for j in mb_indices]
            if extras:
                combined = ak.concatenate([base, *extras], axis=0)
            else:
                combined = base
            mixed[branch].append(combined)
            if branch == "pflow_pt":
                event_count = int(ak.num(combined, axis=0))

        if event_count is None:
            event_count = int(ak.num(mixed["pflow_pt"][-1], axis=0))
        npflow.append(event_count)

    mixed["npflow"] = ak.Array(npflow)
    return mixed


def assemble_output(arr_hs, mixed_pf):
    output = {
        branch: ak.Array(values) for branch, values in mixed_pf.items() if branch != "npflow"
    }
    output["npflow"] = mixed_pf["npflow"]
    for branch in arr_hs.fields:
        if branch.startswith("truth") or branch in {"ntruth", "eventNumber"}:
            if branch in {"truth_ind", "ntruth_ind"}:
                continue
            output[branch] = arr_hs[branch]
    return output


def write_root(path: Path, arrays: dict):
    with uproot.recreate(path) as fout:
        fout["evt_tree"] = arrays


def parse_args():
    parser = argparse.ArgumentParser(description="Mix HS and MB PF ntuples before post-processing.")
    parser.add_argument("--hs", required=True, type=Path, help="Path to HS ROOT file")
    parser.add_argument("--mb", required=True, type=Path, help="Path to MB ROOT file")
    parser.add_argument("--output", required=True, type=Path, help="Output ROOT file path")
    parser.add_argument("--w-hs", required=True, type=float, help="Weight for HS sample")
    parser.add_argument("--w-mb", required=True, type=float, help="Weight for MB sample")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for MB sampling")
    parser.add_argument("--max-events", type=int, default=None, help="Optional number of HS events to mix")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.w_hs <= 0 or args.w_mb <= 0:
        raise ValueError("Weights must be positive.")
    if not args.hs.exists():
        raise FileNotFoundError(f"HS file not found: {args.hs}")
    if not args.mb.exists():
        raise FileNotFoundError(f"MB file not found: {args.mb}")

    ratio = args.w_mb / args.w_hs
    rng = np.random.default_rng(args.seed)

    mb_arrays = load_arrays(args.mb)

    if args.max_events is None:
        with uproot.open(args.hs) as hs_file:
            hs_tree = hs_file["evt_tree"]
            hs_entries = hs_tree.num_entries
        max_events = min(hs_entries, len(mb_arrays["pflow_pt"]))
    else:
        max_events = args.max_events

    hs_arrays = load_arrays(args.hs, max_events)
    mb_arrays = load_arrays(args.mb)

    pf_branches = pick_common_branches(hs_arrays, mb_arrays)
    if not pf_branches:
        raise RuntimeError("No common PF branches to mix.")

    mixed_pf = mix_pf_collections(hs_arrays, mb_arrays, ratio, rng, pf_branches)
    output_arrays = assemble_output(hs_arrays, mixed_pf)

    write_root(args.output, output_arrays)

    print(f"Mixed file written to {args.output}")
    print(f"HS events processed: {len(hs_arrays['pflow_pt'])}")
    print(f"Effective MB overlays per HS event: {ratio:.3f}")


if __name__ == "__main__":
    main()
