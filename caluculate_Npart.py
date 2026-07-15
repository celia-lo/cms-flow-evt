#!/usr/bin/env python3
import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


def compute_npart(
    path: Path,
    tree_name: str,
    prefix: str,
    pt_min: float = 0.0,
    eta_max: float | None = None,
) -> ak.Array:
    """
    Compute N_part per event as the number of particles in a given collection
    passing simple kinematic cuts.

    Assumes branches:
        <prefix>_pt
        <prefix>_eta
    exist in the tree.
    """
    with uproot.open(path) as f:
        tree = f[tree_name]
        arrays = tree.arrays([f"{prefix}_pt", f"{prefix}_eta"], library="ak")

    pt = arrays[f"{prefix}_pt"]
    eta = arrays[f"{prefix}_eta"]

    mask = pt >= pt_min
    if eta_max is not None:
        mask = mask & (ak.abs(eta) <= eta_max)

    # N_part = how many particles per event pass the mask
    n_part = ak.sum(mask, axis=1)
    return n_part


def main():
    parser = argparse.ArgumentParser(description="Compute N_part per event.")
    parser.add_argument("file", type=Path, help="Input ROOT file")
    parser.add_argument(
        "--tree",
        default="evt_tree",
        help="Tree name (e.g. 'evt_tree' for mixed ntuples, 'Events' for MiniAOD)",
    )
    parser.add_argument(
        "--prefix",
        default="fastsim",
        help="Particle collection prefix: e.g. 'fastsim' or 'pflow'",
    )
    parser.add_argument(
        "--pt-min",
        type=float,
        default=0.0,
        help="Minimum pT cut for counting (GeV)",
    )
    parser.add_argument(
        "--eta-max",
        type=float,
        default=None,
        help="Maximum |eta| cut for counting (if not set, no eta cut)",
    )

    args = parser.parse_args()

    n_part = compute_npart(
        args.file,
        tree_name=args.tree,
        prefix=args.prefix,
        pt_min=args.pt_min,
        eta_max=args.eta_max,
    )

    n_part_np = ak.to_numpy(n_part)

    print(f"File: {args.file}")
    print(f"Tree: {args.tree}, prefix: {args.prefix}")
    print(f"Events: {len(n_part_np)}")
    print(f"Mean N_part:   {float(np.mean(n_part_np)):.3f}")
    print(f"Median N_part: {float(np.median(n_part_np)):.3f}")
    print(f"Std N_part:    {float(np.std(n_part_np)):.3f}")
    print("First 10 N_part values:", n_part_np[:10])


if __name__ == "__main__":
    main()
