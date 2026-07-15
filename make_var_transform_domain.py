"""Compute var_transform statistics for the domain-translation model.

Reads pflow_* branches from one or two ROOT files (ALEPH and/or DELPHI) and
computes mean/std for the common feature set [ptrel, eta, phi, npart, ht, met_x, met_y].
Statistics are combined across both datasets if both are provided.

Usage:
  python make_var_transform_domain.py \
      --aleph  my_sample/QQB/train_evt.root \
      --delphi my_sample/delphi/train_evt.root \
      --out    configs/var_transform_domain.yaml

  # Single-file mode (ALEPH only, before DELPHI data is available):
  python make_var_transform_domain.py \
      --aleph my_sample/QQB/train_evt.root \
      --out   configs/var_transform_domain.yaml
"""

import argparse
import math
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import yaml

MAX_PARTICLES = 400


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = None
        self.max = None

    def update(self, data):
        arr = np.asarray(data, dtype=np.float64)
        if arr.size == 0:
            return
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += arr.size
        self.sum += arr.sum()
        self.sumsq += (arr ** 2).sum()
        mn, mx = float(arr.min()), float(arr.max())
        self.min = mn if self.min is None else min(self.min, mn)
        self.max = mx if self.max is None else max(self.max, mx)

    def finalize(self):
        if self.count == 0:
            return {"min": 0.0, "max": 1.0, "mean": 0.0, "std": 1.0}
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean ** 2, 0.0)
        std = math.sqrt(var) if var > 0 else 1.0
        return {
            "min": float(round(self.min, 3)),
            "max": float(round(self.max, 3)),
            "mean": float(round(mean, 3)),
            "std": float(round(std, 3)),
        }


def process_file(path: Path, stats: dict, label: str):
    print(f"[{label}] Processing {path}", flush=True)
    branches = ["npflow", "pflow_pt", "pflow_eta", "pflow_phi"]

    selected = 0
    for arrays in uproot.iterate(f"{path}:evt_tree", branches, step_size=10_000, library="ak"):
        npflow = ak.to_numpy(arrays["npflow"])
        mask = (npflow > 0) & (npflow < MAX_PARTICLES)
        if not mask.any():
            continue

        pflow_pt = arrays["pflow_pt"][mask]
        pflow_eta = arrays["pflow_eta"][mask]
        pflow_phi = arrays["pflow_phi"][mask]
        npflow = npflow[mask]

        pflow_ht = ak.sum(pflow_pt, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ptrel = pflow_pt / pflow_ht

        stats["ptrel"].update(ak.to_numpy(ak.flatten(ptrel)))
        stats["eta"].update(ak.to_numpy(ak.flatten(pflow_eta)))
        stats["phi"].update(ak.to_numpy(ak.flatten(pflow_phi)))
        stats["ht"].update(ak.to_numpy(pflow_ht))
        stats["npart"].update(npflow)

        met_x = ak.sum(pflow_pt * np.cos(pflow_phi), axis=-1)
        met_y = ak.sum(pflow_pt * np.sin(pflow_phi), axis=-1)
        stats["met_x"].update(ak.to_numpy(met_x))
        stats["met_y"].update(ak.to_numpy(met_y))

        selected += int(mask.sum())

    print(f"[{label}] Selected {selected} events", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aleph", type=Path, default=None, help="ALEPH ROOT file")
    p.add_argument("--delphi", type=Path, default=None, help="DELPHI ROOT file")
    p.add_argument("--out", type=Path, default=Path("configs/var_transform_domain.yaml"))
    args = p.parse_args()

    if args.aleph is None and args.delphi is None:
        p.error("At least one of --aleph or --delphi must be provided")

    stats = {
        name: RunningStats()
        for name in ["ptrel", "ht", "eta", "phi", "npart", "met_x", "met_y"]
    }

    if args.aleph is not None:
        process_file(args.aleph, stats, "ALEPH")
    if args.delphi is not None:
        process_file(args.delphi, stats, "DELPHI")

    results = {name: stats[name].finalize() for name in stats}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(results, args.out.open("w"), sort_keys=False)
    print(f"Wrote {args.out}")
    for k, v in results.items():
        print(f"  {k}: mean={v['mean']:.4f} std={v['std']:.4f} "
              f"min={v['min']:.4f} max={v['max']:.4f}")


if __name__ == "__main__":
    main()
