import math
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import yaml

MAX_PARTICLES = 2048
ROOT_FILE = Path(
    "/global/cfs/cdirs/m3246/ylo/parnassus-hep/cms-flow-evt/my_sample/minbias_2024_60PU_noPtcut/train_evt.root"
)
OUT_YAML = Path("configs/var_transform_minbias_60PU.yaml")

# Secondary vertex particles (K0S, Lambda, B-mesons) can have vx/vy up to ±300mm
# and vz up to ±800mm. Clamp these to prevent ±400σ normalized values in the transformer.
# B-mesons: cτ~0.5mm, typical lab decay length ~1-3mm → ±3mm captures most
# K0S at 1 GeV: cτ=27mm, γβ~7 → decay length ~190mm → clamped to 10mm (clips K0S far decays)
# Lambda: similar
VTX_XY_CLAMP = 3.0   # mm — captures B-meson secondaries, clips K0S/Lambda far decays
VTX_Z_CLAMP  = 50.0  # mm — covers K0S typical lab decay lengths in z

# MET in 60PU minbias can reach -642 GeV with std=26 GeV → -24.5σ without clamping
MET_CLAMP = 100.0   # GeV


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = None
        self.max = None

    def update(self, data):
        arr = np.asarray(data)
        if arr.size == 0:
            return
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        self.count += arr.size
        self.sum += arr.sum(dtype=np.float64)
        self.sumsq += np.square(arr, dtype=np.float64).sum(dtype=np.float64)
        mn = float(arr.min())
        mx = float(arr.max())
        self.min = mn if self.min is None else min(self.min, mn)
        self.max = mx if self.max is None else max(self.max, mx)

    def finalize(self):
        if self.count == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 1.0}
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        std = math.sqrt(var) if var > 0 else 1.0
        return {
            "min": float(round(self.min, 3)),
            "max": float(round(self.max, 3)),
            "mean": float(round(mean, 4)),
            "std": float(round(std, 4)),
        }


stats = {name: RunningStats() for name in
         ["ptrel", "ht", "eta", "phi", "vx", "vy", "vz", "npart", "met_x", "met_y"]}

branches = [
    "truth_pt", "truth_eta", "truth_phi",
    "truth_vx", "truth_vy", "truth_vz", "ntruth",
    "pflow_pt", "pflow_phi", "npflow",
]

tree = f"{ROOT_FILE}:evt_tree"
selected = 0

for arrays in uproot.iterate(tree, branches, step_size=10000, library="ak"):
    ntruth = arrays["ntruth"]
    npflow = arrays["npflow"]
    mask = (ntruth > 0) & (ntruth < MAX_PARTICLES) & (npflow > 0) & (npflow < MAX_PARTICLES)
    if not ak.any(mask):
        continue

    pt      = arrays["truth_pt"][mask]
    eta     = arrays["truth_eta"][mask]
    phi     = arrays["truth_phi"][mask]
    vx_raw  = ak.to_numpy(ak.flatten(arrays["truth_vx"][mask]))
    vy_raw  = ak.to_numpy(ak.flatten(arrays["truth_vy"][mask]))
    vz_raw  = ak.to_numpy(ak.flatten(arrays["truth_vz"][mask]))
    ntruth  = ntruth[mask]

    pflow_pt  = arrays["pflow_pt"][mask]
    pflow_phi = arrays["pflow_phi"][mask]
    npflow    = npflow[mask]

    selected += len(ntruth)

    # per-particle truth features
    truth_ht = ak.sum(pt, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ptrel = pt / truth_ht
    stats["ptrel"].update(ak.to_numpy(ak.flatten(ptrel)))
    stats["eta"].update(ak.to_numpy(ak.flatten(eta)))
    stats["phi"].update(ak.to_numpy(ak.flatten(phi)))

    # vx/vy: clamp to ±VTX_XY_CLAMP to avoid extreme secondary vertex outliers
    stats["vx"].update(np.clip(vx_raw, -VTX_XY_CLAMP, VTX_XY_CLAMP))
    stats["vy"].update(np.clip(vy_raw, -VTX_XY_CLAMP, VTX_XY_CLAMP))
    # vz: clamp to ±VTX_Z_CLAMP
    stats["vz"].update(np.clip(vz_raw, -VTX_Z_CLAMP, VTX_Z_CLAMP))

    # event-level quantities from pflow
    pflow_ht = ak.to_numpy(ak.sum(pflow_pt, axis=-1))
    stats["npart"].update(ak.to_numpy(npflow))
    stats["ht"].update(pflow_ht[pflow_ht > 0])

    met_x = ak.to_numpy(ak.sum(pflow_pt * np.cos(pflow_phi), axis=-1))
    met_y = ak.to_numpy(ak.sum(pflow_pt * np.sin(pflow_phi), axis=-1))
    stats["met_x"].update(np.clip(met_x, -MET_CLAMP, MET_CLAMP))
    stats["met_y"].update(np.clip(met_y, -MET_CLAMP, MET_CLAMP))

print(f"Selected events: {selected}")

results = {name: stats[name].finalize() for name in stats}

print("\nNormalized extremes (max |clamped_value - mean| / std):")
for name, vals in results.items():
    mn, mx, mean, std = vals["min"], vals["max"], vals["mean"], vals["std"]
    worst = max(abs(mn - mean), abs(mx - mean)) / std
    print(f"  {name:8s}: min={mn:10.4f}  max={mx:10.4f}  mean={mean:9.4f}  std={std:8.4f}  max_normalized={worst:.1f}σ")

yaml.safe_dump(results, OUT_YAML.open("w"), sort_keys=False)
print(f"\nWrote {OUT_YAML}")
print(f"\nNote: vx/vy clamped at ±{VTX_XY_CLAMP}mm, vz at ±{VTX_Z_CLAMP}mm, met at ±{MET_CLAMP}GeV")
print("Runtime clamping MUST match: set vtx_xy_clamp, vtx_z_clamp, met_clamp in configs/part.yaml")
