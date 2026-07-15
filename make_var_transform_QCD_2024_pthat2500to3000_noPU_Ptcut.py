import math
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import yaml

MAX_PARTICLES = 2048
ROOT_FILE = Path(
    "/global/cfs/cdirs/m3246/ylo/parnassus-hep/cms-flow-evt/my_sample/QCD_2024_pthat2500to3000_noPU_Ptcut/train_evt.root"
)
OUT_YAML = Path("configs/var_transform_QCD_2024_pthat2500to3000_noPU_Ptcut.yaml")

# MET clamped at ±1000 GeV before stat computation (and at runtime in datasetloader.py)
# so extreme QCD met outliers (up to ±6415 GeV) don't inflate std and cause NaN gradients.
MET_CLAMP = 1000.0


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
            "mean": float(round(mean, 3)),
            "std": float(round(std, 3)),
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
    vx      = arrays["truth_vx"][mask]
    vy      = arrays["truth_vy"][mask]
    vz      = arrays["truth_vz"][mask]
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
    stats["vx"].update(ak.to_numpy(ak.flatten(vx)))
    stats["vy"].update(ak.to_numpy(ak.flatten(vy)))
    stats["vz"].update(ak.to_numpy(ak.flatten(vz)))

    # event-level quantities from pflow (evt model targets)
    pflow_ht = ak.to_numpy(ak.sum(pflow_pt, axis=-1))
    stats["npart"].update(ak.to_numpy(npflow))
    # ht: apply log before computing stats so fn: log in var_transform
    # compresses extreme high-pT QCD outliers (raw max ~17813 GeV → ~30σ without log,
    # but only ~4σ after log-transform).
    stats["ht"].update(np.log(pflow_ht[pflow_ht > 0]))

    met_x = ak.to_numpy(ak.sum(pflow_pt * np.cos(pflow_phi), axis=-1))
    met_y = ak.to_numpy(ak.sum(pflow_pt * np.sin(pflow_phi), axis=-1))
    # clamp met at ±MET_CLAMP before computing stats — matches QQB's natural met range
    # and prevents extreme outliers (up to ±6415 GeV) from inflating std.
    stats["met_x"].update(np.clip(met_x, -MET_CLAMP, MET_CLAMP))
    stats["met_y"].update(np.clip(met_y, -MET_CLAMP, MET_CLAMP))

print(f"Selected events: {selected}")

results = {name: stats[name].finalize() for name in stats}

# Write YAML with fn: log annotation for ht (stats are in log-space)
# and met_clamp metadata (informational — runtime clamping done in datasetloader.py)
out = {}
for name, vals in results.items():
    entry = dict(vals)
    if name == "ht":
        entry["fn"] = "log"
    out[name] = entry

yaml.safe_dump(out, OUT_YAML.open("w"), sort_keys=False)
print(f"Wrote {OUT_YAML}")
print()
print("Note: met_x/met_y stats computed from values clamped at ±1000 GeV.")
print("Runtime clamping at ±1000 GeV must be applied in datasetloader.py")
print("(see _get_scaled_global_data) to match these stats.")
