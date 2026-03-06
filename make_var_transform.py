import math
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
import yaml

MAX_PARTICLES = 5000
ROOT_FILE = Path(
    "/global/cfs/cdirs/m3246/ylo/parnassus-hep/cms-flow-evt/my_sample/Parnassus_ntuple_BH1_RPScan_merge.root"
)
OUT_YAML = Path("configs/var_transform_blackhole.yaml")


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
    "truth_vx", "truth_vy", "truth_vz", "ntruth"
]

tree = f"{ROOT_FILE}:evt_tree"
selected = 0

for arrays in uproot.iterate(tree, branches, step_size=10000, library="ak"):
    ntruth = arrays["ntruth"]
    mask = (ntruth > 0) & (ntruth < MAX_PARTICLES)
    if not ak.any(mask):
        continue

    pt = arrays["truth_pt"][mask]
    eta = arrays["truth_eta"][mask]
    phi = arrays["truth_phi"][mask]
    vx = arrays["truth_vx"][mask]
    vy = arrays["truth_vy"][mask]
    vz = arrays["truth_vz"][mask]
    ntruth = ntruth[mask]

    selected += len(ntruth)

    stats["npart"].update(ak.to_numpy(ntruth))
    stats["eta"].update(ak.to_numpy(ak.flatten(eta)))
    stats["phi"].update(ak.to_numpy(ak.flatten(phi)))
    stats["vx"].update(ak.to_numpy(ak.flatten(vx)))
    stats["vy"].update(ak.to_numpy(ak.flatten(vy)))
    stats["vz"].update(ak.to_numpy(ak.flatten(vz)))

    ht = ak.sum(pt, axis=-1)
    stats["ht"].update(ak.to_numpy(ht))

    with np.errstate(divide="ignore", invalid="ignore"):
        ptrel = pt / ht
    stats["ptrel"].update(ak.to_numpy(ak.flatten(ptrel)))

    met_x = ak.sum(pt * np.cos(phi), axis=-1)
    met_y = ak.sum(pt * np.sin(phi), axis=-1)
    stats["met_x"].update(ak.to_numpy(met_x))
    stats["met_y"].update(ak.to_numpy(met_y))

print(f"Selected events: {selected}")

results = {name: stats[name].finalize() for name in stats}
yaml.safe_dump(results, OUT_YAML.open("w"), sort_keys=False)
print(f"Wrote {OUT_YAML}")
