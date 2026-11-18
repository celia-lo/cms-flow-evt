#!/usr/bin/env python3
"""Convert a Delphes output file to the Parnassus evt_tree layout expected by
postprocess_eval.py (-i <this file> -d <reference ntuple> -dl).

Replacement for parnassus_core/preprocessing/delphes_to_root.py with two fixes:
  * neutral hadrons: Delphes ParticleFlowCandidate stores PID 0 for neutral
    hadrons, which the pdgid_class_dict mapped to class 5 (residual); here the
    class is assigned from (PID, charge): e 1, mu 2, photon 4, other charged 0,
    other neutral 3 -- the Parnassus convention.
  * truth eta: the old script wrote Particle.Rapidity as truth_eta.  Truth
    branches are only read for their presence by postprocess_eval.py (the
    truth comes from -d), but they are written correctly here anyway.
Positions and impact parameters are converted from mm (Delphes) to cm.

  python convert_delphes_to_ntuple.py delphes_out.root ntuple_out.root
"""
import argparse
import numpy as np
import awkward as ak
import uproot

ap = argparse.ArgumentParser()
ap.add_argument("input"); ap.add_argument("output")
ap.add_argument("-n", "--num_entries", type=int, default=None)
args = ap.parse_args()

P, T = "ParticleFlowCandidate/ParticleFlowCandidate.", "Particle/Particle."


def classify(pid, charge):
    apid = np.abs(pid)
    cls = ak.where(charge != 0, 0, 3)
    cls = ak.where(apid == 11, 1, cls)
    cls = ak.where(apid == 13, 2, cls)
    cls = ak.where(apid == 22, 4, cls)
    return ak.values_astype(cls, np.float32)


tree = uproot.open(args.input)["Delphes"]
a = tree.arrays([P + k for k in ("PT", "Eta", "Phi", "PID", "Charge", "X", "Y", "Z", "D0", "ErrorD0", "DZ", "ErrorDZ")]
                + [T + k for k in ("PT", "Eta", "Phi", "PID", "Charge", "X", "Y", "Z", "Status")]
                + ["Event/Event.Number"], entry_stop=args.num_entries)
f32 = lambda x: ak.values_astype(x, np.float32)
fs = a[T + "Status"] == 1  # final-state entries of the generator record Delphes stored
truth = ak.zip({"pt": f32(a[T + "PT"][fs]), "eta": f32(a[T + "Eta"][fs]), "phi": f32(a[T + "Phi"][fs]),
                "class": classify(a[T + "PID"][fs], a[T + "Charge"][fs]),
                "vx": f32(a[T + "X"][fs] / 10.0), "vy": f32(a[T + "Y"][fs] / 10.0), "vz": f32(a[T + "Z"][fs] / 10.0)})
pflow = ak.zip({"pt": f32(a[P + "PT"]), "eta": f32(a[P + "Eta"]), "phi": f32(a[P + "Phi"]),
                "class": classify(a[P + "PID"], a[P + "Charge"]),
                "vx": f32(a[P + "X"] / 10.0), "vy": f32(a[P + "Y"] / 10.0), "vz": f32(a[P + "Z"] / 10.0),
                "d0": f32(a[P + "D0"] / 10.0), "d0Error": f32(a[P + "ErrorD0"] / 10.0),
                "z0": f32(a[P + "DZ"] / 10.0), "z0Error": f32(a[P + "ErrorDZ"] / 10.0)})
evn = ak.values_astype(ak.flatten(a["Event/Event.Number"]), np.int64)
with uproot.recreate(args.output) as f:
    f["evt_tree"] = {"truth": truth, "pflow": pflow, "eventNumber": evn}
cls = ak.to_numpy(ak.flatten(pflow["class"]))
print(f"wrote {len(evn)} events; <npflow> {ak.mean(ak.num(pflow.pt)):.1f}; pflow class counts "
      f"{dict(zip(*[x.tolist() for x in np.unique(cls, return_counts=True)]))}")
