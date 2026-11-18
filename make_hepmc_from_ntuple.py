#!/usr/bin/env python3
"""Write the truth particles of a Parnassus ntuple (evt_tree) as a HepMC3 ASCII file
for Delphes.  Streaming replacement for parnassus_core/preprocessing/root_to_hepmc.py:
same content (one status-1 particle per truth entry, pdgId + mass from the ntuple,
particles grouped by production vertex), but writes event by event and converts the
ntuple vertex positions from cm to the HepMC mm convention.

  python make_hepmc_from_ntuple.py my_sample/QQB_noptcut/test_evt.root my_sample/QQB_noptcut/test_evt.hepmc [-n N]
"""
import argparse
import numpy as np
import uproot
import pyhepmc
from pyhepmc import GenEvent, GenParticle, GenVertex, FourVector, Units
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("input"); ap.add_argument("output")
ap.add_argument("-n", "--num_entries", type=int, default=None)
args = ap.parse_args()

tree = uproot.open(args.input)["evt_tree"]
vars_ = ["truth_pt", "truth_eta", "truth_phi", "truth_mass", "truth_pdgId", "truth_vx", "truth_vy", "truth_vz", "eventNumber"]
n_written = 0
with pyhepmc.open(args.output, "w") as out:
    for chunk in tree.iterate(vars_, library="np", step_size=2000, entry_stop=args.num_entries):
        for i in tqdm(range(len(chunk["eventNumber"])), leave=False):
            pt, eta, phi, m, pid = (chunk[k][i] for k in ("truth_pt", "truth_eta", "truth_phi", "truth_mass", "truth_pdgId"))
            vx, vy, vz = (10.0 * chunk[k][i] for k in ("truth_vx", "truth_vy", "truth_vz"))  # cm -> mm
            # eta is pseudorapidity: p = pt*cosh(eta), E = sqrt(p^2 + m^2).  (The old
            # root_to_hepmc.py used the rapidity formula mt*sinh(eta), inflating E of massive particles.)
            px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
            e = np.sqrt(pt**2 * np.cosh(eta) ** 2 + m**2)
            evt = GenEvent(Units.GEV, Units.MM)
            evt.event_number = int(chunk["eventNumber"][i])
            vertices = {}
            for j in range(len(pt)):
                key = (round(float(vx[j]), 4), round(float(vy[j]), 4), round(float(vz[j]), 4))
                v = vertices.get(key)
                if v is None:
                    v = GenVertex(FourVector(*key, 0.0))
                    beam = GenParticle(FourVector(0.0, 0.0, 1e-4, 1e-4), 0, 4)  # dummy incoming, status 4 (beam)
                    v.add_particle_in(beam)
                    evt.add_vertex(v)
                    vertices[key] = v
                v.add_particle_out(GenParticle(FourVector(float(px[j]), float(py[j]), float(pz[j]), float(e[j])), int(pid[j]), 1))
            out.write(evt)
            n_written += 1
print(f"wrote {n_written} events to {args.output}")
