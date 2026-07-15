#!/usr/bin/env python3
import uproot
import sys

path = "my_sample/QQB/ZM4212_39_AL.root"
if len(sys.argv) > 1:
    path = sys.argv[1]

with uproot.open(path) as f:
    print(f"\nFile: {path}")
    print("Top-level keys:")
    for key in f.keys():
        print(f"  {key}")

    # NanoAODs usually have an "Events" tree; inspect it if present
    events = f["events"]
    print(f"\nEntries in Events tree: {events.num_entries}")

    # Grab a few representative branches
    nano_branches = [
        "nMuon", "Muon_pt", "nElectron", "Electron_eta",
        "nJet", "Jet_btagDeepFlavB", "nGenPart", "GenPart_pdgId",
    ]
    available = [name for name in nano_branches if name in events.keys()]
    missing = sorted(set(nano_branches) - set(available))

    print("\nSample NanoAOD-style branches found:")
    for name in available:
        print(f"  {name}")

    print("\nMissing expected branches:")
    for name in missing:
        print(f"  {name}")
