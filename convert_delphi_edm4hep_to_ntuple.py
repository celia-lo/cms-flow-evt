#!/usr/bin/env python3
"""
Convert a Delphi EDM4hep/PODIO ROOT file into a training ntuple.

Differences from the generic convert_edm4hep_to_ntuple.py:
 - Truth collection defaults to sDST_LUJ_GenParticles (LUJET generator output)
 - Reco  collection defaults to fDST_MAIN_Particles
 - Truth particles are filtered to generatorStatus == 1 (stable final-state)
 - Pflow classification uses charge only (PDG field is 0 for Delphi reco)
 - Track d0/z0 default-collection is fDST_TRAC_Tracks

Usage:
  python convert_delphi_edm4hep_to_ntuple.py <file_or_list.txt> <output.root>

Example (single file):
  python convert_delphi_edm4hep_to_ntuple.py \\
    my_sample/delphi/mc/cluster_16212141_incl/run_incl_16212141_0.edm4hep.root \\
    my_sample/delphi/train_evt.root

Example (file list):
  python convert_delphi_edm4hep_to_ntuple.py delphi_files.txt my_sample/delphi/train_evt.root
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import uproot
import awkward as ak


MAX_ABS_ETA = 2.7
TRUTH_PT_MIN = 0.5
PFLOW_PT_MIN = 1.0
TRUTH_NEUTRINOS = {12, 14, 16}
PDGID_CLASS = {
    211: 0, -211: 0,
    11: 1, -11: 1,
    13: 2, -13: 2,
    130: 3, 310: 3, 311: 3, 321: 3, -321: 3,
    22: 4,
    12: -1, 14: -1, 16: -1, -12: -1, -14: -1, -16: -1,
}
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Delphi EDM4hep ROOT samples into evt_tree ntuples."
    )
    p.add_argument("input", type=Path, help="Input EDM4hep ROOT file or .txt file list")
    p.add_argument("output", type=Path, help="Output ROOT file (evt_tree)")
    p.add_argument("--tree", default="events", help="TTree name (default: events)")
    p.add_argument(
        "--truth-collection", default="sDST_LUJ_GenParticles",
        help="EDM4hep truth collection (default: sDST_LUJ_GenParticles)",
    )
    p.add_argument(
        "--pflow-collection", default="fDST_MAIN_Particles",
        help="EDM4hep reco collection (default: fDST_MAIN_Particles)",
    )
    p.add_argument(
        "--track-collection", default="fDST_TRAC_Tracks",
        help="Track collection for d0/z0 lookup (default: fDST_TRAC_Tracks)",
    )
    p.add_argument(
        "--gen-status", type=int, default=1,
        help="Keep only truth particles with this generatorStatus (default: 1 = stable)",
    )
    p.add_argument("--entry-start", type=int, default=0)
    p.add_argument("--entry-stop", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=20000)
    return p.parse_args()


def resolve_input_files(input_path: Path) -> list[Path]:
    if input_path.suffix.lower() != ".txt":
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    resolved: list[Path] = []
    base_dir = input_path.parent
    with input_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cand = Path(line)
            if not cand.is_absolute():
                cand = (base_dir / cand).resolve()
            resolved.append(cand)
    if not resolved:
        raise ValueError(f"No ROOT files listed in {input_path}")
    return resolved


def normalize_keys(keys: Iterable[str]) -> list[str]:
    return [k.decode() if isinstance(k, bytes) else k for k in keys]


class BranchResolver:
    """Resolve EDM4hep branch names — handles collection/collection.field layout."""

    def __init__(self, keys: Iterable[str]):
        self.keys = list(keys)
        self.key_set = set(self.keys)

    def scalar(self, collection: str, field: str) -> str:
        return self._resolve(collection, field)

    def vector(self, collection: str, field: str, axis: str) -> str:
        return self._resolve(collection, f"{field}.{axis}")

    def _resolve(self, collection: str, suffix: str) -> str:
        candidates = [
            f"{collection}.{suffix}",
            f"{collection}.core.{suffix}",
            f"{collection}#{suffix}",
            f"{collection}#.core.{suffix}",
        ]
        for cand in candidates:
            if cand in self.key_set:
                return cand
        for cand in self.key_set:
            if cand.startswith(collection) and cand.endswith(suffix):
                return cand
        raise KeyError(f"Could not find branch for {collection}.{suffix}")


def compute_pt_eta_phi(
    px: ak.Array, py: ak.Array, pz: ak.Array
) -> tuple[ak.Array, ak.Array, ak.Array]:
    pt = np.sqrt(px * px + py * py)
    p = np.sqrt(px * px + py * py + pz * pz) + EPS
    eta = 0.5 * np.log((p + pz + EPS) / (p - pz + EPS))
    phi = np.arctan2(py, px)
    return pt, eta, phi


def map_pdg_classes(pdg: ak.Array) -> ak.Array:
    flat = ak.to_numpy(ak.flatten(pdg, axis=None))
    mapper = np.vectorize(lambda x: PDGID_CLASS.get(int(x), 5), otypes=[np.int32])
    mapped = mapper(flat)
    counts = ak.to_numpy(ak.num(pdg, axis=1))
    if mapped.size == 0:
        return ak.values_astype(ak.zeros_like(pdg), np.int32)
    return ak.unflatten(mapped, counts)


def classify_pflow_data(
    charge: ak.Array,
    pdg: ak.Array | None,
    electron_indices: ak.Array | None,
    muon_indices: ak.Array | None,
    photon_indices: ak.Array | None,
) -> ak.Array:
    """Classify Delphi reco particles using detector PID collections (data/no-truth).

    Truth-verified purities from 5005-event MC sample:
      - fDST_ELID_ElectronID params[0]>=5 AND fDST_HAID_dEdx params[0]>1.3
            → class 1 (electron, 78% purity)
      - fDST_MUID_MuonID params[0]>=7  → class 2 (muon, 52% purity)
      - sDST_PHOT_PhotonID             → class 4 (photon, 96% purity)
      - Default: charge != 0 → class 0, charge == 0 → class 3
    Note: fDST/sDST_MAIN_Particles are identical, so sDST_PHOT indices index into fDST.
    """
    if pdg is not None:
        flat_pdg = ak.to_numpy(ak.flatten(pdg, axis=None))
        if np.any(flat_pdg != 0):
            return map_pdg_classes(pdg)

    cls = ak.where(
        charge != 0,
        ak.full_like(charge, 0, dtype=np.int32),
        ak.full_like(charge, 3, dtype=np.int32),
    )

    cls_list = ak.to_list(cls)
    for idx_list, label in [
        (photon_indices, 4),
        (electron_indices, 1),
        (muon_indices, 2),
    ]:
        if idx_list is not None:
            for evt_cls, indices in zip(cls_list, ak.to_list(idx_list)):
                for i in indices:
                    if i < len(evt_cls):
                        evt_cls[i] = label

    return ak.values_astype(ak.Array(cls_list), np.int32)


def fill_constant_like(reference: ak.Array, value: float, dtype=np.float32) -> ak.Array:
    counts = ak.to_numpy(ak.num(reference, axis=1))
    return ak.Array([np.full(int(c), value, dtype=dtype) for c in counts])


def gather_first_track_param(
    track_values: ak.Array,
    track_refs: ak.Array,
    begins: ak.Array,
    ends: ak.Array,
    default: float = -999.0,
) -> ak.Array:
    gathered = []
    for values, refs, start_arr, end_arr in zip(track_values, track_refs, begins, ends):
        event_vals = []
        refs_list = ak.to_list(refs)
        values_list = ak.to_list(values)
        for b, e in zip(start_arr, end_arr):
            if e > b and b < len(refs_list):
                idx = refs_list[b]
                if idx < len(values_list):
                    event_vals.append(float(values_list[idx]))
                else:
                    event_vals.append(default)
            else:
                event_vals.append(default)
        gathered.append(np.array(event_vals, dtype=np.float32))
    return ak.Array(gathered)


def main() -> int:
    args = parse_args()
    input_files = resolve_input_files(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = uproot.recreate(args.output)

    first_chunk = True
    processed = 0
    next_event_number = args.entry_start or 0

    tc = args.truth_collection
    pc = args.pflow_collection
    trc = args.track_collection

    for file_idx, input_path in enumerate(input_files):
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        print(f"[convert_delphi] Processing {input_path}", flush=True)
        with uproot.open(input_path) as infile:
            if args.tree not in infile:
                raise KeyError(f"TTree '{args.tree}' not found in {input_path}")
            tree = infile[args.tree]
            keys = normalize_keys(tree.keys())
            key_set = set(keys)
            resolver = BranchResolver(keys)

            def try_resolve(func, *fargs) -> str | None:
                try:
                    return func(*fargs)
                except KeyError:
                    return None

            # Truth branches
            truth_px_key = resolver.vector(tc, "momentum", "x")
            truth_py_key = resolver.vector(tc, "momentum", "y")
            truth_pz_key = resolver.vector(tc, "momentum", "z")
            truth_vx_key = resolver.vector(tc, "vertex", "x")
            truth_vy_key = resolver.vector(tc, "vertex", "y")
            truth_vz_key = resolver.vector(tc, "vertex", "z")
            truth_pdg_key = resolver.scalar(tc, "PDG")
            truth_mass_key = try_resolve(resolver.scalar, tc, "mass")
            truth_gen_status_key = try_resolve(resolver.scalar, tc, "generatorStatus")

            # Pflow branches
            pflow_px_key = resolver.vector(pc, "momentum", "x")
            pflow_py_key = resolver.vector(pc, "momentum", "y")
            pflow_pz_key = resolver.vector(pc, "momentum", "z")
            pflow_vx_key = try_resolve(resolver.vector, pc, "referencePoint", "x")
            pflow_vy_key = try_resolve(resolver.vector, pc, "referencePoint", "y")
            pflow_vz_key = try_resolve(resolver.vector, pc, "referencePoint", "z")
            pflow_charge_key = try_resolve(resolver.scalar, pc, "charge")
            pflow_pdg_key = try_resolve(resolver.scalar, pc, "PDG")
            pflow_mass_key = try_resolve(resolver.scalar, pc, "mass")

            # Track d0/z0 links (optional)
            track_idx_key = f"_{pc}_tracks/_{pc}_tracks.index"
            track_idx_key = track_idx_key if track_idx_key in key_set else None
            track_begin_key = f"{pc}/{pc}.tracks_begin"
            track_begin_key = track_begin_key if track_begin_key in key_set else None
            track_end_key = f"{pc}/{pc}.tracks_end"
            track_end_key = track_end_key if track_end_key in key_set else None
            ts_prefix = f"_{trc}_trackStates/_{trc}_trackStates"
            track_d0_key = f"{ts_prefix}.D0" if f"{ts_prefix}.D0" in key_set else None
            track_z0_key = f"{ts_prefix}.Z0" if f"{ts_prefix}.Z0" in key_set else None

            # Delphi PID: each ID collection links back to reco particle indices.
            #   fDST_ELID_ElectronID  = all charged tracks (bookkeeping, likelihood=0) — not used
            # Truth-verified PID collections (purities from 5005-event MC sample):
            #   fDST_ELID_ElectronID params[0]>=5  = 67% electron purity
            #     + fDST_HAID_dEdx    params[0]>1.3 = 78% combined electron purity → class 1
            #   fDST_MUID_MuonID     params[0]>=7  = 52% muon purity (was 28% unthresholded) → class 2
            #   sDST_PHOT_PhotonID                 = 96% photon purity → class 4
            #   fDST_EL_ElectronExtra              = 1.5% electron purity — not used
            elid_reco_key   = "_fDST_ELID_ElectronID_particle/_fDST_ELID_ElectronID_particle.index"
            elid_reco_key   = elid_reco_key if elid_reco_key in key_set else None
            elid_params_key = "_fDST_ELID_ElectronID_parameters"
            elid_params_key = elid_params_key if elid_params_key in key_set else None
            elid_begin_key  = "fDST_ELID_ElectronID/fDST_ELID_ElectronID.parameters_begin"
            elid_begin_key  = elid_begin_key if elid_begin_key in key_set else None
            dedx_reco_key   = "_fDST_HAID_dEdx_particle/_fDST_HAID_dEdx_particle.index"
            dedx_reco_key   = dedx_reco_key if dedx_reco_key in key_set else None
            dedx_params_key = "_fDST_HAID_dEdx_parameters"
            dedx_params_key = dedx_params_key if dedx_params_key in key_set else None
            dedx_begin_key  = "fDST_HAID_dEdx/fDST_HAID_dEdx.parameters_begin"
            dedx_begin_key  = dedx_begin_key if dedx_begin_key in key_set else None
            muon_idx_key    = "_fDST_MUID_MuonID_particle/_fDST_MUID_MuonID_particle.index"
            muon_idx_key    = muon_idx_key if muon_idx_key in key_set else None
            muon_params_key = "_fDST_MUID_MuonID_parameters"
            muon_params_key = muon_params_key if muon_params_key in key_set else None
            muon_begin_key  = "fDST_MUID_MuonID/fDST_MUID_MuonID.parameters_begin"
            muon_begin_key  = muon_begin_key if muon_begin_key in key_set else None
            photon_idx_key  = "_sDST_PHOT_PhotonID_particle/_sDST_PHOT_PhotonID_particle.index"
            photon_idx_key  = photon_idx_key if photon_idx_key in key_set else None


            if truth_gen_status_key is None:
                print(
                    f"[convert_delphi] WARNING: generatorStatus not found in {tc}; "
                    "all truth particles will be kept",
                    flush=True,
                )

            required_branches = sorted({
                k for k in [
                    truth_px_key, truth_py_key, truth_pz_key,
                    truth_vx_key, truth_vy_key, truth_vz_key, truth_pdg_key,
                    truth_mass_key, truth_gen_status_key,
                    pflow_px_key, pflow_py_key, pflow_pz_key,
                    pflow_vx_key, pflow_vy_key, pflow_vz_key,
                    pflow_charge_key, pflow_pdg_key, pflow_mass_key,
                    track_idx_key, track_begin_key, track_end_key,
                    track_d0_key, track_z0_key,
                    elid_reco_key, elid_params_key, elid_begin_key,
                    dedx_reco_key, dedx_params_key, dedx_begin_key,
                    muon_idx_key, muon_params_key, muon_begin_key,
                    photon_idx_key,
                ] if k is not None
            })

            total_entries = tree.num_entries
            if file_idx == 0:
                start = args.entry_start or 0
                stop = args.entry_stop if args.entry_stop is not None else total_entries
            else:
                start = 0
                stop = total_entries

            if start >= stop:
                continue

            for chunk in tree.iterate(
                required_branches,
                library="ak",
                step_size=args.chunk_size,
                entry_start=start,
                entry_stop=stop,
            ):
                chunk_size = len(chunk[truth_px_key])
                processed += chunk_size

                truth_px = chunk[truth_px_key]
                truth_py = chunk[truth_py_key]
                truth_pz = chunk[truth_pz_key]
                truth_vx = chunk[truth_vx_key]
                truth_vy = chunk[truth_vy_key]
                truth_vz = chunk[truth_vz_key]
                truth_pdg = chunk[truth_pdg_key]
                truth_mass = chunk[truth_mass_key] if truth_mass_key else None
                truth_gen_status = (
                    chunk[truth_gen_status_key] if truth_gen_status_key else None
                )

                pflow_px = chunk[pflow_px_key]
                pflow_py = chunk[pflow_py_key]
                pflow_pz = chunk[pflow_pz_key]
                pflow_vx = chunk[pflow_vx_key] if pflow_vx_key else None
                pflow_vy = chunk[pflow_vy_key] if pflow_vy_key else None
                pflow_vz = chunk[pflow_vz_key] if pflow_vz_key else None
                pflow_charge = chunk[pflow_charge_key] if pflow_charge_key else None
                pflow_pdg = chunk[pflow_pdg_key] if pflow_pdg_key else None
                pflow_mass = chunk[pflow_mass_key] if pflow_mass_key else None

                track_refs = chunk[track_idx_key] if track_idx_key else None
                track_begin = chunk[track_begin_key] if track_begin_key else None
                track_end = chunk[track_end_key] if track_end_key else None
                track_d0 = chunk[track_d0_key] if track_d0_key else None
                track_z0 = chunk[track_z0_key] if track_z0_key else None

                photon_indices = chunk[photon_idx_key] if photon_idx_key else None

                # electrons: ELID params[0]>=5 AND dEdx params[0]>1.3 → 78% purity
                electron_indices = None
                if elid_reco_key and elid_params_key and elid_begin_key:
                    elid_reco   = chunk[elid_reco_key]
                    elid_params = chunk[elid_params_key]
                    elid_begin  = chunk[elid_begin_key]
                    use_dedx = dedx_reco_key and dedx_params_key and dedx_begin_key
                    if use_dedx:
                        dedx_reco_l   = ak.to_list(chunk[dedx_reco_key])
                        dedx_params_l = ak.to_list(chunk[dedx_params_key])
                        dedx_begin_l  = ak.to_list(chunk[dedx_begin_key])
                    el_idx_list = []
                    for i_evt, (reco_list, params_list, begin_list) in enumerate(zip(
                        ak.to_list(elid_reco), ak.to_list(elid_params), ak.to_list(elid_begin)
                    )):
                        dedx_map = ({r: dedx_params_l[i_evt][b]
                                     for r, b in zip(dedx_reco_l[i_evt], dedx_begin_l[i_evt])}
                                    if use_dedx else {})
                        el_idx_list.append([
                            r for r, b in zip(reco_list, begin_list)
                            if int(params_list[b]) >= 5 and dedx_map.get(r, 0) > 1.3
                        ])
                    electron_indices = ak.Array(el_idx_list)

                # muons: MUID params[0] >= 7 → 52% purity
                muon_indices = None
                if muon_idx_key and muon_params_key and muon_begin_key:
                    mu_idx_list = []
                    for reco_list, params_list, begin_list in zip(
                        ak.to_list(chunk[muon_idx_key]),
                        ak.to_list(chunk[muon_params_key]),
                        ak.to_list(chunk[muon_begin_key]),
                    ):
                        mu_idx_list.append([
                            r for r, b in zip(reco_list, begin_list)
                            if params_list[b] >= 7
                        ])
                    muon_indices = ak.Array(mu_idx_list)

                # ---- truth selection ----
                truth_pt, truth_eta, truth_phi = compute_pt_eta_phi(
                    truth_px, truth_py, truth_pz
                )
                truth_cls = map_pdg_classes(truth_pdg)

                flat_pdg = ak.flatten(truth_pdg, axis=None)
                neutrino_mask = ak.unflatten(
                    ak.Array(np.isin(np.abs(ak.to_numpy(flat_pdg)), list(TRUTH_NEUTRINOS))),
                    ak.num(truth_pdg, axis=1),
                )
                eta_abs = ak.unflatten(
                    ak.Array(np.abs(ak.to_numpy(ak.flatten(truth_eta, axis=None)))),
                    ak.num(truth_eta, axis=1),
                )

                truth_keep = (
                    (eta_abs <= MAX_ABS_ETA)
                    & (~neutrino_mask)
                    & (truth_cls != 5)
                    & (truth_pt >= TRUTH_PT_MIN)
                )
                if truth_gen_status is not None:
                    truth_keep = truth_keep & (truth_gen_status == args.gen_status)

                truth_pt = truth_pt[truth_keep]
                truth_eta = truth_eta[truth_keep]
                truth_phi = truth_phi[truth_keep]
                truth_vx = truth_vx[truth_keep]
                truth_vy = truth_vy[truth_keep]
                truth_vz = truth_vz[truth_keep]
                truth_cls = truth_cls[truth_keep]
                truth_mass = truth_mass[truth_keep] if truth_mass is not None else None
                truth_pdg = truth_pdg[truth_keep]

                if truth_mass is None:
                    truth_mass = fill_constant_like(truth_pt, -999.0)

                # ---- pflow selection ----
                pflow_pt, pflow_eta, pflow_phi = compute_pt_eta_phi(
                    pflow_px, pflow_py, pflow_pz
                )
                # classify before eta/pt cut so PID indices are still valid
                pflow_cls = classify_pflow_data(
                    pflow_charge, pflow_pdg, electron_indices, muon_indices, photon_indices
                )

                pflow_eta_abs = ak.unflatten(
                    ak.Array(np.abs(ak.to_numpy(ak.flatten(pflow_eta, axis=None)))),
                    ak.num(pflow_eta, axis=1),
                )
                pflow_keep = (pflow_eta_abs <= MAX_ABS_ETA) & (pflow_pt >= PFLOW_PT_MIN)

                # d0/z0 from linked tracks before applying pflow_keep
                pflow_d0 = None
                pflow_z0 = None
                if (
                    track_d0 is not None
                    and track_z0 is not None
                    and track_refs is not None
                    and track_begin is not None
                    and track_end is not None
                ):
                    pflow_d0 = gather_first_track_param(
                        track_d0, track_refs, track_begin, track_end
                    )
                    pflow_z0 = gather_first_track_param(
                        track_z0, track_refs, track_begin, track_end
                    )

                pflow_pt = pflow_pt[pflow_keep]
                pflow_eta = pflow_eta[pflow_keep]
                pflow_phi = pflow_phi[pflow_keep]
                pflow_vx = pflow_vx[pflow_keep] if pflow_vx is not None else None
                pflow_vy = pflow_vy[pflow_keep] if pflow_vy is not None else None
                pflow_vz = pflow_vz[pflow_keep] if pflow_vz is not None else None
                pflow_cls = pflow_cls[pflow_keep]
                pflow_mass = pflow_mass[pflow_keep] if pflow_mass is not None else None
                pflow_pdg_out = pflow_pdg[pflow_keep] if pflow_pdg is not None else None
                pflow_d0 = pflow_d0[pflow_keep] if pflow_d0 is not None else None
                pflow_z0 = pflow_z0[pflow_keep] if pflow_z0 is not None else None

                if pflow_d0 is None:
                    pflow_d0 = fill_constant_like(pflow_pt, -999.0)
                if pflow_z0 is None:
                    pflow_z0 = fill_constant_like(pflow_pt, -999.0)

                out_chunk = {
                    "truth_pt": truth_pt,
                    "truth_eta": truth_eta,
                    "truth_phi": truth_phi,
                    "truth_vx": truth_vx,
                    "truth_vy": truth_vy,
                    "truth_vz": truth_vz,
                    "truth_class": ak.values_astype(truth_cls, np.int32),
                    "truth_mass": truth_mass,
                    "truth_pdgId": ak.values_astype(truth_pdg, np.int32),
                    "pflow_pt": pflow_pt,
                    "pflow_eta": pflow_eta,
                    "pflow_phi": pflow_phi,
                    "pflow_vx": pflow_vx if pflow_vx is not None else fill_constant_like(pflow_pt, 0.0),
                    "pflow_vy": pflow_vy if pflow_vy is not None else fill_constant_like(pflow_pt, 0.0),
                    "pflow_vz": pflow_vz if pflow_vz is not None else fill_constant_like(pflow_pt, 0.0),
                    "pflow_class": ak.values_astype(pflow_cls, np.int32),
                    "pflow_mass": pflow_mass if pflow_mass is not None else fill_constant_like(pflow_pt, -999.0),
                    "pflow_pdgId": (
                        ak.values_astype(pflow_pdg_out, np.int32)
                        if pflow_pdg_out is not None
                        else fill_constant_like(pflow_pt, 0, dtype=np.int32)
                    ),
                    "pflow_d0": pflow_d0,
                    "pflow_d0Error": fill_constant_like(pflow_pt, -999.0),
                    "pflow_z0": pflow_z0,
                    "pflow_z0Error": fill_constant_like(pflow_pt, -999.0),
                    "ntruth": ak.values_astype(ak.num(truth_pt, axis=1), np.int32),
                    "npflow": ak.values_astype(ak.num(pflow_pt, axis=1), np.int32),
                    "eventNumber": np.arange(
                        next_event_number, next_event_number + chunk_size, dtype=np.int64
                    ),
                }
                next_event_number += chunk_size

                if first_chunk:
                    output["evt_tree"] = out_chunk
                    first_chunk = False
                else:
                    output["evt_tree"].extend(out_chunk)

                print(
                    f"[convert_delphi] processed {processed} events total",
                    flush=True,
                )

        if args.entry_stop is not None:
            break

    output.close()
    print(f"[convert_delphi] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
