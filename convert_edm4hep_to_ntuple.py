#!/usr/bin/env python3
"""
Convert an EDM4hep/PODIO ROOT file into a training ntuple with the branches
listed in configs/evt.yaml (truth_* and pflow_* features).

The script reads the EDM4hep collections, computes cylindrical coordinates and
basic class labels, then writes an evt_tree TTree compatible with the existing
training pipeline.

To run the script:
python convert_edm4hep_to_ntuple.py qqb_train_files.txt my_sample/QQB/train_evt.root --tree events --truth-collection MCParticles --pflow-collection RecoParticles --event-number-branch EventHeader.eventNumber
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
    211: 0,
    -211: 0,
    11: 1,
    -11: 1,
    13: 2,
    -13: 2,
    130: 3,
    310: 3,
    311: 3,
    321: 3,
    -321: 3,
    22: 4,
    12: -1,
    14: -1,
    16: -1,
    -12: -1,
    -14: -1,
    -16: -1,
}
FLOW_TYPE_TO_CLASS = {
    0: 5,   # OTHER -> residual
    1: 0,   # CHARGED
    2: 3,   # NEUTRAL
    3: 4,   # PHOTON
    4: 3,   # LCAL (treat as neutral hadron)
    5: 2,   # MUON
    6: -1,  # NEUTRINO
    7: 0,   # TRACK-like charged object
    8: 3,   # CLUSTER-like neutral object
    9: 0,
    10: 3,
    11: 5,
}
PID_TYPE_TO_CLASS = {
    0: 0,  # charged hadron-like
    1: 1,  # electron-like
    2: 2,  # muon-like
    3: 0,  # secondary charged tracks -> treat as charged hadrons
    4: 4,  # photon-like
    5: 3,  # neutral hadron-like
    6: 3,
    7: 3,
    8: 3,
}
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert EDM4hep ROOT samples into evt_tree ntuples."
    )
    parser.add_argument("input", type=Path, help="Input EDM4hep ROOT file")
    parser.add_argument("output", type=Path, help="Output ROOT file (evt_tree)")
    parser.add_argument(
        "--tree", default="events", help="Input TTree name (default: events)"
    )
    parser.add_argument(
        "--truth-collection",
        default="MCParticles",
        help="EDM4hep collection to treat as truth (default: MCParticles)",
    )
    parser.add_argument(
        "--pflow-collection",
        default="FlowElements",
        help="EDM4hep collection to treat as PF candidates (default: FlowElements)",
    )
    parser.add_argument(
        "--event-number-branch",
        default="EventHeader.eventNumber",
        help="Optional scalar branch to copy as eventNumber if present.",
    )
    parser.add_argument(
        "--entry-start",
        type=int,
        default=0,
        help="Entry index to start reading from (default: 0)",
    )
    parser.add_argument(
        "--entry-stop",
        type=int,
        default=None,
        help="Optional entry index to stop reading (exclusive).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=20000,
        help="Number of events to process per chunk (default: 20000).",
    )
    parser.add_argument(
        "--allow-missing-branches",
        action="store_true",
        help="Skip missing EDM4hep branches instead of failing fast.",
    )
    parser.add_argument(
        "--pid-collection",
        default="ParticleID",
        help="Collection name with ParticleID records to derive PF classes (default: ParticleID).",
    )
    parser.add_argument(
        "--track-collection",
        default="Tracks",
        help="Collection name for tracks linked to RecoParticles (default: Tracks).",
    )
    return parser.parse_args()


def resolve_input_files(input_path: Path) -> list[Path]:
    """Return a list of ROOT files to convert.

    If the input path ends with .txt, treat it as a list of file paths (one per
    line, allowing relative paths and ignoring blank/comment lines). Otherwise,
    assume it is a single ROOT file.
    """

    if input_path.suffix.lower() != ".txt":
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    resolved: list[Path] = []
    base_dir = input_path.parent
    with input_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = (base_dir / candidate).resolve()
            resolved.append(candidate)

    if not resolved:
        raise ValueError(f"No ROOT files listed in {input_path}")

    return resolved


def normalize_keys(keys: Iterable[str]) -> list[str]:
    normed = []
    for key in keys:
        if isinstance(key, bytes):
            key = key.decode()
        normed.append(key)
    return normed


class BranchResolver:
    """Resolve EDM4hep branch names even when PODIO inserts helper tokens."""

    def __init__(self, keys: Iterable[str]):
        self.keys = list(keys)
        self.key_set = set(self.keys)

    def scalar(self, collection: str, field: str) -> str:
        suffix = field
        return self._resolve(collection, suffix)

    def vector(self, collection: str, field: str, axis: str) -> str:
        suffix = f"{field}.{axis}"
        return self._resolve(collection, suffix)

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
        prefix_variants = [collection, f"{collection}#"]
        for cand in list(self.key_set):
            for prefix in prefix_variants:
                if cand.startswith(prefix) and cand.endswith(suffix):
                    return cand
        raise KeyError(f"Could not find branch for {collection}.{suffix}")


def compute_pt_eta_phi(px: ak.Array, py: ak.Array, pz: ak.Array) -> tuple[ak.Array, ak.Array, ak.Array]:
    pt = np.sqrt(px * px + py * py)
    p = np.sqrt(px * px + py * py + pz * pz) + EPS
    eta = 0.5 * np.log((p + pz + EPS) / (p - pz + EPS))
    phi = np.arctan2(py, px)
    return pt, eta, phi


def map_pdg_classes(pdg: ak.Array | None) -> ak.Array | None:
    if pdg is None:
        return None
    flat = ak.to_numpy(ak.flatten(pdg, axis=None))
    mapper = np.vectorize(lambda x: PDGID_CLASS.get(int(x), 5), otypes=[np.int32])
    mapped = mapper(flat)
    counts = ak.to_numpy(ak.num(pdg, axis=1))
    if mapped.size == 0:
        return ak.values_astype(ak.zeros_like(pdg), np.int32)
    return ak.unflatten(mapped, counts)


def classify_truth(pdg: ak.Array) -> ak.Array:
    classes = map_pdg_classes(pdg)
    if classes is None:
        raise RuntimeError("truth PDG IDs are required for classification")
    return classes


def classify_pflow(
    flow_type: ak.Array | None,
    charge: ak.Array | None,
    pdg: ak.Array | None,
    pid_types: ak.Array | None = None,
) -> ak.Array:
    if pid_types is not None:
        flat = ak.to_numpy(ak.flatten(pid_types, axis=None))
        if flat.size == 0:
            return ak.values_astype(pid_types, np.int32)
        mapper = np.vectorize(lambda x: PID_TYPE_TO_CLASS.get(int(x), 3), otypes=[np.int32])
        mapped = mapper(flat)
        counts = ak.to_numpy(ak.num(pid_types, axis=1))
        return ak.unflatten(mapped, counts)
    if pdg is not None:
        classes = map_pdg_classes(pdg)
        if classes is not None:
            return classes
    if flow_type is None:
        base = charge if charge is not None else ak.Array(np.array([], dtype=np.int32))
        cls = ak.full_like(base, 5, dtype=np.int32)
        return cls

    cls = ak.full_like(flow_type, 5, dtype=np.int32)
    for type_id, label in FLOW_TYPE_TO_CLASS.items():
        cls = ak.where(flow_type == type_id, label, cls)

    if charge is not None:
        cls = ak.where(charge == 0, ak.where(cls == 5, 3, cls), cls)
        cls = ak.where(charge != 0, ak.where(cls == 5, 0, cls), cls)

    return ak.values_astype(cls, np.int32)


def chunk_event_count(chunk: ak.Array | dict[str, ak.Array]) -> int:
    if isinstance(chunk, ak.Array):
        return len(chunk)
    for array in chunk.values():
        try:
            return len(array)
        except TypeError:
            continue
    raise RuntimeError("Chunk did not contain any array branches to determine event count.")


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


def fill_constant_like(reference: ak.Array, value: float, dtype=np.float32) -> ak.Array:
    counts = ak.to_numpy(ak.num(reference, axis=1))
    return ak.Array([np.full(int(count), value, dtype=dtype) for count in counts])


def build_branch_map(
    tree_keys: list[str],
    truth_collection: str,
    pflow_collection: str,
    pid_collection: str,
    track_collection: str,
    allow_missing: bool,
) -> dict[str, str | None]:
    resolver = BranchResolver(tree_keys)
    branches: dict[str, str | None] = {}
    key_set = set(tree_keys)

    def optional(func, *args, allow: bool | None = None) -> str | None:
        flag = allow_missing if allow is None else allow
        try:
            return func(*args)
        except KeyError:
            if flag:
                return None
            raise

    branches["truth_px"] = optional(resolver.vector, truth_collection, "momentum", "x")
    branches["truth_py"] = optional(resolver.vector, truth_collection, "momentum", "y")
    branches["truth_pz"] = optional(resolver.vector, truth_collection, "momentum", "z")
    branches["truth_vx"] = optional(resolver.vector, truth_collection, "vertex", "x")
    branches["truth_vy"] = optional(resolver.vector, truth_collection, "vertex", "y")
    branches["truth_vz"] = optional(resolver.vector, truth_collection, "vertex", "z")
    branches["truth_pdg"] = optional(resolver.scalar, truth_collection, "PDG")
    branches["truth_charge"] = optional(resolver.scalar, truth_collection, "charge")
    branches["truth_mass"] = optional(resolver.scalar, truth_collection, "mass", allow=True)

    branches["pflow_px"] = optional(resolver.vector, pflow_collection, "momentum", "x")
    if branches["pflow_px"] is None:
        branches["pflow_px"] = optional(resolver.vector, pflow_collection, "p4", "px")
    branches["pflow_py"] = optional(resolver.vector, pflow_collection, "momentum", "y")
    if branches["pflow_py"] is None:
        branches["pflow_py"] = optional(resolver.vector, pflow_collection, "p4", "py")
    branches["pflow_pz"] = optional(resolver.vector, pflow_collection, "momentum", "z")
    if branches["pflow_pz"] is None:
        branches["pflow_pz"] = optional(resolver.vector, pflow_collection, "p4", "pz")
    branches["pflow_vx"] = optional(resolver.vector, pflow_collection, "referencePoint", "x")
    branches["pflow_vy"] = optional(resolver.vector, pflow_collection, "referencePoint", "y")
    branches["pflow_vz"] = optional(resolver.vector, pflow_collection, "referencePoint", "z")
    branches["pflow_charge"] = optional(resolver.scalar, pflow_collection, "charge", allow=True)
    branches["pflow_pdg"] = optional(resolver.scalar, pflow_collection, "PDG", allow=True)
    branches["pflow_type"] = optional(resolver.scalar, pflow_collection, "type", allow=True)
    branches["pflow_mass"] = optional(resolver.scalar, pflow_collection, "mass", allow=True)
    tracks_direct = f"_{pflow_collection}_tracks/_{pflow_collection}_tracks.index"
    branches["pflow_tracks"] = tracks_direct if tracks_direct in key_set else None
    begin_key = f"{pflow_collection}/{pflow_collection}.tracks_begin"
    end_key = f"{pflow_collection}/{pflow_collection}.tracks_end"
    branches["pflow_tracks_begin"] = begin_key if begin_key in key_set else None
    branches["pflow_tracks_end"] = end_key if end_key in key_set else None

    pid_prefix = pid_collection.rstrip("/")
    pid_type_key = f"{pid_prefix}/{pid_prefix}.type"
    pid_index_key = f"_{pid_prefix}_particle/_{pid_prefix}_particle.index"
    branches["pid_type"] = pid_type_key if pid_type_key in key_set else None
    branches["pid_index"] = pid_index_key if pid_index_key in key_set else None

    track_prefix = track_collection.rstrip("/")
    track_state_prefix = f"_{track_prefix}_trackStates/_{track_prefix}_trackStates"
    track_d0_key = f"{track_state_prefix}.D0"
    track_z0_key = f"{track_state_prefix}.Z0"
    branches["track_d0"] = track_d0_key if track_d0_key in key_set else None
    branches["track_z0"] = track_z0_key if track_z0_key in key_set else None

    return branches


def main() -> int:
    args = parse_args()

    input_files = resolve_input_files(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = uproot.recreate(args.output)
    first_chunk = True
    processed = 0
    next_event_number = args.entry_start or 0

    for file_idx, input_path in enumerate(input_files):
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        print(f"[convert_edm4hep] Processing {input_path}", flush=True)
        with uproot.open(input_path) as infile:
            if args.tree not in infile:
                raise KeyError(f"TTree '{args.tree}' not found in {input_path}")
            tree = infile[args.tree]
            keys = normalize_keys(tree.keys())

            branch_map = build_branch_map(
                keys,
                args.truth_collection,
                args.pflow_collection,
                args.pid_collection,
                args.track_collection,
                args.allow_missing_branches,
            )
            required_labels = [
                "truth_px",
                "truth_py",
                "truth_pz",
                "truth_vx",
                "truth_vy",
                "truth_vz",
                "truth_pdg",
                "pflow_px",
                "pflow_py",
                "pflow_pz",
                "pflow_vx",
                "pflow_vy",
                "pflow_vz",
            ]
            missing = [label for label in required_labels if branch_map.get(label) is None]
            if missing:
                raise RuntimeError(
                    "Missing required EDM4hep branches: "
                    + ", ".join(missing)
                    + ". Rerun with --allow-missing-branches only for optional fields "
                    "(charges/types) or specify the correct collection names."
                )
            if (
                branch_map.get("pflow_type") is None
                and branch_map.get("pflow_charge") is None
                and branch_map.get("pflow_pdg") is None
                and branch_map.get("pid_type") is None
            ):
                raise RuntimeError(
                    "Cannot determine pflow_class without PID types, FlowElement type, charge, or PDG. "
                    "Please provide at least one of them."
                )

            if args.event_number_branch and args.event_number_branch in keys:
                available_event_branch = args.event_number_branch
            else:
                available_event_branch = None
                if args.event_number_branch:
                    print(
                        f"[convert_edm4hep] Branch '{args.event_number_branch}' not found; "
                        "event numbers will be synthesized.",
                        flush=True,
                    )

            required_branches = sorted(
                {
                    name
                    for key, name in branch_map.items()
                    if name is not None
                }
            )
            if available_event_branch:
                required_branches.append(available_event_branch)

            if not required_branches:
                raise RuntimeError("No EDM4hep branches resolved; nothing to convert.")

            total_entries = tree.num_entries
            if file_idx == 0:
                start = args.entry_start or 0
                stop = args.entry_stop if args.entry_stop is not None else total_entries
            else:
                start = 0
                stop = total_entries

            if start >= stop:
                continue

            chunk_iter = tree.iterate(
                required_branches,
                library="ak",
                step_size=args.chunk_size,
                entry_start=start,
                entry_stop=stop,
            )

            for chunk in chunk_iter:
                chunk_size = chunk_event_count(chunk)
                processed += chunk_size

                truth_px = chunk[branch_map["truth_px"]] if branch_map["truth_px"] else None
                truth_py = chunk[branch_map["truth_py"]] if branch_map["truth_py"] else None
                truth_pz = chunk[branch_map["truth_pz"]] if branch_map["truth_pz"] else None
                truth_vx = chunk[branch_map["truth_vx"]] if branch_map["truth_vx"] else None
                truth_vy = chunk[branch_map["truth_vy"]] if branch_map["truth_vy"] else None
                truth_vz = chunk[branch_map["truth_vz"]] if branch_map["truth_vz"] else None
                truth_pdg = chunk[branch_map["truth_pdg"]] if branch_map["truth_pdg"] else None
                truth_charge = (
                    chunk[branch_map["truth_charge"]] if branch_map["truth_charge"] else None
                )
                truth_mass = (
                    chunk[branch_map["truth_mass"]] if branch_map.get("truth_mass") else None
                )

                pflow_px = chunk[branch_map["pflow_px"]] if branch_map["pflow_px"] else None
                pflow_py = chunk[branch_map["pflow_py"]] if branch_map["pflow_py"] else None
                pflow_pz = chunk[branch_map["pflow_pz"]] if branch_map["pflow_pz"] else None
                pflow_vx = chunk[branch_map["pflow_vx"]] if branch_map["pflow_vx"] else None
                pflow_vy = chunk[branch_map["pflow_vy"]] if branch_map["pflow_vy"] else None
                pflow_vz = chunk[branch_map["pflow_vz"]] if branch_map["pflow_vz"] else None
                pflow_charge = (
                    chunk[branch_map["pflow_charge"]] if branch_map["pflow_charge"] else None
                )
                pflow_pdg = chunk[branch_map["pflow_pdg"]] if branch_map["pflow_pdg"] else None
                pflow_type = (
                    chunk[branch_map["pflow_type"]] if branch_map["pflow_type"] else None
                )
                pflow_mass = (
                    chunk[branch_map["pflow_mass"]] if branch_map.get("pflow_mass") else None
                )
                pid_types = (
                    chunk[branch_map["pid_type"]] if branch_map.get("pid_type") else None
                )
                pid_indices = (
                    chunk[branch_map["pid_index"]] if branch_map.get("pid_index") else None
                )
                track_indices = (
                    chunk[branch_map["pflow_tracks"]] if branch_map.get("pflow_tracks") else None
                )
                track_begin = (
                    chunk[branch_map["pflow_tracks_begin"]]
                    if branch_map.get("pflow_tracks_begin")
                    else None
                )
                track_end = (
                    chunk[branch_map["pflow_tracks_end"]]
                    if branch_map.get("pflow_tracks_end")
                    else None
                )
                track_d0 = chunk[branch_map["track_d0"]] if branch_map.get("track_d0") else None
                track_z0 = chunk[branch_map["track_z0"]] if branch_map.get("track_z0") else None

                truth_pt, truth_eta, truth_phi = compute_pt_eta_phi(truth_px, truth_py, truth_pz)
                pflow_pt, pflow_eta, pflow_phi = compute_pt_eta_phi(pflow_px, pflow_py, pflow_pz)
                if first_chunk:
                    print("[convert_edm4hep][debug] track branch keys:", branch_map.get("pflow_tracks"), branch_map.get("pflow_tracks_begin"), branch_map.get("track_d0"))

                truth_cls = classify_truth(truth_pdg)
                flat_truth = ak.flatten(truth_pdg, axis=None)
                neutrino_mask_flat = np.isin(
                    np.abs(ak.to_numpy(flat_truth)), list(TRUTH_NEUTRINOS)
                )
                neutrino_mask = ak.Array(neutrino_mask_flat)
                neutrino_mask = ak.unflatten(neutrino_mask, ak.num(truth_pdg, axis=1))
                truth_eta_flat = ak.flatten(truth_eta, axis=None)
                truth_eta_abs = ak.Array(np.abs(ak.to_numpy(truth_eta_flat)))
                truth_eta_abs = ak.unflatten(truth_eta_abs, ak.num(truth_eta, axis=1))
                truth_keep = (
                    (truth_eta_abs <= MAX_ABS_ETA)
                    & (~neutrino_mask)
                    & (truth_cls != 5)
                    & (truth_pt >= TRUTH_PT_MIN)
                )
                truth_pt = truth_pt[truth_keep]
                truth_eta = truth_eta[truth_keep]
                truth_phi = truth_phi[truth_keep]
                truth_vx = truth_vx[truth_keep] if truth_vx is not None else None
                truth_vy = truth_vy[truth_keep] if truth_vy is not None else None
                truth_vz = truth_vz[truth_keep] if truth_vz is not None else None
                truth_cls = truth_cls[truth_keep]
                truth_mass = truth_mass[truth_keep] if truth_mass is not None else None
                truth_pdg = truth_pdg[truth_keep] if truth_pdg is not None else None

                if truth_mass is None:
                    truth_mass = fill_constant_like(truth_pt, -999.0)
                if truth_pdg is None:
                    truth_pdg = fill_constant_like(truth_pt, 0, dtype=np.int32)

                pid_class = None
                if pid_types is not None and pid_indices is not None:
                    sequential = ak.local_index(pid_indices, axis=1)
                    if ak.all(pid_indices == sequential):
                        pid_class = ak.values_astype(pid_types, np.int32)
                    else:
                        pid_class = ak.Array(
                            [
                                np.array([types[idx] for idx in indices], dtype=np.int32)
                                for types, indices in zip(pid_types, pid_indices)
                            ]
                        )
                else:
                    pid_class = None

                pflow_cls = classify_pflow(pflow_type, pflow_charge, pflow_pdg, pid_class)
                pflow_eta_flat = ak.flatten(pflow_eta, axis=None)
                pflow_eta_abs = ak.Array(np.abs(ak.to_numpy(pflow_eta_flat)))
                pflow_eta_abs = ak.unflatten(pflow_eta_abs, ak.num(pflow_eta, axis=1))
                pflow_d0 = None
                pflow_z0 = None
                if (
                    track_d0 is not None
                    and track_z0 is not None
                    and track_indices is not None
                    and track_begin is not None
                    and track_end is not None
                ):
                    pflow_d0 = gather_first_track_param(
                        track_d0, track_indices, track_begin, track_end
                    )
                    pflow_z0 = gather_first_track_param(
                        track_z0, track_indices, track_begin, track_end
                    )
                    if first_chunk:
                        print("[convert_edm4hep][debug] sample d0", ak.to_list(pflow_d0[0][:5]))

                pflow_keep = (pflow_eta_abs <= MAX_ABS_ETA) & (pflow_pt >= PFLOW_PT_MIN)
                pflow_pt = pflow_pt[pflow_keep]
                pflow_eta = pflow_eta[pflow_keep]
                pflow_phi = pflow_phi[pflow_keep]
                pflow_vx = pflow_vx[pflow_keep] if pflow_vx is not None else None
                pflow_vy = pflow_vy[pflow_keep] if pflow_vy is not None else None
                pflow_vz = pflow_vz[pflow_keep] if pflow_vz is not None else None
                pflow_cls = pflow_cls[pflow_keep]
                pflow_d0 = pflow_d0[pflow_keep] if pflow_d0 is not None else None
                pflow_z0 = pflow_z0[pflow_keep] if pflow_z0 is not None else None
                pflow_mass = pflow_mass[pflow_keep] if pflow_mass is not None else None
                pflow_pdg = pflow_pdg[pflow_keep] if pflow_pdg is not None else None

                if pflow_d0 is None:
                    pflow_d0 = fill_constant_like(pflow_pt, -999.0)
                if pflow_z0 is None:
                    pflow_z0 = fill_constant_like(pflow_pt, -999.0)
                pflow_d0_err = fill_constant_like(pflow_pt, -999.0)
                pflow_z0_err = fill_constant_like(pflow_pt, -999.0)

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
                    "pflow_vx": pflow_vx,
                    "pflow_vy": pflow_vy,
                    "pflow_vz": pflow_vz,
                    "pflow_class": ak.values_astype(pflow_cls, np.int32),
                    "pflow_mass": pflow_mass if pflow_mass is not None else fill_constant_like(pflow_pt, -999.0),
                    "pflow_pdgId": ak.values_astype(pflow_pdg, np.int32)
                    if pflow_pdg is not None
                    else fill_constant_like(pflow_pt, 0, dtype=np.int32),
                    "pflow_d0": pflow_d0,
                    "pflow_d0Error": pflow_d0_err,
                    "pflow_z0": pflow_z0,
                    "pflow_z0Error": pflow_z0_err,
                    "ntruth": ak.values_astype(ak.num(truth_pt, axis=1), np.int32),
                    "npflow": ak.values_astype(ak.num(pflow_pt, axis=1), np.int32),
                }

                if available_event_branch and available_event_branch in chunk:
                    out_chunk["eventNumber"] = chunk[available_event_branch]
                else:
                    out_chunk["eventNumber"] = np.arange(
                        next_event_number, next_event_number + chunk_size, dtype=np.int64
                    )
                    next_event_number += chunk_size

                out_chunk = {k: v for k, v in out_chunk.items() if v is not None}

                if first_chunk:
                    output["evt_tree"] = out_chunk
                    first_chunk = False
                else:
                    output["evt_tree"].extend(out_chunk)

                print(
                    f"[convert_edm4hep] processed {processed} events total",
                    flush=True,
                )

        if args.entry_stop is not None:
            break

    output.close()
    print(f"[convert_edm4hep] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
