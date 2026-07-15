#!/usr/bin/env python3
"""
Convert an SLD minidst_translated MC parquet file into a training ntuple
(evt_tree, same schema as convert_delphi_edm4hep_to_ntuple.py).

Input is *not* EDM4hep — it's the Jazelle mini-DST banks already translated
1:1 into parquet (see datasets/minidst_translated/parquet/ in the sld data
repo). Only files from MC runs (tagged mc74_96r18 / mcwab_96r14) carry the
MCPART/MCHEAD/MCPNT truth banks needed here; real-data files have no truth
and cannot be used as input to this converter.

Known limitations (see comments inline for reasoning):
 - truth_class / pflow_class use a charge-only fallback (0=charged hadron,
   3=neutral hadron) refined with whatever PID banks are actually available
   (PHCRID likelihoods, PHKELID electron score, PHWIC muon link). There is
   no photon-ID bank in SLD's minidst, so photons are NOT separated from
   neutral hadrons (both land in class 3) until someone builds an EM/hadronic
   shower-shape discriminant from PHKLUS.elayer.
 - MCPART.ptype is SLD/Jazelle's internal particle-type code, not a PDG code.
   No verified decode table for it was available at the time this script was
   written (it should live in the Jazelle manual under
   scanned_documents/SLD_software_users_guide/Jazelle_manual). It is passed
   through as truth_pdgId purely for bookkeeping -- do not treat it as PDG.
 - "Stable final state" truth particles are approximated as MCPART entries
   with no children (id not referenced as any other particle's parent_id),
   excluding JETSET/LUND string-fragmentation bookkeeping codes {91,92,93}
   (and their signed variants), which is a well-established convention of
   the JETSET/PYTHIA generator family and not SLD-specific guesswork. This
   is *not* the same as a generatorStatus==1 flag (SLD's MCPART has none) and
   may still admit other non-terminal bookkeeping entries.
 - Truth neutrinos are not excluded (no verified way to identify them from
   ptype alone yet), unlike the CMS/Delphi converters' TRUTH_NEUTRINOS cut.

Usage:
  python convert_sld_to_ntuple.py <file_or_list.txt> <output.root>

Example (single file):
  python convert_sld_to_ntuple.py \\
    /global/cfs/cdirs/m3246/ylo/sld/datasets/minidst_translated/parquet/qf1569.qf1569.5ne-mc74_96r18_mdst.7b1.parquet \\
    my_sample/sld/train_evt.root

Example (file list, one path per line):
  python convert_sld_to_ntuple.py sld_mc_files.txt my_sample/sld/train_evt.root
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import pyarrow.parquet as pq
import uproot

MAX_ABS_ETA = 2.7
TRUTH_PT_MIN = 0.5
PFLOW_PT_MIN = 1.0
EPS = 1e-9

# JETSET/LUND string-fragmentation bookkeeping pseudo-particles (not physical
# final-state particles even though they can appear as MCPART leaves).
FRAGMENTATION_PTYPES = {91, 92, 93, -91, -92, -93}

# Fill in once the Jazelle ptype -> PDG table has been transcribed from the
# manual. Any code present here overrides the charge-based fallback for
# truth_class. Left empty on purpose -- see module docstring.
PTYPE_CLASS: dict[int, int] = {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert SLD minidst_translated MC parquet into evt_tree ntuples."
    )
    p.add_argument("input", type=Path, help="Input MC parquet file or .txt file list")
    p.add_argument("output", type=Path, help="Output ROOT file (evt_tree)")
    p.add_argument("--chunk-size", type=int, default=20000,
                    help="Events per output write chunk (default: 20000)")
    p.add_argument("--max-events", type=int, default=None,
                    help="Stop after this many input events (for smoke tests)")
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
        raise ValueError(f"No parquet files listed in {input_path}")
    return resolved


def compute_pt_eta_phi(px: ak.Array, py: ak.Array, pz: ak.Array):
    pt = np.sqrt(px * px + py * py)
    p = np.sqrt(px * px + py * py + pz * pz) + EPS
    eta = 0.5 * np.log((p + pz + EPS) / (p - pz + EPS))
    phi = np.arctan2(py, px)
    return pt, eta, phi


def compute_mass(e: ak.Array, px: ak.Array, py: ak.Array, pz: ak.Array) -> ak.Array:
    p2 = px * px + py * py + pz * pz
    m2 = e * e - p2
    return np.sqrt(ak.where(m2 > 0, m2, 0.0))


def truth_final_state_mask(ids: ak.Array, parent_ids: ak.Array, ptype: ak.Array) -> ak.Array:
    """Per event: keep MCPART entries with no children, excluding JETSET
    fragmentation bookkeeping codes. Done with a per-event Python loop
    (same pattern used elsewhere in this repo for id-based linking) since
    awkward has no built-in "is my id referenced elsewhere in this sublist"
    reduction.
    """
    out = []
    for id_list, parent_list, ptype_list in zip(
        ak.to_list(ids), ak.to_list(parent_ids), ak.to_list(ptype)
    ):
        parent_set = set(parent_list)
        out.append(
            np.array(
                [
                    (pid not in parent_set) and (pt not in FRAGMENTATION_PTYPES)
                    for pid, pt in zip(id_list, ptype_list)
                ],
                dtype=bool,
            )
        )
    return ak.Array(out)


def gather_by_id(
    ref_ids: ak.Array, table_ids: ak.Array, table_values: ak.Array, missing: float
) -> ak.Array:
    """For each event, map ref_ids (0 = no link, matching SLD's PHPOINT
    convention) through {table id -> table value} built from a separate,
    independently-indexed bank (e.g. PHPOINT.phcrid_id -> PHCRID.id)."""
    out = []
    for refs, ids, values in zip(
        ak.to_list(ref_ids), ak.to_list(table_ids), ak.to_list(table_values)
    ):
        lut = dict(zip(ids, values))
        out.append(np.array([lut.get(r, missing) if r != 0 else missing for r in refs], dtype=np.float32))
    return ak.Array(out)


def classify_pflow(
    charge: ak.Array,
    kelid_ref: ak.Array, kelid_ids: ak.Array, kelid_prob: ak.Array,
    crid_ref: ak.Array, crid_ids: ak.Array,
    crid_e: ak.Array, crid_mu: ak.Array, crid_pi: ak.Array, crid_k: ak.Array, crid_p: ak.Array,
    wic_ref: ak.Array,
) -> ak.Array:
    """Charge-based default (0=charged hadron, 3=neutral hadron), refined by
    whatever PID banks are linked via PHPOINT:
      - PHCRID (Cherenkov likelihoods): highest of (llik_e, llik_mu, llik_pi,
        llik_k, llik_p) wins -> electron/muon/otherwise unchanged. This is a
        standard likelihood-ratio PID, not a threshold guess.
      - PHKELID.prob (raw int16-range electron score, sign/median-0 convention
        observed in the data -- not documented, treated as a coarse
        electron flag via prob > 0) when CRID didn't already resolve it.
      - PHWIC link presence (muon system) as a coarse muon flag when neither
        of the above resolved it.
    No photon-ID bank exists in this data, so neutrals stay class 3 (see
    module docstring).
    """
    cls = ak.where(charge != 0, ak.zeros_like(charge, dtype=np.int32), ak.full_like(charge, 3, dtype=np.int32))
    cls = ak.values_astype(cls, np.int32)

    has_crid = crid_ref != 0
    e_l = gather_by_id(crid_ref, crid_ids, crid_e, -1e9)
    mu_l = gather_by_id(crid_ref, crid_ids, crid_mu, -1e9)
    pi_l = gather_by_id(crid_ref, crid_ids, crid_pi, -1e9)
    k_l = gather_by_id(crid_ref, crid_ids, crid_k, -1e9)
    p_l = gather_by_id(crid_ref, crid_ids, crid_p, -1e9)
    best_is_e = (e_l >= mu_l) & (e_l >= pi_l) & (e_l >= k_l) & (e_l >= p_l)
    best_is_mu = (mu_l >= e_l) & (mu_l >= pi_l) & (mu_l >= k_l) & (mu_l >= p_l)
    cls = ak.where(has_crid & best_is_e, 1, cls)
    cls = ak.where(has_crid & best_is_mu & ~best_is_e, 2, cls)

    kelid_prob_g = gather_by_id(kelid_ref, kelid_ids, kelid_prob, 0.0)
    unresolved_by_crid = ~has_crid
    cls = ak.where(unresolved_by_crid & (kelid_ref != 0) & (kelid_prob_g > 0), 1, cls)

    still_default_charged = (charge != 0) & (cls == 0)
    cls = ak.where(still_default_charged & unresolved_by_crid & (kelid_ref == 0) & (wic_ref != 0), 2, cls)

    return ak.values_astype(cls, np.int32)


def fill_constant_like(reference: ak.Array, value: float, dtype=np.float32) -> ak.Array:
    counts = ak.to_numpy(ak.num(reference, axis=1))
    return ak.Array([np.full(int(c), value, dtype=dtype) for c in counts])


def process_table(table) -> dict:
    mcpart = table["MCPART"]
    truth_id = mcpart.id
    truth_parent = mcpart.parent_id
    truth_ptype = mcpart.ptype
    truth_charge = mcpart.charge
    truth_px, truth_py, truth_pz, truth_e = mcpart.px, mcpart.py, mcpart.pz, mcpart.e
    truth_vx, truth_vy, truth_vz = mcpart.xt_x, mcpart.xt_y, mcpart.xt_z

    truth_pt, truth_eta, truth_phi = compute_pt_eta_phi(truth_px, truth_py, truth_pz)
    truth_mass = compute_mass(truth_e, truth_px, truth_py, truth_pz)

    fs_mask = truth_final_state_mask(truth_id, truth_parent, truth_ptype)
    eta_abs = np.abs(truth_eta)
    truth_keep = fs_mask & (eta_abs <= MAX_ABS_ETA) & (truth_pt >= TRUTH_PT_MIN)

    truth_cls_default = ak.where(
        truth_charge != 0, ak.zeros_like(truth_charge, dtype=np.int32), ak.full_like(truth_charge, 3, dtype=np.int32)
    )
    if PTYPE_CLASS:
        flat = ak.to_numpy(ak.flatten(truth_ptype, axis=None))
        mapped = np.array([PTYPE_CLASS.get(int(x), -999) for x in flat], dtype=np.int32)
        mapped_ak = ak.unflatten(ak.Array(mapped), ak.num(truth_ptype, axis=1))
        truth_cls = ak.where(mapped_ak != -999, mapped_ak, truth_cls_default)
    else:
        truth_cls = truth_cls_default
    truth_cls = ak.values_astype(truth_cls, np.int32)

    out_truth = {
        "truth_pt": truth_pt[truth_keep],
        "truth_eta": truth_eta[truth_keep],
        "truth_phi": truth_phi[truth_keep],
        "truth_vx": truth_vx[truth_keep],
        "truth_vy": truth_vy[truth_keep],
        "truth_vz": truth_vz[truth_keep],
        "truth_class": truth_cls[truth_keep],
        "truth_mass": truth_mass[truth_keep],
        "truth_pdgId": ak.values_astype(truth_ptype[truth_keep], np.int32),
    }

    phpsum = table["PHPSUM"]
    phpoint = table["PHPOINT"]
    pflow_px, pflow_py, pflow_pz = phpsum.px, phpsum.py, phpsum.pz
    pflow_vx, pflow_vy, pflow_vz = phpsum.x, phpsum.y, phpsum.z
    pflow_charge = phpsum.charge

    pflow_pt, pflow_eta, pflow_phi = compute_pt_eta_phi(pflow_px, pflow_py, pflow_pz)

    phkelid = table["PHKELID"]
    phcrid = table["PHCRID"]

    pflow_cls = classify_pflow(
        pflow_charge,
        phpoint.phkelid_id, phkelid.id, phkelid.prob,
        phpoint.phcrid_id, phcrid.id,
        phcrid.llik_e, phcrid.llik_mu, phcrid.llik_pi, phcrid.llik_k, phcrid.llik_p,
        phpoint.phwic_id,
    )

    pflow_eta_abs = np.abs(pflow_eta)
    pflow_keep = (pflow_eta_abs <= MAX_ABS_ETA) & (pflow_pt >= PFLOW_PT_MIN)

    out_pflow = {
        "pflow_pt": pflow_pt[pflow_keep],
        "pflow_eta": pflow_eta[pflow_keep],
        "pflow_phi": pflow_phi[pflow_keep],
        "pflow_vx": pflow_vx[pflow_keep],
        "pflow_vy": pflow_vy[pflow_keep],
        "pflow_vz": pflow_vz[pflow_keep],
        "pflow_class": pflow_cls[pflow_keep],
    }

    out = {**out_truth, **out_pflow}
    out["ntruth"] = ak.values_astype(ak.num(out["truth_pt"], axis=1), np.int32)
    out["npflow"] = ak.values_astype(ak.num(out["pflow_pt"], axis=1), np.int32)
    return out


def main() -> int:
    args = parse_args()
    input_files = resolve_input_files(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = uproot.recreate(args.output)

    first_chunk = True
    processed = 0
    next_event_number = 0
    needed_columns = [
        "MCPART", "PHPSUM", "PHPOINT", "PHKELID", "PHCRID",
    ]

    for input_path in input_files:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        print(f"[convert_sld] Processing {input_path}", flush=True)

        arrow_table = pq.read_table(input_path, columns=needed_columns)
        n_rows = arrow_table.num_rows
        stop = n_rows if args.max_events is None else min(n_rows, args.max_events - processed)
        if stop <= 0:
            break

        for start in range(0, stop, args.chunk_size):
            end = min(start + args.chunk_size, stop)
            batch = arrow_table.slice(start, end - start)
            events = ak.from_arrow(batch)
            out_chunk = process_table(events)

            chunk_size = len(out_chunk["ntruth"])
            out_chunk["eventNumber"] = np.arange(
                next_event_number, next_event_number + chunk_size, dtype=np.int64
            )
            next_event_number += chunk_size
            processed += chunk_size

            if first_chunk:
                output["evt_tree"] = out_chunk
                first_chunk = False
            else:
                output["evt_tree"].extend(out_chunk)

            print(f"[convert_sld] processed {processed} events total", flush=True)

        if args.max_events is not None and processed >= args.max_events:
            break

    output.close()
    print(f"[convert_sld] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
