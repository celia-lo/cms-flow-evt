import argparse

import numpy as np
import fastjet as fj
import awkward as ak

from tqdm import tqdm
from itertools import product

import multiprocessing as mp

import uproot
import sys
import os
from contextlib import contextmanager

from parnassus_core.utils.jet_helper import Jet, get_cluster_sequence
from parnassus_core.utils.matching import match_objects, reshape_phi

###  0: charged hadrons
###  1: electrons
###  2: muons
###  3: neutral hadrons
###  4: photons
###  5: residual
### -1: neutrinos


PARTICLE_VARS = ["pt", "eta", "phi", "vx", "vy", "vz", "class"]
PARTICLE_IP_VARS = ["d0", "d0Error", "z0", "z0Error"]

JET_VARS = ["pt", "eta", "phi", "d2", "c2"]

EVENT_VARS = ["eventNumber"]

MATCHING_VARS = [
    "tr_idx",
    "pf_idx",
]

CATEGORY_MASKS = {
    "all": lambda x: x > -1000,
    "jet": lambda x: x > -1,
    "bkg": lambda x: x == -1,
}

parser = argparse.ArgumentParser(description="Preprocess full event files")
parser.add_argument("-i", "--input", type=str, required=False, help="Input file")
parser.add_argument(
    "-d", "--data", type=str, required=True, help="Input test data file"
)
parser.add_argument(
    "-o", "--output", type=str, help="Output filename", default="out.root"
)
parser.add_argument("-es", "--entry_start", type=int, help="Starting event", default=0)
parser.add_argument(
    "-n", "--n_events", type=int, help="Number of events to process", default=100
)
parser.add_argument("-dr", type=float, help="Delta R for jet clustering", default=0.5)
parser.add_argument("-df", action="store_true", help="Treat as diffusion file")
parser.add_argument("-dl", action="store_true", help="Treat as Delphes file")
parser.add_argument("--eta", type=float, help="Eta cut", default=3.0)
parser.add_argument("--ip", help="Use impact parameters", action="store_true")
parser.add_argument(
    "-m", "--matching", action="store_true", help="Enable particle matching"
)
parser.add_argument(
    "--dr_cut",
    type=float,
    help="Delta R cut for particle matching",
    default=0.6,
)


def init_out_data_dict(ip=False, matching=False):
    out_data_evt = {
        f"{name}_{var}": []
        for name, var in product(
            ["pflow", "truth", "fastsim"], PARTICLE_VARS + ["jet_idx"]
        )
    }

    particle_vars = PARTICLE_VARS + ["jet_idx"]
    event_vars = ["eventNumber"]

    if ip:
        particle_vars.extend(PARTICLE_IP_VARS)
        for var in PARTICLE_IP_VARS:
            out_data_evt[f"pflow_{var}"] = []
            out_data_evt[f"fastsim_{var}"] = []

    if matching:
        event_vars.extend(
            [
                f"{collection}_hung_cost_{category}"
                for collection, category in product(
                    ["fastsim", "pflow"], ["all", "jet", "bkg"]
                )
            ]
        )
        for category, var in product(["all", "jet", "bkg"], MATCHING_VARS):
            out_data_evt[f"fastsim_{var}_{category}"] = []
            out_data_evt[f"pflow_{var}_{category}"] = []
            particle_vars.append(f"{var}_{category}")

    for var in event_vars:
        out_data_evt[var] = []

    for name, var in product(["pflow", "truth", "fastsim"], JET_VARS):
        out_data_evt[f"{name}_jet_{var}"] = []

    return out_data_evt, particle_vars, event_vars


def load_file(filename, entry_start=0, n_events=None, df=False, ip=False):
    truth_data = {var: None for var in PARTICLE_VARS}
    pflow_data = {var: None for var in PARTICLE_VARS + (PARTICLE_IP_VARS if ip else [])}
    with uproot.open(filename, num_workers=4) as file:
        tree = file["evt_tree"]
        tree_keys = set(tree.keys())
        if n_events is None:
            n_events = tree.num_entries
        varlist_ = PARTICLE_VARS
        if "truth_ind" in tree_keys:
            varlist_ = PARTICLE_VARS + ["ind"]
        ip_vars = PARTICLE_IP_VARS if ip else []
        truth_vars = [f"truth_{var}" for var in varlist_]
        pflow_prefix = "fastsim" if df else "pflow"
        pflow_var_template = varlist_ + ip_vars

        def prefixed_vars(prefix):
            return [f"{prefix}_{var}" for var in pflow_var_template]

        pflow_vars = prefixed_vars(pflow_prefix)
        missing = [var for var in pflow_vars if var not in tree_keys]
        if missing:
            if df:
                fallback_prefix = "pflow"
                fallback_vars = prefixed_vars(fallback_prefix)
                if all(var in tree_keys for var in fallback_vars):
                    print(
                        "Warning: fastsim_* branches not found, "
                        "falling back to pflow_* branches."
                    )
                    pflow_prefix = fallback_prefix
                    pflow_vars = fallback_vars
                    missing = []
            if missing:
                missing_str = ", ".join(missing)
                raise KeyError(
                    f"Missing branches for prefix '{pflow_prefix}': {missing_str}"
                )
        arrays = tree.arrays(
            truth_vars + pflow_vars + ["eventNumber"],
            library="np",
            entry_stop=n_events + entry_start,
            entry_start=entry_start,
        )
        for var in truth_vars:
            truth_data[var.split("_")[-1]] = arrays[var]
        for var in pflow_vars:
            pflow_data[var.split("_")[-1]] = arrays[var]
        event_number = arrays["eventNumber"]
        truth_data["eventNumber"] = event_number
        pflow_data["eventNumber"] = event_number
    return truth_data, pflow_data, event_number


def to_ak(pt, eta, phi):
    return ak.Array(
        {
            "px": pt * np.cos(phi),
            "py": pt * np.sin(phi),
            "pz": pt * np.sinh(eta),
            "E": pt * np.cosh(eta),
        },
        with_name="Momentum4D",
    )


def find_repeats(arr):
    return {el: np.argwhere(el == arr).flatten() for el in np.unique(arr)}


def cluster_jets(pt, eta, phi, jetdef, ptmin=20):
    particles = to_ak(pt, eta, phi)
    cs = get_cluster_sequence(
        jetdef, particles, user_indices=list(range(len(particles)))
    )
    jets = cs.inclusive_jets(ptmin=ptmin)
    jets = fj.sorted_by_pt(jets)
    jets = [Jet(j, 0.5, calc_substructure=True) for j in jets]
    jets = [j for j in jets if j.nconstituents >= 2]

    used_indices = set()

    jet_idxs = np.zeros(len(pt), dtype=int)
    for jet_idx, jet in enumerate(jets):
        particle_idx = jet.constituents_idx
        jet_idxs[particle_idx] = jet_idx
        used_indices.update(particle_idx)
    particle_idx = np.arange(len(pt))
    particle_idx = particle_idx[~np.isin(particle_idx, list(used_indices))]
    jet_idxs[particle_idx] = -1

    return jets, jet_idxs


def match_event(
    truth_data,
    fs_evt_data,
    pflow_data,
    truth_jet_idx,
    pflow_jet_idx,
    fastsim_jet_idx,
    dr_cut=0.6,
):
    out_dict = dict()
    tr_eta = truth_data["eta"]
    tr_phi = reshape_phi(truth_data["phi"])

    fs_eta = fs_evt_data["eta"]
    fs_phi = reshape_phi(fs_evt_data["phi"])
    fs_len = len(fs_eta)

    pf_eta = pflow_data["eta"]
    pf_phi = reshape_phi(pflow_data["phi"])
    pf_len = len(pf_eta)
    for category, mask in CATEGORY_MASKS.items():
        tr_mask = mask(truth_jet_idx)
        fs_mask = mask(fastsim_jet_idx)
        pf_mask = mask(pflow_jet_idx)
        fs_input_idx, fs_truth_idx, fs_hung_cost = match_objects(
            fs_eta[fs_mask],
            fs_phi[fs_mask],
            tr_eta[tr_mask],
            tr_phi[tr_mask],
            dr_cut=dr_cut,
            pad_to_len=fs_len,
        )
        pf_input_idx, pf_truth_idx, pf_hung_cost = match_objects(
            pf_eta[pf_mask],
            pf_phi[pf_mask],
            tr_eta[tr_mask],
            tr_phi[tr_mask],
            dr_cut=dr_cut,
            pad_to_len=pf_len,
        )
        out_dict[f"fastsim_hung_cost_{category}"] = fs_hung_cost
        out_dict[f"fastsim_tr_idx_{category}"] = fs_truth_idx
        out_dict[f"fastsim_pf_idx_{category}"] = fs_input_idx

        out_dict[f"pflow_hung_cost_{category}"] = pf_hung_cost
        out_dict[f"pflow_tr_idx_{category}"] = pf_truth_idx
        out_dict[f"pflow_pf_idx_{category}"] = pf_input_idx
    return out_dict


def process_events(
    batch_event_numbers,
    evt_truth_data,
    evt_pflow_data,
    fs_data,
    dl_flag=False,
    dr=0.5,
    eta_cut=3.0,
    ip=False,
    matching=False,
    dr_cut=0.6,
):
    jetdef = fj.JetDefinition(fj.antikt_algorithm, dr)

    out_data_evt, _, _ = init_out_data_dict(ip=ip, matching=matching)

    for i in range(len(batch_event_numbers)):
        truth_data = {
            var: evt_truth_data[var][i]
            for var in evt_truth_data.keys()
            if var != "eventNumber"
        }
        pflow_data = {
            var: evt_pflow_data[var][i]
            for var in evt_pflow_data.keys()
            if var != "eventNumber"
        }
        truth_event_number = evt_truth_data["eventNumber"][i]
        pflow_event_number = evt_pflow_data["eventNumber"][i]
        if truth_event_number != pflow_event_number:
            raise ValueError("Event numbers do not match")

        if not dl_flag:
            fs_ind_arr = fs_data.get("ind")
            if fs_ind_arr is not None:
                fs_ind = fs_ind_arr[i].astype(bool)
            else:
                fs_ind = np.ones_like(fs_data["pt"][i], dtype=bool)
            fs_evt_data = {
                var: fs_data[var][i][fs_ind]
                for var in fs_data.keys()
                if var != "eventNumber"
            }
            fs_data["pt"] = fs_data["pt"]  # / 1000
        else:
            fs_evt_data = {
                var: fs_data[var][i] for var in fs_data.keys() if var != "eventNumber"
            }

        fs_evt_mask = (
            (fs_evt_data["pt"] > 0)
            & (fs_evt_data["pt"] < 1e4)
            & (np.abs(fs_evt_data["eta"]) <= eta_cut)
        )
        for key, val in fs_evt_data.items():
            fs_evt_data[key] = val[fs_evt_mask]
        pf_evt_mask = np.abs(pflow_data["eta"]) <= eta_cut
        for key, val in pflow_data.items():
            pflow_data[key] = val[pf_evt_mask]

        # Jet clustering
        pf_jets, pf_jet_indices = cluster_jets(
            pflow_data["pt"], pflow_data["eta"], pflow_data["phi"], jetdef
        )
        tr_jets, tr_jet_indices = cluster_jets(
            truth_data["pt"], truth_data["eta"], truth_data["phi"], jetdef
        )
        fs_jets, fs_jet_indices = cluster_jets(
            fs_evt_data["pt"], fs_evt_data["eta"], fs_evt_data["phi"], jetdef
        )

        for key in truth_data.keys():
            out_data_evt[f"truth_{key}"].append(truth_data[key])
        for key in pflow_data.keys():
            out_data_evt[f"pflow_{key}"].append(pflow_data[key])
            out_data_evt[f"fastsim_{key}"].append(fs_evt_data[key])
        out_data_evt["truth_jet_idx"].append(tr_jet_indices)
        out_data_evt["pflow_jet_idx"].append(pf_jet_indices)
        out_data_evt["fastsim_jet_idx"].append(fs_jet_indices)

        for name, jet_collection in zip(
            ["pflow", "truth", "fastsim"], [pf_jets, tr_jets, fs_jets]
        ):
            out_data_evt[f"{name}_jet_pt"].append(
                np.array([jet.pt() for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_eta"].append(
                np.array([jet.eta() for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_phi"].append(
                np.array([jet.phi() for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_d2"].append(
                np.array([jet.substructure["d2"] for jet in jet_collection])
            )
            out_data_evt[f"{name}_jet_c2"].append(
                np.array([jet.substructure["c2"] for jet in jet_collection])
            )

        if matching:
            matched_data = match_event(
                truth_data,
                fs_evt_data,
                pflow_data,
                tr_jet_indices,
                pf_jet_indices,
                fs_jet_indices,
                dr_cut=dr_cut,
            )
            for key, val in matched_data.items():
                out_data_evt[key].append(val)
        out_data_evt["eventNumber"].append(truth_event_number)
    return out_data_evt


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    """
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(to, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def process_events_wrapper(args):
    with stdout_redirected():
        return process_events(*args)


def extract_batch_data(
    batch_event_numbers,
    evt_event_numbers,
    fs_event_numbers,
    evt_truth_data,
    evt_pflow_data,
    fs_data,
):
    fs_idx = np.isin(fs_event_numbers, batch_event_numbers)
    evt_idx = np.isin(evt_event_numbers, batch_event_numbers)
    evt_truth_data = {key: val[evt_idx] for key, val in evt_truth_data.items()}
    evt_pflow_data = {key: val[evt_idx] for key, val in evt_pflow_data.items()}
    fs_data = {key: val[fs_idx] for key, val in fs_data.items()}
    return evt_truth_data, evt_pflow_data, fs_data


def main():
    args = parser.parse_args()

    print(f"Loading event file: {args.input.split('/')[-1]}")
    _, fs_data, fs_event_number = load_file(
        args.input, n_events=args.n_events, df=args.df, ip=args.ip
    )
    print(f"Loading data file: {'/'.join(args.data.split('/')[-2:])}")
    evt_truth_data, evt_pflow_data, evt_event_number = load_file(
        args.data, n_events=args.n_events, ip=args.ip
    )

    print("All variables loaded")

    out_data_evt, particle_vars, event_vars = init_out_data_dict(
        ip=args.ip, matching=args.matching
    )

    goodEventNumbers = np.intersect1d(evt_event_number, fs_event_number)

    if args.n_events > len(goodEventNumbers) or args.n_events == -1:
        args.n_events = len(goodEventNumbers)
    if len(goodEventNumbers) == 0:
        print(
            "ERROR: Event numbers do not match in the input files,"
            " are you sure you are using the same data file as for evaluation?\n"
            " Exiting..."
        )
        sys.exit(1)
    print(
        f"Having {len(goodEventNumbers)} good events, processing {args.n_events} of them"
    )

    batch_size = 1000
    n_batches = args.n_events // batch_size
    n_batches += 1 if args.n_events % batch_size != 0 else 0

    input_batches = np.array_split(goodEventNumbers[: args.n_events], n_batches)
    input_batched_data = [
        (
            batch,
            *extract_batch_data(
                batch,
                evt_event_number,
                fs_event_number,
                evt_truth_data,
                evt_pflow_data,
                fs_data,
            ),
            args.dl,
            args.dr,
            args.eta,
            args.ip,
            args.matching,
            args.dr_cut,
        )
        for batch in input_batches
    ]
    with mp.Pool(processes=20) as pool:
        results = list(
            tqdm(
                pool.imap(process_events_wrapper, input_batched_data),
                total=n_batches,
            )
        )

    for result in results:
        for key, val in result.items():
            out_data_evt[key].extend(val)

    output_path = f"{args.output.replace('.root', '')}_{args.n_events // 1000}k_test_eta{int(args.eta * 10)}Eval.root"
    with uproot.recreate(output_path) as f:
        # Convert to awkward arrays first

        # Write in batches
        write_batch_size = 1000
        total_events = len(out_data_evt[next(iter(out_data_evt.keys()))])

        for start_idx in tqdm(
            range(0, total_events, write_batch_size), desc="Writing to file"
        ):
            end_idx = min(start_idx + write_batch_size, total_events)
            # batch_data = {key: val[start_idx:end_idx] for key, val in all_data.items()}
            particle_data = {
                collection: ak.zip(
                    {
                        key.replace(f"{collection}_", "", 1): ak.Array(
                            val[start_idx:end_idx]
                        )
                        for key, val in out_data_evt.items()
                        if key.startswith(collection)
                        and key.replace(f"{collection}_", "", 1) in particle_vars
                    }
                )
                for collection in ["truth", "pflow", "fastsim"]
            }
            jet_data = {
                f"{collection}_jets": ak.zip(
                    {
                        key.replace(f"{collection}_", "", 1): ak.Array(
                            val[start_idx:end_idx]
                        )
                        for key, val in out_data_evt.items()
                        if key.startswith(f"{collection}_jet") and "idx" not in key
                    }
                )
                for collection in ["truth", "pflow", "fastsim"]
            }
            event_data = {
                key: ak.Array(val[start_idx:end_idx])
                for key, val in out_data_evt.items()
                if key in event_vars
            }

            # Combine all data
            batch_data = event_data | particle_data | jet_data

            if start_idx == 0:
                f["evt_tree"] = batch_data
            else:
                f["evt_tree"].extend(batch_data)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
