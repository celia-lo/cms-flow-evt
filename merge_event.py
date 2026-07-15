import os
import numpy as np
import uproot
import awkward as ak

INFILE = "my_sample/Parnassus_ntuple_minbias_noPU_test.root"
INTREE = "evt_tree"
OUTFILE = "my_sample/Parnassus_ntuple_minbias_noPU_test_x10.root"

GROUP_SIZE = 10
CHUNK_EVENTS = 200_000

JAGGED_BRANCHES = [
    "pflow_pt","pflow_eta","pflow_phi","pflow_mass","pflow_pdgId",
    "pflow_vx","pflow_vy","pflow_vz","pflow_d0","pflow_d0Error",
    "pflow_z0","pflow_z0Error","pflow_class",
    "truth_pt","truth_eta","truth_phi","truth_mass","truth_pdgId",
    "truth_vx","truth_vy","truth_vz","truth_class",
]

SCALAR_BRANCHES = {"eventNumber": "first"}

def concat_groups_jagged(arr: ak.Array, group_size: int) -> ak.Array:
    n_trim = (len(arr) // group_size) * group_size
    if n_trim == 0:
        return arr[:0]
    layout = arr[:n_trim].layout.to_ListOffsetArray64()
    offsets = np.asarray(layout.offsets, dtype=np.int64)
    lengths = offsets[1:] - offsets[:-1]
    grouped_lengths = lengths.reshape(-1, group_size).sum(axis=1)
    new_offsets = np.concatenate([[0], np.cumsum(grouped_lengths)])
    new_content = layout.content[: int(new_offsets[-1])]
    new_layout = ak.contents.ListOffsetArray(ak.index.Index64(new_offsets), new_content)
    return ak.Array(new_layout)

def aggregate_scalar(arr: ak.Array, group_size: int, mode: str) -> np.ndarray:
    flat = ak.to_numpy(arr)
    if flat.size == 0:
        return flat
    reshaped = flat.reshape(-1, group_size)
    if mode == "first":
        reduced = reshaped[:, 0]
    else:
        raise ValueError(f"Unsupported aggregation mode '{mode}'")
    return reduced.astype(flat.dtype, copy=False)

def main():
    if not os.path.exists(INFILE):
        raise FileNotFoundError(INFILE)

    with uproot.open(INFILE) as f:
        tree = f[INTREE]
        total = tree.num_entries
        total_groups = total // GROUP_SIZE
        remainder = total % GROUP_SIZE

        jagged = [b for b in JAGGED_BRANCHES if b in tree.keys()]
        scalars = {k: v for k, v in SCALAR_BRANCHES.items() if k in tree.keys()}

        if not jagged:
            raise RuntimeError("No PF/truth jagged branches found.")

        print(f"Input events: {total}")
        print(f"Writing {total_groups} grouped events (dropping {remainder}).")
        print(f"Jagged branches: {jagged}")
        print(f"Scalar branches: {scalars}")

        branches = jagged + list(scalars.keys())
        out_file = uproot.recreate(OUTFILE)
        carry = None
        written = 0

        for chunk in tree.iterate(branches, library="ak", step_size=CHUNK_EVENTS):
            arrays = {k: chunk[k] for k in branches}

            if carry is not None:
                arrays = {k: ak.concatenate([carry[k], arrays[k]], axis=0) for k in branches}
                carry = None

            n_events = len(next(iter(arrays.values())))
            n_trim = (n_events // GROUP_SIZE) * GROUP_SIZE
            if n_trim == 0:
                carry = arrays
                continue
            if n_trim < n_events:
                carry = {k: arrays[k][n_trim:] for k in branches}
                arrays = {k: arrays[k][:n_trim] for k in branches}

            out_chunk = {k: concat_groups_jagged(arrays[k], GROUP_SIZE) for k in jagged}
            n_groups = len(out_chunk[jagged[0]])

            for name, mode in scalars.items():
                out_chunk[name] = aggregate_scalar(arrays[name], GROUP_SIZE, mode)

            out_chunk["npflow"] = ak.to_numpy(ak.num(out_chunk["pflow_pt"])).astype(np.int32)
            out_chunk["ntruth"] = ak.to_numpy(ak.num(out_chunk["truth_pt"])).astype(np.int32)

            if INTREE not in out_file:
                out_file[INTREE] = out_chunk
            else:
                out_file[INTREE].extend(out_chunk)

            written += n_groups
            print(f"wrote {written}/{total_groups}", flush=True)

        out_file.close()

    print(f"Done. Output written to {OUTFILE}")

if __name__ == "__main__":
    main()
