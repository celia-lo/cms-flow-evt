import numpy as np
import awkward as ak
import uproot


def analyze_sample(path, max_events=10_000):
    with uproot.open(path) as file:
        tree = file["evt_tree"]
        arrays = tree.arrays(
            ["npflow", "ntruth", "pflow_pt", "truth_pt"],
            entry_stop=max_events,
            library="ak",
        )

    npflow = ak.to_numpy(arrays["npflow"])
    ntruth = ak.to_numpy(arrays["ntruth"])
    pf_ht = ak.to_numpy(ak.sum(arrays["pflow_pt"], axis=1))
    truth_ht = ak.to_numpy(ak.sum(arrays["truth_pt"], axis=1))

    count = len(npflow)
    return {
        "events": count,
        "mean_npflow": float(npflow.mean()),
        "median_npflow": float(np.median(npflow)),
        "frac_npflow_eq0": float((npflow == 0).sum()) / count,
        "frac_npflow_gt5": float((npflow > 5).sum()) / count,
        "frac_npflow_gt10": float((npflow > 10).sum()) / count,
        "mean_ntruth": float(ntruth.mean()),
        "median_ntruth": float(np.median(ntruth)),
        "mean_pf_ht": float(pf_ht.mean()),
        "median_pf_ht": float(np.median(pf_ht)),
        "mean_truth_ht": float(truth_ht.mean()),
        "median_truth_ht": float(np.median(truth_ht)),
    }


def print_stats(label, stats):
    print(f"=== {label} ===")
    print(f"events              : {stats['events']}")
    print(f"mean npflow         : {stats['mean_npflow']:.2f}")
    print(f"median npflow       : {stats['median_npflow']:.1f}")
    print(f"fraction npflow == 0: {stats['frac_npflow_eq0']:.3f}")
    print(f"fraction npflow > 5 : {stats['frac_npflow_gt5']:.3f}")
    print(f"fraction npflow > 10: {stats['frac_npflow_gt10']:.3f}")
    print(f"mean ntruth         : {stats['mean_ntruth']:.1f}")
    print(f"median ntruth       : {stats['median_ntruth']:.1f}")
    print(f"mean pf HT          : {stats['mean_pf_ht']:.1f}")
    print(f"median pf HT        : {stats['median_pf_ht']:.1f}")
    print(f"mean truth HT       : {stats['mean_truth_ht']:.1f}")
    print(f"median truth HT     : {stats['median_truth_ht']:.1f}")
    print()


if __name__ == "__main__":
    DATASETS = {
        "minbias_noPU": "my_sample/Parnassus_ntuple_minbias_noPU_merge.root",
        "minbias_noPU_rmPt": "my_sample/Parnassus_ntuple_minbias_noPU_rmPt_merge.root",
    }
    for label, path in DATASETS.items():
        stats = analyze_sample(path)
        print_stats(label, stats)

