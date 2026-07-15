#!/usr/bin/env python3
"""Check whether minbias(60PU) ~ N_interactions * minbias(noPU).

Constructs synthetic 60PU events by stacking N noPU events per synthetic
event, then overlays key distributions against the real 60PU sample.

If the 60PU file contains a 'nPU' branch, that per-event integer is used
directly as the stack count.  Otherwise falls back to --n-interactions.

Example:
    python compare_60pu_stacked_nopu.py --n-events 5000
    python compare_60pu_stacked_nopu.py --n-events 5000 --n-interactions 60  # fixed fallback
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.stats import ks_2samp


PFLOW_BRANCHES = ["pflow_pt", "pflow_eta", "pflow_phi", "pflow_class"]
LOAD_BRANCHES = ["npflow"] + PFLOW_BRANCHES
LOAD_BRANCHES_PU = LOAD_BRANCHES + ["nPU"]  # extra branch in new 60PU files


# ---------------------------------------------------------------------------
# Cache helpers — save/load pre-cut arrays so overlay is only run once
# ---------------------------------------------------------------------------

def save_cache(cache_dir: Path, pu_dict: dict, synthetic: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("pu", pu_dict), ("syn", synthetic)]:
        arr = ak.Array({b: data[b] for b in PFLOW_BRANCHES + ["npflow"]})
        form, length, container = ak.to_buffers(arr)
        np.savez(
            cache_dir / f"{name}.npz",
            _form=np.array([form.to_json()]),
            _length=np.array([length]),
            **container,
        )
    print(f"Cache saved to {cache_dir}/")


def load_cache(cache_dir: Path) -> tuple[dict, dict]:
    result = []
    for name in ["pu", "syn"]:
        raw = np.load(cache_dir / f"{name}.npz", allow_pickle=False)
        form = ak.forms.from_json(str(raw["_form"][0]))
        length = int(raw["_length"][0])
        container = {k: raw[k] for k in raw.files if not k.startswith("_")}
        arr = ak.from_buffers(form, length, container)
        result.append({f: arr[f] for f in arr.fields})
    print(f"Cache loaded from {cache_dir}/")
    return result[0], result[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare 60PU vs stacked noPU.")
    p.add_argument("--pu-dir", type=Path,
                   default=Path("my_sample/minbias_2024_60PU_noPtcut/N_int_stored/ntuple_pu_1M_split_3000_v2"))
    p.add_argument("--nopu-dir", type=Path, default=Path("my_sample/minbias_2024_noPU_noPtcut"))
    p.add_argument("--n-interactions", type=float, default=None,
                   help="Fallback fixed N if nPU branch is absent; omit to use per-event nPU")
    p.add_argument("--poisson", action="store_true",
                   help="When using --n-interactions fallback: sample N ~ Poisson(n_interactions)")
    p.add_argument("--n-events", type=int, default=5000,
                   help="Number of 60PU events to compare")
    p.add_argument("--max-nopu", type=int, default=200_000,
                   help="Max noPU events to load into pool")
    p.add_argument("--output-dir", type=Path, default=Path("comparison_60pu_stacked"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Directory to save/load pre-cut arrays. If it exists, skip stacking.")
    p.add_argument("--apply-chs", action="store_true",
                   help="Apply CHS-like cleaning to stacked noPU: remove charged particles "
                        "(class 0,1,2) from all but the primary noPU event")
    p.add_argument("--charged-only", action="store_true",
                   help="Compare only charged particles (class 0,1,2) in both samples")
    p.add_argument("--pt-cut", type=float, default=None,
                   help="Apply pflow_pt > PT_CUT to both samples before comparing (e.g. 1.0)")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def load_dir(dirpath: Path, branches: list[str], max_events: int | None = None) -> ak.Array:
    files = sorted(dirpath.glob("*.root"))
    if not files:
        raise FileNotFoundError(f"No ROOT files found in {dirpath}")
    sources = [f"{f}:evt_tree" for f in files]
    # filter to branches that actually exist in the first file
    with uproot.open(sources[0]) as f:
        available = set(f.keys())
    branches = [b for b in branches if b in available]
    return uproot.concatenate(sources, expressions=branches, entry_stop=max_events, library="ak")


def stack_nopu(
    nopu: ak.Array,
    npu_values: np.ndarray,
    rng: np.random.Generator,
    apply_chs: bool = False,
) -> dict[str, list]:
    """Stack noPU events into synthetic 60PU events using per-event nPU counts.

    With apply_chs=True: the first sampled noPU event is the 'primary' (all
    particles kept); charged particles (class 0,1,2) are removed from the
    remaining N-1 pileup events, mimicking CHS.
    """
    pool_size = len(nopu["pflow_pt"])
    n_events = len(npu_values)
    # neutral mask applied to pileup events under CHS
    NEUTRAL_CLASSES = {3, 4}

    out_branches = {b: [] for b in PFLOW_BRANCHES}
    npflow_list: list[int] = []

    label = "CHS-cleaned" if apply_chs else "plain"
    print(f"Stacking {n_events} synthetic events [{label}] "
          f"(mean nPU={npu_values.mean():.1f}, pool={pool_size})...")

    for i in range(n_events):
        if i % 500 == 0:
            print(f"  {i}/{n_events}", flush=True)
        k = max(int(npu_values[i]), 1)
        idxs = rng.integers(0, pool_size, size=k)

        event_parts: dict[str, list] = {b: [] for b in PFLOW_BRANCHES}
        for j, idx in enumerate(idxs):
            is_primary = (j == 0)
            for branch in PFLOW_BRANCHES:
                arr = nopu[branch][idx]
                if apply_chs and not is_primary and branch == "pflow_class":
                    # will apply mask after collecting class array
                    event_parts[branch].append(arr)
                elif apply_chs and not is_primary:
                    event_parts[branch].append(arr)
                else:
                    event_parts[branch].append(arr)

        if apply_chs and k > 1:
            # build neutral mask for pileup events (index 1 onward)
            prim_class = event_parts["pflow_class"][0]
            pu_classes = event_parts["pflow_class"][1:]
            pu_masks = [ak.Array([c in NEUTRAL_CLASSES for c in ak.to_list(cls)]) for cls in pu_classes]

            for branch in PFLOW_BRANCHES:
                primary_part = event_parts[branch][0]
                pu_parts = [event_parts[branch][j + 1][pu_masks[j]] for j in range(len(pu_masks))]
                all_parts = [primary_part] + pu_parts
                combined = ak.concatenate(all_parts)
                out_branches[branch].append(combined)
        else:
            for branch in PFLOW_BRANCHES:
                parts = event_parts[branch]
                combined = ak.concatenate(parts) if len(parts) > 1 else parts[0]
                out_branches[branch].append(combined)

        npflow_list.append(int(ak.num(out_branches["pflow_pt"][-1], axis=0)))

    out_branches["npflow"] = ak.Array(npflow_list)
    for b in PFLOW_BRANCHES:
        out_branches[b] = ak.Array(out_branches[b])
    return out_branches


CHARGED_CLASSES = {0, 1, 2}


def filter_charged(data: dict, key: str = "pflow_class") -> dict:
    """Keep only charged particles (class 0,1,2) in each event."""
    mask = ak.Array([[c in CHARGED_CLASSES for c in ak.to_list(ev)] for ev in data[key]])
    filtered = {}
    for branch in PFLOW_BRANCHES:
        filtered[branch] = data[branch][mask]
    filtered["npflow"] = ak.sum(mask, axis=1)
    return filtered


def filter_pt(data: dict, pt_cut: float) -> dict:
    """Keep only particles with pflow_pt > pt_cut in each event."""
    mask = data["pflow_pt"] > pt_cut
    filtered = {}
    for branch in PFLOW_BRANCHES:
        filtered[branch] = data[branch][mask]
    filtered["npflow"] = ak.sum(mask, axis=1)
    return filtered


def plot_comparison(
    var: str,
    data_a: np.ndarray,
    data_b: np.ndarray,
    label_a: str,
    label_b: str,
    out_path: Path,
    bins: int | np.ndarray,
    xlabel: str | None = None,
    dpi: int = 150,
) -> None:
    finite_a = data_a[np.isfinite(data_a)]
    finite_b = data_b[np.isfinite(data_b)]
    if finite_a.size == 0 or finite_b.size == 0:
        return

    if not isinstance(bins, np.ndarray):
        combined = np.concatenate([finite_a, finite_b])
        lo, hi = np.percentile(combined, [0.5, 99.5])
        if lo >= hi:
            lo, hi = combined.min(), combined.max()
        bins = np.linspace(lo, hi, bins + 1)

    ks_stat, ks_p = ks_2samp(finite_a, finite_b)

    counts_a, _ = np.histogram(finite_a, bins=bins)
    counts_b, _ = np.histogram(finite_b, bins=bins)
    norm_a = counts_a / counts_a.sum()
    norm_b = counts_b / counts_b.sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    # fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(6, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    ax.step(bins[:-1], norm_a, where="post", lw=2, label=f"{label_a}")
    ax.step(bins[:-1], norm_b, where="post", lw=2, label=f"{label_b}", linestyle="--")
    ax.set_ylabel("Normalized", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_title(var, fontsize=13)
    # ax.set_xlabel(xlabel or var, fontsize=13)

    # # ratio panel
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     ratio = np.where(norm_b > 0, norm_a / norm_b, np.nan)
    # ax_ratio.step(bins[:-1], ratio, where="post", lw=1.5, color="black")
    # ax_ratio.axhline(1.0, color="gray", lw=1, linestyle="--")
    # ax_ratio.set_ylim(0.0, 5.0)
    # ax_ratio.set_ylabel("PU / stacked", fontsize=11)
    # ax_ratio.set_xlabel(xlabel or var, fontsize=13)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or args.output_dir / "cache"
    cache_exists = (cache_dir / "pu.npz").exists()

    if cache_exists:
        pu_dict_raw, synthetic = load_cache(cache_dir)
        n_events_cached = len(pu_dict_raw["npflow"])
        print(f"  {n_events_cached} events in cache")
        label_syn = "stacked noPU (nPU/event)"
    else:
        print(f"Loading {args.n_events} 60PU events from {args.pu_dir} ...")
        pu = load_dir(args.pu_dir, LOAD_BRANCHES_PU, max_events=args.n_events)
        n_loaded = len(pu["npflow"])
        print(f"  loaded {n_loaded} events, mean npflow={float(ak.mean(pu['npflow'])):.1f}")

        if "nPU" in pu.fields:
            npu_values = np.asarray(ak.to_numpy(pu["nPU"]), dtype=int)
            label_syn = "stacked noPU (nPU/event)"
            print(f"  using per-event nPU branch: mean={npu_values.mean():.1f}, "
                  f"min={npu_values.min()}, max={npu_values.max()}")
        else:
            fallback_n = args.n_interactions if args.n_interactions is not None else 20.0
            print(f"  nPU branch not found — falling back to n_interactions={fallback_n}")
            if args.poisson:
                npu_values = rng.poisson(fallback_n, size=n_loaded).astype(int)
            else:
                base = int(fallback_n)
                frac = fallback_n - base
                npu_values = np.array(
                    [base + (1 if rng.random() < frac else 0) for _ in range(n_loaded)], dtype=int
                )
            label_syn = f"stacked noPU (N={fallback_n:.0f})"

        mean_n = npu_values.mean()
        nopu_needed = min(int(mean_n * n_loaded * 2), args.max_nopu)
        print(f"Loading up to {nopu_needed} noPU events from {args.nopu_dir} ...")
        nopu = load_dir(args.nopu_dir, LOAD_BRANCHES, max_events=nopu_needed)
        print(f"  loaded {len(nopu['npflow'])} events, mean npflow={float(ak.mean(nopu['npflow'])):.1f}")
        synthetic = stack_nopu(nopu, npu_values, rng, apply_chs=args.apply_chs)
        pu_dict_raw = {b: pu[b] for b in PFLOW_BRANCHES}
        pu_dict_raw["npflow"] = pu["npflow"]
        save_cache(cache_dir, pu_dict_raw, synthetic)

    # apply charged-only / pt filters to pre-cut raw arrays
    pu_dict = {k: v for k, v in pu_dict_raw.items()}

    if args.charged_only:
        print("Filtering to charged particles only (class 0,1,2)...")
        pu_dict = filter_charged(pu_dict)
        synthetic = filter_charged(synthetic)
        label_suffix = " [charged]"
    else:
        label_suffix = ""

    if args.pt_cut is not None:
        print(f"Applying pT > {args.pt_cut} GeV cut...")
        pu_dict = filter_pt(pu_dict, args.pt_cut)
        synthetic = filter_pt(synthetic, args.pt_cut)
        # label_suffix += f" pT>{args.pt_cut:.0f}"

    label_pu = f"PU{label_suffix}"
    if args.apply_chs:
        label_syn = label_syn.replace("stacked", "stacked+CHS")
    label_syn = f"{label_syn}{label_suffix}"

    # --- event-level variables ---
    ev_vars = {
        "npflow": (
            np.asarray(ak.to_numpy(pu_dict["npflow"]), dtype=float),
            np.asarray(ak.to_numpy(synthetic["npflow"]), dtype=float),
        ),
        "ht": (
            np.asarray(ak.to_numpy(ak.sum(pu_dict["pflow_pt"], axis=1)), dtype=float),
            np.asarray(ak.to_numpy(ak.sum(synthetic["pflow_pt"], axis=1)), dtype=float),
        ),
    }

    # --- particle-level variables (flattened) ---
    part_vars = {}
    for branch in PFLOW_BRANCHES:
        arr_pu = np.asarray(ak.to_numpy(ak.flatten(pu_dict[branch])), dtype=float)
        arr_syn = np.asarray(ak.to_numpy(ak.flatten(synthetic[branch])), dtype=float)
        part_vars[branch] = (arr_pu, arr_syn)

    # custom bins
    custom_bins: dict[str, np.ndarray | int] = {
        "npflow": np.linspace(0, 5000, 101),
        "ht": np.linspace(0, 3000, 61),
        "pflow_pt": np.linspace(0, 20, 81),
        "pflow_eta": np.linspace(-3, 3, 61),
        "pflow_phi": np.linspace(-np.pi, np.pi, 65),
        "pflow_class": np.arange(-0.5, 6.5, 1.0),
    }

    xlabels = {
        "npflow": "N pflow particles",
        "ht": "HT = Σ pflow pT  [GeV]",
        "pflow_pt": "pflow pT  [GeV]",
        "pflow_eta": "pflow η",
        "pflow_phi": "pflow φ",
        "pflow_class": "pflow class",
    }

    print("\nPlotting...")
    all_vars = {**ev_vars, **part_vars}
    for var, (arr_pu, arr_syn) in all_vars.items():
        print(f"  {var}:  60PU mean={arr_pu.mean():.2f},  stacked mean={arr_syn.mean():.2f}")
        plot_comparison(
            var,
            arr_pu,
            arr_syn,
            label_pu,
            label_syn,
            args.output_dir / f"{var}.png",
            bins=custom_bins.get(var, 60),
            xlabel=xlabels.get(var),
            dpi=args.dpi,
        )

    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
