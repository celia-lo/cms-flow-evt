#!/usr/bin/env python3
"""Check whether QCD(60PU) ~ QCD(noPU) + N×minbias(noPU).

For each QCD(60PU) event with nPU=k, takes the paired QCD(noPU) event and
overlays k randomly drawn minbias(noPU) events on top, then compares
distributions between real and synthetic 60PU.

Example:
    python compare_qcd_overlay.py --n-events 2000 --pt-cut 1.0 --output-dir comparison_qcd_overlay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import fastjet as fj
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.stats import ks_2samp

from parnassus_core.utils.jet_helper import Jet, get_cluster_sequence
from parnassus_core.utils.matching import reshape_phi


PFLOW_BRANCHES = ["pflow_pt", "pflow_eta", "pflow_phi", "pflow_class"]
LOAD_BRANCHES  = ["npflow"] + PFLOW_BRANCHES
LOAD_BRANCHES_PU = LOAD_BRANCHES + ["nPU"]
CHARGED_CLASSES = {0, 1, 2}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def save_cache(cache_dir: Path, pu_dict: dict, synthetic: dict, manifest: dict) -> None:
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
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
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


def load_manifest(cache_dir: Path) -> dict | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text())


def build_manifest(args: argparse.Namespace) -> dict:
    return {
        "qcd_nopu_tag":  "fastsim" if args.qcd_nopu_fastsim else "full-sim",
        "qcd_nopu_source": str(args.qcd_nopu_fastsim) if args.qcd_nopu_fastsim else str(args.qcd_nopu_dir),
        "mb_nopu_tag":   "fastsim" if args.mb_nopu_fastsim else "full-sim",
        "mb_nopu_source": str(args.mb_nopu_fastsim) if args.mb_nopu_fastsim else str(args.mb_nopu_dir),
        "n_events": args.n_events,
        "max_mb":   args.max_mb,
        "pt_cut_hs": args.pt_cut_hs,
        "pt_cut_mb": args.pt_cut_mb,
        "seed": args.seed,
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_fastsim_file(filepath: Path, max_events: int | None = None) -> dict:
    """Load postprocessed fastsim file (fastsim_pt/eta/phi/class flat branches)."""
    FASTSIM_BRANCHES = ["nfastsim", "fastsim_pt", "fastsim_eta", "fastsim_phi", "fastsim_class"]
    with uproot.open(f"{filepath}:evt_tree") as t:
        entry_stop = min(max_events, t.num_entries) if max_events else t.num_entries
        data = t.arrays(FASTSIM_BRANCHES, entry_stop=entry_stop, library="ak")
    out = {
        "pflow_pt":    data["fastsim_pt"],
        "pflow_eta":   data["fastsim_eta"],
        "pflow_phi":   data["fastsim_phi"],
        "pflow_class": data["fastsim_class"],
        "npflow":      data["nfastsim"],
    }
    print(f"  loaded {len(out['npflow'])} fastsim events from {filepath.name}, "
          f"mean npflow={float(ak.mean(out['npflow'])):.1f}")
    return out


def load_dir(dirpath: Path, branches: list[str], max_events: int | None = None) -> ak.Array:
    files = sorted(dirpath.glob("*.root"))
    if not files:
        raise FileNotFoundError(f"No ROOT files in {dirpath}")
    # skip empty or corrupt files (first 4 bytes not ROOT magic)
    good = []
    for f in files:
        if f.stat().st_size < 100:
            continue
        try:
            with uproot.open(f"{f}:evt_tree") as t:
                if t.num_entries > 0:
                    good.append(f)
        except Exception:
            pass
    if not good:
        raise RuntimeError(f"No valid ROOT files found in {dirpath}")
    skipped = len(files) - len(good)
    if skipped:
        print(f"  [skipped {skipped} corrupt/empty files in {dirpath.name}]")
    sources = [f"{f}:evt_tree" for f in good]
    with uproot.open(sources[0]) as f:
        available = set(f.keys())
    branches = [b for b in branches if b in available]
    return uproot.concatenate(sources, expressions=branches, entry_stop=max_events, library="ak")


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------

def overlay_qcd(
    qcd_nopu: ak.Array,
    mb_nopu: ak.Array,
    npu_values: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """For each QCD(noPU) event i, overlay npu_values[i] random minbias(noPU) events."""
    n_events  = len(npu_values)
    mb_size   = len(mb_nopu["pflow_pt"])

    out = {b: [] for b in PFLOW_BRANCHES}
    npflow_list: list[int] = []

    print(f"Overlaying {n_events} QCD events "
          f"(mean nPU={npu_values.mean():.1f}, MB pool={mb_size})...")
    for i in range(n_events):
        if i % 500 == 0:
            print(f"  {i}/{n_events}", flush=True)
        k = max(int(npu_values[i]), 0)
        mb_idxs = rng.integers(0, mb_size, size=k) if k > 0 else []

        for branch in PFLOW_BRANCHES:
            parts = [qcd_nopu[branch][i]]
            for j in mb_idxs:
                parts.append(mb_nopu[branch][j])
            combined = ak.concatenate(parts) if len(parts) > 1 else parts[0]
            out[branch].append(combined)

        npflow_list.append(int(ak.num(out["pflow_pt"][-1], axis=0)))

    out["npflow"] = ak.Array(npflow_list)
    for b in PFLOW_BRANCHES:
        out[b] = ak.Array(out[b])
    return out


# ---------------------------------------------------------------------------
# Jet reconstruction (mirrors postprocess_eval.py's cluster_jets)
# ---------------------------------------------------------------------------

def to_ak(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray) -> ak.Array:
    return ak.Array(
        {
            "px": pt * np.cos(phi),
            "py": pt * np.sin(phi),
            "pz": pt * np.sinh(eta),
            "E":  pt * np.cosh(eta),
        },
        with_name="Momentum4D",
    )


def cluster_jets(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, jetdef, radius: float, ptmin: float = 20.0) -> list[Jet]:
    particles = to_ak(pt, eta, phi)
    cs = get_cluster_sequence(jetdef, particles, user_indices=list(range(len(particles))))
    jets = fj.sorted_by_pt(cs.inclusive_jets(ptmin=ptmin))
    jets = [Jet(j, radius, calc_substructure=True) for j in jets]
    jets = [j for j in jets if j.nconstituents >= 2]
    return jets


def cluster_events(data: dict, radius: float, ptmin: float) -> dict:
    """Cluster each event's pflow particles into jets and return flat jet-level arrays."""
    jetdef = fj.JetDefinition(fj.antikt_algorithm, radius)
    n_events = len(data["npflow"])
    njet = np.zeros(n_events, dtype=int)
    jet_pt, jet_eta, jet_phi, jet_d2, jet_c2 = [], [], [], [], []

    print(f"Clustering jets for {n_events} events (R={radius}, ptmin={ptmin} GeV)...")
    for i in range(n_events):
        if i % 500 == 0:
            print(f"  {i}/{n_events}", flush=True)
        pt  = np.asarray(ak.to_numpy(data["pflow_pt"][i]),  dtype=float)
        if pt.size == 0:
            continue
        eta = np.asarray(ak.to_numpy(data["pflow_eta"][i]), dtype=float)
        phi = np.asarray(ak.to_numpy(data["pflow_phi"][i]), dtype=float)
        jets = cluster_jets(pt, eta, phi, jetdef, radius, ptmin=ptmin)
        njet[i] = len(jets)
        for j in jets:
            jet_pt.append(j.pt())
            jet_eta.append(j.eta())
            jet_phi.append(j.phi())
            jet_d2.append(j.substructure["d2"])
            jet_c2.append(j.substructure["c2"])

    return {
        "njet":    njet,
        "jet_pt":  np.asarray(jet_pt),
        "jet_eta": np.asarray(jet_eta),
        # fastjet's jet.phi() is in [0, 2pi); wrap to the standard (-pi, pi] convention
        # used elsewhere in this codebase (see parnassus_core.performance.performance.Data.load_from_dict).
        "jet_phi": reshape_phi(np.asarray(jet_phi)),
        "jet_d2":  np.asarray(jet_d2),
        "jet_c2":  np.asarray(jet_c2),
    }


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def filter_pt(data: dict, pt_cut: float) -> dict:
    mask = data["pflow_pt"] > pt_cut
    filtered = {b: data[b][mask] for b in PFLOW_BRANCHES}
    filtered["npflow"] = ak.sum(mask, axis=1)
    return filtered


def filter_charged(data: dict) -> dict:
    mask = ak.Array(
        [[c in CHARGED_CLASSES for c in ak.to_list(ev)] for ev in data["pflow_class"]]
    )
    filtered = {b: data[b][mask] for b in PFLOW_BRANCHES}
    filtered["npflow"] = ak.sum(mask, axis=1)
    return filtered


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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
    a = data_a[np.isfinite(data_a)]
    b = data_b[np.isfinite(data_b)]
    if a.size == 0 or b.size == 0:
        return

    if not isinstance(bins, np.ndarray):
        combined = np.concatenate([a, b])
        lo, hi = np.percentile(combined, [0.5, 99.5])
        if lo >= hi:
            lo, hi = combined.min(), combined.max()
        bins = np.linspace(lo, hi, bins + 1)

    ks_stat, ks_p = ks_2samp(a, b)
    ca, _ = np.histogram(a, bins=bins)
    cb, _ = np.histogram(b, bins=bins)
    na = ca / ca.sum()
    nb = cb / cb.sum()

    fig, (ax, ax_r) = plt.subplots(
        2, 1, figsize=(6, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax.step(bins[:-1], na, where="post", lw=2, label=label_a)
    ax.step(bins[:-1], nb, where="post", lw=2, label=label_b, linestyle="--")
    ax.set_ylabel("Normalized", fontsize=13)
    # ax.legend(fontsize=10, title=f"KS={ks_stat:.3f}  p={ks_p:.2g}", title_fontsize=10)
    ax.legend(fontsize=10)
    ax.set_title(var, fontsize=13)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nb > 0, na / nb, np.nan)
    ax_r.step(bins[:-1], ratio, where="post", lw=1.5, color="black")
    ax_r.axhline(1.0, color="gray", lw=1, linestyle="--")
    ax_r.set_ylim(0.0, 3.0)
    ax_r.set_ylabel("PU / overlay", fontsize=11)
    ax_r.set_xlabel(xlabel or var, fontsize=13)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare QCD(60PU) vs QCD(noPU)+N×MB(noPU).")
    p.add_argument("--qcd-pu-dir", type=Path,
                   default=Path("my_sample/QCD_2024_pthat2500to3000_60PU_noPtcut/nPU_stored/qcd_pthat2500to3000_ntuple_pu_1M_split_1000_3d"))
    p.add_argument("--qcd-nopu-dir", type=Path,
                   default=Path("my_sample/QCD_2024_pthat2500to3000_noPU_noPtcut/qcd_pthat2500to3000_ntuple_nopu_1M_split_1000_3d"))
    p.add_argument("--mb-nopu-dir", type=Path,
                   default=Path("my_sample/minbias_2024_noPU_noPtcut"))
    p.add_argument("--qcd-nopu-fastsim", type=Path, default=None,
                   help="Fastsim eval output ROOT file to use instead of --qcd-nopu-dir")
    p.add_argument("--mb-nopu-fastsim", type=Path, default=None,
                   help="Fastsim eval output ROOT file to use instead of --mb-nopu-dir")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="Directory to save/load pre-cut arrays. If it exists, skip overlay.")
    p.add_argument("--n-events", type=int, default=2000)
    p.add_argument("--max-mb", type=int, default=200_000,
                   help="Max minbias noPU events to load as pool")
    p.add_argument("--pt-cut", type=float, default=1.0,
                   help="pT cut applied to QCD 60PU after loading")
    p.add_argument("--pt-cut-hs", type=float, default=1.0,
                   help="pT cut applied to QCD noPU (hard scatter) before overlay")
    p.add_argument("--pt-cut-mb", type=float, default=1.2,
                   help="pT cut applied to MB noPU before overlay")
    p.add_argument("--charged-only", action="store_true")
    p.add_argument("--jet-radius", type=float, default=0.5,
                   help="Anti-kt jet radius for reconstructing jets from the (post pT-cut) particles")
    p.add_argument("--jet-ptmin", type=float, default=20.0,
                   help="Minimum jet pT [GeV] kept after clustering")
    p.add_argument("--output-dir", type=Path, default=Path("comparison_qcd_overlay"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or args.output_dir / "cache"
    cache_exists = (cache_dir / "pu.npz").exists()

    current_manifest = build_manifest(args)

    if cache_exists:
        pu_dict_raw, synthetic = load_cache(cache_dir)
        print(f"  {len(pu_dict_raw['npflow'])} events in cache")
        cached_manifest = load_manifest(cache_dir)
        if cached_manifest is None:
            print(f"  WARNING: {cache_dir}/manifest.json missing (pre-dates manifest tracking). "
                  f"Cannot verify what this cache was built from; assuming current CLI args, "
                  f"which may mislabel the plots if they differ from the run that built the cache.")
            manifest = current_manifest
        else:
            manifest = cached_manifest
            if manifest != current_manifest:
                print(f"  WARNING: cache in {cache_dir} was built with different arguments than "
                      f"this invocation. Reusing cached data (NOT re-loading), and labeling plots "
                      f"from the cache's manifest, not the current CLI args:")
                print(f"    cached:  {manifest}")
                print(f"    current: {current_manifest}")
    else:
        print(f"Loading {args.n_events} QCD 60PU events...")
        qcd_pu = load_dir(args.qcd_pu_dir, LOAD_BRANCHES_PU, max_events=args.n_events)
        n_loaded = len(qcd_pu["npflow"])
        npu_values = np.asarray(ak.to_numpy(qcd_pu["nPU"]), dtype=int)
        print(f"  loaded {n_loaded} events, mean npflow={float(ak.mean(qcd_pu['npflow'])):.1f}")
        print(f"  nPU: mean={npu_values.mean():.1f}, min={npu_values.min()}, max={npu_values.max()}")

        print(f"Loading {args.n_events} QCD noPU events...")
        if args.qcd_nopu_fastsim is not None:
            print(f"  (fastsim: {args.qcd_nopu_fastsim})")
            qcd_nopu = load_fastsim_file(args.qcd_nopu_fastsim, max_events=n_loaded)
        else:
            qcd_nopu = load_dir(args.qcd_nopu_dir, LOAD_BRANCHES, max_events=n_loaded)
            print(f"  loaded {len(qcd_nopu['npflow'])} events, mean npflow={float(ak.mean(qcd_nopu['npflow'])):.1f}")

        print(f"Loading up to {args.max_mb} minbias noPU events...")
        if args.mb_nopu_fastsim is not None:
            print(f"  (fastsim: {args.mb_nopu_fastsim})")
            mb_nopu = load_fastsim_file(args.mb_nopu_fastsim, max_events=args.max_mb)
        else:
            mb_nopu = load_dir(args.mb_nopu_dir, LOAD_BRANCHES, max_events=args.max_mb)
            print(f"  loaded {len(mb_nopu['npflow'])} events, mean npflow={float(ak.mean(mb_nopu['npflow'])):.1f}")

        if args.pt_cut_hs is not None:
            print(f"Applying pT > {args.pt_cut_hs} GeV to QCD noPU (hard scatter)...")
            qcd_nopu = {b: v for b, v in filter_pt({b: qcd_nopu[b] for b in PFLOW_BRANCHES + ["npflow"]}, args.pt_cut_hs).items()}
        if args.pt_cut_mb is not None:
            print(f"Applying pT > {args.pt_cut_mb} GeV to MB noPU...")
            mb_nopu = {b: v for b, v in filter_pt({b: mb_nopu[b] for b in PFLOW_BRANCHES + ["npflow"]}, args.pt_cut_mb).items()}

        synthetic = overlay_qcd(qcd_nopu, mb_nopu, npu_values[:len(qcd_nopu["npflow"])], rng)
        pu_dict_raw = {b: qcd_pu[b] for b in PFLOW_BRANCHES}
        pu_dict_raw["npflow"] = qcd_pu["npflow"]
        manifest = current_manifest
        save_cache(cache_dir, pu_dict_raw, synthetic, manifest)

    # --- build working dicts ---
    pu_dict = {k: v for k, v in pu_dict_raw.items()}

    label_suffix = ""
    if args.charged_only:
        print("Filtering to charged only (class 0,1,2)...")
        pu_dict   = filter_charged(pu_dict)
        synthetic = filter_charged(synthetic)
        label_suffix += " [charged]"

    if args.pt_cut is not None:
        print(f"Applying pT > {args.pt_cut} GeV...")
        pu_dict   = filter_pt(pu_dict,   args.pt_cut)
        synthetic = filter_pt(synthetic, args.pt_cut)
        # label_suffix += f" pT>{args.pt_cut:.1f}"

    label_pu  = f"QCD PU{label_suffix}"
    hs_tag = manifest["qcd_nopu_tag"]
    mb_tag = manifest["mb_nopu_tag"]
    label_syn = f"QCD noPU ({hs_tag}) + N×MB noPU ({mb_tag}){label_suffix}"

    # --- event-level ---
    ev_vars = {
        "npflow": (
            np.asarray(ak.to_numpy(pu_dict["npflow"]), dtype=float),
            np.asarray(ak.to_numpy(synthetic["npflow"]), dtype=float),
        ),
        "ht": (
            np.asarray(ak.to_numpy(ak.sum(pu_dict["pflow_pt"], axis=1)), dtype=float),
            np.asarray(ak.to_numpy(ak.sum(synthetic["pflow_pt"], axis=1)), dtype=float),
        ),
        "met_x": (
            np.asarray(ak.to_numpy(ak.sum(pu_dict["pflow_pt"] * np.cos(pu_dict["pflow_phi"]), axis=1)), dtype=float),
            np.asarray(ak.to_numpy(ak.sum(synthetic["pflow_pt"] * np.cos(synthetic["pflow_phi"]), axis=1)), dtype=float),
        ),
        "met_y": (
            np.asarray(ak.to_numpy(ak.sum(pu_dict["pflow_pt"] * np.sin(pu_dict["pflow_phi"]), axis=1)), dtype=float),
            np.asarray(ak.to_numpy(ak.sum(synthetic["pflow_pt"] * np.sin(synthetic["pflow_phi"]), axis=1)), dtype=float),
        ),
    }

    # --- particle-level ---
    part_vars = {}
    for branch in PFLOW_BRANCHES:
        part_vars[branch] = (
            np.asarray(ak.to_numpy(ak.flatten(pu_dict[branch])),   dtype=float),
            np.asarray(ak.to_numpy(ak.flatten(synthetic[branch])), dtype=float),
        )

    # --- jet-level (reconstruct jets from the same post-cut particles) ---
    jets_pu  = cluster_events(pu_dict,   args.jet_radius, args.jet_ptmin)
    jets_syn = cluster_events(synthetic, args.jet_radius, args.jet_ptmin)

    ev_vars["njet"] = (
        jets_pu["njet"].astype(float),
        jets_syn["njet"].astype(float),
    )
    jet_vars = {
        "jet_pt":  (jets_pu["jet_pt"],  jets_syn["jet_pt"]),
        "jet_eta": (jets_pu["jet_eta"], jets_syn["jet_eta"]),
        "jet_phi": (jets_pu["jet_phi"], jets_syn["jet_phi"]),
        "jet_d2":  (jets_pu["jet_d2"],  jets_syn["jet_d2"]),
        "jet_c2":  (jets_pu["jet_c2"],  jets_syn["jet_c2"]),
    }

    custom_bins: dict = {
        "npflow":      np.linspace(0,    800, 81),
        "ht":          np.linspace(3000,    8000, 51),
        "met_x":       np.linspace(-80,     80, 50),
        "met_y":       np.linspace(-80,     80, 50),
        "njet":        np.arange(-0.5,      20.5, 1.0),
        "pflow_pt":    np.linspace(1.5,    20, 81),
        "pflow_eta":   np.linspace(-3,      3, 61),
        "pflow_phi":   np.linspace(-np.pi, np.pi, 65),
        "pflow_class": np.arange(-0.5, 6.5, 1.0),
        "jet_pt":      np.linspace(0,     1000, 100),
        "jet_eta":     np.linspace(-3,       3, 50),
        "jet_phi":     np.linspace(-np.pi, np.pi, 50),
        "jet_d2":      np.linspace(-3,       4, 50),
        "jet_c2":      np.linspace(0,      0.5, 50),
    }
    xlabels = {
        "npflow":      "N pflow particles",
        "ht":          "HT = Σ pflow pT  [GeV]",
        "met_x":       "MET x  [GeV]",
        "met_y":       "MET y  [GeV]",
        "njet":        "N jets",
        "pflow_pt":    "pflow pT  [GeV]",
        "pflow_eta":   "pflow η",
        "pflow_phi":   "pflow φ",
        "pflow_class": "pflow class",
        "jet_pt":      "jet pT  [GeV]",
        "jet_eta":     "jet η",
        "jet_phi":     "jet φ",
        "jet_d2":      "jet D2",
        "jet_c2":      "jet C2",
    }

    PT_PLOT_MIN = 1.5  # normalize pflow_pt shape above this threshold only

    print("\nPlotting...")
    for var, (arr_pu, arr_syn) in {**ev_vars, **part_vars, **jet_vars}.items():
        plot_pu, plot_syn = arr_pu, arr_syn
        if var == "pflow_pt":
            plot_pu  = arr_pu[arr_pu   > PT_PLOT_MIN]
            plot_syn = arr_syn[arr_syn > PT_PLOT_MIN]
        print(f"  {var}:  60PU mean={arr_pu.mean():.2f},  overlay mean={arr_syn.mean():.2f}")
        plot_comparison(
            var, plot_pu, plot_syn, label_pu, label_syn,
            args.output_dir / f"{var}.png",
            bins=custom_bins.get(var, 60),
            xlabel=xlabels.get(var),
            dpi=args.dpi,
        )

    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
