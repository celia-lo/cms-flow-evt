#!/usr/bin/env python3
"""Overlay fast-sim distributions from two postprocessed ROOT files.

Reference (blue, filled): PF or fast-sim collection from a large mixed sample.
Mixed (red, line): fast-sim collection from the prefiltered/mixed file.

The script generates event-, jet-, and particle-level panels similar to the
figures in the CMS 2011 validation plots, plus residual panels computed as
fastsim - pflow within each file.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


CmsColor = "#1f77b4"
FmColor = "crimson"


EVENT_PLOTS = [
    {"branch": "nfastsim", "bins": np.arange(-0.5, 1000.5, 10), "label": r"$N_{\mathrm{PF}}$"},
    {
        "branch": "fastsim_hung_cost_all",
        "bins": np.linspace(0, 1, 51),
        "label": r"Hungarian cost (all)",
    },
    {
        "branch": "fastsim_hung_cost_jet",
        "bins": np.linspace(0, 1, 51),
        "label": r"Hungarian cost (jet)",
    },
    {
        "branch": "fastsim_hung_cost_bkg",
        "bins": np.linspace(0, 1, 51),
        "label": r"Hungarian cost (bkg)",
    },
]

JET_PLOTS = [
    {"branch": "fastsim_jets_jet_pt", "bins": np.linspace(0, 200, 121), "label": r"Jet $p_T$ [GeV]"},
    {"branch": "fastsim_jets_jet_eta", "bins": np.linspace(-3, 3, 121), "label": r"Jet $\eta$"},
    {"branch": "fastsim_jets_jet_phi", "bins": np.linspace(-np.pi, np.pi, 121), "label": r"Jet $\phi$"},
    {"branch": "fastsim_jets_jet_d2", "bins": np.linspace(-3, 3, 121), "label": r"Jet $d_2$"},
    {"branch": "fastsim_jets_jet_c2", "bins": np.linspace(0, 2, 121), "label": r"Jet $c_2$"},
]

JET_RESIDUALS = [
    {
        "fast_branch": "fastsim_jets_jet_pt",
        "pf_branch": "pflow_jets_jet_pt",
        "bins": np.linspace(-100, 100, 121),
        "label": r"Residual jet $p_T$ [GeV]",
    },
    {
        "fast_branch": "fastsim_jets_jet_eta",
        "pf_branch": "pflow_jets_jet_eta",
        "bins": np.linspace(-1.5, 1.5, 121),
        "label": r"Residual jet $\eta$",
    },
    {
        "fast_branch": "fastsim_jets_jet_phi",
        "pf_branch": "pflow_jets_jet_phi",
        "bins": np.linspace(-np.pi, np.pi, 121),
        "label": r"Residual jet $\phi$",
    },
    {
        "fast_branch": "fastsim_jets_jet_d2",
        "pf_branch": "pflow_jets_jet_d2",
        "bins": np.linspace(-2, 2, 121),
        "label": r"Residual jet $d_2$",
    },
    {
        "fast_branch": "fastsim_jets_jet_c2",
        "pf_branch": "pflow_jets_jet_c2",
        "bins": np.linspace(-1, 1, 121),
        "label": r"Residual jet $c_2$",
    },
]

PARTICLE_PLOTS = [
    {"branch": "fastsim_pt", "bins": np.linspace(0, 200, 121), "label": r"Particle $p_T$ [GeV]", "logy": True},
    {"branch": "fastsim_eta", "bins": np.linspace(-3, 3, 121), "label": r"Particle $\eta$"},
    {"branch": "fastsim_phi", "bins": np.linspace(-np.pi, np.pi, 121), "label": r"Particle $\phi$"},
    {"branch": "fastsim_vx", "bins": np.linspace(-0.5, 0.5, 121), "label": r"Particle $v_x$ [cm]"},
    {"branch": "fastsim_vy", "bins": np.linspace(-0.5, 0.5, 121), "label": r"Particle $v_y$ [cm]"},
    {"branch": "fastsim_vz", "bins": np.linspace(-15, 15, 121), "label": r"Particle $v_z$ [cm]"},
    {"branch": "fastsim_class", "bins": np.arange(-0.5, 6.5, 1.0), "label": "Particle class"},
]

PARTICLE_RESIDUALS = [
    {
        "fast_branch": "fastsim_pt",
        "pf_branch": "pflow_pt",
        "bins": np.linspace(-100, 100, 121),
        "label": r"Residual particle $p_T$ [GeV]",
    },
    {
        "fast_branch": "fastsim_eta",
        "pf_branch": "pflow_eta",
        "bins": np.linspace(-1.5, 1.5, 121),
        "label": r"Residual particle $\eta$",
    },
    {
        "fast_branch": "fastsim_phi",
        "pf_branch": "pflow_phi",
        "bins": np.linspace(-np.pi, np.pi, 121),
        "label": r"Residual particle $\phi$",
    },
    {
        "fast_branch": "fastsim_vx",
        "pf_branch": "pflow_vx",
        "bins": np.linspace(-0.2, 0.2, 121),
        "label": r"Residual particle $v_x$ [cm]",
    },
    {
        "fast_branch": "fastsim_vy",
        "pf_branch": "pflow_vy",
        "bins": np.linspace(-0.2, 0.2, 121),
        "label": r"Residual particle $v_y$ [cm]",
    },
    {
        "fast_branch": "fastsim_vz",
        "pf_branch": "pflow_vz",
        "bins": np.linspace(-5, 5, 121),
        "label": r"Residual particle $v_z$ [cm]",
    },
]


def load_flat(path: Path, branch: str, entry_stop: int | None) -> np.ndarray:
    with uproot.open(path) as f:
        arr = f["evt_tree"][branch].array(entry_stop=entry_stop, library="ak")
    if isinstance(arr, ak.Array):
        arr = ak.flatten(arr, axis=None)
    return np.asarray(arr, dtype=np.float64)


def load_residual(path: Path, fast_branch: str, pf_branch: str, entry_stop: int | None) -> np.ndarray:
    with uproot.open(path) as f:
        arrays = f["evt_tree"].arrays([fast_branch, pf_branch], entry_stop=entry_stop, library="ak")
    diff = arrays[fast_branch] - arrays[pf_branch]
    return np.asarray(ak.flatten(diff, axis=None), dtype=np.float64)


def ensure_bins(bins, data):
    if callable(bins):
        return bins(data)
    return bins


def plot_grid(configs, loader, mixed_path, ref_path, entry_stop, title, outfile):
    n = len(configs)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    axes = axes.flatten()

    for ax, cfg in zip(axes, configs):
        if loader == "residual":
            ref_vals = load_residual(ref_path, cfg["fast_branch"], cfg["pf_branch"], entry_stop)
            mixed_vals = load_residual(mixed_path, cfg["fast_branch"], cfg["pf_branch"], entry_stop)
        else:
            ref_vals = load_flat(ref_path, cfg["branch"], entry_stop)
            mixed_vals = load_flat(mixed_path, cfg["branch"], entry_stop)

        bins = ensure_bins(cfg["bins"], ref_vals)
        ax.hist(
            ref_vals,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=CmsColor,
            alpha=0.45,
            label="CMS PF",
        )
        ax.hist(
            mixed_vals,
            bins=bins,
            density=True,
            histtype="step",
            color=FmColor,
            linewidth=2.0,
            label="Parnassus FM",
        )
        ax.set_xlabel(cfg["label"])
        ax.set_ylabel("a.u.")
        if cfg.get("logy"):
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        fig.delaxes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}")


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay mixed fast-sim vs reference PF distributions.")
    parser.add_argument("--mixed", required=True, type=Path, help="Path to mixed fast-sim file")
    parser.add_argument("--reference", required=True, type=Path, help="Path to reference PF file")
    parser.add_argument("--output-dir", type=Path, default=Path("mixed_vs_reference_plots"))
    parser.add_argument("--max-events", type=int, default=None, help="Optional event cap")
    return parser.parse_args()


def main():
    args = parse_args()

    plot_grid(
        EVENT_PLOTS,
        loader="simple",
        mixed_path=args.mixed,
        ref_path=args.reference,
        entry_stop=args.max_events,
        title="Event-level features",
        outfile=args.output_dir / "event_features.png",
    )
    plot_grid(
        JET_PLOTS,
        loader="simple",
        mixed_path=args.mixed,
        ref_path=args.reference,
        entry_stop=args.max_events,
        title="Jet-level features",
        outfile=args.output_dir / "jet_features.png",
    )
    plot_grid(
        JET_RESIDUALS,
        loader="residual",
        mixed_path=args.mixed,
        ref_path=args.reference,
        entry_stop=args.max_events,
        title="Jet residuals (fastsim - PF)",
        outfile=args.output_dir / "jet_residuals.png",
    )
    plot_grid(
        PARTICLE_PLOTS,
        loader="simple",
        mixed_path=args.mixed,
        ref_path=args.reference,
        entry_stop=args.max_events,
        title="Particle-level features",
        outfile=args.output_dir / "particle_features.png",
    )
    plot_grid(
        PARTICLE_RESIDUALS,
        loader="residual",
        mixed_path=args.mixed,
        ref_path=args.reference,
        entry_stop=args.max_events,
        title="Particle residuals (fastsim - PF)",
        outfile=args.output_dir / "particle_residuals.png",
    )


if __name__ == "__main__":
    main()
