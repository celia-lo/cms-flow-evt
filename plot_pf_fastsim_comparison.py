#!/usr/bin/env python3
"""
Compare ALEPH PF truth, Parnassus fast-sim, and Delphes fast-sim in a single figure set.
"""

import argparse
import math
from pathlib import Path

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np
import uproot


ALEPH_COLOR = "#1f77b4"
PARNASSUS_COLOR = "crimson"
DELPHES_COLOR = "#2ca02c"


def load_branch(path, branch, entry_stop):
    with uproot.open(path) as f:
        arr = f["evt_tree"][branch].array(entry_stop=entry_stop, library="ak")
    if isinstance(arr, ak.Array):
        arr = ak.flatten(arr, axis=None)
    return np.asarray(arr, dtype=np.float64)


def load_residual(pf_path, sim_path, branch, entry_stop):
    with uproot.open(pf_path) as f_ref, uproot.open(sim_path) as f_sim:
        ref = f_ref["evt_tree"][branch].array(entry_stop=entry_stop, library="ak")
        sim = f_sim["evt_tree"][branch].array(entry_stop=entry_stop, library="ak")
    diff = ak.flatten(sim - ref, axis=None)
    return np.asarray(diff, dtype=np.float64)


EVENT_PLOTS = [
    ("npflow", np.arange(-0.5, 1000.5, 10), r"$N_{\mathrm{PF}}$"),
]

JET_PLOTS = [
    ("pflow_pt", np.linspace(0, 200, 121), r"PF jet constituent $p_T$ [GeV]"),
    ("pflow_eta", np.linspace(-3, 3, 121), r"PF jet constituent $\eta$"),
    ("pflow_phi", np.linspace(-np.pi, np.pi, 121), r"PF jet constituent $\phi$"),
]

PARTICLE_PLOTS = [
    ("truth_pt", np.linspace(0, 200, 121), r"Truth particle $p_T$ [GeV]", True),
    ("truth_eta", np.linspace(-3, 3, 121), r"Truth particle $\eta$", False),
    ("truth_phi", np.linspace(-np.pi, np.pi, 121), r"Truth particle $\phi$", False),
    ("truth_vz", np.linspace(-15, 15, 121), r"Truth particle $v_z$ [cm]", False),
]


def plot_grid(configs, loader, args, title, outfile):
    n = len(configs)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows), squeeze=False)
    axes = axes.flatten()

    for ax, cfg in zip(axes, configs):
        branch = cfg[0]
        bins = cfg[1]
        label = cfg[2]
        logy = cfg[3] if len(cfg) > 3 else False

        pf_vals = load_branch(args.pf, branch, args.max_events)
        parn_vals = load_branch(args.parnassus, branch, args.max_events)
        delp_vals = load_branch(args.delphes, branch, args.max_events)

        ax.hist(
            pf_vals,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=ALEPH_COLOR,
            alpha=0.4,
            label="ALEPH PF",
        )
        ax.hist(
            parn_vals,
            bins=bins,
            density=True,
            histtype="step",
            color=PARNASSUS_COLOR,
            linewidth=2.0,
            label="Parnassus",
        )
        ax.hist(
            delp_vals,
            bins=bins,
            density=True,
            histtype="step",
            color=DELPHES_COLOR,
            linewidth=2.0,
            label="Delphes",
        )
        ax.set_xlabel(label)
        ax.set_ylabel("a.u.")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    for ax in axes[n:]:
        fig.delaxes(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=14, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PF truth vs Parnassus vs Delphes.")
    parser.add_argument("--pf", required=True, type=Path, help="ALEPH PF reference (evt_tree)")
    parser.add_argument("--parnassus", required=True, type=Path, help="Parnassus fast-sim")
    parser.add_argument("--delphes", required=True, type=Path, help="Delphes fast-sim")
    parser.add_argument("--output-dir", type=Path, default=Path("aleph_fastcomp_plots"))
    parser.add_argument("--max-events", type=int, default=None, help="Event cap")
    return parser.parse_args()


def main():
    args = parse_args()

    plot_grid(
        EVENT_PLOTS,
        loader="simple",
        args=args,
        title="Event-level comparison",
        outfile=args.output_dir / "event_features.png",
    )
    plot_grid(
        JET_PLOTS,
        loader="simple",
        args=args,
        title="Jet-level comparison",
        outfile=args.output_dir / "jet_features.png",
    )
    plot_grid(
        PARTICLE_PLOTS,
        loader="simple",
        args=args,
        title="Particle-level comparison",
        outfile=args.output_dir / "particle_features.png",
    )


if __name__ == "__main__":
    main()
