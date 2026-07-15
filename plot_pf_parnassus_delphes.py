#!/usr/bin/env python3
"""
Reproduce Plot_postprocess-style panels with ALEPH PF, Parnassus fast-sim, and Delphes fast-sim.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from parnassus_core.experiments.cms_2011.performance.performance import (
    CMS2011Performance,
    CMS2011PerformanceConfig,
    CMS2011PlotHelper,
    CMS2011Sample,
)
from parnassus_core.performance import labels as perf_labels

# Re-map the sample label away from the CMS-oriented ttbar text.
perf_labels.SAMPLE_LABELS["ttbar"] = r"$Z\to q\bar{q}$"
perf_labels.SAMPLE_LABELS["zqqb"] = r"$Z\to q\bar{q}$"
perf_labels.VAR_LABELS.setdefault("m_vis", r"$M_{\mathrm{vis}}$")
perf_labels.VAR_LABELS.setdefault("thrust", r"$T$")
perf_labels.RESIDUAL_VAR_LABELS.setdefault(
    "m_vis_res", r"$M_{\mathrm{vis}}^{\mathrm{reco}} - M_{\mathrm{vis}}^{\mathrm{truth}}$"
)
perf_labels.RESIDUAL_VAR_LABELS.setdefault(
    "thrust_res", r"$T^{\mathrm{reco}} - T^{\mathrm{truth}}$"
)
perf_labels.VAR_LABELS["ht"] = r"$H_T$ [GeV]"
perf_labels.VAR_LABELS["met_x"] = r"$E_x^{\mathrm{miss}}$ [GeV]"
perf_labels.VAR_LABELS["met_y"] = r"$E_y^{\mathrm{miss}}$ [GeV]"
perf_labels.VAR_LABELS["m_vis"] = r"$M_{\mathrm{vis}}$ [GeV]"
perf_labels.VAR_LABELS["pt"] = r"$p_T$ [GeV]"
perf_labels.VAR_LABELS["vx"] = r"$v_x$ [cm]"
perf_labels.VAR_LABELS["vy"] = r"$v_y$ [cm]"
perf_labels.VAR_LABELS["vz"] = r"$v_z$ [cm]"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay ALEPH PF, Parnassus, and Delphes fast-sim distributions."
    )
    parser.add_argument(
        "--parnassus",
        required=True,
        type=str,
        help="Path to Parnassus postprocessed ROOT (evt_tree) file.",
    )
    parser.add_argument(
        "--delphes",
        required=True,
        type=str,
        help="Path to Delphes-converted ROOT (evt_tree) file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="aleph_pf_parn_delphes_plots",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=22,
        help="Matplotlib font size for the helper.",
    )
    return parser.parse_args()


def build_config(parnassus_path, delphes_path):
    config_dict = {
        "samples": {
            "zqqb": [
                {
                    "name": "parnassus",
                    "label": "Parnassus",
                    "path": str(parnassus_path),
                    "pflow_type": "fm",
                },
                {
                    "name": "delphes",
                    "label": "Delphes",
                    "path": str(delphes_path),
                    "pflow_type": "fm",
                },
            ]
        }
    }
    return CMS2011PerformanceConfig.from_dict(config_dict)


def save_fig(fig, path):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def recenter_legend(fig, ncol=3, fontsize=22):
    handles = None
    labels = None
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is None:
            continue
        handles, labels = ax.get_legend_handles_labels()
        lg.remove()
        break
    if handles:
        sp = fig.subplotpars
        axes_center = (sp.left + sp.right) / 2
        fig.legend(
            handles=handles,
            labels=labels,
            loc="center",
            bbox_to_anchor=(axes_center, 0.90),
            ncol=ncol,
            frameon=True,
            fancybox=False,
            edgecolor="0.7",
            facecolor="white",
            fontsize=fontsize,
            handlelength=2.5,
            columnspacing=2.5,
            borderpad=0.4,
        )


def remove_subplot_legends(fig):
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is not None:
            lg.remove()


def main():
    args = parse_args()
    config = build_config(args.parnassus, args.delphes)
    perf = CMS2011Performance(config)
    perf.load()
    perf.process()
    def _compute_mass_thrust(particle_dict):
        masses, thrusts = [], []
        for pt, eta, phi in zip(
            particle_dict["pt"], particle_dict["eta"], particle_dict["phi"]
        ):
            pt = np.asarray(pt)
            if pt.size == 0:
                masses.append(0.0)
                thrusts.append(0.0)
                continue
            eta = np.asarray(eta)
            phi = np.asarray(phi)
            px = pt * np.cos(phi)
            py = pt * np.sin(phi)
            pz = pt * np.sinh(eta)
            energy = np.sqrt(px**2 + py**2 + pz**2)
            mass_sq = max(
                energy.sum() ** 2 - (px.sum() ** 2 + py.sum() ** 2 + pz.sum() ** 2), 0.0
            )
            masses.append(np.sqrt(mass_sq))
            momenta = np.stack([px, py, pz], axis=1)
            norms = np.linalg.norm(momenta, axis=1)
            denom = norms.sum()
            if denom <= 0:
                thrusts.append(0.0)
                continue
            axis = momenta[np.argmax(norms)] / (norms.max() + 1e-12)
            for _ in range(50):
                dots = momenta @ axis
                signs = np.where(dots >= 0, 1.0, -1.0)
                axis_new = (momenta * signs[:, None]).sum(axis=0)
                norm = np.linalg.norm(axis_new)
                if norm <= 1e-12:
                    break
                axis_new /= norm
                if np.linalg.norm(axis_new - axis) < 1e-6:
                    axis = axis_new
                    break
                axis = axis_new
            thrusts.append(np.abs(momenta @ axis).sum() / denom)
        return np.asarray(masses, dtype=np.float32), np.asarray(thrusts, dtype=np.float32)

    for sample in perf.sample_data_dict.values():
        for data in sample.data_dict.values():
            mass_vals, thrust_vals = _compute_mass_thrust(data.particle_data_all)
            data.event_data["m_vis"] = mass_vals
            data.event_data["thrust"] = thrust_vals

        truth_mass = sample.data_dict["TR"].event_data["m_vis"]
        truth_thrust = sample.data_dict["TR"].event_data["thrust"]
        for data in sample.data_dict.values():
            if data.source.name == "TR":
                data.event_data["m_vis_res"] = np.zeros_like(truth_mass)
                data.event_data["thrust_res"] = np.zeros_like(truth_thrust)
            else:
                data.event_data["m_vis_res"] = data.event_data["m_vis"] - truth_mass
                data.event_data["thrust_res"] = data.event_data["thrust"] - truth_thrust

    style_dict = {
        "PF": {"linewidth": 2, "color": "#1f77b4", "alpha": 0.35},
        "parnassus": {"linewidth": 2, "color": "crimson"},
        "delphes": {"linewidth": 2, "color": "#2ca02c", "linestyle": "--"},
    }

    plot_helper = CMS2011PlotHelper(
        perf, plot_truth=False, style_dict=style_dict, font_size=args.font_size
    )

    # ── Bins (all figures consolidated here) ──────────────────────────────
    event_bins = plot_helper.bins_config.event_bins
    # Fig2: npart, njet, HT
    event_bins.setdefault("npart", {})[CMS2011Sample.ZQQB] = np.linspace(0, 60, 31)
    event_bins.setdefault("njet", {})[CMS2011Sample.ZQQB] = np.arange(0, 7)
    event_bins.setdefault("ht", {})[CMS2011Sample.ZQQB] = np.linspace(0, 200, 101)
    # Fig3: MET, M_vis, thrust
    event_bins.setdefault("met_x", {})[CMS2011Sample.ZQQB] = np.linspace(-40, 40, 81)
    event_bins.setdefault("met_y", {})[CMS2011Sample.ZQQB] = np.linspace(-40, 40, 81)
    event_bins.setdefault("m_vis", {})[CMS2011Sample.ZQQB] = np.linspace(0, 200, 121)
    event_bins.setdefault("thrust", {})[CMS2011Sample.ZQQB] = np.linspace(0.4, 1.0, 61)

    event_res_bins = plot_helper.bins_config.event_res_bins
    event_res_bins.setdefault("npart_res", {})[CMS2011Sample.ZQQB] = np.linspace(-30, 10, 41)
    event_res_bins.setdefault("ht_res", {})[CMS2011Sample.ZQQB] = np.linspace(-1, 1, 101)
    event_res_bins.setdefault("met_x_res", {})[CMS2011Sample.ZQQB] = np.linspace(-10, 10, 101)
    event_res_bins.setdefault("met_y_res", {})[CMS2011Sample.ZQQB] = np.linspace(-10, 10, 101)
    event_res_bins.setdefault("m_vis_res", {})[CMS2011Sample.ZQQB] = np.linspace(-100, 100, 101)
    event_res_bins.setdefault("thrust_res", {})[CMS2011Sample.ZQQB] = np.linspace(-0.3, 0.3, 101)

    # Fig4: jet pt and residuals
    jet_bins = plot_helper.bins_config.jet_bins
    jet_bins.setdefault("pt", {})[CMS2011Sample.ZQQB] = np.linspace(20, 140, 61)
    jet_res_bins = plot_helper.bins_config.jet_res_bins
    jet_res_bins.setdefault("pt_res", {})[CMS2011Sample.ZQQB] = np.linspace(-1, 1, 101)
    jet_res_bins.setdefault("eta_res", {})[CMS2011Sample.ZQQB] = np.linspace(-0.2, 0.2, 101)
    jet_res_bins.setdefault("phi_res", {})[CMS2011Sample.ZQQB] = np.linspace(-0.2, 0.2, 101)
    jet_res_bins.setdefault("d2_res", {})[CMS2011Sample.ZQQB] = np.linspace(-2, 2, 101)
    jet_res_bins.setdefault("c2_res", {})[CMS2011Sample.ZQQB] = np.linspace(-0.5, 0.5, 101)

    # Fig5/6: particle pt and vz residual
    particle_bins = plot_helper.bins_config.particle_bins
    particle_res_bins = plot_helper.bins_config.particle_res_bins
    particle_bins.setdefault("pt", {})[CMS2011Sample.ZQQB] = np.linspace(0, 100, 101)
    particle_res_bins.setdefault("vz_res", {})[CMS2011Sample.ZQQB] = np.linspace(-1, 1, 101)

    # ── Fig2: npart, njet, HT (features + residuals) ───────────────────────
    event_counts = ["npart", "njet", "ht"]
    fig = plot_helper.plot_features("event_data", event_counts, bins=event_bins)
    recenter_legend(fig, ncol=3, fontsize=args.font_size)
    save_fig(fig, os.path.join(args.output_dir, "event_counts.png"))

    event_counts_res = [f"{var}_res" for var in event_counts]
    fig = plot_helper.plot_features(
        "event_data", event_counts_res, bins=event_res_bins,
        log_scale_vars=["ht_res"],
    )
    remove_subplot_legends(fig)
    save_fig(fig, os.path.join(args.output_dir, "event_counts_residuals.png"))

    # ── Fig3: MET, M_vis, thrust (features + residuals) ────────────────────
    event_met = ["met_x", "met_y", "m_vis", "thrust"]
    fig = plot_helper.plot_features("event_data", event_met, bins=event_bins)
    recenter_legend(fig, ncol=3, fontsize=args.font_size)
    save_fig(fig, os.path.join(args.output_dir, "event_met_mvis_thrust.png"))

    event_met_res = [f"{var}_res" for var in event_met]
    fig = plot_helper.plot_features(
        "event_data", event_met_res, bins=event_res_bins,
        log_scale_vars=event_met_res,
    )
    remove_subplot_legends(fig)
    save_fig(fig, os.path.join(args.output_dir, "event_met_mvis_thrust_residuals.png"))

    # ── Combined event figure (MandT) ──────────────────────────────────────
    aleph_event_vars = ["npart", "njet", "ht", "met_x", "met_y", "m_vis", "thrust"]
    fig = plot_helper.plot_features("event_data", aleph_event_vars, bins=event_bins)
    recenter_legend(fig, ncol=3, fontsize=args.font_size)
    save_fig(fig, os.path.join(args.output_dir, "MandT_feature.png"))

    aleph_event_res_vars = [f"{var}_res" for var in aleph_event_vars]
    aleph_res_log_vars = ["ht_res", "met_x_res", "met_y_res", "m_vis_res", "thrust_res"]
    fig = plot_helper.plot_features(
        "event_data", aleph_event_res_vars, bins=event_res_bins,
        log_scale_vars=aleph_res_log_vars,
    )
    remove_subplot_legends(fig)
    save_fig(fig, os.path.join(args.output_dir, "MandT_residual.png"))

    # ── Fig4: jet observables (features + residuals) ───────────────────────
    fig = plot_helper.plot_jet_features(log_scale_vars=["pt", "d2", "c2"])
    recenter_legend(fig, ncol=3, fontsize=args.font_size)
    save_fig(fig, os.path.join(args.output_dir, "jet_features.png"))

    fig = plot_helper.plot_jet_features(
        residuals=True,
        log_scale_vars=["pt_res", "eta_res", "phi_res", "d2_res", "c2_res"],
    )
    remove_subplot_legends(fig)
    save_fig(fig, os.path.join(args.output_dir, "jet_residuals.png"))

    # ── Fig5: particle kinematics pt/eta/phi (features + residuals) ────────
    particle_kinematics = ["pt", "eta", "phi"]
    fig = plot_helper.plot_features(
        "particle_data_all",
        particle_kinematics,
        bins=particle_bins,
        log_scale_vars=["pt"],
    )
    recenter_legend(fig, ncol=3, fontsize=args.font_size)
    save_fig(fig, os.path.join(args.output_dir, "particle_kinematics.png"))

    particle_kin_res = [f"{var}_res" for var in particle_kinematics]
    fig = plot_helper.plot_features(
        "particle_data_all",
        particle_kin_res,
        bins=particle_res_bins,
        log_scale_vars=particle_kin_res,
        xlabel_prefix="Residual ",
    )
    remove_subplot_legends(fig)
    save_fig(fig, os.path.join(args.output_dir, "particle_kinematics_residuals.png"))

    # ── Fig6: particle vertices vx/vy/vz (all log scale) ─────────────────
    particle_vertices = ["vx", "vy", "vz"]
    fig = plot_helper.plot_features(
        "particle_data_all",
        particle_vertices,
        bins=particle_bins,
        log_scale_vars=particle_vertices,
    )
    recenter_legend(fig, ncol=3, fontsize=args.font_size)
    save_fig(fig, os.path.join(args.output_dir, "particle_vertices.png"))

    particle_vert_res = [f"{var}_res" for var in particle_vertices]
    fig = plot_helper.plot_features(
        "particle_data_all",
        particle_vert_res,
        bins=particle_res_bins,
        log_scale_vars=particle_vert_res,
        xlabel_prefix="Residual ",
    )
    remove_subplot_legends(fig)
    save_fig(fig, os.path.join(args.output_dir, "particle_vertices_residuals.png"))

    # ── Particle class confusion matrices + overall histogram ──────────────
    fig = plot_helper.plot_classes()
    save_fig(fig, os.path.join(args.output_dir, "particle_classes.png"))


if __name__ == "__main__":
    main()
