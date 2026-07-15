
#!/usr/bin/env python3
"""Compare two postprocessed CMS Flow Matching samples and save plots."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from parnassus_core.experiments.cms_2011.performance.performance import (
    CMS2011Performance,
    CMS2011PerformanceConfig,
    CMS2011Sample,
    CMS2011PlotHelper,
)
from parnassus_core.performance.config import Source


def build_config(sample_key: str, entries: list[dict]) -> CMS2011PerformanceConfig:
    config_dict = {"samples": {sample_key: entries}}
    return CMS2011PerformanceConfig.from_dict(config_dict)


def load_performance(config: CMS2011PerformanceConfig) -> CMS2011Performance:
    perf = CMS2011Performance(config)
    perf.load()
    perf.process()
    return perf


def make_style_dict(names: list[str], colors: list[str]) -> dict:
    style = {}
    for name, color in zip(names, colors):
        style[name] = {"color": color, "linewidth": 2.0}
    style[Source.PF.name] = {"color": "#1f77b4", "alpha": 0.35}
    return style


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two postprocessed ROOT files")
    parser.add_argument("first", type=Path, help="Path to the first postprocessed ROOT file")
    parser.add_argument("second", type=Path, help="Path to the second postprocessed ROOT file")
    parser.add_argument("--first-label", default="Mixed HS⊕MB", help="Legend label for the first file")
    parser.add_argument("--second-label", default="Reference Mixed QCD", help="Legend label for the second file")
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_plots"), help="Directory for output PNGs")
    parser.add_argument("--sample-key", default="ttbar", help="CMS2011 sample key (default: ttbar)")
    parser.add_argument("--num-events", type=int, default=None, help="Optional number of events to load from each file")
    args = parser.parse_args()

    for path in (args.first, args.second):
        if not path.exists():
            parser.error(f"File not found: {path}")

    entries = [
        {
            "name": "first",
            "label": args.first_label,
            "path": str(args.first),
            "pflow_type": "fm",
            "num_events": args.num_events,
        },
        {
            "name": "second",
            "label": args.second_label,
            "path": str(args.second),
            "pflow_type": "fm",
            "num_events": args.num_events,
        },
    ]

    config = build_config(args.sample_key, entries)
    performance = load_performance(config)

    style_dict = make_style_dict(["first", "second"], ["crimson", "#2ca02c"])
    plot_helper = CMS2011PlotHelper(performance, plot_truth=False, style_dict=style_dict)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        (plot_helper.plot_event_features, {}, "event_features"),
        (plot_helper.plot_event_features, {"residuals": True}, "event_residuals"),
        (plot_helper.plot_jet_features, {"log_scale_vars": ["pt", "d2", "c2"]}, "jet_features"),
        (plot_helper.plot_jet_features, {"residuals": True}, "jet_residuals"),
        (
            plot_helper.plot_particle_features,
            {"log_scale_vars": ["pt", "vx", "vy", "vz"]},
            "particle_features",
        ),
        (
            plot_helper.plot_particle_features,
            {"residuals": True},
            "particle_residuals",
        ),
    ]

    for plot_fn, kwargs, stem in plots:
        fig = plot_fn(**kwargs)
        fig.tight_layout()
        output_path = output_dir / f"{stem}.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
