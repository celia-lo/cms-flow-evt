#!/usr/bin/env python3
"""Plot particle class confusion matrices for one or two postprocessed ROOT files."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from parnassus_core.experiments.cms_2011.performance.performance import (
    CMS2011Performance,
    CMS2011PerformanceConfig,
    CMS2011PlotHelper,
)
from parnassus_core.performance.config import Source


def make_style_dict(names: list[str], colors: list[str]) -> dict:
    style = {}
    for name, color in zip(names, colors):
        style[name] = {"color": color, "linewidth": 2.0}
    style[Source.PF.name] = {"color": "#1f77b4", "alpha": 0.35}
    return style


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot particle class confusion matrices from postprocessed ROOT files."
    )
    parser.add_argument("first", type=Path, help="First postprocessed ROOT file")
    parser.add_argument(
        "second", type=Path, nargs="?", default=None,
        help="Optional second postprocessed ROOT file"
    )
    parser.add_argument("--first-label", default="Parnassus", help="Legend label for the first file")
    parser.add_argument("--second-label", default="Parnassus (12M)", help="Legend label for the second file")
    parser.add_argument("--output", type=Path, default=Path("particle_classes.png"), help="Output PNG path")
    parser.add_argument("--sample-key", default="zqqb", help="Sample key (default: zqqb)")
    parser.add_argument("--num-events", type=int, default=None, help="Optional event cap")
    return parser.parse_args()


def main():
    args = parse_args()

    entries = [
        {
            "name": "first",
            "label": args.first_label,
            "path": str(args.first),
            "pflow_type": "fm",
            "num_events": args.num_events,
        },
    ]
    names = ["first"]
    colors = ["crimson"]

    if args.second is not None:
        entries.append(
            {
                "name": "second",
                "label": args.second_label,
                "path": str(args.second),
                "pflow_type": "fm",
                "num_events": args.num_events,
            }
        )
        names.append("second")
        colors.append("#2ca02c")

    config = CMS2011PerformanceConfig.from_dict(
        {"samples": {args.sample_key: entries}}
    )
    performance = CMS2011Performance(config)
    performance.load()
    performance.process()

    style_dict = make_style_dict(names, colors)
    plot_helper = CMS2011PlotHelper(performance, plot_truth=False, style_dict=style_dict)

    fig = plot_helper.plot_classes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
