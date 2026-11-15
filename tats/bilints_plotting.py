"""Plotting functions for BilinTS experiments.

Generates publication-quality PDF figures matching the style of
``tats.plotting`` (same rcParams, colour palette, layout conventions).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Style constants (mirrors tats.plotting)
# ---------------------------------------------------------------------------

_FONT_TITLE = 14
_FONT_LABEL = 12
_FONT_LEGEND = 10
_FONT_TICK = 10
_FIGSIZE = (8, 5)
_GRID_ALPHA = 0.3
_PALETTE = "tab10"

# BilinTS variants get a visually distinct accent colour
_BILINTS_COLOR = "#1f77b4"     # default tab:blue
_BILINTS_WARM_COLOR = "#2ca02c"  # tab:green
_BASELINE_COLORS = {
    "Random": "#7f7f7f",
    "DescMatch": "#ff7f0e",
    "ParamMatch": "#d62728",
    "EpsGreedy": "#9467bd",
    "UCB": "#8c564b",
    "RepeatedTS": "#e377c2",
}


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "axes.titlesize": _FONT_TITLE,
            "axes.labelsize": _FONT_LABEL,
            "legend.fontsize": _FONT_LEGEND,
            "xtick.labelsize": _FONT_TICK,
            "ytick.labelsize": _FONT_TICK,
            "figure.figsize": _FIGSIZE,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _ensure_parent(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _color_for(name: str) -> str:
    """Return a colour for a selector name, highlighting BilinTS variants."""
    if name == "BilinTS-Cold":
        return _BILINTS_COLOR
    if name == "BilinTS-Warm":
        return _BILINTS_WARM_COLOR
    return _BASELINE_COLORS.get(name, "#17becf")


def _get_colors(names: list[str]) -> list[str]:
    return [_color_for(n) for n in names]


# ---------------------------------------------------------------------------
# 1. Main results: grouped bar chart of TSA across selectors
# ---------------------------------------------------------------------------

def plot_main_results(
    main: dict,
    output_path: str | Path,
    title: str = "BFCL Tool Selection Accuracy",
) -> None:
    """Grouped bar chart comparing selectors on mean TSA +/- std.

    Parameters
    ----------
    main:
        ``{selector_name: {"tsa_mean": float, "tsa_std": float}}``.
    output_path:
        PDF save path.
    title:
        Figure title.
    """
    _apply_style()
    out = _ensure_parent(output_path)

    names = list(main.keys())
    means = [main[n]["tsa_mean"] for n in names]
    stds = [main[n]["tsa_std"] for n in names]
    colors = _get_colors(names)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black",
                  linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Tool Selection Accuracy (TSA)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=_GRID_ALPHA, zorder=0)

    # Annotate bars with values
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{m:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Degradation matrix: grouped bar chart per degradation level
# ---------------------------------------------------------------------------

def plot_degradation_matrix(
    degradation: dict,
    output_path: str | Path,
    title: str = "TSA under Documentation Degradation",
) -> None:
    """Grouped bar chart with one cluster per degradation level.

    Parameters
    ----------
    degradation:
        ``{level: {selector: {"tsa_mean": float, "tsa_std": float}}}``.
        Levels are typically ``["Full", "L1", "L2", "L3"]``.
    output_path:
        PDF save path.
    title:
        Figure title.
    """
    _apply_style()
    out = _ensure_parent(output_path)

    levels = list(degradation.keys())
    # Collect the union of selector names preserving first-seen order
    selectors: list[str] = []
    for lvl in levels:
        for s in degradation[lvl]:
            if s not in selectors:
                selectors.append(s)

    n_levels = len(levels)
    n_selectors = len(selectors)
    colors = _get_colors(selectors)

    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.8 / n_selectors
    x_base = np.arange(n_levels)

    for i, sel in enumerate(selectors):
        means = []
        stds = []
        for lvl in levels:
            entry = degradation[lvl].get(sel, {"tsa_mean": 0.0, "tsa_std": 0.0})
            means.append(entry["tsa_mean"])
            stds.append(entry["tsa_std"])
        offset = (i - (n_selectors - 1) / 2) * bar_width
        ax.bar(
            x_base + offset, means, bar_width,
            yerr=stds, capsize=3,
            color=colors[i], edgecolor="black", linewidth=0.4,
            label=sel, zorder=3,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(levels)
    ax.set_xlabel("Degradation Level")
    ax.set_ylabel("Tool Selection Accuracy (TSA)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=_GRID_ALPHA, zorder=0)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Per-round TSA line chart (repeated BFCL)
# ---------------------------------------------------------------------------

def plot_per_round_tsa(
    repeated: dict,
    output_path: str | Path,
    title: str = "Per-Round TSA (Repeated BFCL)",
) -> None:
    """Line chart of TSA per interaction round.

    Parameters
    ----------
    repeated:
        ``{selector: {"tsa_mean": float, "tsa_std": float,
        "per_round_tsa": [float, ...]}}``.
    output_path:
        PDF save path.
    title:
        Figure title.
    """
    _apply_style()
    out = _ensure_parent(output_path)

    selectors = list(repeated.keys())
    colors = _get_colors(selectors)
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, sel in enumerate(selectors):
        tsa_per_round = repeated[sel]["per_round_tsa"]
        rounds = np.arange(1, len(tsa_per_round) + 1)
        marker = markers[i % len(markers)]
        ax.plot(
            rounds, tsa_per_round,
            marker=marker, markersize=6,
            linewidth=1.5, color=colors[i],
            label=sel,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel("Tool Selection Accuracy (TSA)")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(rounds)
    ax.legend(loc="best", ncol=2)
    ax.grid(True, alpha=_GRID_ALPHA)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Ablation: horizontal bar chart
# ---------------------------------------------------------------------------

def plot_ablation(
    ablation: dict,
    output_path: str | Path,
    title: str = "BilinTS Ablation Study",
) -> None:
    """Horizontal bar chart showing ablation impact on TSA.

    Parameters
    ----------
    ablation:
        ``{variant_name: {"tsa_mean": float, "tsa_std": float}}``.
        The full model should be named ``"BilinTS (full)"`` or similar;
        it will be coloured differently from the ablated variants.
    output_path:
        PDF save path.
    title:
        Figure title.
    """
    _apply_style()
    out = _ensure_parent(output_path)

    names = list(ablation.keys())
    means = [ablation[n]["tsa_mean"] for n in names]
    stds = [ablation[n]["tsa_std"] for n in names]

    # Full model gets blue, ablations get grey
    colors = [
        _BILINTS_COLOR if "full" in n.lower() or n.startswith("BilinTS") and "no" not in n.lower()
        else "#aaaaaa"
        for n in names
    ]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.6 * len(names))))

    ax.barh(y, means, xerr=stds, capsize=3,
            color=colors, edgecolor="black", linewidth=0.4, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Tool Selection Accuracy (TSA)")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=_GRID_ALPHA, zorder=0)
    ax.invert_yaxis()  # highest at top

    # Annotate with numeric values
    for yi, m in zip(y, means):
        ax.text(m + 0.015, yi, f"{m:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Feature weights across degradation levels
# ---------------------------------------------------------------------------

_DEFAULT_FEATURE_NAMES = [
    "desc_sim", "name_sim", "param_match", "n_params",
    "desc_len", "n_cand", "req_ratio", "has_desc",
]


def plot_feature_weights(
    theta_dict: dict[str, list | np.ndarray],
    feature_names: list[str] | None = None,
    output_path: str | Path = "figures/bilints_feature_weights.pdf",
    title: str = "Learned Feature Weights by Degradation Level",
) -> None:
    """Grouped bar chart (or heatmap) of theta weights across degradation levels.

    Parameters
    ----------
    theta_dict:
        ``{degradation_level: theta_vector}``.  Each theta vector has one
        weight per feature.
    feature_names:
        Human-readable feature names.  Defaults to the 8-feature set.
    output_path:
        PDF save path.
    title:
        Figure title.
    """
    _apply_style()
    out = _ensure_parent(output_path)

    if feature_names is None:
        feature_names = _DEFAULT_FEATURE_NAMES

    levels = list(theta_dict.keys())
    n_features = len(feature_names)
    n_levels = len(levels)

    # Build matrix (levels x features)
    matrix = np.array([np.asarray(theta_dict[lvl])[:n_features] for lvl in levels])

    # Use heatmap when there are many features, grouped bars otherwise
    if n_features > 10:
        # Heatmap
        fig, ax = plt.subplots(figsize=(max(8, 0.8 * n_features), max(4, 0.6 * n_levels)))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r")
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.set_yticks(range(n_levels))
        ax.set_yticklabels(levels)
        fig.colorbar(im, ax=ax, label="Weight")
        ax.set_title(title)
    else:
        # Grouped bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_width = 0.8 / n_levels
        x_base = np.arange(n_features)
        cmap = plt.get_cmap("tab10")
        level_colors = [cmap(i) for i in range(n_levels)]

        for i, lvl in enumerate(levels):
            offset = (i - (n_levels - 1) / 2) * bar_width
            ax.bar(
                x_base + offset, matrix[i], bar_width,
                color=level_colors[i], edgecolor="black", linewidth=0.4,
                label=lvl, zorder=3,
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels(feature_names, rotation=35, ha="right")
        ax.set_ylabel("Weight")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=_GRID_ALPHA, zorder=0)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Learning curves: cumulative TSA over interactions
# ---------------------------------------------------------------------------

def plot_learning_curves(
    results: dict[str, list[list[float]]],  # name -> [seed_0_curve, seed_1_curve, ...]
    output_path: str,
    title: str = "Learning Curves",
) -> None:
    """Plot cumulative TSA over interactions with mean +/- std bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, curves in results.items():
        curves_arr = np.array(curves)  # (n_seeds, n_steps)
        mean = curves_arr.mean(axis=0)
        std = curves_arr.std(axis=0)
        steps = np.arange(1, len(mean) + 1)
        line, = ax.plot(steps, mean, label=name, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=line.get_color())

    ax.set_xlabel("Interaction", fontsize=12)
    ax.set_ylabel("Cumulative TSA", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# 7. Feature weight heatmap across conditions
# ---------------------------------------------------------------------------

def plot_feature_weight_heatmap(
    weights: dict[str, dict[str, float]],  # condition -> {feature_name: weight}
    output_path: str,
    title: str = "Feature Weights Across Conditions",
) -> None:
    """Plot feature weights as a heatmap across degradation levels or methods."""
    conditions = list(weights.keys())
    feature_names = list(next(iter(weights.values())).keys())
    data = np.array([[weights[c][f] for f in feature_names] for c in conditions])

    fig, ax = plt.subplots(figsize=(10, max(3, len(conditions) * 0.6 + 1)))
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto",
                   vmin=-np.abs(data).max(), vmax=np.abs(data).max())

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=10)

    # Annotate cells with values
    for i in range(len(conditions)):
        for j in range(len(feature_names)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="Weight")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Convenience: generate all plots from a single results JSON
# ---------------------------------------------------------------------------

def generate_all_plots(
    results_path: str | Path,
    output_dir: str | Path = "figures",
) -> None:
    """Load results JSON and generate all available BilinTS figures.

    Parameters
    ----------
    results_path:
        Path to ``bilints_results.json``.
    output_dir:
        Directory where PDFs will be written.
    """
    results_path = Path(results_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    if "main" in results:
        plot_main_results(results["main"], output_dir / "bilints_main.pdf")
    if "degradation" in results:
        plot_degradation_matrix(results["degradation"], output_dir / "bilints_degradation.pdf")
    if "repeated" in results:
        plot_per_round_tsa(results["repeated"], output_dir / "bilints_repeated.pdf")
    if "ablation" in results:
        plot_ablation(results["ablation"], output_dir / "bilints_ablation.pdf")
    if "feature_weights" in results:
        plot_feature_weights(results["feature_weights"], output_path=output_dir / "bilints_feature_weights.pdf")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    rpath = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/bilints_results.json")
    odir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures")
    generate_all_plots(rpath, odir)
    print(f"Plots saved to {odir}/")
