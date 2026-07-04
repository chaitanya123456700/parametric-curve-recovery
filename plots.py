"""
plots.py — diagnostic visualizations for the Flam pipeline result.

Usage:
    python plots.py xy_data.csv

Produces (saved as PNGs, and shown if run interactively):

  1. fit_overlay.png          Data scatter + fitted curve in (x, y) space.
                              The single most important plot: does the
                              curve actually pass through the cloud?

  2. residual_histogram.png   Histogram of per-point distances to the fitted
                              curve. Should be tightly clustered near 0 with
                              no heavy tail -- a tail means some points
                              aren't explained by this curve at all.

  3. residual_vs_t.png        Residual plotted against each point's matched
                              t. Flat/patternless = good fit everywhere.
                              A residual that grows with t, or spikes at
                              specific t ranges, points at a systematic
                              problem (e.g. amplitude/M slightly off, or a
                              wrong period count on sin(0.3t)).

  4. phase_convergence.png    True L1 cost at each pipeline stage (seed ->
                              scout -> refined -> polished), log-scale.
                              Should fall monotonically -- a plateau or
                              increase between phases is a bug signal.

  5. multiseed_agreement.png  Final (theta, M, X) from each independent
                              Phase 4 seed run, plotted as points. Tight
                              clustering = the multi-seed agreement claim,
                              visualized instead of just reported as a number.

  6. phase1a_diagnostic.png   Phase 1a's candidates: radius-space cost vs
                              true (x,y) cost, log-log. This is the plot
                              that actually shows the bug: candidates near
                              the degenerate M~=0 basin have LOW radius cost
                              but HIGH true cost -- visually demonstrating
                              why selecting by radius cost alone fails, and
                              why grounding against Phase 0 fixes it.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from common import load_data, curve_xy, curve_residuals, curve_residual_sum, DEFAULT_T_GRID
from phase1a_seed import phase1a_seed
from phase1_scout import phase1_scout
from phase2_refine import phase2_refine
from phase3_polish import phase3_polish
from phase4_validate import phase4_validate


def plot_fit_overlay(data_xy, params, savepath="fit_overlay.png"):
    theta, M, X = params
    t_dense = np.linspace(6, 60, 4000)
    cx, cy = curve_xy(t_dense, theta, M, X)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(data_xy[:, 0], data_xy[:, 1], s=6, alpha=0.4, color="tab:blue",
               label=f"data (n={len(data_xy)})")
    ax.plot(cx, cy, color="tab:red", linewidth=1.5,
            label=f"fitted curve (θ={np.rad2deg(theta):.2f}°, M={M:.4f}, X={X:.2f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Fit overlay: data cloud vs recovered curve")
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    print(f"saved {savepath}")
    return fig


def plot_residual_histogram(data_xy, params, savepath="residual_histogram.png"):
    dist = curve_residuals(params, data_xy)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(dist, bins=60, color="tab:blue", edgecolor="white", linewidth=0.3)
    ax.axvline(dist.mean(), color="tab:red", linestyle="--",
               label=f"mean = {dist.mean():.4f}")
    ax.set_xlabel("distance from point to fitted curve")
    ax.set_ylabel("count")
    ax.set_title("Residual distribution (nearest-point-on-curve distance)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    print(f"saved {savepath}")
    return fig


def plot_residual_vs_t(data_xy, params, savepath="residual_vs_t.png", t_grid=DEFAULT_T_GRID):
    """Residual against each point's matched t (nearest point on the dense
    curve sweep), to check for systematic/localized mismatch."""
    from scipy.spatial import cKDTree
    theta, M, X = params
    cx, cy = curve_xy(t_grid, theta, M, X)
    tree = cKDTree(np.column_stack([cx, cy]))
    dist, idx = tree.query(data_xy)
    t_match = t_grid[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(t_match, dist, s=6, alpha=0.4, color="tab:purple")
    ax.set_xlabel("matched t")
    ax.set_ylabel("residual (distance to curve)")
    ax.set_title("Residual vs t -- flat = no systematic bias across the range")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    print(f"saved {savepath}")
    return fig


def plot_phase_convergence(data_xy, seed_params, scout_params, refined_params,
                            final_params, savepath="phase_convergence.png"):
    stages = ["Phase 1a\n(seed)", "Phase 1\n(scout)", "Phase 2\n(refined)", "Phase 3\n(polished)"]
    costs = [
        curve_residual_sum(seed_params, data_xy),
        curve_residual_sum(scout_params, data_xy),
        curve_residual_sum(refined_params, data_xy),
        curve_residual_sum(final_params, data_xy),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(stages, costs, marker="o", color="tab:green")
    ax.set_yscale("log")
    ax.set_ylabel("true L1 residual sum (log scale)")
    ax.set_title("Cost across pipeline phases")
    for x, y in zip(stages, costs):
        ax.annotate(f"{y:.3g}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    print(f"saved {savepath}")
    return fig


def plot_multiseed_agreement(report, savepath="multiseed_agreement.png"):
    runs = report["runs"]
    thetas = [np.rad2deg(r[0][0]) for r in runs]
    Ms = [r[0][1] for r in runs]
    Xs = [r[0][2] for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, vals, name in zip(axes, [thetas, Ms, Xs], ["theta (deg)", "M", "X"]):
        ax.scatter(range(len(vals)), vals, color="tab:orange", s=60, zorder=3)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"run {i}" for i in range(len(vals))])
        ax.set_title(name)
        ax.grid(alpha=0.3)
    fig.suptitle("Final parameter estimate across independent seed runs (tight = agreement)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    print(f"saved {savepath}")
    return fig


def plot_phase1a_diagnostic(candidates, best_params, savepath="phase1a_diagnostic.png"):
    """
    Shows the actual bug: candidates with the LOWEST radius-space cost
    are not the ones with the lowest TRUE cost -- visual proof that
    selecting by radius cost alone is the wrong criterion.
    """
    radius_costs = np.array([c[1] for c in candidates])
    true_costs = np.array([c[2] for c in candidates])
    is_best = np.array([np.array_equal(c[0], best_params) for c in candidates])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(radius_costs[~is_best], true_costs[~is_best], s=70, color="tab:gray",
               label="other candidates", zorder=2)
    ax.scatter(radius_costs[is_best], true_costs[is_best], s=140, color="tab:red",
               marker="*", label="selected (lowest true cost)", zorder=3)
    ax.set_xlabel("radius-space cost (what Phase 1a's raw objective sees)")
    ax.set_ylabel("true (x,y) cost -- what actually matters")
    ax.set_yscale("log")
    ax.set_title("Phase 1a candidates: radius cost is NOT a reliable proxy for true fit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    print(f"saved {savepath}")
    return fig


def main(csv_path):
    data = load_data(csv_path)

    print("Re-running pipeline to collect intermediate results for plotting...")
    seed_params, seed_true_cost, candidates = phase1a_seed(data)
    scout_params, scout_cost, _ = phase1_scout(data, seed_params)
    refined_params, refine_cost, _ = phase2_refine(data, scout_params)
    final_params, final_l1 = phase3_polish(data, refined_params)
    report = phase4_validate(data)

    plot_fit_overlay(data, final_params)
    plot_residual_histogram(data, final_params)
    plot_residual_vs_t(data, final_params)
    plot_phase_convergence(data, seed_params, scout_params, refined_params, final_params)
    plot_multiseed_agreement(report)
    plot_phase1a_diagnostic(candidates, seed_params)

    print("\nAll plots saved.")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "xy_data.csv"
    main(csv_path)
