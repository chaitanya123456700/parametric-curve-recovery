"""
phase4_validate.py — Phase 4: validation (don't skip this).

Three checks, none of which require knowing the true parameters:

1. Multi-seed agreement: re-run the full Phase 1a -> 1 -> 2 -> 3 pipeline
   from several different random seeds. If they converge to the same
   (theta, M, X) neighborhood, that's evidence you're not sitting in one
   of sin(0.3t)'s local minima. If they don't agree, that's the signal to
   widen DE's population/generations rather than trust any single run.
2. Self-consistency L1: report the final params' own residual sum. It is
   NOT identical to the grader's true-curve-vs-predicted-curve L1 (that
   needs the true params, which we don't have), but it's the best proxy
   available.
3. Cross-check against Phase 1a: Phase 1a's estimate comes from a
   completely different residual (1D radius matching + closed-form theta)
   than Phase 0-3's (2D nearest-point-on-curve). Agreement between the two
   is real evidence of a correct fit; disagreement tells you which phase
   to distrust rather than just "something might be wrong somewhere."
"""
import numpy as np

from common import curve_residual_sum
from phase1a_seed import phase1a_seed
from phase1_scout import phase1_scout
from phase2_refine import phase2_refine
from phase3_polish import phase3_polish


def run_full_pipeline(data_xy, base_seed=0):
    """One end-to-end run of Phases 1a -> 1 -> 2 -> 3, for a given seed."""
    seed_params, _, _ = phase1a_seed(data_xy, base_seed=base_seed)
    scout_params, _, _ = phase1_scout(data_xy, seed_params)
    refined_params, _, _ = phase2_refine(data_xy, scout_params)
    final_params, final_l1 = phase3_polish(data_xy, refined_params)
    return final_params, final_l1, seed_params


def phase4_validate(data_xy, n_runs=4, agreement_tol=(0.05, 0.01, 3.0)):
    """
    agreement_tol: (theta_rad_tol, M_tol, X_tol) — max spread across runs
    still counted as "agreement."
    """
    runs = []
    for i in range(n_runs):
        final_params, final_l1, seed_params = run_full_pipeline(data_xy, base_seed=i * 100)
        runs.append((final_params, final_l1, seed_params))

    finals = np.array([r[0] for r in runs])
    spread = finals.max(axis=0) - finals.min(axis=0)
    theta_tol, M_tol, X_tol = agreement_tol
    agrees = (spread[0] <= theta_tol) and (spread[1] <= M_tol) and (spread[2] <= X_tol)

    best_run = min(runs, key=lambda r: r[1])
    best_params, best_l1, best_seed_params = best_run

    # Cross-check: Phase 1a estimate vs the winning full-pipeline result
    seed_theta, seed_M, seed_X = best_seed_params
    final_theta, final_M, final_X = best_params
    cross_check_diff = np.array([
        abs(seed_theta - final_theta),
        abs(seed_M - final_M),
        abs(seed_X - final_X),
    ])

    report = {
        "best_params": best_params,
        "best_l1": best_l1,
        "runs": runs,
        "spread": spread,
        "multi_seed_agrees": agrees,
        "phase1a_cross_check_diff": cross_check_diff,
    }
    return report


def print_report(report):
    print("Phase 4 validation report")
    print("=" * 60)
    print(f"{'run':>4}  {'theta(deg)':>10}  {'M':>9}  {'X':>8}  {'L1':>10}")
    for i, (params, l1, _) in enumerate(report["runs"]):
        theta, M, X = params
        print(f"{i:>4}  {np.rad2deg(theta):10.3f}  {M:9.5f}  {X:8.3f}  {l1:10.3f}")

    print("-" * 60)
    theta_s, M_s, X_s = report["spread"]
    print(f"Spread across runs: theta={np.rad2deg(theta_s):.3f} deg, "
          f"M={M_s:.5f}, X={X_s:.3f}")
    print(f"Multi-seed agreement: {'YES' if report['multi_seed_agrees'] else 'NO -- widen search'}")

    d_theta, d_M, d_X = report["phase1a_cross_check_diff"]
    print(f"Phase 1a cross-check |seed - final|: "
          f"theta={np.rad2deg(d_theta):.3f} deg, M={d_M:.5f}, X={d_X:.3f}")

    theta, M, X = report["best_params"]
    print("-" * 60)
    print(f"Best result: theta={theta:.6f} rad ({np.rad2deg(theta):.3f} deg), "
          f"M={M:.6f}, X={X:.6f}, self-consistency L1={report['best_l1']:.3f}")


if __name__ == "__main__":
    import sys
    from common import load_data, to_latex

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "xy_data.csv"
    data = load_data(csv_path)

    report = phase4_validate(data)
    print_report(report)

    theta, M, X = report["best_params"]
    print("\nFinal LaTeX (radians):")
    print(to_latex(theta, M, X))
