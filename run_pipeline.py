"""
run_pipeline.py — end-to-end entry point.

Usage:
    python run_pipeline.py xy_data.csv

Runs Phase 1a -> Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 (which itself
re-runs 1a-3 across several seeds for validation) and writes:
    - console report (per-phase params + costs, validation table)
    - result.json           (final params + metadata)
    - final_answer.tex      (LaTeX string, radians, ready to paste in README)
"""
import json
import sys

import numpy as np

from common import load_data, to_latex, curve_residual_sum
from phase1a_seed import phase1a_seed
from phase1_scout import phase1_scout
from phase2_refine import phase2_refine
from phase3_polish import phase3_polish
from phase4_validate import phase4_validate, print_report


def main(csv_path):
    print(f"Loading data from {csv_path} ...")
    data = load_data(csv_path)
    print(f"  {len(data)} points loaded.\n")

    print("Phase 1a: invariant-based seeding (radius matching, multi-restart, "
          "grounded against Phase 0 objective)...")
    seed_params, seed_true_cost, candidates = phase1a_seed(data)
    theta, M, X = seed_params
    print(f"  seed -> theta={np.rad2deg(theta):.3f} deg, M={M:.6f}, X={X:.3f}  "
          f"(true cost={seed_true_cost:.3f})\n")

    print("Phase 1: global scout (DE-from-seed + cold DE + RANSAC consensus)...")
    scout_params, scout_cost, scout_candidates = phase1_scout(data, seed_params)
    theta, M, X = scout_params
    print(f"  scout -> theta={np.rad2deg(theta):.3f} deg, M={M:.6f}, X={X:.3f}  "
          f"(cost={scout_cost:.3f})\n")

    print("Phase 2: smooth local refinement (TRF + huber)...")
    refined_params, refine_cost, _ = phase2_refine(data, scout_params)
    theta, M, X = refined_params
    print(f"  refined -> theta={np.rad2deg(theta):.3f} deg, M={M:.6f}, X={X:.3f}  "
          f"(huber cost={refine_cost:.4f})\n")

    print("Phase 3: strict L1 polish (Nelder-Mead)...")
    final_params, final_l1 = phase3_polish(data, refined_params)
    theta, M, X = final_params
    print(f"  polished -> theta={np.rad2deg(theta):.3f} deg, M={M:.6f}, X={X:.3f}  "
          f"(true L1={final_l1:.3f})\n")

    print("Phase 4: validation (multi-seed re-run + cross-check)...")
    report = phase4_validate(data)
    print_report(report)

    best_params = report["best_params"]
    best_l1 = report["best_l1"]
    theta, M, X = best_params
    latex = to_latex(theta, M, X)

    result = {
        "theta_rad": theta,
        "theta_deg": float(np.rad2deg(theta)),
        "M": M,
        "X": X,
        "self_consistency_l1": best_l1,
        "multi_seed_agrees": bool(report["multi_seed_agrees"]),
        "latex": latex,
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)
    with open("final_answer.tex", "w") as f:
        f.write(latex + "\n")

    print("\nWrote result.json and final_answer.tex")
    print("\nFinal LaTeX (radians):")
    print(latex)


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "xy_data.csv"
    main(csv_path)
