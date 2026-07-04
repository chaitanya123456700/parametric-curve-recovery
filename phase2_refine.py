"""
phase2_refine.py — Phase 2: smooth local refinement.

Trust Region Reflective (TRF) with loss='huber': huber/soft_l1 approximates
the true L1 objective while staying differentiable enough for TRF to
descend quickly. 'lm' is not an option here -- scipy's LM implementation
only supports loss='linear' and doesn't support bounds at all, and
theta/M/X all have hard bounds. TRF's "reflective" behavior (reflecting
proposed steps back into the feasible region rather than stepping outside
bounds) is exactly what's needed for a bound-constrained problem like this.

This is still a LOCAL refinement -- it inherits whatever neighborhood
Phase 1 landed in. The periodicity risk from sin(0.3t) doesn't go away
here; it's handled upstream (Phase 1's multi-seed scouting) and validated
downstream (Phase 4).
"""
from scipy.optimize import least_squares

from common import curve_residuals, LS_LOWER, LS_UPPER


def phase2_refine(data_xy, x0, f_scale=1.0):
    """
    x0: starting (theta, M, X), typically Phase 1's scout result.
    f_scale: huber loss transition scale (residual units); default 1.0
             is a reasonable start given point-cloud spacing, tune if the
             fit is systematically pulled toward outliers or too timid.
    """
    result = least_squares(
        curve_residuals, x0=x0, args=(data_xy,),
        bounds=(LS_LOWER, LS_UPPER), method="trf", loss="huber", f_scale=f_scale,
    )
    return result.x, result.cost, result


if __name__ == "__main__":
    import sys
    import numpy as np
    from common import load_data, curve_residual_sum
    from phase1a_seed import phase1a_seed
    from phase1_scout import phase1_scout

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "xy_data.csv"
    data = load_data(csv_path)

    seed_params, _, _ = phase1a_seed(data)
    scout_params, _, _ = phase1_scout(data, seed_params)

    refined_params, cost, _ = phase2_refine(data, scout_params)
    theta, M, X = refined_params
    true_l1 = curve_residual_sum(refined_params, data)
    print(f"Phase 2 refined: theta={theta:.6f} rad ({np.rad2deg(theta):.3f} deg), "
          f"M={M:.6f}, X={X:.6f}")
    print(f"  huber cost={cost:.4f}   true L1 residual sum={true_l1:.3f}")
