"""
phase3_polish.py — Phase 3: strict L1 polish.

Phase 2's huber loss is a smooth surrogate for L1, needed because TRF
requires enough smoothness to descend efficiently. Nelder-Mead is
derivative-free, so it can optimize the TRUE L1 objective directly with no
smoothing -- the right tool for a final-polish role with only 3 parameters
(well within Nelder-Mead's comfort zone). It has no defense against
sin(0.3t)'s local minima on its own, which is why it's fed Phase 2's
already-refined estimate rather than a cold start, and why the simplex is
restarted once to guard against degeneration/stalling (no convergence
guarantee on non-convex objectives in general).

Because curve_residuals() recomputes correspondence fresh from the current
(theta, M, X) on every call, there's no stale correspondence carried over
from Phase 2 -- each phase gets correspondence appropriate to its own
current parameters automatically.
"""
import numpy as np
from scipy.optimize import minimize

from common import curve_residual_sum, PARAM_BOUNDS


def _clip(params):
    return np.array([
        np.clip(params[0], *PARAM_BOUNDS[0]),
        np.clip(params[1], *PARAM_BOUNDS[1]),
        np.clip(params[2], *PARAM_BOUNDS[2]),
    ])


def phase3_polish(data_xy, x0, restarts=1):
    """
    x0: starting (theta, M, X), typically Phase 2's refined result.
    restarts: re-launch Nelder-Mead from the previous result this many
              extra times as a cheap guard against simplex stall/degeneration.
    """
    params = np.asarray(x0, dtype=float)
    result = None
    for _ in range(restarts + 1):
        result = minimize(
            lambda p: curve_residual_sum(_clip(p), data_xy),
            x0=params, method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000},
        )
        params = _clip(result.x)
    return params, result.fun


if __name__ == "__main__":
    import sys
    from common import load_data
    from phase1a_seed import phase1a_seed
    from phase1_scout import phase1_scout
    from phase2_refine import phase2_refine

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "xy_data.csv"
    data = load_data(csv_path)

    seed_params, _, _ = phase1a_seed(data)
    scout_params, _, _ = phase1_scout(data, seed_params)
    refined_params, _, _ = phase2_refine(data, scout_params)

    final_params, final_l1 = phase3_polish(data, refined_params)
    theta, M, X = final_params
    print(f"Phase 3 polished: theta={theta:.6f} rad ({np.rad2deg(theta):.3f} deg), "
          f"M={M:.6f}, X={X:.6f}")
    print(f"  true L1 residual sum={final_l1:.3f}")
