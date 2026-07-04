"""
phase1a_seed.py — Phase 1a: rotation-invariant radius seeding.

Uses the invariant (no theta dependence):

    (x - X)^2 + (y - 42)^2 = t^2 + exp(2M|t|) * sin^2(0.3t)

to search a cheap 2D (M, X) space instead of the full 3D (theta, M, X)
space, then recovers theta in closed form (median over matched points) —
no iteration needed for theta at all.

--- THE BUG AND THE FIX -----------------------------------------------------
True params: theta=23.7 deg, M=0.021, X=41.3.
First run returned: theta=50 deg, M=-0.0005, X=82.6.  Wrong, and not a
coding bug — a real failure mode of radius-only matching.

Why: as M -> 0, rho_theory(t) = sqrt(t^2 + exp(2M|t|)sin^2(0.3t)) ~= t,
i.e. the theoretical radius curve degenerates into an almost-perfect ramp
covering the *entire* [6, 60] range. Once that happens, nearly any X that
pushes the data's radii into roughly that same range finds a near-perfect
nearest-neighbor match in radius space, because the ramp is dense enough to
match almost anything. The optimizer found a degenerate cheat -- fake out
the radius check with M ~= 0 instead of solving for the real M. Radius
matching is correspondence-free the way Phase 0 is, but it never checks
whether the resulting curve actually looks like the data in (x, y) space —
only whether radii individually land somewhere plausible.

Fix implemented below: run several random restarts of the (M, X) search,
and for each restart compute the closed-form theta, then *select the
winner by the real Phase 0 (x, y) objective* (common.curve_residual_sum),
not by the radius-space loss. Radius space is used only to propose
candidates cheaply; Phase 0 is what grounds and judges them.
-----------------------------------------------------------------------------
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree

from common import (
    M_BOUNDS, X_BOUNDS, THETA_BOUNDS, Y_OFFSET, DEFAULT_T_GRID,
    curve_residual_sum,
)


def radius_residuals(params, data_xy, t_grid=DEFAULT_T_GRID):
    """
    Residual in radius space, plus the matched t for each data point
    (used for the closed-form theta and later as a Phase 1 seed helper).
    """
    M, X = params
    rho_theory = np.sqrt(
        t_grid ** 2 + np.exp(2 * M * np.abs(t_grid)) * np.sin(0.3 * t_grid) ** 2
    )
    rho_data = np.sqrt((data_xy[:, 0] - X) ** 2 + (data_xy[:, 1] - Y_OFFSET) ** 2)
    tree = cKDTree(rho_theory.reshape(-1, 1))
    dist, idx = tree.query(rho_data.reshape(-1, 1))
    return dist, t_grid[idx]


def theta_closed_form(data_xy, M, X, t_match):
    """
    theta drops out of the complex-multiplication view of the rotation:
    x' + i*y' = (u + i*v) * e^{i*theta}. Median across all points is
    robust to individual correspondence errors.
    """
    xp = data_xy[:, 0] - X
    yp = data_xy[:, 1] - Y_OFFSET
    u = t_match
    v = np.exp(M * np.abs(t_match)) * np.sin(0.3 * t_match)
    theta_i = np.mod(np.arctan2(yp, xp) - np.arctan2(v, u), 2 * np.pi)
    return np.median(theta_i)


def _radius_obj(params, data_xy):
    dist, _ = radius_residuals(params, data_xy)
    return dist.sum()


def _one_global_restart(data_xy, popsize, maxiter, seed):
    """
    Global DE search over the FULL (M, X) box. Reliably finds the
    *global* minimum of the radius-space objective -- which, as the bug
    writeup explains, is exactly the degenerate M~=0 ramp basin. Kept as
    one of the candidates (it's cheap and sometimes still fine), but never
    trusted alone.
    """
    result = differential_evolution(
        _radius_obj, bounds=[M_BOUNDS, X_BOUNDS], args=(data_xy,),
        popsize=popsize, maxiter=maxiter, seed=seed, polish=True,
    )
    return result.x, result.fun


def _one_local_restart(data_xy, M0, X0, seed):
    """
    LOCAL search (Nelder-Mead) started away from the degenerate basin.
    Unlike DE, a local method started far from M~=0 stays in its own
    basin instead of homing in on the single global optimum of the
    (flawed) radius objective every time -- this is what actually
    produces diverse candidates to check against the real Phase 0 cost.
    """
    result = minimize(
        lambda p: _radius_obj(np.clip(p, [M_BOUNDS[0], X_BOUNDS[0]],
                                       [M_BOUNDS[1], X_BOUNDS[1]]), data_xy),
        x0=[M0, X0], method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000},
    )
    M_est = float(np.clip(result.x[0], *M_BOUNDS))
    X_est = float(np.clip(result.x[1], *X_BOUNDS))
    return np.array([M_est, X_est]), result.fun


def _to_full_candidate(data_xy, mx_params, radius_cost):
    M_est, X_est = mx_params
    _, t_match = radius_residuals(mx_params, data_xy)
    theta_est = theta_closed_form(data_xy, M_est, X_est, t_match)
    theta_est = np.clip(theta_est, *THETA_BOUNDS)
    return np.array([theta_est, M_est, X_est])


def phase1a_seed(data_xy, n_restarts=8, popsize=20, maxiter=60, base_seed=0):
    """
    FIXED VERSION.

    A single global DE search over (M, X) reliably converges to the SAME
    degenerate M~=0 basin every time -- that basin genuinely is the global
    optimum of the flawed radius-space objective, so re-running DE alone
    doesn't produce diversity. The actual fix has two parts:

      1. Propose diverse candidates: one global DE pass (kept for
         reference/comparison) PLUS several LOCAL (Nelder-Mead) restarts
         launched from scattered starting points spread across the (M, X)
         box, including points deliberately away from M~=0. A local method
         started in a different basin stays there instead of being pulled
         into the one global radius-space optimum.
      2. Ground every candidate in the real Phase 0 objective
         (nearest-point-on-curve in (x, y) space) and select by THAT, not
         by radius-space loss. A candidate that only "cheats" the radius
         check will show a bad true_cost once actually compared against
         the data in (x, y) space.

    Returns
    -------
    best_params : np.ndarray [theta, M, X]
    best_true_cost : float          (Phase 0 residual sum for the winner)
    all_candidates : list of (params, radius_cost, true_cost)  for the writeup
    """
    rng = np.random.default_rng(base_seed)
    candidates = []

    # (a) one global DE pass -- included so the writeup can show the bug directly
    de_mx, de_radius_cost = _one_global_restart(data_xy, popsize, maxiter, base_seed)
    de_params = _to_full_candidate(data_xy, de_mx, de_radius_cost)
    candidates.append((de_params, de_radius_cost, curve_residual_sum(de_params, data_xy)))

    # (b) scattered local restarts -- the actual diversity-producing step
    M_starts = np.concatenate([
        [M_BOUNDS[0] * 0.8, M_BOUNDS[1] * 0.8],           # near the bounds, away from 0
        rng.uniform(*M_BOUNDS, size=n_restarts - 2),       # random spread
    ])
    X_starts = rng.uniform(*X_BOUNDS, size=n_restarts)

    for i in range(n_restarts):
        mx, radius_cost = _one_local_restart(data_xy, M_starts[i], X_starts[i], base_seed + i + 1)
        params = _to_full_candidate(data_xy, mx, radius_cost)
        true_cost = curve_residual_sum(params, data_xy)
        candidates.append((params, radius_cost, true_cost))

    best = min(candidates, key=lambda c: c[2])  # select by TRUE (x,y) cost, never radius cost
    best_params, best_radius_cost, best_true_cost = best
    return best_params, best_true_cost, candidates


if __name__ == "__main__":
    from common import load_data
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "xy_data.csv"
    data = load_data(csv_path)

    best_params, best_true_cost, all_candidates = phase1a_seed(data)

    print("Phase 1a candidates (radius-space cost -> true (x,y) cost):")
    for params, r_cost, t_cost in sorted(all_candidates, key=lambda c: c[2]):
        theta, M, X = params
        flag = "  <-- selected" if np.array_equal(params, best_params) else ""
        print(
            f"  theta={np.rad2deg(theta):6.2f} deg  M={M:+.5f}  X={X:6.2f}  "
            f"radius_cost={r_cost:9.3f}  true_cost={t_cost:9.3f}{flag}"
        )

    theta, M, X = best_params
    print("\nSelected Phase 1a seed (grounded against Phase 0 objective):")
    print(f"  theta = {theta:.6f} rad ({np.rad2deg(theta):.3f} deg)")
    print(f"  M     = {M:.6f}")
    print(f"  X     = {X:.6f}")
    print(f"  true Phase-0 residual sum = {best_true_cost:.3f}")
