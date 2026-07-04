"""
phase1_scout.py — Phase 1: global scout over the full (theta, M, X) space.

Two complementary searches, both seeded from Phase 1a's estimate rather
than starting cold, plus extra cold-started seeds so we can tell a real
global optimum from a lucky basin around the Phase 1a seed:

  1. Differential evolution over the full 3D box, with part of the initial
     population placed near the Phase 1a seed (small random perturbations)
     and the rest random, on a coarser t-grid for speed.
  2. RANSAC-style consensus voting: repeatedly sample a small random subset
     of points, turn it into a candidate, score it by how many of the FULL
     1500 points land near the resulting curve (using Phase 0's exact
     residual function). Keep the best-consensus candidate. This is the
     fixed version of "solve 3 points exactly" from the assignment's
     Newton-Raphson idea -- instead of trusting one fragile triplet, we
     vote across many.

sin(0.3t) gives ~2.6 periods across t in (6, 60), so real local minima
exist. If the DE-from-seed run and a cold DE run don't land in the same
neighborhood, that's the signal to widen population/generations before
trusting the result -- Phase 4 checks this explicitly.
"""
import numpy as np
from scipy.optimize import differential_evolution

from common import (
    PARAM_BOUNDS, THETA_BOUNDS, M_BOUNDS, X_BOUNDS, COARSE_T_GRID,
    curve_residuals, curve_residual_sum,
)


def _de_objective(params, data_xy, t_grid):
    return curve_residuals(params, data_xy, t_grid).sum()


def de_scout(data_xy, seed_estimate=None, popsize=25, maxiter=150,
             t_grid=COARSE_T_GRID, rng_seed=0):
    """
    Differential evolution over the full 3D (theta, M, X) box.
    If seed_estimate is given, DE's `init` population is biased toward it
    (small perturbations) while still keeping random diversity, so a bad
    seed can still be escaped.
    """
    rng = np.random.default_rng(rng_seed)

    init = "latinhypercube"
    if seed_estimate is not None:
        pop_size_total = popsize * 3  # matches DE's internal len(bounds)*popsize
        n_seeded = pop_size_total // 2
        n_random = pop_size_total - n_seeded

        theta0, M0, X0 = seed_estimate
        seeded = np.column_stack([
            np.clip(theta0 + rng.normal(0, 0.05, n_seeded), *THETA_BOUNDS),
            np.clip(M0 + rng.normal(0, 0.003, n_seeded), *M_BOUNDS),
            np.clip(X0 + rng.normal(0, 2.0, n_seeded), *X_BOUNDS),
        ])
        random_pop = np.column_stack([
            rng.uniform(*THETA_BOUNDS, n_random),
            rng.uniform(*M_BOUNDS, n_random),
            rng.uniform(*X_BOUNDS, n_random),
        ])
        init = np.vstack([seeded, random_pop])

    result = differential_evolution(
        _de_objective, bounds=PARAM_BOUNDS, args=(data_xy, t_grid),
        popsize=popsize, maxiter=maxiter, init=init, seed=rng_seed, polish=True,
    )
    return result.x, result.fun


def ransac_consensus(data_xy, n_trials=300, sample_size=3, inlier_thresh=1.0,
                      t_grid=COARSE_T_GRID, rng_seed=0):
    """
    RANSAC-style complement: sample a handful of points, propose a rough
    candidate by a cheap local search seeded at the sample's own radius/
    angle stats, then score the candidate by inlier count on the FULL
    dataset using Phase 0's exact residual. Keeps the best-consensus
    candidate rather than trusting a single triplet.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(data_xy)
    best_params, best_inliers = None, -1

    for _ in range(n_trials):
        idx = rng.choice(n, size=sample_size, replace=False)
        sample = data_xy[idx]

        # Cheap local proposal: small bounded local search around a random
        # start, minimizing residual on the SAMPLE only (fast).
        theta0 = rng.uniform(*THETA_BOUNDS)
        M0 = rng.uniform(*M_BOUNDS)
        X0 = rng.uniform(*X_BOUNDS)

        result = differential_evolution(
            lambda p: curve_residuals(p, sample, t_grid).sum(),
            bounds=PARAM_BOUNDS, popsize=10, maxiter=25,
            seed=int(rng.integers(0, 1_000_000)), polish=True,
        )
        candidate = result.x

        # Score on the FULL point cloud, not just the sample.
        full_dist = curve_residuals(candidate, data_xy, t_grid)
        inliers = int((full_dist < inlier_thresh).sum())

        if inliers > best_inliers:
            best_inliers, best_params = inliers, candidate

    return best_params, best_inliers


def phase1_scout(data_xy, seed_estimate):
    """
    Runs DE-from-seed, a cold DE run, and RANSAC consensus; returns the
    candidate with the lowest true Phase 0 residual sum, plus all
    candidates for the Phase 4 agreement check.
    """
    de_seeded_params, de_seeded_cost = de_scout(
        data_xy, seed_estimate=seed_estimate, rng_seed=1
    )
    de_cold_params, de_cold_cost = de_scout(
        data_xy, seed_estimate=None, rng_seed=2
    )
    ransac_params, ransac_inliers = ransac_consensus(data_xy, rng_seed=3)
    ransac_cost = curve_residual_sum(ransac_params, data_xy)

    candidates = {
        "de_seeded": (de_seeded_params, de_seeded_cost),
        "de_cold": (de_cold_params, de_cold_cost),
        "ransac": (ransac_params, ransac_cost),
    }
    best_name = min(candidates, key=lambda k: candidates[k][1])
    best_params, best_cost = candidates[best_name]
    return best_params, best_cost, candidates


if __name__ == "__main__":
    import sys
    from common import load_data
    from phase1a_seed import phase1a_seed

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "xy_data.csv"
    data = load_data(csv_path)

    seed_params, _, _ = phase1a_seed(data)
    best_params, best_cost, candidates = phase1_scout(data, seed_params)

    print("Phase 1 candidates:")
    for name, (params, cost) in candidates.items():
        theta, M, X = params
        print(f"  {name:10s}: theta={np.rad2deg(theta):6.2f} deg  M={M:+.5f}  "
              f"X={X:6.2f}  cost={cost:9.3f}")

    theta, M, X = best_params
    print(f"\nSelected Phase 1 scout result: theta={theta:.6f} rad, "
          f"M={M:.6f}, X={X:.6f}, cost={best_cost:.3f}")
