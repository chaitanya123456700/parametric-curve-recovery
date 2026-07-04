"""
common.py — Phase 0: core model + correspondence-free objective.

Every later phase imports from here. Nothing in this file assumes row order
or point count means anything about `t` — correspondence is always derived
fresh from whatever (theta, M, X) is currently being tested.

    x(t) = t*cos(theta) - exp(M*|t|)*sin(0.3t)*sin(theta) + X
    y(t) = 42 + t*sin(theta) + exp(M*|t|)*sin(0.3t)*cos(theta)

t in (6, 60), theta in (0 deg, 50 deg), M in (-0.05, 0.05), X in (0, 100).
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Bounds / constants (single source of truth — every phase imports these,
# nobody re-types them and risks a degrees/radians slip)
# ---------------------------------------------------------------------------
T_MIN, T_MAX = 6.0, 60.0
THETA_BOUNDS = (0.0, np.deg2rad(50.0))   # radians — np.sin/np.cos want radians
M_BOUNDS = (-0.05, 0.05)
X_BOUNDS = (0.0, 100.0)
Y_OFFSET = 42.0

PARAM_BOUNDS = [THETA_BOUNDS, M_BOUNDS, X_BOUNDS]
LS_LOWER = [THETA_BOUNDS[0], M_BOUNDS[0], X_BOUNDS[0]]
LS_UPPER = [THETA_BOUNDS[1], M_BOUNDS[1], X_BOUNDS[1]]

DEFAULT_T_GRID = np.linspace(T_MIN, T_MAX, 8000)
COARSE_T_GRID = np.linspace(T_MIN, T_MAX, 2000)  # for cheaper Phase 1 scouting


def curve_xy(t, theta, M, X):
    """Evaluate the parametric curve at parameter value(s) t."""
    v = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - v * np.sin(theta) + X
    y = Y_OFFSET + t * np.sin(theta) + v * np.cos(theta)
    return x, y


def curve_residuals(params, data_xy, t_grid=DEFAULT_T_GRID):
    """
    Correspondence-free residual: for each data point, distance to the
    nearest point on the candidate curve (dense t-sweep). Order- and
    count-independent, which is what makes it valid on shuffled data.
    Recomputed fresh every call -> no stale correspondence ever leaks
    between phases.
    """
    theta, M, X = params
    cx, cy = curve_xy(t_grid, theta, M, X)
    tree = cKDTree(np.column_stack([cx, cy]))
    dist, _ = tree.query(data_xy)
    return dist


def curve_residual_sum(params, data_xy, t_grid=DEFAULT_T_GRID):
    """Convenience scalar wrapper (strict L1 objective) for scalar optimizers."""
    return curve_residuals(params, data_xy, t_grid).sum()


def load_data(csv_path):
    """Load the (x, y) point cloud. No t / index column expected or used."""
    df = pd.read_csv(csv_path)
    return df[["x", "y"]].to_numpy(dtype=float)


def to_latex(theta, M, X):
    """
    Final answer in the assignment's Desmos example style: radians, numbers
    substituted directly into the equation string.
    """
    return (
        r"x(t) = t\cos(%.6f) - e^{%.6f|t|}\sin(0.3t)\sin(%.6f) + %.6f, \quad "
        r"y(t) = 42 + t\sin(%.6f) + e^{%.6f|t|}\sin(0.3t)\cos(%.6f)"
        % (theta, M, theta, X, theta, M, theta)
    )
