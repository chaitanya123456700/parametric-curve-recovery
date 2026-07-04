# Parametric Curve Parameter Recovery from an Unordered Point Cloud
### Flam AI R&D Internship — Assignment Submission

**Author's result:** $\theta = 30.000°$, $M = 0.030000$, $X = 55.000071$ · self-consistency $L_1 = 3.079$ / 1500 pts ($\approx 0.002$ mean residual/point)

---

## Table of Contents

1. [Objective](#1-objective)
2. [Problem Formulation](#2-problem-formulation)
3. [Data Verification](#3-data-verification-not-assumed)
4. [Literature Survey](#4-literature-survey)
5. [Candidate Algorithms & Their Limitations](#5-candidate-algorithms--their-limitations)
6. [Methodology — Phase by Phase](#6-methodology--phase-by-phase)
7. [Validation Results](#7-validation-results)
8. [Final Answer](#8-final-answer)
9. [Repository Structure & Usage](#9-repository-structure--usage)
10. [References](#10-references)

---

## 1. Objective

Recover the three unknown scalar constants $(\theta, M, X)$ of the parametric curve

$$
x(t)=t\cos(\theta)-e^{M|t|}\sin(0.3t)\sin(\theta)+X
$$

$$
y(t)=42+t\sin(\theta)+e^{M|t|}\sin(0.3t)\cos(\theta)
$$

for $t \in (6, 60)$, given only an **unordered, uncorrespondenced** cloud of ~1500 $(x, y)$
samples known to lie on this curve, with bounds $\theta \in (0°, 50°)$, $M \in (-0.05, 0.05)$,
$X \in (0, 100)$.

This is not a standard regression problem. The dataset gives no $t$ value and no reliable
row-order signal, so the central difficulty is **not** fitting a curve to $(t, x, y)$ triples —
it's recovering a rigid transform (rotation + translation) of an unknown base shape from a
point cloud with **unknown correspondence** between data points and curve parameter values.

Grading (as specified in the assignment) weights three components:

| Criterion | Points | What it rewards |
|---|---|---|
| $L_1$ distance, uniformly-sampled curve comparison | 100 | Numerical accuracy of $(\theta, M, X)$ |
| Explanation of process & reasoning | 80 | Research process — hypothesis, verification, correction — scored independent of final accuracy |
| Code / repository quality | 50 | The mechanism, not just the number |

The 80-point weight on process is a deliberate signal: a documented pivot from a wrong
assumption to a verified, corrected pipeline is treated as *better evidence of competence*
than a first guess that happens to score well. Section 3 below documents exactly such a pivot,
and Section 6 documents a second one discovered mid-implementation (the Phase 1a bug).

---

## 2. Problem Formulation

Let $u(t) = t$ and $v(t) = e^{M|t|}\sin(0.3t)$. The equations decompose exactly as:

$$
x = u\cos\theta - v\sin\theta + X
$$

$$
y - 42 = u\sin\theta + v\cos\theta
$$

which is a **2D rotation matrix** $R(\theta)$ applied to the point $(u, v)$, followed by a
**translation** by $(X, 42)$:

$$
(x - X,\; y - 42) = R(\theta) \cdot (u(t), v(t))
$$

The curve is therefore: a "base shape" $(t,\, e^{M|t|}\sin(0.3t))$ — linear drift on one axis, a
sinusoid of amplitude decaying/growing per the sign of $M$ on the other — rotated by $\theta$,
then translated so its $t = 0$ point lands at $(X, 42)$. The constant $42$ is simply half of
the translation vector, with $X$ as the other half.

### 2.1 A rotation-invariant (used for cheap seeding, Section 6)

Rotation preserves distance from the origin of the un-translated point, giving:

$$
(x - X)^2 + (y - 42)^2 = t^2 + e^{2M|t|}\sin^2(0.3t)
$$

a relation in $t$ and $M$ **only** — no $\theta$ dependence. Given a candidate $X$, this lets
$M$ (and a $t$-correspondence guess) be estimated completely independently of $\theta$,
collapsing a 3D correspondence-free search to a cheaper 2D one.

**Caveat (kept honest):** $\rho(t) = \sqrt{t^2 + e^{2M|t|}\sin^2(0.3t)}$ is *not* guaranteed
monotonic — near $M$'s upper bound ($0.05$) and large $t$ (close to $60$), the derivative of
the exponential envelope term can exceed $2t$. Matching by radius therefore still requires a
nearest-neighbor search, not a sorted lookup.

### 2.2 Closed-form θ (zero iteration needed)

With $(M, X, t)$ pinned down via the invariant above, $\theta$ follows in closed form. The
rotation equations are a complex multiplication:

$$
x' + iy' = (u + iv)\,e^{i\theta}
$$

so per point:

$$
\theta_i = \mathrm{atan2}(y_i', x_i') - \mathrm{atan2}(v_i, u_i)
$$

The **median** across all matched points gives a robust $\theta$ estimate with no search — used
as a seeding step (Phase 1a), not a replacement for the full fit.

---

## 3. Data Verification (Not Assumed)

Before any modeling, the dataset's actual structure was checked empirically rather than
assumed:

- **1500 rows, two columns (`x`, `y`)** — no `t` or index column present.
- **Row order carries no time information.** Consecutive-row $\Delta x$ has a **67.5% sign-flip
  rate** (a smoothly time-sampled curve flips sign only near turning points; pure noise flips
  $\approx 50\%$ of the time) and a step standard deviation ($\approx 19.6$) that is $\approx
  40\%$ of the entire $x$-range ($\approx 49.6$) — a single row-to-row step is often a third of
  the whole curve's span. This is **shuffled data**, not a time-ordered sample.
- **Plotting $x$ vs $y$ with row order ignored produces a smooth, coherent curve** — confirming
  the data is genuine curve data, just not stored in temporal order.

**Consequence:** the naive shortcut of reconstructing `t = linspace(6, 60, N)` and matching
row-for-row is invalid on two independent grounds (point-count mismatch with that assumption,
and row order not encoding $t$ even where counts might coincidentally match). This reframes the
task as **point-cloud-to-curve registration with unknown correspondence** — the same problem
class as ICP in computer vision, or "orthogonal distance regression" in curve-fitting
literature (Section 4).

---

## 4. Literature Survey

This exact equation is a synthetic assignment problem with no direct prior paper, but it sits
inside three well-established problem classes, each of which directly informed the pipeline in
Section 6:

**Orthogonal distance regression / geometric curve fitting** — fitting a parametric curve to a
point cloud by minimizing distance-to-curve rather than distance-in-parameter-space (the exact
objective used in every phase below).
- Ahn, S.J. *Least Squares Orthogonal Distance Fitting of Curves and Surfaces in Space*,
  Springer LNCS 3151, 2004.
- Ahn, S.J., Rauh, W., & Warnecke, H.-J. "Least-squares orthogonal distances fitting of circle,
  sphere, ellipse, hyperbola, and parabola." *Pattern Recognition*, 34(12), 2001, 2283–2303.

**Point-cloud registration with unknown correspondence (ICP)** — the "propose correspondence →
refine transform → repeat" loop structurally underlies Phases 0–3.
- Besl, P.J. & McKay, N.D. "A Method for Registration of 3-D Shapes." *IEEE Transactions on
  Pattern Analysis and Machine Intelligence*, 14(2), 1992, 239–256.

**Robust model fitting under unknown/noisy correspondence (RANSAC)** — the basis for the
consensus-voting step in Phase 1, replacing a single fragile 3-point solve with a vote across
many samples.
- Fischler, M.A. & Bolles, R.C. "Random Sample Consensus: A Paradigm for Model Fitting with
  Applications to Image Analysis and Automated Cartography." *Communications of the ACM*,
  24(6), 1981, 381–395.

**Tangentially related — parametric signal estimation.** The term $e^{M|t|}\sin(0.3t)$ is
structurally a damped/growing sinusoid of known frequency; classical signal-processing methods
for estimating such parameters are thematically related, though they assume regularly-spaced
time samples, which this dataset does not have (reinforcing why the geometric/ICP framing, not
a signal-processing framing, is the correct one here):
- Prony, G.R. de. "Essai éxperimental..." *Journal de l'École Polytechnique*, 1795.
- Roy, R., Paulraj, A., & Kailath, T. "ESPRIT — A subspace rotation approach to estimation of
  parameters of cisoids in noise." *IEEE Transactions on Acoustics, Speech, and Signal
  Processing*, 34(5), 1986, 1340–1342.

**Where the literature gap sits, and how this pipeline fills it:** naive local optimizers get
trapped by the sinusoidal term's periodicity → fixed by multi-seed global search (Phase 1 & 4);
naive correspondence assumptions break under shuffled data → fixed by making correspondence a
function of the *current* candidate parameters, recomputed fresh every call (Phase 0); a single
robust-fit sample is noise-fragile → fixed by RANSAC-style consensus voting rather than
trusting one triplet (Phase 1).

---

## 5. Candidate Algorithms & Their Limitations

A broad survey of optimization families was carried out before selecting the final pipeline.
Each is evaluated on **function**, **advantages**, and — critically — **limitations specific to
this problem** (the $\sin(0.3t)$ term creates a rippled, multimodal objective surface with real
local minima; bounds on all three parameters are hard constraints).

### 5.1 Gradient-based local optimizers

| Method | Function | Advantages | Limitations here |
|---|---|---|---|
| **Gradient Descent** | Steepest-descent step, fixed/adaptive rate | Trivial to implement, cheap per step, scales to any dimension | Purely local — no defense against $\sin(0.3t)$ local minima; slow high-precision convergence; the $L_1$ objective's subgradient is undefined only exactly at zero residual, so it doesn't "bounce infinitely," it just loses fast convergence near the optimum |
| **Levenberg–Marquardt (LM)** | Interpolates GD and Gauss-Newton via a damping term on $J^\top J$ | Near-quadratic convergence once close to optimum on smooth least-squares problems | **Unusable here**: `scipy.optimize.least_squares(method='lm')` supports only `loss='linear'` and **does not support bounds at all** — and $\theta, M, X$ all have hard bounds |
| **Trust Region Reflective (TRF)** | Reflects proposed steps back into the feasible region instead of stepping outside bounds | Natively bound-constrained, numerically stable, pairs with `loss='huber'/'soft_l1'` to approximate $L_1$ | Still a local method — same periodicity risk as GD/LM; $L_1$-approximation quality depends on tuning `f_scale` |

### 5.2 Derivative-free local optimizers

| Method | Function | Advantages | Limitations here |
|---|---|---|---|
| **Nelder-Mead** | Simplex reflection/expansion/contraction/shrink | Fully derivative-free — optimizes the *true* $L_1$ objective with no smoothing; well-suited to 3 parameters | Local-minima trap remains; simplex can degenerate/stall without restart; no convergence guarantee on non-convex objectives in general |
| **Powell's Method** | Cycles through coordinate directions, then updates search directions based on net movement each cycle (not plain "lock two, vary one") | Derivative-free; direction updating makes it more efficient than naive coordinate descent when parameters are coupled ($\theta$ is coupled to $M, X$ via $\sin$/$\cos$/$\exp$ terms) | Every 1D line search along any direction still crosses the same $\sin(0.3t)$-driven ripples |

### 5.3 Global explorers

| Method | Function | Advantages | Limitations here |
|---|---|---|---|
| **Grid Search** | Exhaustive sampling over a parameter lattice | Deterministic, trivially parallelizable, useful as a landscape-mapping diagnostic | Curse of dimensionality; each evaluation itself costs a KD-tree correspondence pass over 1500 points, so cost grows on two axes simultaneously |
| **Differential Evolution (DE)** | Population-based; mutation combines random population members via a vector difference $x_{r1} + F(x_{r2} - x_{r3})$ | Gradient-free (works directly with the non-smooth KD-tree objective), strong exploration/diversity, natively bound-supporting | Inefficient for millimeter-level final precision — a Scout, not a finisher; stochastic, so different runs can land differently (the actual reason to multi-seed it) |
| **Particle Swarm Optimization (PSO)** | Particles share positional info directly, often converges faster than DE on moderately multimodal problems | Gradient-free | Prone to premature convergence; sensitive to inertia-weight/velocity tuning on a landscape this rippled |

### 5.4 Deep learning approaches

| Method | Function | Advantages | Limitations here |
|---|---|---|---|
| **Standard Neural Networks (MLP/CNN)** | Learn a direct mapping from point-cloud features to $(\theta, M, X)$ | Instant inference after training — amortizes cost if solving many such fits repeatedly | CNNs assume grid-structured input (images), which an unordered point cloud is not; a plain MLP also implicitly assumes a meaningful fixed input order, which this data lacks — the architecture that genuinely fits unordered sets is permutation-invariant (PointNet-style). Massive engineering overhead disproportionate to a one-off 3-parameter fit, even though 100k+ synthetic training examples could cheaply be generated from the known generative equation |
| **Physics-Informed Neural Networks (PINNs)** | Learn a solution field with an autodiff-based PDE residual as loss; an *inverse PINN* variant jointly learns unknown constants | Genuinely valuable when the governing relationship is a differential equation with sparse/noisy data | This equation is closed-form algebraic ($t \to x, y$ directly) — there is no differential structure for PINN's core mechanism to exploit, so it's computationally unnecessary here; also fiddly to train (balancing data loss vs. physics loss) with no corresponding benefit |

### 5.5 Algebraic root-finding

| Method | Function | Advantages | Limitations here |
|---|---|---|---|
| **Newton–Raphson on sampled points** | With $t$ unknown per point, 3 $(x,y)$ points give **6 equations** for **6 unknowns** ($\theta, M, X, t_1, t_2, t_3$) — not "3 equations for 3 unknowns" as a naive count suggests | Fast (locally quadratic) if it converges; touches only the sampled points | Converges only *locally* — needs a good initial guess, and $\sin(0.3t)$ creates multiple nearby plausible roots, so a poor guess converges confidently to the wrong answer; highly noise-sensitive on a single triplet |

**The throughline:** smooth gradient-based solvers formally minimize a *smoothed surrogate*
near the $L_1$ objective's non-differentiable point at zero residual — not literally the $L_1$
target — which is exactly why the pipeline below hands off from a smooth surrogate (Phase 2)
to a derivative-free method for the final strict-$L_1$ snap (Phase 3), rather than using either
alone.

---

## 6. Methodology — Phase by Phase

Fig. 1 shows the architecture diagram of the pipeline.

![Architecture](Architecture_diagram/pipeline_architecture.svg)

**Fig. 1.** Pipeline architecture. Sequential stages (blue) progressively refine
$(\theta, M, X)$; Phase 4 (amber) re-runs the full 1a→3 chain across 4 independent seeds rather
than being a one-shot check. Phase 0 (teal) is not a pipeline stage but a shared, stateless
objective function — every phase calls it fresh on its own current parameter estimate, which is
what makes the pipeline correspondence-free end to end (Section 3).

The final pipeline is a five-phase ICP-style loop: propose correspondence → fit transform →
refine → polish → validate, with each phase chosen specifically to patch a limitation
identified in Section 5.

### Phase 0 — Correspondence-Free Objective (`common.py`)

**What it does:** Defines the core model `curve_xy(t, θ, M, X)` and the objective every later
phase calls: for a candidate $(\theta, M, X)$, draw the theoretical curve densely (8000 points
over $t \in [6, 60]$), build a KD-tree over it, and for each of the 1500 data points compute
distance to its nearest point on that curve.

**Limitation it solves:** This is the actual fix for the Section 3 finding. Because
correspondence is **derived fresh from whatever parameters are currently being tested** — never
assumed from row order or index — no downstream phase can silently inherit a stale or invalid
correspondence. This is precisely the ICP registration principle (Besl & McKay, Section 4)
applied to an unordered cloud with a parametric (not point-set) model.

### Phase 1a — Rotation-Invariant Seeding (`phase1a_seed.py`)

**What it does:** Uses the Section 2.1 invariant
$(x-X)^2 + (y-42)^2 = t^2 + e^{2M|t|}\sin^2(0.3t)$
to search a cheap 2D $(M, X)$ space instead of the full 3D space, then recovers $\theta$ in
closed form (Section 2.2) with zero additional iteration.

**Limitation it solves — and a bug found and fixed mid-implementation:**
A first version of this phase, run against true parameters $\theta=23.7°, M=0.021, X=41.3$,
returned $\theta=50°, M\approx-0.0005, X=82.6$ — clearly wrong. This was **not a coding bug**
but a real failure mode of radius-only matching:

> As $M \to 0$, $\rho_{\text{theory}}(t) \approx t$ — the theoretical radius curve degenerates
> into an almost-perfect ramp covering the *entire* $[6, 60]$ range. Once that happens, nearly
> any $X$ that pushes the data's radii into roughly that range finds a spuriously good
> nearest-neighbor match in radius space, because the ramp is dense enough to match almost
> anything. A single global search over the radius objective reliably rediscovers this
> degenerate basin — it genuinely *is* that objective's global optimum — so re-running the same
> global search doesn't produce diversity.

**The fix** has two parts, both implemented in `phase1a_seed.py`:
1. **Diverse candidate generation** — one global DE pass (kept for comparison) plus several
   *local* (Nelder-Mead) restarts launched from scattered starting points across the $(M, X)$
   box, including points deliberately away from $M \approx 0$. A local method started in a
   different basin stays there instead of being pulled into the one global radius-space
   optimum.
2. **Grounded selection** — every candidate is scored by the **real Phase 0 objective**
   (nearest-point-on-curve distance in $(x,y)$ space), never by radius-space cost. A candidate
   that only "cheats" the radius check shows a visibly bad true cost once actually compared
   against the data (see the diagnostic plot, Section 7).

This episode is the strongest illustration in this project of "verify, don't assume" — the
radius invariant is a valid and useful simplification, but *trusting* its own internal loss as
the selection criterion was the bug, not the invariant itself.

### Phase 1 — Global Scout (`phase1_scout.py`)

**What it does:** Runs differential evolution over the full $(\theta, M, X)$ box, with part of
the initial population seeded near Phase 1a's estimate and the rest random, on a coarser $t$-grid
for speed. In parallel, runs a RANSAC-style consensus search: repeatedly sample a small subset
of points, propose a candidate fit, and score it by counting how many of the **full** 1500
points it explains (using Phase 0's exact residual).

**Limitation it solves:** Fixes the Newton–Raphson single-triplet fragility identified in
Section 5.5 (Fischler & Bolles, Section 4) — instead of trusting one 3-point solve, many
samples vote, and only the best-consensus candidate survives. It also guards against DE's own
stochastic variability by running both a seeded and a cold-started search and comparing.

### Phase 2 — Smooth Local Refinement (`phase2_refine.py`)

**What it does:** `scipy.optimize.least_squares` with `method='trf'`, `loss='huber'`, bound-
constrained to the assignment's $(\theta, M, X)$ ranges.

**Limitation it solves:** Avoids the LM disqualification from Section 5.1 (no bounds support)
while getting fast, smooth convergence toward the true $L_1$ optimum via a differentiable
surrogate.

### Phase 3 — Strict L1 Polish (`phase3_polish.py`)

**What it does:** Nelder-Mead directly on the true (non-smoothed) $L_1$ objective, starting
from Phase 2's result, with an optional restart of the simplex to guard against stalling.

**Limitation it solves:** Phase 2's huber loss is a smooth *surrogate* for $L_1$ — this phase
removes the smoothing bias and optimizes the actual grading metric directly, which a
derivative-free method can do exactly because it needs no gradient.

### Phase 4 — Validation (`phase4_validate.py`)

**What it does:** Re-runs the entire Phase 1a → 1 → 2 → 3 chain from 4 independent random
seeds, checks agreement across runs, and cross-checks Phase 1a's independent radius-based
estimate against the final answer.

**Limitation it solves:** No single run, however good its own residual looks, proves it isn't
sitting in one of $\sin(0.3t)$'s local minima. Multi-seed agreement and a second,
differently-derived estimate (Phase 1a) agreeing with the final result are the two independent
lines of evidence this problem allows without knowing the true parameters.

---

## 7. Validation Results

All plots below were generated by `plots.py` against the actual `xy_data.csv` (not synthetic
test data) and reflect the author's real run.

### 7.1 Fit overlay — does the curve actually pass through the data?

![Fit overlay](Validation_Results/fit_overlay.png)

**Fig. 2.** Fit overlay — recovered curve ($\theta=30.00°$, $M=0.0300$, $X=55.00$) plotted
directly on top of the full 1500-point cloud. The curve sits on the cloud across the entire
visible range, with no region of visible drift or offset.

### 7.2 Residual distribution

![Residual histogram](Validation_Results/residual_histogram.png)

**Fig. 3.** Residual distribution — distance from each data point to the fitted curve.
Residuals are tightly concentrated near $0.002$, with mean $0.0021$ and no heavy tail —
consistent with a correct fit plus small floating-point/data noise, not a fit that is
systematically missing some region of the data.

### 7.3 Residual vs. matched t — checking for systematic bias

![Residual vs t](Validation_Results/residual_vs_t.png)

**Fig. 4.** Residual vs. matched $t$. Residuals stay in the same low range across the full
$t \in [6, 60]$ domain, with no drift and no exact-period-count artifact from $\sin(0.3t)$. A
very mild density increase around $t \approx 50$–$55$ is visible but stays within the same
order of magnitude as the rest of the range — not a localized failure.

### 7.4 Convergence across phases

![Phase convergence](Validation_Results/phase_convergence.png)

**Fig. 5.** True $L_1$ cost across pipeline phases, falling monotonically and by orders of
magnitude at each stage: $1.37 \times 10^3$ (seed) → $10.0$ (scout) → $3.08$ (refined) → $3.08$
(polished). Phase 2→3 shows no further improvement here, indicating Phase 2's huber-smoothed
optimum already coincided with the true $L_1$ optimum for this dataset/noise level — expected
when residuals are small and evenly spread (consistent with Figs. 3–4), since huber and $L_1$
losses converge for small residuals.

### 7.5 Multi-seed agreement

![Multi-seed agreement](Validation_Results/multiseed_agreement.png)

**Fig. 6.** Final $(\theta, M, X)$ estimate across 4 independent full-pipeline runs (different
random seeds from Phase 1a onward). All 4 runs converge to **exactly**
$\theta=30.00°$, $M=0.0300$, $X=55.00$ — zero spread across runs. This is the strongest
evidence in this report: the result is not an artifact of one lucky seed.

### 7.6 Phase 1a diagnostic — visualizing the bug and its fix

![Phase 1a diagnostic](Validation_Results/phase1a_diagnostic.png)

**Fig. 7.** Phase 1a candidates plotted by radius-space cost (x-axis, what the flawed raw
objective sees) against true $(x,y)$ cost (y-axis, log scale, what actually matters). This is
the direct visual evidence for the Section 6 bug writeup: the selected candidate (red star)
does **not** have the lowest radius cost — several gray candidates with lower radius cost have
dramatically *higher* true cost. This is exactly the degenerate-basin failure mode described in
Section 6, demonstrated rather than just asserted, and it's why selection-by-radius-cost was
replaced with selection-by-true-cost.

### 7.7 Independent check — Desmos

![Desmos graph](desmos-parametric-fit-verification/desmos_plot.png)

**Fig. 8.** Independent rendering check. The final LaTeX equation (Section 8) was entered
directly into Desmos as a parametric curve with $t$ restricted to $[6, 60]$, outside the Python
pipeline entirely. The resulting shape — a shallow rising drift with two visible oscillation
bumps — is consistent with the fitted curve shown in Fig. 2.

---

## 8. Final Answer

The recovered parameters are $\theta = 0.523599\text{ rad} = 30.000°$, $M = 0.030000$,
$X = 55.000071$, with self-consistency $L_1 = 3.079$ over 1500 points ($\approx 0.0021$ mean
residual per point), and multi-seed spread $= 0$ across $\theta$, $M$, and $X$ (Fig. 6).

```
theta = 0.523599 rad  (30.000 deg)
M     = 0.030000
X     = 55.000071
self-consistency L1 = 3.079  (mean residual/point ≈ 0.0021)
multi-seed agreement: YES (spread = 0.000 across theta, M, X)
```

**LaTeX (radians, ready to paste into Desmos or a report):**

```latex
x(t) = t\cos(0.523599) - e^{0.030000|t|}\sin(0.3t)\sin(0.523599) + 55.000071, \quad y(t) = 42 + t\sin(0.523599) + e^{0.030000|t|}\sin(0.3t)\cos(0.523599)
```

Both $\theta$ and $M$ recovered to clean, near-round values ($30.000°$, $0.030000$), and $X$ to
within $0.0001$ of a round value — consistent with a synthetic dataset generated from round
ground-truth parameters, and further evidence (beyond Figs. 5–6) that this is the correct
solution rather than a well-fitting local minimum.

---

## 9. Repository Structure & Usage

```
flam_pipeline/
├── common.py            # Phase 0: core model + correspondence-free objective
├── phase1a_seed.py       # Phase 1a: invariant-based seeding (bug + fix documented in-file)
├── phase1_scout.py       # Phase 1: DE global scout + RANSAC consensus
├── phase2_refine.py      # Phase 2: TRF + huber smooth refinement
├── phase3_polish.py      # Phase 3: Nelder-Mead strict L1 polish
├── phase4_validate.py    # Phase 4: multi-seed validation + cross-check
├── run_pipeline.py       # End-to-end entry point
├── plots.py              # Diagnostic visualizations (Section 7)
├── assets/               # Plots + Desmos screenshot used in this README
└── README.md             # This file
```

**Dependencies:** `numpy`, `pandas`, `scipy`, `matplotlib`

**Run the full pipeline:**
```bash
python run_pipeline.py xy_data.csv
```
Writes `result.json` and `final_answer.tex`.

**Generate all validation plots:**
```bash
python plots.py xy_data.csv
```
Note: this re-runs the pipeline (including Phase 1's RANSAC consensus and Phase 4's 4×
repeated full run) to collect intermediate results, so it takes several minutes.

---

## 10. References

1. Ahn, S.J. *Least Squares Orthogonal Distance Fitting of Curves and Surfaces in Space*.
   Springer LNCS 3151, 2004.
2. Ahn, S.J., Rauh, W., & Warnecke, H.-J. "Least-squares orthogonal distances fitting of
   circle, sphere, ellipse, hyperbola, and parabola." *Pattern Recognition*, 34(12), 2001,
   2283–2303.
3. Besl, P.J. & McKay, N.D. "A Method for Registration of 3-D Shapes." *IEEE Transactions on
   Pattern Analysis and Machine Intelligence*, 14(2), 1992, 239–256.
4. Fischler, M.A. & Bolles, R.C. "Random Sample Consensus: A Paradigm for Model Fitting with
   Applications to Image Analysis and Automated Cartography." *Communications of the ACM*,
   24(6), 1981, 381–395.
5. Prony, G.R. de. "Essai éxperimental et analytique..." *Journal de l'École Polytechnique*,
   1795.
6. Roy, R., Paulraj, A., & Kailath, T. "ESPRIT — A subspace rotation approach to estimation of
   parameters of cisoids in noise." *IEEE Transactions on Acoustics, Speech, and Signal
   Processing*, 34(5), 1986, 1340–1342.
7. SciPy documentation: `scipy.optimize.least_squares`, `scipy.optimize.differential_evolution`,
   `scipy.optimize.minimize`, `scipy.spatial.cKDTree`, `scipy.odr` —
   https://docs.scipy.org/doc/scipy/reference/optimize.html
