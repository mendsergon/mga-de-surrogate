# Differential Evolution + Neural Network Surrogate for MGA Trajectory Optimization
### v1.1.0 — MGA-1DSM Extension

A from-scratch implementation of **differential evolution** with a **neural network surrogate model** for interplanetary Multiple Gravity Assist (MGA) trajectory optimization. The optimizer solves Lambert's problem at each leg of a candidate trajectory, computes v∞ at each planetary encounter, and accumulates the total Δv cost including gravity-assist feasibility constraints. A multilayer perceptron is trained on accumulated fitness evaluations and used to pre-screen each generation's trial population, so only the most promising candidates trigger a full Lambert solve. Using **numpy** for the optimizer and **scikit-learn** for the surrogate, the system solves the ESA GTOP **Cassini1** problem (Earth–Venus–Venus–Earth–Jupiter–Saturn) and simpler benchmarks, with a 10-test validation suite covering DE correctness, MGA physics, and surrogate behaviour.

v1.1.0 adds the **MGA-1DSM formulation** — one deep-space maneuver per transfer leg — implemented via a universal-variable Keplerian propagator. Both unpowered MGA and MGA-1DSM are benchmarked against the same ESA GTOP Cassini1 problem under identical optimizer settings, isolating the effect of the trajectory transcription from optimizer configuration.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Optimizer](https://img.shields.io/badge/Optimizer-Differential_Evolution-green)
![Surrogate](https://img.shields.io/badge/Surrogate-MLP_Regressor-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Core Features

- **Differential Evolution from scratch** — standard DE/rand/1/bin (Storn & Price, 1997) with bound reflection, multi-run statistics, and convergence history tracking
- **Multiple Gravity Assist fitness function** with physically correct flyby feasibility constraints derived from the hyperbolic turn-angle bound
- **Izzo's Lambert solver** for every transfer leg — quartic-convergence Householder iteration with analytical derivatives
- **Analytical planetary ephemeris** from JPL approximate Keplerian elements
- **Neural network surrogate** — scikit-learn MLP regressor with normalized inputs, fitness clipping, and rank-correlation scoring
- **Surrogate-assisted DE** — pre-screening strategy that evaluates only the top-k predicted candidates per generation
- **Three benchmark problems** — Earth→Mars direct (2D), Earth-Venus-Earth-Jupiter (4D), ESA GTOP Cassini1 EVVEJS (6D)
- **Validation suite** — 10 tests covering DE correctness, MGA physics, and surrogate behaviour

---

## Getting Started

### Clone the Repository

```bash
git clone git@github.com:mendsergon/mga-de-surrogate.git
cd mga-de-surrogate
```

Or using HTTPS:
```bash
git clone https://github.com/mendsergon/mga-de-surrogate.git
cd mga-de-surrogate
```

### Requirements

```
numpy
scikit-learn
```

No astrodynamics libraries are used. No astropy, no poliastro, no pykep, no pygmo.

### Run

```bash
python main.py                              # Earth→Mars (fast default)
python main.py --problem evej               # Earth-Venus-Earth-Jupiter
python main.py --problem cassini1           # Cassini1 EVVEJS (slow)
python main.py --runs 10 --budget 150000    # recommended Cassini1 settings
python main.py --de-only                    # skip surrogate for baseline
python main.py --surrogate-only             # skip plain DE
python main.py --validate                   # run validation suite
```

### Validation Only

```bash
python validate.py
```

### Programmatic Usage

```python
import numpy as np
from de import differential_evolution
from de_surrogate import surrogate_de
from mga import cassini1_fitness, CASSINI1_BOUNDS, CASSINI1_SEQUENCE, decode_mission

# Plain DE
x, f, hist = differential_evolution(
    cassini1_fitness, CASSINI1_BOUNDS,
    pop_size=60, max_gen=500, seed=42, verbose=True
)

# Surrogate-assisted DE
x, f, hist = surrogate_de(
    cassini1_fitness, CASSINI1_BOUNDS,
    pop_size=60, max_gen=500, max_evals=30000, seed=42,
    warmup_evals=500, screen_fraction=0.3, retrain_every=10,
)

info = decode_mission(x, CASSINI1_SEQUENCE)
print(f"Total Δv: {info['total_dv']:.3f} km/s")
print(f"Departure: {info['departure']}, arrival: {info['arrival']}")
```

---

## The Optimizer

### Differential Evolution

The solver implements DE/rand/1/bin, the standard variant from Storn, R. and Price, K. (1997). *Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces.* Journal of Global Optimization, 11(4), 341-359.

For each candidate xᵢ in the population, a trial vector is built in three steps:

```
Mutation:   v = x_a + F · (x_b − x_c)    for random distinct a, b, c ≠ i
Crossover:  trial_j = v_j if rand() < CR, else x_i_j  (at least one dim forced)
Selection:  x_i ← trial if fitness(trial) < fitness(x_i)
```

where F ∈ [0.4, 0.9] is the scale factor and CR ∈ [0.7, 0.95] is the crossover rate. The algorithm is gradient-free, naturally parallel, and robust to local optima — properties that make it well-suited to the highly nonlinear, multimodal landscapes of interplanetary trajectory design.

### Bound Handling

Out-of-bound values are reflected back into the feasible region: values below the lower bound are mapped to `2·lb − x`, values above the upper bound mapped symmetrically. This preserves exploration pressure near the boundaries without the truncation bias that simple clipping introduces.

### Default Parameters

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| Population size | 10×D | Standard DE guideline |
| F (scale factor) | 0.7 | Balanced exploration / exploitation |
| CR (crossover rate) | 0.9 | High crossover for multimodal landscapes |
| Convergence tolerance | 1e-8 | On population fitness standard deviation |

---

## The MGA Fitness Function

### Decision Vector

For a planet sequence `[p0, p1, ..., p_{N-1}]`, the decision vector is:

```
x = [t0, T1, T2, ..., T_{N-1}]
```

where `t0` is the departure date in MJD2000 and `Tk` is the transfer time (days) for leg k.

### Fitness Computation

1. Compute planet state vectors at each encounter from the analytical ephemeris
2. Solve Lambert's problem for each leg, yielding departure and arrival velocities
3. Launch Δv = ‖v_departure − v_planet₀‖  (hyperbolic excess at origin)
4. At each intermediate planet, compute `v∞_in = v_arrival − v_planet` and `v∞_out = v_departure_next − v_planet`, then add the flyby cost
5. Arrival Δv = ‖v_arrival_final − v_planet_final‖  (rendezvous)

The total cost is the sum of launch, flyby, and arrival Δv contributions.

### Flyby Cost Model

An unpowered gravity-assist flyby at minimum periapsis radius r_p rotates the hyperbolic excess velocity by at most:

```
sin(δ_max / 2) = 1 / (1 + r_p · v∞² / μ_planet)
```

Let α be the angle between v∞_in and v∞_out:

```
α ≤ δ_max  :  cost = ||v∞_out| − |v∞_in||
α  > δ_max :  cost = √(v₁² + v₂² − 2·v₁·v₂·cos(α − δ_max))
```

The first case handles feasible flybys where the only cost is the magnitude mismatch. The second case computes the minimum impulse required at periapsis when the requested turn exceeds the maximum achievable rotation. This model correctly reports zero cost for geometrically feasible flybys and high cost for impossible geometries.

### Minimum Flyby Radii

| Planet | r_min [km] | Notes |
|--------|-----------|-------|
| Mercury | 2,640 | surface + 200 km |
| Venus | 6,352 | surface + 300 km |
| Earth | 6,678 | surface + 300 km |
| Mars | 3,590 | surface + 200 km |
| Jupiter | 500,444 | 6 R_J (radiation belt avoidance) |
| Saturn | 61,268 | surface + 1,000 km |

---

## The Surrogate

### Architecture

Multilayer perceptron via `sklearn.neural_network.MLPRegressor`:

- **Hidden layers**: (128, 128, 64) ReLU neurons
- **Optimizer**: Adam with learning rate 1e-3
- **Early stopping**: enabled with 15% validation fraction
- **Max iterations**: 500

### Input/Output Handling

- **Inputs**: decision vectors mapped to [0, 1] using problem bounds
- **Outputs**: fitness values clipped to a cap (default 100 km/s) to prevent extreme penalty values (1e6 for infeasible solutions) from destroying training gradients, then standardized

### Training Reservoir

Observations are accumulated in a rolling reservoir capped at 10,000 entries with FIFO eviction. The surrogate is retrained every `retrain_every` generations (default 10) on the current reservoir.

### Surrogate-Assisted DE Loop

At each generation:

1. Generate the full trial population via mutation and crossover (as in standard DE)
2. If the surrogate is trained and the warmup evaluation budget has been exceeded:
   - Predict fitness for all trials with the surrogate
   - Rank by predicted fitness
   - Evaluate the top `screen_fraction` (default 30%) with the real fitness function
3. Otherwise evaluate all trials (warmup phase)
4. Add new observations to the reservoir and retrain on schedule

### Evaluation Metrics

The surrogate reports three metrics on held-out data:

- **MAE** — mean absolute error
- **RMSE** — root mean square error
- **Rank correlation** (Spearman) — whether the surrogate preserves fitness ordering, which is what actually matters for pre-screening

---

## Benchmark Problems

### Earth → Mars Direct (2 variables)

The simplest benchmark — a direct two-body transfer with no flybys. Decision vector is `[t0_MJD2000, T_days]`. Used for fast testing and cross-validation against the independent porkchop plot reference.

### Earth → Venus → Earth → Jupiter (4 variables)

A three-flyby transfer that approximates a fragment of the Cassini trajectory structure. Intermediate difficulty — verifies that DE handles flyby feasibility constraints before scaling up to Cassini1.

### Cassini1 — EVVEJS (6 variables)

The ESA GTOP Cassini1 benchmark — Earth–Venus–Venus–Earth–Jupiter–Saturn. Decision vector bounds match the ESA problem definition. This implementation uses an **unpowered-MGA formulation** (no deep-space impulses within transfer legs), so the achievable minimum is higher than the reference MGA-1DSM best-known solution of 4.930 km/s. The problem geometry, planet sequence, and variable bounds are identical to the ESA version.

---

## Results

### Earth → Mars Direct (5 runs × 5,000 evaluations)

| Metric | DE | Surrogate-DE |
|--------|-----|--------------|
| Best Δv | 5.608 km/s | 5.608 km/s |
| Mean Δv | 5.608 km/s | 5.608 km/s |
| Std | 0.000 | 0.000 |
| Avg evaluations | 1,064 | 3,335 |
| Avg time | 0.18 s | 43.62 s |
| Reference | 5.610 km/s | — |

Both optimizers find the global minimum consistently across all 5 runs, within 0.04% of the independent porkchop plot reference (5.611 km/s). The surrogate provides no benefit on this 2D problem — DE converges in approximately 1,000 evaluations, leaving no budget pressure for the surrogate to exploit. Surrogate warmup and retraining overhead make it 240× slower for identical solution quality.

**Best mission:**

| Property | Value |
|----------|-------|
| Departure | 2026-10-30 |
| Arrival | 2027-09-06 |
| Time of flight | 311 days |
| Total Δv | 5.608 km/s |

---

### Earth → Venus → Earth → Jupiter (10 runs × 50,000 evaluations)

| Metric | DE | Surrogate-DE |
|--------|-----|--------------|
| Best Δv | 11.743 km/s | 11.743 km/s |
| Mean Δv | 12.085 km/s | 12.852 km/s |
| Median Δv | 12.085 km/s | 12.427 km/s |
| Worst Δv | 12.427 km/s | 15.891 km/s |
| Std | 0.342 | 1.310 |
| Avg time | 6.32 s | 446.14 s |

DE finds exactly two distinct basins across 10 runs — 11.743 km/s (5 runs) and 12.427 km/s (5 runs) — with zero variance within each group. The std=0.342 arises entirely from the 50/50 split between the two basins, confirming the EVEJ landscape is genuinely bimodal at population 40. DE terminates at 9,000–11,000 evaluations out of a 50,000 budget, again demonstrating that population size rather than evaluation count is the binding constraint — the same pattern observed on Cassini1.

The surrogate is slightly harmful at 4D: mean Δv increases by 6.4% and std rises from 0.342 to 1.310 km/s. Several surrogate-DE runs exhaust the full 50,000-evaluation budget without triggering the convergence criterion, indicating the surrogate is slowing convergence rather than accelerating it. This is a qualitatively different failure mode from Cassini1 where the surrogate caused active divergence — here it merely delays stagnation without improving the final result.

**Best mission:**

| Leg | Transfer time | Arrival |
|-----|--------------|---------|
| Earth → Venus | 130 d | 2026-12-15 |
| Venus → Earth | 316 d | 2027-10-27 |
| Earth → Jupiter | 1,058 d | 2030-09-19 |

- **Departure:** 2026-08-07
- **Duration:** 1,504 days (4.12 years)
- **Total Δv:** 11.743 km/s

---

### Cassini1 — EVVEJS (10 runs × 150,000 evaluations)

| Metric | DE | Surrogate-DE |
|--------|-----|--------------|
| Best Δv | 9.419 km/s | 9.419 km/s |
| Mean Δv | 9.711 km/s | 11.669 km/s |
| Median Δv | 9.419 km/s | 9.426 km/s |
| Worst Δv | 12.337 km/s | 20.188 km/s |
| Std | 0.875 | 3.356 |
| Avg time | 48.04 s | 608.50 s |
| Reference (MGA-1DSM) | 4.930 km/s | — |

**Best mission:**

| Leg | Transfer time | Arrival |
|-----|--------------|---------|
| Earth → Venus | 180 d | 1998-05-18 |
| Venus → Venus | 415 d | 1999-07-06 |
| Venus → Earth | 53 d | 1999-08-28 |
| Earth → Jupiter | 1,057 d | 2002-07-20 |
| Jupiter → Saturn | 4,634 d | 2015-03-28 |

- **Departure:** 1997-11-18 — within 34 days of the real Cassini launch (1997-10-15), confirming correct problem geometry
- **Duration:** 6,339 days (17.36 years)
- **Total Δv:** 9.419 km/s

#### Why the result differs from the 4.930 km/s reference

Two independent reasons account for the gap:

**Formulation difference.** This implementation uses unpowered MGA only — no deep-space impulses within transfer legs. The ESA GTOP best-known solution uses the MGA-1DSM formulation, which allows one deep-space impulse per leg. The search space here is strictly smaller, so the achievable minimum is higher by construction.

**Population size is the binding constraint, not budget.** At 150,000 evaluations with population 60, DE converges and triggers its termination criterion at 39,000–65,000 evaluations — well before the budget is exhausted. All 10 runs either stagnate at 9.419 km/s or find a different local basin (12.337 km/s in run 7, std=0.875). More budget does not help because DE has already stagnated; the fix is a larger population (150–200) that maintains diversity long enough to explore the multimodal landscape. This is the primary path to improvement within the current formulation.

#### Why the surrogate hurts on Cassini1

The surrogate-assisted DE performs significantly worse: mean Δv increases by 20.2% and std rises from 0.875 to 3.356 km/s. The MLP learns a smooth approximation of a jagged fitness surface and its pre-screening systematically filters out candidates in unexplored basins. The result is reduced population diversity and premature stagnation. High variance across runs (9.419 to 20.188 km/s) confirms the surrogate is actively destabilizing the search.

Notably, increasing the budget from 30,000 to 150,000 evaluations improves the surrogate mean from 13.641 to 11.669 km/s — more observations help the MLP learn a better landscape approximation — but it remains worse than plain DE at any budget tested. The surrogate is demonstrably useful only when the landscape is smooth enough for an MLP to learn useful fitness ordering. Cassini1 does not satisfy this condition at population 60.

**Planned improvement:** an adaptive fallback that monitors held-out rank correlation during the run and reverts to plain DE evaluation when ρ drops below a threshold, combined with population sizes of 150–200 for improved landscape coverage.

---

### Cassini1-1DSM — EVVEJS (10 runs × 150,000 evaluations)

| Metric | DE |
|--------|-----|
| Best Δv | 9.419 km/s |
| Mean Δv | 9.419 km/s |
| Median Δv | 9.419 km/s |
| Worst Δv | 9.419 km/s |
| Std | 0.000 |
| Avg evaluations | 109,823 |
| Avg time | 475.3 s |
| Reference (MGA-1DSM) | 4.930 km/s |

**Best mission (with DSM placements):**

| Leg | Transfer time | η (DSM fraction) | Arrival |
|-----|--------------|-----------------|---------|
| Earth → Venus | 180 d | 0.35 | 1998-05-17 |
| Venus → Venus | 415 d | 0.26 | 1999-07-06 |
| Venus → Earth | 53 d | 0.46 | 1999-08-27 |
| Earth → Jupiter | 1,057 d | 0.78 | 2002-07-19 |
| Jupiter → Saturn | 4,634 d | 0.73 | 2015-03-28 |

- **Departure:** 1997-11-18
- **Duration:** 6,339 days (17.36 years)
- **Total Δv:** 9.419 km/s

#### Formulation comparison: unpowered MGA vs MGA-1DSM

| Metric | Unpowered MGA (6D) | MGA-1DSM (11D) |
|--------|-------------------|----------------|
| Best Δv | 9.419 km/s | 9.419 km/s |
| Mean Δv | 9.711 km/s | 9.419 km/s |
| Std | 0.875 | 0.000 |
| Population | 60 | 110 |
| Avg time per run | 48.04 s | 475.3 s |
| Evals to convergence | 40,000–65,000 | 100,000–120,000 |

The 1DSM formulation finds the same best Δv as unpowered MGA. The DSM placements are non-trivial (η values away from 0 and 1) but the resulting DSM burns are near-zero — the optimizer has found the same trajectory family and placed DSMs that contribute almost nothing to the total cost. This confirms the population constraint finding extends to the 1DSM case: the extra 5 η variables expand the search space without helping escape the dominant local basin at 9.419 km/s. The correct fix remains population 150–200, not a different trajectory transcription.

The 1DSM formulation is approximately 10× slower per run (475s vs 48s) due to the Keplerian propagation sub-stepping required to locate DSM points on each leg.

---

### Surrogate Scaling Across Dimensionality

The three benchmark problems together reveal a systematic pattern in surrogate effectiveness:

| Problem | Dims | Surrogate mean Δv change | Surrogate time penalty | Conclusion |
|---------|------|--------------------------|----------------------|------------|
| Earth→Mars | 2 | 0.0% | 240× slower | Unnecessary |
| EVEJ | 4 | +6.4% | 70× slower | Slightly harmful |
| Cassini1 | 6 | +20.2% | 13× slower | Significantly harmful |

Surrogate damage scales with dimensionality. As the landscape becomes more complex, the MLP's smooth approximation increasingly misrepresents the fitness surface, causing pre-screening to filter out promising candidates in unexplored basins. The time penalty per unit of quality degradation worsens at higher dimensions because surrogate retraining overhead grows while the benefit decreases. This systematic relationship motivates the adaptive fallback: monitor rank correlation ρ during the run and disable pre-screening when the surrogate is demonstrably not learning useful fitness ordering.

---

## Validation

The solver passes 10 tests covering DE correctness, MGA physics, and surrogate behaviour:

| Test | Description | Result |
|------|-------------|--------|
| DE sphere (D=10) | Convex smooth function | ✅ best = 1.365e-08 (target < 1e-04) |
| DE Rosenbrock (D=10) | Narrow curved valley | ✅ best = 4.351e-02 (target < 1.0) |
| DE Ackley (D=10) | Flat with central well | ✅ best = 1.231e-04 (target < 1e-02) |
| Earth→Mars convergence | Cross-check vs porkchop reference | ✅ best = 5.608 km/s (ref 5.610) |
| Cassini1 feasibility | Random 6D points produce finite Δv | ✅ 100/100 finite |
| Flyby zero cost | Identical v∞ in/out | ✅ cost = 0.000e+00 |
| Flyby reversal cost | 180° v∞ flip | ✅ cost = 4.143 km/s |
| Flyby small rotation | 3° turn at Jupiter | ✅ cost = 0 |
| Surrogate rank correlation | Spearman ρ on quadratic | ✅ ρ = 0.996, MAE = 0.52 |
| Surrogate-assisted DE | End-to-end on Earth→Mars | ✅ 5.608 km/s |

---

## Accuracy — What Is Correct and What Is Approximate

### Optimizer Correctness

Differential evolution is a stochastic global optimizer — it produces a best-found-so-far, not a provably optimal result. Convergence is assessed by running multiple independent seeds and reporting best, mean, median, worst, and standard deviation. The implementation is validated on standard test functions where the global optimum is known, and against the independent porkchop plot reference for Earth-Mars.

### MGA Physics

The fitness function uses the patched-conics approximation with an unpowered-flyby model:

- Each transfer leg is computed as a two-body Keplerian arc via Lambert's problem (exact to machine precision)
- Flybys are instantaneous and preserve the hyperbolic excess magnitude
- The turn-angle bound `sin(δ_max/2) = 1/(1 + r_p·v∞²/μ)` is physically exact for unpowered encounters
- Deep-space impulses within a transfer leg are not modelled

### What Is Not Modelled

| Simplification | Effect |
|----------------|--------|
| Unpowered MGA only | No deep-space impulses; results higher than MGA-1DSM reference |
| Patched conics | No N-body perturbations during transfer |
| No planetary arrival spirals | Δv reported is hyperbolic excess, not launch vehicle Δv |
| Prograde transfers only | Retrograde available via Lambert API but not in MGA fitness |
| Fixed surrogate hyperparameters | screen_fraction and retrain_every not adapted to landscape |

---

## Project Structure

| File | Description |
|------|-------------|
| `de.py` | Differential evolution (DE/rand/1/bin) with multi-run statistics and convergence history |
| `testfuncs.py` | Standard test functions — sphere, Rosenbrock, Rastrigin, Ackley |
| `lambert.py` | Izzo's Lambert solver — called by every MGA fitness evaluation |
| `ephemeris.py` | Analytical planetary ephemeris from JPL Keplerian elements |
| `mga.py` | MGA fitness functions, flyby cost model, unpowered MGA and MGA-1DSM problem definitions |
| `kepler.py` | Universal-variable Keplerian propagator — used by MGA-1DSM to locate DSM points |
| `surrogate.py` | MLP surrogate with input normalization and training reservoir |
| `de_surrogate.py` | Surrogate-assisted DE with pre-screening strategy |
| `benchmark.py` | Benchmark harness comparing plain DE against surrogate-assisted DE |
| `validate.py` | 10-test validation suite |
| `main.py` | CLI entry point with problem selection and budget options |

---

## Performance

| Problem | Dimensions | Budget | DE runtime | Surrogate-DE runtime |
|---------|-----------|--------|------------|---------------------|
| Earth → Mars | 2 | 5,000 | 0.18 s | 43.62 s |
| EVEJ | 4 | 50,000 | 6.32 s | 446.14 s |
| Cassini1 | 6 | 150,000 | 48.04 s | 608.50 s |
| Cassini1-1DSM | 11 | 150,000 | 475.3 s | — |

Measured on a single CPU core (Apple M1). The 1DSM formulation is ~10× slower than unpowered MGA due to Keplerian propagation sub-stepping for DSM point location. The computation is embarrassingly parallel but parallelisation is not implemented.

---

## Known Limitations

- **Population constraint dominates both formulations.** Both unpowered MGA (6D, pop=60) and MGA-1DSM (11D, pop=110) stagnate at 9.419 km/s. The extra DSM variables do not help escape local basins at these population sizes. Population 150–200 is the recommended fix.
- **Population size is the binding constraint on Cassini1.** At pop=60 DE stagnates at 40,000–65,000 evaluations regardless of budget. Population 150–200 is the recommended fix, not a larger budget.
- **Surrogate degrades on multimodal problems.** Pre-screening is beneficial only when the landscape is smooth enough for the MLP to learn useful fitness ordering. On Cassini1 at pop=60, the surrogate actively harms convergence. An adaptive rank-correlation fallback is the planned fix.
- **Default DE parameters only.** Self-adaptive variants (jDE, SHADE, L-SHADE) are not implemented.
- **No launch C3 filtering.** All departure Δv values are reported regardless of launch vehicle capability.
- **Single-objective.** Pareto frontiers over Δv vs flight time are not supported.

---

## Future Work

- Increase population to 150–200 for Cassini1 to maintain diversity and escape local basins
- Adaptive surrogate fallback: disable pre-screening when held-out rank correlation drops below threshold
- Self-adaptive DE variants (jDE, SHADE) for improved convergence on high-dimensional problems
- Pareto frontier over total Δv vs flight time
- C++ port with OpenCL parallelization for GPU-accelerated population evaluation — planned as part of the Solar Mission Planner

---

## References

1. Storn, R. and Price, K. (1997). *Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces.* Journal of Global Optimization, 11(4), 341-359.
2. Jin, Y. (2011). *Surrogate-assisted evolutionary computation: Recent advances and future challenges.* Swarm and Evolutionary Computation, 1(2), 61-70.
3. Vasile, M. and De Pascale, P. (2006). *Preliminary Design of Multiple Gravity-Assist Trajectories.* Journal of Spacecraft and Rockets, 43(4), 794-805.
4. Izzo, D. (2015). *Revisiting Lambert's problem.* Celestial Mechanics and Dynamical Astronomy, 121(1), 1-15.
5. Zuo, M. et al. (2020). *A case learning-based differential evolution algorithm for global optimization of interplanetary trajectory design.* Applied Soft Computing, 94, 106451.
6. ESA GTOP database: https://www.esa.int/gsp/ACT/projects/gtop/

---

## License

This project is licensed under the MIT License.

## Acknowledgments

Problem definitions and reference values from ESA Advanced Concepts Team GTOP database. Orbital parameters and physical constants derived from NASA/JPL solar system data.