# Differential Evolution + Neural Network Surrogate for MGA Trajectory Optimization
### v1.0.0 — DE with MLP Pre-Screening

A from-scratch implementation of **differential evolution** with a **neural network surrogate model** for interplanetary Multiple Gravity Assist (MGA) trajectory optimization. The optimizer solves Lambert's problem at each leg of a candidate trajectory, computes v∞ at each planetary encounter, and accumulates the total Δv cost including gravity-assist feasibility constraints. A multilayer perceptron is trained on accumulated fitness evaluations and used to pre-screen each generation's trial population, so only the most promising candidates trigger a full Lambert solve. Using **numpy** for the optimizer and **scikit-learn** for the surrogate, the system solves the ESA GTOP **Cassini1** problem (Earth–Venus–Venus–Earth–Jupiter–Saturn) and simpler benchmarks, with a 6-test validation suite covering DE correctness, MGA physics, and surrogate behaviour.

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
- **Validation suite** — 6 tests covering DE convergence, Earth-Mars cross-check, Cassini1 feasibility, flyby cost sanity, and surrogate behaviour

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
python main.py --runs 10 --budget 30000     # custom settings
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

The first case handles feasible flybys where the only cost is the magnitude mismatch that a pure unpowered flyby cannot provide. The second case computes the minimum impulse required at periapsis when the requested turn exceeds the maximum achievable rotation. This model correctly reports zero cost for geometrically feasible flybys and high cost for impossible geometries like 180° reversals.

### Minimum Flyby Radii

| Planet | r_min [km] | Notes |
|--------|-----------|-------|
| Mercury | 2,640 | surface + 200 km |
| Venus | 6,352 | surface + 300 km |
| Earth | 6,678 | surface + 300 km |
| Mars | 3,590 | surface + 200 km |
| Jupiter | 500,444 | 6 R_J (radiation belt avoidance) |
| Saturn | 61,268 | surface + 1000 km |

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

The simplest benchmark — a direct two-body transfer with no flybys. Decision vector is `[t0_MJD2000, T_days]`. Useful for fast testing and cross-validation against an independent porkchop plot reference.

### Earth → Venus → Earth → Jupiter (4 variables)

A three-flyby transfer that approximates a fragment of the Cassini trajectory structure. Intermediate difficulty — verifies that DE handles flyby feasibility constraints before scaling up to the full Cassini1 problem.

### Cassini1 — EVVEJS (6 variables)

The ESA GTOP Cassini1 benchmark — Earth–Venus–Venus–Earth–Jupiter–Saturn. Decision vector bounds match the ESA problem definition. This implementation uses an **unpowered-MGA formulation** (no deep-space impulses within transfer legs), so the achievable minimum is higher than the reference MGA-1DSM best-known solution of ≈4.93 km/s. The problem geometry, planet sequence, and variable bounds are identical to the ESA version.

---

## Results

### Earth → Mars Direct

| Metric | Value |
|--------|-------|
| Optimal departure | 2026-10-30 |
| Optimal arrival | 2027-09-06 |
| Time of flight | 311 days |
| Total Δv | 5.608 km/s |
| Convergence | ~52 generations |
| Real evaluations | ~1560 |

This matches the independent porkchop plot minimum (5.611 km/s) to within 0.05%, confirming that the DE optimizer and MGA fitness function are correctly coupled.

### Surrogate-Assisted DE

On the same Earth-Mars problem with a 1500-evaluation budget, surrogate-assisted DE converges to 5.648 km/s in ~1140 real evaluations. The rank correlation of the surrogate on held-out data exceeds 0.99, confirming that the MLP successfully learns the fitness landscape ordering.

---

## Validation

The solver passes 6 tests covering DE correctness, MGA physics, and surrogate behaviour:

| Test | Description | Result |
|------|-------------|--------|
| DE on sphere (D=10) | Convex smooth function | ✅ best = 1.4e-08 (target <1e-04) |
| DE on Rosenbrock (D=10) | Narrow curved valley | ✅ best = 4.4e-02 (target <1.0) |
| DE on Ackley (D=10) | Flat with central well | ✅ best = 1.2e-04 (target <1e-02) |
| Earth→Mars DE convergence | Cross-check against porkchop reference | ✅ best = 5.608 km/s (ref ~5.611) |
| Cassini1 feasibility | Random 6D points produce finite Δv | ✅ 100/100 finite |
| Flyby zero cost | Identical v∞ in/out | ✅ cost = 0 |
| Flyby reversal cost | 180° v∞ flip | ✅ cost = 4.143 km/s |
| Flyby small rotation | 3° turn at Jupiter | ✅ cost = 0 |
| Surrogate rank correlation | Spearman ρ on quadratic | ✅ ρ = 0.996 |
| Surrogate-assisted DE | End-to-end on Earth→Mars | ✅ 5.648 km/s in 1140 evals |

---

## Accuracy — What Is Correct and What Is Approximate

### Optimizer Correctness

Differential evolution is a stochastic global optimizer — it does not produce a provably optimal result, only a best-found-so-far. Convergence is measured by comparing multiple independent runs with different seeds and reporting best, mean, median, worst, and standard deviation. The implementation is validated on standard test functions (sphere, Rosenbrock, Ackley) where the global optimum is known, and against the independent porkchop plot reference for Earth→Mars.

### MGA Physics

The fitness function uses the patched-conics approximation with an **unpowered-flyby** model. This means:

- Each transfer leg is computed as a two-body Keplerian arc via Lambert's problem (exact to machine precision)
- Flybys are instantaneous and preserve the hyperbolic excess magnitude
- The turn-angle bound `sin(δ_max/2) = 1/(1 + r_p·v∞²/μ)` is physically exact for unpowered encounters
- Deep-space impulses within a transfer leg are **not** modelled

The reference ESA GTOP Cassini1 best-known solution (≈4.93 km/s) uses the MGA-1DSM formulation, which allows one deep-space impulse per leg. Results from this implementation will be higher than the reference because the search space is strictly smaller.

### What Is Not Modelled

| Simplification | Effect |
|----------------|--------|
| Unpowered MGA only | No deep-space impulses; results higher than MGA-1DSM reference |
| Patched conics | No N-body perturbations during transfer |
| No planetary arrival spirals | Δv reported is hyperbolic excess, not launch vehicle Δv |
| Prograde transfers only | Retrograde available via Lambert API but not exposed in MGA fitness |
| No Mercury arrival problems | Planet sequence must be chosen from Mercury-Saturn |

---

## Project Structure

| File | Description |
|------|-------------|
| `de.py` | Differential evolution (DE/rand/1/bin) with multi-run statistics and convergence history |
| `testfuncs.py` | Standard test functions — sphere, Rosenbrock, Rastrigin, Ackley |
| `lambert.py` | Izzo's Lambert solver — called by every MGA fitness evaluation |
| `ephemeris.py` | Analytical planetary ephemeris from JPL Keplerian elements |
| `mga.py` | MGA fitness functions, flyby cost model, problem definitions (Earth-Mars, EVEJ, Cassini1) |
| `surrogate.py` | MLP surrogate with input normalization and training reservoir |
| `de_surrogate.py` | Surrogate-assisted DE with pre-screening strategy |
| `benchmark.py` | Benchmark harness comparing plain DE against surrogate-assisted DE |
| `validate.py` | 6-test validation suite |
| `main.py` | CLI entry point with problem selection and budget options |

---

## Performance

| Problem | Dimensions | Typical Budget | Runtime per Run |
|---------|-----------|---------------|-----------------|
| Earth → Mars | 2 | 3,000 evals | ~30 s |
| EVEJ | 4 | 15,000 evals | ~2 min |
| Cassini1 | 6 | 30,000 evals | ~5 min |

Measured on a single CPU core. The computation is embarrassingly parallel but parallelisation is not implemented — each Lambert solve is already fast enough that the overhead of thread dispatch would not pay off at this scale.

---

## Known Limitations

- **Unpowered MGA only.** Deep-space maneuvers within transfer legs are not modelled, so Cassini1 results will not match the MGA-1DSM reference of 4.93 km/s. This is a deliberate scope choice to keep the fitness function simple and physically transparent.
- **Default DE parameters only.** Self-adaptive variants (jDE, SHADE, L-SHADE) are not implemented. F and CR are fixed at 0.7 and 0.9.
- **No launch C3 filtering.** All departure Δv values are reported regardless of realistic launch vehicle capability.
- **Single-objective.** The optimizer minimizes total Δv only. Pareto frontiers over Δv vs flight time are not supported.
- **Surrogate is optional but not adaptive.** `screen_fraction` and `retrain_every` are fixed hyperparameters, not learned from the observed landscape.

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
