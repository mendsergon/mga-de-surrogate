"""
Differential Evolution (DE/rand/1/bin)

Storn, R. and Price, K. (1997). Differential Evolution — A Simple and
Efficient Heuristic for Global Optimization over Continuous Spaces.
Journal of Global Optimization, 11(4), 341-359.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class DEHistory:
    """Convergence history for a DE run."""
    best_fitness: List[float] = field(default_factory=list)
    mean_fitness: List[float] = field(default_factory=list)
    n_evals: List[int] = field(default_factory=list)
    best_vector: Optional[np.ndarray] = None


def _reflect_bounds(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Reflect out-of-bound values back into the feasible region."""
    x = np.where(x < lb, 2.0 * lb - x, x)
    x = np.where(x > ub, 2.0 * ub - x, x)
    return np.clip(x, lb, ub)


def differential_evolution(
    fitness: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    pop_size: int = 50,
    F: float = 0.7,
    CR: float = 0.9,
    max_gen: int = 500,
    max_evals: Optional[int] = None,
    tol: float = 1e-8,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, DEHistory]:
    """
    Minimize fitness(x) over bounded search space via DE/rand/1/bin.

    Parameters
    ----------
    fitness   : objective function, takes 1-D array, returns scalar
    bounds    : array of shape (D, 2) with [lower, upper] per dimension
    pop_size  : population size (typically 10×D)
    F         : mutation scale factor ∈ [0.4, 0.9]
    CR        : crossover rate ∈ [0.7, 0.95]
    max_gen   : maximum generations
    max_evals : maximum fitness evaluations (overrides max_gen)
    tol       : convergence tolerance on population fitness std
    seed      : RNG seed
    verbose   : print progress every 50 generations
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=np.float64)
    D = bounds.shape[0]
    lb, ub = bounds[:, 0], bounds[:, 1]

    # Initialize population uniformly
    pop = lb + rng.random((pop_size, D)) * (ub - lb)
    fit = np.array([fitness(x) for x in pop])
    n_evals = pop_size

    history = DEHistory()
    history.best_fitness.append(float(np.min(fit)))
    history.mean_fitness.append(float(np.mean(fit)))
    history.n_evals.append(n_evals)

    for gen in range(max_gen):
        for i in range(pop_size):
            # Pick three distinct candidates ≠ i
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(idxs, 3, replace=False)

            # Mutation: v = x_a + F * (x_b - x_c)
            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = _reflect_bounds(mutant, lb, ub)

            # Binomial crossover
            cross = rng.random(D) < CR
            if not np.any(cross):
                cross[rng.integers(0, D)] = True
            trial = np.where(cross, mutant, pop[i])

            # Selection
            f_trial = fitness(trial)
            n_evals += 1
            if f_trial < fit[i]:
                pop[i] = trial
                fit[i] = f_trial

            if max_evals is not None and n_evals >= max_evals:
                break

        history.best_fitness.append(float(np.min(fit)))
        history.mean_fitness.append(float(np.mean(fit)))
        history.n_evals.append(n_evals)

        if verbose and (gen + 1) % 50 == 0:
            print(f"  gen {gen+1:4d}  evals={n_evals:6d}  "
                  f"best={np.min(fit):.6e}  mean={np.mean(fit):.6e}")

        if np.std(fit) < tol:
            if verbose:
                print(f"  converged (std < {tol}) at gen {gen+1}")
            break
        if max_evals is not None and n_evals >= max_evals:
            if verbose:
                print(f"  max_evals reached at gen {gen+1}")
            break

    best_idx = int(np.argmin(fit))
    history.best_vector = pop[best_idx].copy()
    return pop[best_idx], float(fit[best_idx]), history


def de_multi_run(
    fitness: Callable[[np.ndarray], float],
    bounds: np.ndarray,
    n_runs: int = 20,
    seed_base: int = 0,
    **de_kwargs,
) -> dict:
    """
    Run DE multiple times with different seeds and aggregate statistics.
    Standard practice for reporting stochastic optimizer performance.
    """
    results = []
    histories = []
    for k in range(n_runs):
        x, f, hist = differential_evolution(
            fitness, bounds, seed=seed_base + k, **de_kwargs
        )
        results.append(f)
        histories.append(hist)

    results = np.array(results)
    return {
        "best":   float(np.min(results)),
        "worst":  float(np.max(results)),
        "mean":   float(np.mean(results)),
        "median": float(np.median(results)),
        "std":    float(np.std(results)),
        "all":    results.tolist(),
        "histories": histories,
    }
