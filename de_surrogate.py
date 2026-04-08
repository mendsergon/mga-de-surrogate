"""
Surrogate-Assisted Differential Evolution

Standard DE with neural network pre-screening. Each generation:

  1. Generate trial vectors via mutation and crossover as usual.
  2. If the surrogate is trained, predict fitness for all trials.
  3. Evaluate the real fitness function only for the top-k predicted trials.
  4. Add new observations to the training set and periodically retrain.

This reduces real fitness evaluations (expensive Lambert solves) while
still exploring the search space broadly.

Jin, Y. (2011). Surrogate-assisted evolutionary computation.
Swarm and Evolutionary Computation, 1(2), 61-70.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass, field
from typing import List

from de import DEHistory, _reflect_bounds
from surrogate import Surrogate, DataCollector


@dataclass
class SurrogateHistory(DEHistory):
    """Convergence history with surrogate-specific metrics."""
    real_evals: List[int] = field(default_factory=list)
    surrogate_score: List[dict] = field(default_factory=list)
    retrain_generations: List[int] = field(default_factory=list)


def surrogate_de(
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
    # Surrogate-specific parameters
    warmup_evals: int = 500,
    screen_fraction: float = 0.3,
    retrain_every: int = 10,
    surrogate_hidden: Tuple[int, ...] = (128, 128, 64),
) -> Tuple[np.ndarray, float, SurrogateHistory]:
    """
    Surrogate-assisted DE/rand/1/bin.

    Parameters
    ----------
    fitness         : expensive real fitness function
    bounds          : (D, 2) array of [lower, upper]
    pop_size        : DE population size
    F, CR           : DE mutation and crossover parameters
    max_gen         : maximum generations
    max_evals       : cap on real fitness evaluations
    tol             : convergence tolerance
    seed            : RNG seed
    verbose         : print progress
    warmup_evals    : number of real evaluations before training the surrogate
    screen_fraction : fraction of trial population actually evaluated when
                      surrogate is active (top-k by predicted fitness)
    retrain_every   : retrain surrogate every N generations
    surrogate_hidden: hidden layer sizes for the MLP

    Returns
    -------
    best_x, best_f, history
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=np.float64)
    D = bounds.shape[0]
    lb, ub = bounds[:, 0], bounds[:, 1]

    # Initialize population
    pop = lb + rng.random((pop_size, D)) * (ub - lb)
    fit = np.array([fitness(x) for x in pop])
    n_real_evals = pop_size

    # Training data reservoir and surrogate
    collector = DataCollector(max_size=10000)
    collector.add_batch(pop, fit)

    surrogate = Surrogate(
        bounds=bounds,
        hidden_layers=surrogate_hidden,
        random_state=seed if seed is not None else 0,
    )

    history = SurrogateHistory()
    history.best_fitness.append(float(np.min(fit)))
    history.mean_fitness.append(float(np.mean(fit)))
    history.n_evals.append(n_real_evals)
    history.real_evals.append(n_real_evals)

    k_screen = max(1, int(screen_fraction * pop_size))

    for gen in range(max_gen):
        # --- Generate trial population ---
        trials = np.zeros_like(pop)
        for i in range(pop_size):
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(idxs, 3, replace=False)

            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = _reflect_bounds(mutant, lb, ub)

            cross = rng.random(D) < CR
            if not np.any(cross):
                cross[rng.integers(0, D)] = True
            trials[i] = np.where(cross, mutant, pop[i])

        # --- Decide which trials to evaluate for real ---
        if surrogate.fitted and n_real_evals >= warmup_evals:
            # Surrogate-assisted: predict all, evaluate only top k
            predicted = surrogate.predict(trials)
            # Sort ascending (we minimize)
            order = np.argsort(predicted)
            eval_idx = order[:k_screen]
        else:
            # Warmup phase: evaluate everything
            eval_idx = np.arange(pop_size)

        # --- Real evaluation and selection ---
        for i in eval_idx:
            f_trial = fitness(trials[i])
            n_real_evals += 1
            collector.add(trials[i], f_trial)

            if f_trial < fit[i]:
                pop[i] = trials[i]
                fit[i] = f_trial

            if max_evals is not None and n_real_evals >= max_evals:
                break

        # --- Retrain surrogate periodically ---
        if (gen + 1) % retrain_every == 0 and len(collector) >= 20:
            X_train, y_train = collector.arrays()
            surrogate.fit(X_train, y_train)
            history.retrain_generations.append(gen + 1)

            if surrogate.fitted and verbose:
                score = surrogate.score(X_train, y_train)
                history.surrogate_score.append(score)

        # --- Record history ---
        history.best_fitness.append(float(np.min(fit)))
        history.mean_fitness.append(float(np.mean(fit)))
        history.n_evals.append(n_real_evals)
        history.real_evals.append(n_real_evals)

        if verbose and (gen + 1) % 50 == 0:
            status = "surrogate" if surrogate.fitted else "warmup"
            print(f"  gen {gen+1:4d}  real_evals={n_real_evals:6d}  "
                  f"best={np.min(fit):.6e}  [{status}]")

        if np.std(fit) < tol:
            if verbose:
                print(f"  converged at gen {gen+1}")
            break
        if max_evals is not None and n_real_evals >= max_evals:
            if verbose:
                print(f"  max_evals reached at gen {gen+1}")
            break

    best_idx = int(np.argmin(fit))
    history.best_vector = pop[best_idx].copy()
    return pop[best_idx], float(fit[best_idx]), history
