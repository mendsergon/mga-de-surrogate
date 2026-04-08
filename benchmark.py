"""
Benchmark Harness

Compares pure differential evolution against surrogate-assisted DE on
MGA trajectory problems. Measures:
    - Best / mean / std Δv over N independent runs
    - Real fitness evaluations used
    - Convergence curves (best Δv vs real evaluations)
"""

from __future__ import annotations

import numpy as np
import time
from typing import Callable, Dict, List

from de import differential_evolution
from de_surrogate import surrogate_de
from mga import (
    earth_mars_direct_fitness, EARTH_MARS_BOUNDS,
    evej_fitness, EVEJ_BOUNDS, EVEJ_SEQUENCE,
    cassini1_fitness, CASSINI1_BOUNDS, CASSINI1_SEQUENCE,
    CASSINI1_REFERENCE_DV,
    decode_mission,
)


# --- Problem registry ---

PROBLEMS = {
    "earth-mars": {
        "fitness":  earth_mars_direct_fitness,
        "bounds":   EARTH_MARS_BOUNDS,
        "sequence": ["earth", "mars"],
        "name":     "Earth → Mars direct",
        "reference": 5.61,
    },
    "evej": {
        "fitness":  evej_fitness,
        "bounds":   EVEJ_BOUNDS,
        "sequence": EVEJ_SEQUENCE,
        "name":     "Earth → Venus → Earth → Jupiter",
        "reference": None,
    },
    "cassini1": {
        "fitness":  cassini1_fitness,
        "bounds":   CASSINI1_BOUNDS,
        "sequence": CASSINI1_SEQUENCE,
        "name":     "Cassini1 (EVVEJS)",
        "reference": CASSINI1_REFERENCE_DV,
    },
}


# --- Single benchmark ---

def run_benchmark(
    problem_name: str,
    n_runs: int = 5,
    max_evals: int = 20000,
    pop_size: int = 60,
    use_surrogate: bool = False,
    verbose: bool = False,
) -> Dict:
    """
    Run an optimizer (plain DE or surrogate-DE) on a problem
    multiple times and collect statistics.
    """
    prob = PROBLEMS[problem_name]

    results = []
    evals_used = []
    times = []
    histories = []
    best_x = None
    best_f = np.inf

    label = "surrogate-DE" if use_surrogate else "DE"
    print(f"\n  Running {label} on {prob['name']}")
    print(f"    {n_runs} runs × {max_evals} max evals  (pop={pop_size})")

    for k in range(n_runs):
        t0 = time.time()

        if use_surrogate:
            x, f, hist = surrogate_de(
                prob["fitness"], prob["bounds"],
                pop_size=pop_size,
                max_gen=10000,
                max_evals=max_evals,
                seed=k,
                warmup_evals=min(500, max_evals // 4),
                screen_fraction=0.3,
                retrain_every=10,
                verbose=verbose,
            )
        else:
            x, f, hist = differential_evolution(
                prob["fitness"], prob["bounds"],
                pop_size=pop_size,
                max_gen=10000,
                max_evals=max_evals,
                seed=k,
                verbose=verbose,
            )

        dt = time.time() - t0
        results.append(f)
        evals_used.append(hist.n_evals[-1])
        times.append(dt)
        histories.append(hist)

        if f < best_f:
            best_f = f
            best_x = x

        print(f"    run {k+1}/{n_runs}  Δv = {f:7.3f} km/s  "
              f"evals = {hist.n_evals[-1]:6d}  time = {dt:5.1f} s")

    results = np.array(results)
    return {
        "problem":     problem_name,
        "optimizer":   label,
        "best":        float(np.min(results)),
        "worst":       float(np.max(results)),
        "mean":        float(np.mean(results)),
        "median":      float(np.median(results)),
        "std":         float(np.std(results)),
        "n_runs":      n_runs,
        "max_evals":   max_evals,
        "all_results": results.tolist(),
        "evals_used":  evals_used,
        "avg_time_s":  float(np.mean(times)),
        "best_x":      best_x.tolist() if best_x is not None else None,
        "histories":   histories,
        "sequence":    prob["sequence"],
        "reference":   prob["reference"],
    }


# --- Comparison printout ---

def print_comparison(de_stats: Dict, surr_stats: Dict) -> None:
    """Pretty-print DE vs surrogate-DE comparison."""
    name = PROBLEMS[de_stats["problem"]]["name"]
    ref = de_stats["reference"]

    print(f"\n{'='*64}")
    print(f"  COMPARISON: {name}")
    print(f"{'='*64}")
    if ref is not None:
        print(f"  Reference (best known): {ref:.3f} km/s")
    print(f"  Budget: {de_stats['max_evals']} real evaluations per run\n")

    header = f"  {'metric':<22} {'DE':>15} {'surrogate-DE':>18}"
    print(header)
    print("  " + "-" * 58)
    for metric in ["best", "mean", "median", "worst", "std"]:
        print(f"  {metric:<22} {de_stats[metric]:>13.3f}   {surr_stats[metric]:>16.3f}")
    print(f"  {'avg time (s)':<22} {de_stats['avg_time_s']:>13.2f}   "
          f"{surr_stats['avg_time_s']:>16.2f}")

    # Acceleration metric
    if surr_stats["mean"] > 0:
        improvement = (de_stats["mean"] - surr_stats["mean"]) / de_stats["mean"]
        print(f"\n  Mean Δv improvement (surrogate vs plain): {improvement*100:+.1f}%")


# --- Mission printout ---

def print_mission(stats: Dict) -> None:
    """Show the best mission found."""
    if stats["best_x"] is None:
        return

    info = decode_mission(np.array(stats["best_x"]), stats["sequence"])
    print(f"\n  Best mission found:")
    print(f"    Departure: {info['departure']}")
    print(f"    Arrival:   {info['arrival']}")
    print(f"    Duration:  {info['total_days']:.0f} days "
          f"({info['total_years']:.2f} years)")
    print(f"    Total Δv:  {info['total_dv']:.3f} km/s")
    if len(info['legs']) > 1:
        print(f"    Legs:")
        for leg in info["legs"]:
            print(f"      {leg['from']:8s} → {leg['to']:8s}  "
                  f"{leg['tof_days']:6.0f} d  (arr {leg['arrival_date']})")
