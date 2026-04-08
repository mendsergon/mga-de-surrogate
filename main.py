#!/usr/bin/env python3
"""
GTOP DE + Surrogate Benchmark Runner

Usage:
    python main.py                              # Earth→Mars (fast default)
    python main.py --problem cassini1           # Cassini1 (slow, 6D)
    python main.py --problem evej               # Earth-Venus-Earth-Jupiter
    python main.py --validate                   # run validation suite
    python main.py --runs 10 --budget 30000     # custom settings
    python main.py --de-only                    # skip surrogate
    python main.py --surrogate-only             # skip plain DE
"""

from __future__ import annotations

import argparse
import sys

from benchmark import run_benchmark, print_comparison, print_mission, PROBLEMS


def main():
    parser = argparse.ArgumentParser(
        description="Differential evolution + surrogate benchmark on MGA trajectory problems",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--problem", "-p",
        choices=list(PROBLEMS.keys()),
        default="earth-mars",
        help="Problem to solve (default: earth-mars)",
    )
    parser.add_argument(
        "--runs", "-r",
        type=int, default=5,
        help="Number of independent runs (default: 5)",
    )
    parser.add_argument(
        "--budget", "-b",
        type=int, default=None,
        help="Max real fitness evaluations per run",
    )
    parser.add_argument(
        "--pop", type=int, default=None,
        help="Population size (default: 10×D)",
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Run validation suite and exit",
    )
    parser.add_argument(
        "--de-only",
        action="store_true",
        help="Run only plain DE (skip surrogate)",
    )
    parser.add_argument(
        "--surrogate-only",
        action="store_true",
        help="Run only surrogate-assisted DE (skip plain)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output inside each optimizer run",
    )

    args = parser.parse_args()

    if args.validate:
        from validate import run_all
        run_all()
        return

    # Defaults depend on problem size
    prob = PROBLEMS[args.problem]
    D = prob["bounds"].shape[0]
    default_budgets = {
        "earth-mars": 3000,
        "evej":       15000,
        "cassini1":   30000,
    }
    budget = args.budget or default_budgets.get(args.problem, 10000)
    pop_size = args.pop or max(20, 10 * D)

    print(f"\n{'='*64}")
    print(f"  GTOP-DE-SURROGATE BENCHMARK")
    print(f"{'='*64}")
    print(f"  Problem:      {prob['name']}")
    print(f"  Dimensions:   {D}")
    print(f"  Population:   {pop_size}")
    print(f"  Budget:       {budget} real evaluations per run")
    print(f"  Runs:         {args.runs}")
    if prob['reference'] is not None:
        print(f"  Reference:    {prob['reference']:.3f} km/s (best known)")

    de_stats = None
    surr_stats = None

    if not args.surrogate_only:
        de_stats = run_benchmark(
            args.problem,
            n_runs=args.runs,
            max_evals=budget,
            pop_size=pop_size,
            use_surrogate=False,
            verbose=args.verbose,
        )

    if not args.de_only:
        surr_stats = run_benchmark(
            args.problem,
            n_runs=args.runs,
            max_evals=budget,
            pop_size=pop_size,
            use_surrogate=True,
            verbose=args.verbose,
        )

    # Results
    if de_stats is not None and surr_stats is not None:
        print_comparison(de_stats, surr_stats)
        best_stats = de_stats if de_stats["best"] < surr_stats["best"] else surr_stats
        print_mission(best_stats)
    elif de_stats is not None:
        print(f"\n  DE best: {de_stats['best']:.3f} km/s")
        print(f"  DE mean: {de_stats['mean']:.3f} km/s")
        print_mission(de_stats)
    elif surr_stats is not None:
        print(f"\n  Surrogate-DE best: {surr_stats['best']:.3f} km/s")
        print(f"  Surrogate-DE mean: {surr_stats['mean']:.3f} km/s")
        print_mission(surr_stats)

    print()


if __name__ == "__main__":
    main()
