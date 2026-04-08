"""
Validation Suite

Tests DE, MGA fitness, and surrogate correctness.

    1. DE on standard test functions (sphere, Rosenbrock, Ackley)
    2. MGA Earth→Mars direct (should match porkchop minimum ~5.6 km/s)
    3. MGA Cassini1 feasibility (random points produce finite cost)
    4. Flyby cost model sanity (identical vectors = 0, reversal = high)
    5. Surrogate rank-correlation on a quadratic
    6. Surrogate-assisted DE end-to-end smoke test
"""

from __future__ import annotations

import numpy as np

from de import differential_evolution
from de_surrogate import surrogate_de
from surrogate import Surrogate
from testfuncs import rastrigin, rosenbrock, sphere, ackley, make_bounds
from mga import (
    cassini1_fitness, CASSINI1_BOUNDS, CASSINI1_SEQUENCE,
    earth_mars_direct_fitness, EARTH_MARS_BOUNDS,
    _flyby_cost, MU_PLANET, R_MIN_FLYBY,
    decode_mission,
)


def _heading(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def _pass_fail(name: str, passed: bool, details: str = "") -> None:
    sym = "✓" if passed else "✗"
    status = "PASS" if passed else "FAIL"
    print(f"  [{sym}] {name:40s} {status}  {details}")


# --- Test 1: DE on test functions ---

def test_de_standard_functions():
    _heading("Test 1: DE on standard test functions (D=10)")

    cases = [
        ("sphere",     sphere,     1e-4),
        ("rosenbrock", rosenbrock, 1.0),
        ("ackley",     ackley,     1e-2),
    ]

    for name, func, threshold in cases:
        bounds = make_bounds(name, 10)
        x, f, hist = differential_evolution(
            func, bounds, pop_size=50, max_gen=500, seed=42
        )
        _pass_fail(
            name,
            f < threshold,
            f"best={f:.3e}  (target <{threshold})"
        )


# --- Test 2: Earth→Mars direct ---

def test_earth_mars_direct():
    _heading("Test 2: DE on Earth→Mars direct transfer")

    x, f, hist = differential_evolution(
        earth_mars_direct_fitness, EARTH_MARS_BOUNDS,
        pop_size=30, max_gen=200, seed=42,
    )

    _pass_fail(
        "Δv < 6 km/s (porkchop ref ~5.6)",
        f < 6.0,
        f"best={f:.3f} km/s"
    )

    info = decode_mission(x, ["earth", "mars"])
    print(f"\n  Departure: {info['departure']}")
    print(f"  Arrival:   {info['arrival']}")
    print(f"  TOF: {info['total_days']:.0f} days")


# --- Test 3: Cassini1 feasibility ---

def test_cassini1_feasibility():
    _heading("Test 3: Cassini1 fitness produces finite values")

    rng = np.random.default_rng(0)
    n_samples = 100
    finite_count = 0
    penalty_val = 1e5

    for _ in range(n_samples):
        x = CASSINI1_BOUNDS[:, 0] + rng.random(6) * (
            CASSINI1_BOUNDS[:, 1] - CASSINI1_BOUNDS[:, 0]
        )
        f = cassini1_fitness(x)
        if np.isfinite(f) and f < penalty_val:
            finite_count += 1

    _pass_fail(
        "Random Cassini1 points finite",
        finite_count > 50,
        f"{finite_count}/{n_samples} finite"
    )


# --- Test 4: Flyby cost sanity ---

def test_flyby_cost():
    _heading("Test 4: Flyby cost model sanity")

    # Same vector in and out: zero cost
    v = np.array([5.0, 0.0, 0.0])
    cost = _flyby_cost(v, v, MU_PLANET["jupiter"], R_MIN_FLYBY["jupiter"])
    _pass_fail("Identical v_inf → zero cost", cost < 1e-9, f"cost={cost:.3e}")

    # 180° flip with equal magnitudes: impossible, high cost
    v_rev = np.array([-5.0, 0.0, 0.0])
    cost = _flyby_cost(v, v_rev, MU_PLANET["jupiter"], R_MIN_FLYBY["jupiter"])
    _pass_fail("180° reversal → high cost", cost > 1.0, f"cost={cost:.3f} km/s")

    # Small angle at Jupiter: easily feasible
    theta = 0.05  # ~3°
    v2 = np.array([5.0 * np.cos(theta), 5.0 * np.sin(theta), 0.0])
    cost = _flyby_cost(v, v2, MU_PLANET["jupiter"], R_MIN_FLYBY["jupiter"])
    _pass_fail("Small rotation → zero cost", cost < 1e-6, f"cost={cost:.3e}")


# --- Test 5: Surrogate rank correlation ---

def test_surrogate():
    _heading("Test 5: Surrogate model on quadratic")

    bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(200, 2))
    y = np.sum(X**2, axis=1)

    surrogate = Surrogate(bounds=bounds, max_iter=200, random_state=0)
    surrogate.fit(X, y)

    X_test = rng.uniform(-5, 5, size=(50, 2))
    y_test = np.sum(X_test**2, axis=1)
    score = surrogate.score(X_test, y_test)

    _pass_fail(
        "Rank correlation > 0.9",
        score["rank_corr"] > 0.9,
        f"ρ={score['rank_corr']:.3f}  MAE={score['mae']:.2f}"
    )


# --- Test 6: Surrogate-assisted DE end-to-end ---

def test_surrogate_de():
    _heading("Test 6: Surrogate-assisted DE end-to-end")

    x, f, hist = surrogate_de(
        earth_mars_direct_fitness, EARTH_MARS_BOUNDS,
        pop_size=30, max_gen=100, max_evals=1500, seed=42,
        warmup_evals=300, screen_fraction=0.3, retrain_every=10,
    )

    _pass_fail(
        "Surrogate-DE converges < 6 km/s",
        np.isfinite(f) and f < 6.0,
        f"best={f:.3f} km/s  real_evals={hist.real_evals[-1]}"
    )


# --- Run all ---

def run_all():
    print("\n" + "=" * 60)
    print("    GTOP-DE-SURROGATE VALIDATION SUITE")
    print("=" * 60)

    test_de_standard_functions()
    test_earth_mars_direct()
    test_cassini1_feasibility()
    test_flyby_cost()
    test_surrogate()
    test_surrogate_de()

    print("\n" + "=" * 60)
    print("    ALL TESTS COMPLETE")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all()
