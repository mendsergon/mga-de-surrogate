"""
Standard test functions for global optimization.

Used to validate the DE implementation before applying it to
interplanetary trajectory problems.
"""

from __future__ import annotations

import numpy as np


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin. Global minimum f(0) = 0. Highly multimodal."""
    x = np.asarray(x)
    A = 10.0
    n = len(x)
    return float(A * n + np.sum(x**2 - A * np.cos(2.0 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock. Global minimum f([1,1,...]) = 0. Narrow curved valley."""
    x = np.asarray(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2))


def sphere(x: np.ndarray) -> float:
    """Sphere. Global minimum f(0) = 0. Simplest convex test."""
    return float(np.sum(np.asarray(x)**2))


def ackley(x: np.ndarray) -> float:
    """Ackley. Global minimum f(0) = 0. Flat outer region with central well."""
    x = np.asarray(x)
    n = len(x)
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(2.0 * np.pi * x))
    return float(-20.0 * np.exp(-0.2 * np.sqrt(s1 / n))
                 - np.exp(s2 / n) + 20.0 + np.e)


BOUNDS = {
    "rastrigin":  (-5.12, 5.12),
    "rosenbrock": (-2.048, 2.048),
    "sphere":     (-5.0, 5.0),
    "ackley":     (-32.768, 32.768),
}


def make_bounds(name: str, dim: int) -> np.ndarray:
    """Build a (dim, 2) bounds array for a named test function."""
    lo, hi = BOUNDS[name]
    return np.tile([lo, hi], (dim, 1))
