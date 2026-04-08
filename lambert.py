"""
Izzo's Lambert Solver (2015)

Izzo, D. (2015). Revisiting Lambert's problem.
Celestial Mechanics and Dynamical Astronomy, 121(1), 1-15.

Given r1, r2, and time of flight, finds the connecting Keplerian arc.
Reformulates the problem in terms of a single parameter x and solves
T(x) = T_target via Householder iteration with analytical derivatives
up to third order (quartic convergence).
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from typing import Tuple, List


# --- Stumpff functions c2, c3 ---

def _stumpff_c2(psi: float) -> float:
    """c2(ψ) = (1 - cos √ψ) / ψ"""
    if abs(psi) < 1e-12:
        return 1.0 / 2.0 - psi / 24.0 + psi**2 / 720.0
    if psi > 0:
        sp = np.sqrt(psi)
        return (1.0 - np.cos(sp)) / psi
    sp = np.sqrt(-psi)
    return (np.cosh(sp) - 1.0) / (-psi)


def _stumpff_c3(psi: float) -> float:
    """c3(ψ) = (√ψ - sin √ψ) / ψ^{3/2}"""
    if abs(psi) < 1e-12:
        return 1.0 / 6.0 - psi / 120.0 + psi**2 / 5040.0
    if psi > 0:
        sp = np.sqrt(psi)
        return (sp - np.sin(sp)) / (psi * sp)
    sp = np.sqrt(-psi)
    return (np.sinh(sp) - sp) / ((-psi) * sp)


# --- Non-dimensional time of flight T(x, λ, N) ---

def _x2tof(x: float, lam: float, N: int) -> float:
    """
    Non-dimensional TOF from free parameter x.

    Battin formulation: elliptic (x < 1) via arccos/arcsin,
    hyperbolic (x > 1) via arccosh/arcsinh.
    Falls back to series expansion for |x - 1| < 1e-4.
    """
    if abs(x - 1.0) < 1e-4:
        return _x2tof_series(x, lam, N)

    d = 1.0 - x * x

    if d > 0:  # elliptic
        alpha = 2.0 * np.arccos(np.clip(x, -1.0, 1.0))
        sin_arg = np.clip(abs(lam) * np.sqrt(d), -1.0, 1.0)
        beta = 2.0 * np.arcsin(sin_arg)
        if lam < 0:
            beta = -beta
        T = ((alpha - np.sin(alpha)) - (beta - np.sin(beta))
             + 2.0 * N * np.pi) / (2.0 * d**1.5)

    else:  # hyperbolic
        alpha_h = 2.0 * np.arccosh(x)
        beta_h = 2.0 * np.arcsinh(np.sqrt(-lam * lam * d))
        if lam < 0:
            beta_h = -beta_h
        T = ((-alpha_h + np.sinh(alpha_h))
             - (-beta_h + np.sinh(beta_h))) / (2.0 * (-d)**1.5)

    return T


def _x2tof_series(x: float, lam: float, N: int) -> float:
    """Near-parabolic TOF via series expansion around x = 1 (Izzo Eq. 18)."""
    xi = 1.0 - x * x
    lam2 = lam * lam

    d = 1.0 / 3.0 * (1.0 - lam2 * lam)
    T = d
    for k in range(1, 20):
        c_k = (3.0 + 5.0 * (k - 1) - 4.0 * (k - 1) * lam2)
        c_k /= ((2.0 * k + 1.0) * (2.0 * (k - 1) + 3.0))
        d = d * c_k * xi
        T += d
        if abs(d) < 1e-16:
            break

    T = 2.0 * T + N * np.pi / max(abs(1.0 - x * x), 1e-30)**1.5
    return T


# --- Derivatives dT/dx, d²T/dx², d³T/dx³ (Izzo §2.4) ---

def _dt_dx(x: float, T: float, lam: float) -> Tuple[float, float, float]:
    """
    First three derivatives of T(x) for Householder iteration.

    T'   = (3Tx − 2 + 2λ³x/y) / (1 − x²)
    T''  = (3T + 5xT' + 2(1−λ²)λ³/y³) / (1 − x²)
    T''' = (7xT'' + 8T' − 6(1−λ²)λ⁵x/y⁵) / (1 − x²)
    """
    lam2 = lam * lam
    lam3 = lam2 * lam
    lam5 = lam2 * lam3
    omx2 = 1.0 - x * x
    y = np.sqrt(max(1.0 - lam2 * omx2, 0.0))

    if abs(omx2) < 1e-14:
        return 0.0, 0.0, 0.0

    inv_omx2 = 1.0 / omx2
    y_safe = max(y, 1e-30)

    dT = (3.0 * T * x - 2.0 + 2.0 * lam3 * x / y_safe) * inv_omx2
    ddT = (3.0 * T + 5.0 * x * dT
           + 2.0 * (1.0 - lam2) * lam3 / (y_safe**3)) * inv_omx2
    dddT = (7.0 * x * ddT + 8.0 * dT
            - 6.0 * (1.0 - lam2) * lam5 * x / (y_safe**5)) * inv_omx2

    return dT, ddT, dddT


# --- Initial guess ---

def _initial_guess(T_target: float, lam: float, N: int) -> float:
    """
    Starting x₀ for iteration. Compares T_target against T(x=0)
    and T_parabolic to select the correct energy branch.
    """
    if N == 0:
        T0 = np.arccos(lam) + lam * np.sqrt(1.0 - lam * lam)
        T_parab = 2.0 / 3.0 * (1.0 - lam**3)

        if T_target <= T_parab:
            x0 = (T_parab / T_target) ** (2.0 / 3.0) - 1.0
        elif T_target >= T0:
            x0 = -((T_target / T0 - 1.0) ** (2.0 / 3.0))
            x0 = max(x0, -0.999)
        else:
            frac = (T_target - T_parab) / (T0 - T_parab)
            x0 = (1.0 - frac) * 0.999 + frac * 0.0
    else:
        x0 = 0.0

    return np.clip(x0, -0.999, 10.0)


# --- Householder iteration ---

def _householder(x0: float, T_target: float, lam: float, N: int,
                 tol: float = 1e-12, max_iter: int = 50) -> float:
    """Third-order Householder iteration for T(x) = T_target (quartic convergence)."""
    x = x0
    for _ in range(max_iter):
        T = _x2tof(x, lam, N)
        f = T - T_target
        if abs(f) < tol:
            break
        dT, ddT, dddT = _dt_dx(x, T, lam)
        if abs(dT) < 1e-30:
            break

        delta = f / dT
        delta /= (1.0 - 0.5 * delta * ddT / dT)
        delta /= (1.0 - delta * (0.5 * ddT / dT
                   - delta * dddT / (6.0 * dT)))
        x -= delta

    return x


# --- Velocity extraction ---

def _compute_velocities(
    r1_vec: np.ndarray, r2_vec: np.ndarray,
    r1: float, r2: float,
    c: float, s: float,
    lam: float, x: float,
    mu: float, dtheta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Recover v1, v2 from solved parameter x via radial/transverse
    decomposition (Izzo §2.3).
    """
    y = np.sqrt(max(1.0 - lam * lam * (1.0 - x * x), 0.0))

    # Unit vectors
    ir1 = r1_vec / r1
    ir2 = r2_vec / r2

    cross = np.cross(ir1, ir2)
    cn = norm(cross)
    ih = np.array([0.0, 0.0, 1.0]) if cn < 1e-14 else cross / cn

    it1 = np.cross(ih, ir1)
    it2 = np.cross(ih, ir2)

    # Velocity components
    gamma = np.sqrt(mu * s / 2.0)
    rho = (r1 - r2) / c if c > 1e-14 else 0.0
    sigma = np.sqrt(max(1.0 - rho * rho, 0.0))
    if np.sin(dtheta) < 0:
        sigma = -sigma

    lyx = lam * y - x
    lxy = y + lam * x

    vr1 = gamma * (lyx - rho * (lam * y + x)) / r1
    vr2 = -gamma * (lyx + rho * (lam * y + x)) / r2
    vt1 = gamma * sigma * lxy / r1
    vt2 = gamma * sigma * lxy / r2

    v1 = vr1 * ir1 + vt1 * it1
    v2 = vr2 * ir2 + vt2 * it2

    return v1, v2


# --- Public API ---

def solve(
    r1_vec: np.ndarray,
    r2_vec: np.ndarray,
    tof: float,
    mu: float,
    prograde: bool = True,
    multi_revs: int = 0,
    tol: float = 1e-12,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Solve Lambert's problem using Izzo's algorithm.

    Parameters
    ----------
    r1_vec     : departure position [km]
    r2_vec     : arrival position [km]
    tof        : time of flight [s]
    mu         : gravitational parameter [km³/s²]
    prograde   : True for prograde transfer
    multi_revs : max complete revolutions to include
    tol        : convergence tolerance

    Returns
    -------
    List of (v1, v2) tuples — departure and arrival velocities [km/s].
    """
    r1_vec = np.asarray(r1_vec, dtype=np.float64)
    r2_vec = np.asarray(r2_vec, dtype=np.float64)

    r1 = norm(r1_vec)
    r2 = norm(r2_vec)

    # Transfer angle
    cos_dtheta = np.clip(np.dot(r1_vec, r2_vec) / (r1 * r2), -1.0, 1.0)
    cross_z = np.cross(r1_vec, r2_vec)[2]

    if prograde:
        dtheta = np.arccos(cos_dtheta) if cross_z >= 0 \
            else 2.0 * np.pi - np.arccos(cos_dtheta)
    else:
        dtheta = np.arccos(cos_dtheta) if cross_z < 0 \
            else 2.0 * np.pi - np.arccos(cos_dtheta)

    # Chord and semi-perimeter
    c = np.sqrt(r1**2 + r2**2 - 2.0 * r1 * r2 * np.cos(dtheta))
    s = (r1 + r2 + c) / 2.0

    # Non-dimensional geometry parameter and target TOF
    lam2 = 1.0 - c / s
    lam = np.sqrt(max(lam2, 0.0))
    if dtheta > np.pi:
        lam = -lam

    T_target = tof * np.sqrt(2.0 * mu / s**3)

    # Solve for each revolution count
    solutions = []
    for N in range(multi_revs + 1):
        try:
            x0 = _initial_guess(T_target, lam, N)
            x = _householder(x0, T_target, lam, N, tol=tol)

            T_check = _x2tof(x, lam, N)
            if abs(T_check - T_target) / max(T_target, 1e-12) > 1e-6:
                continue

            v1, v2 = _compute_velocities(
                r1_vec, r2_vec, r1, r2, c, s, lam, x, mu, dtheta
            )
            solutions.append((v1, v2))

        except (ValueError, RuntimeError, ZeroDivisionError):
            continue

    return solutions
