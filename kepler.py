"""
kepler.py — Two-body Keplerian orbit propagation via universal variables.

Follows Bate, Mueller, White (1971), Chapter 4.

All units: km, km/s, seconds, km^3/s^2.
"""

import numpy as np


def stumpff_c2(psi):
    if psi > 1e-6:
        s = np.sin(np.sqrt(psi) / 2.0)
        return 2.0 * s * s / psi
    elif psi < -1e-6:
        # Clamp to avoid overflow in cosh for large negative psi
        psi_clamped = max(psi, -500.0)
        return (np.cosh(np.sqrt(-psi_clamped)) - 1.0) / (-psi_clamped)
    else:
        return 0.5 - psi / 24.0 + psi**2 / 720.0


def stumpff_c3(psi):
    if psi > 1e-6:
        sp = np.sqrt(psi)
        return (sp - np.sin(sp)) / (psi * sp)
    elif psi < -1e-6:
        # Clamp to avoid overflow in sinh for large negative psi
        psi_clamped = max(psi, -500.0)
        sp = np.sqrt(-psi_clamped)
        return (np.sinh(sp) - sp) / ((-psi_clamped) * sp)
    else:
        return 1.0/6.0 - psi / 120.0 + psi**2 / 5040.0

def _propagate_single(r0, v0, dt, mu):
    """
    Single-step universal variable propagator.

    Uses plain Newton-Raphson on the universal Kepler equation. Converges
    quadratically from the initial guess; Halley's method was unnecessary
    (and the previous Fpp expression was dimensionally inconsistent).

    Convergence uses two tests, either of which accepts:
      1. Relative step:     |dchi| <= TOL_CHI * max(1, |chi|)
      2. Relative residual: |F|    <= TOL_F   * (|target| + r0*|chi|)
    Both tests track the floating-point scale of the equation. A fixed
    |dchi| < 1e-12 test is below ulp(target) for solar-system targets,
    causing limit-cycle non-convergence when NR is otherwise at the answer.
    """
    r0_mag = np.linalg.norm(r0)
    v0_mag = np.linalg.norm(v0)
    vr0    = np.dot(r0, v0) / r0_mag
    alpha  = 2.0 / r0_mag - v0_mag**2 / mu
    sqrt_mu = np.sqrt(mu)

    # Initial guess
    if alpha > 1e-12:
        # Elliptic
        a    = 1.0 / alpha
        chi0 = sqrt_mu * abs(dt) * alpha
        chi0 = np.sign(dt) * min(abs(chi0), 2.0 * np.pi * np.sqrt(a))
    elif alpha < -1e-12:
        # Hyperbolic
        a    = 1.0 / alpha
        chi0 = (np.sign(dt) * np.sqrt(-a) *
                np.log((-2.0 * mu * alpha * dt) /
                       (np.dot(r0, v0) +
                        np.sign(dt) * np.sqrt(-mu * a) * (1.0 - r0_mag * alpha))))
    else:
        # Parabolic
        h    = np.cross(r0, v0)
        p    = np.linalg.norm(h)**2 / mu
        chi0 = np.sqrt(2.0 * p)

    chi    = chi0
    target = sqrt_mu * dt
    c2 = c3 = psi = 0.0
    r_mag = r0_mag

    TOL_CHI = 1e-11
    TOL_F   = 1e-13

    converged = False
    for _ in range(100):
        psi = chi**2 * alpha
        c2  = stumpff_c2(psi)
        c3  = stumpff_c3(psi)

        r_mag = (chi**2 * c2
                 + (r0_mag * vr0 / sqrt_mu) * chi * (1.0 - psi * c3)
                 + r0_mag * (1.0 - psi * c2))

        if r_mag <= 0.0:
            chi *= 0.5
            continue

        rhs = (chi**3 * c3
               + (r0_mag * vr0 / sqrt_mu) * chi**2 * c2
               + r0_mag * chi * (1.0 - psi * c3))

        F    = rhs - target
        dchi = -F / r_mag  # Newton; F' = r_mag

        # Residual-based acceptance: at floating-point floor of the equation.
        scale = abs(target) + r0_mag * abs(chi)
        if abs(F) <= TOL_F * scale:
            converged = True
            break

        chi += dchi

        # Relative-step acceptance: threshold tracks |chi| so it stays above ulp noise.
        if abs(dchi) <= TOL_CHI * max(1.0, abs(chi)):
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"Keplerian propagator did not converge: "
            f"r0={r0_mag:.1f} km, dt={dt:.1f} s, alpha={alpha:.3e}"
        )

    f  = 1.0 - chi**2 * c2 / r0_mag
    g  = dt  - chi**3 * c3 / sqrt_mu

    r1     = f * r0 + g * v0
    r1_mag = np.linalg.norm(r1)

    df = sqrt_mu * chi * (psi * c3 - 1.0) / (r1_mag * r0_mag)
    dg = 1.0 - chi**2 * c2 / r1_mag

    return r1, df * r0 + dg * v0


def propagate(r0, v0, dt, mu, max_step_s=None):
    """
    Propagate a two-body Keplerian orbit using universal variables.

    Long propagations are split into sub-steps to avoid the near-(2π)²
    singularity in the universal Kepler equation. Default max_step is
    20 days, which keeps psi well below (2π)² for all solar system
    transfer orbits (including tight Earth-Venus transfers).

    Args:
        r0         : ndarray shape (3,) — initial position (km)
        v0         : ndarray shape (3,) — initial velocity (km/s)
        dt         : float — propagation time (s), may be negative
        mu         : float — gravitational parameter (km^3/s^2)
        max_step_s : float — max sub-step in seconds (default: 20 days)

    Returns:
        r1, v1 : position and velocity after dt
    """
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    if max_step_s is None:
        max_step_s = 20.0 * 86400.0  # 20 days in seconds

    if abs(dt) <= max_step_s:
        return _propagate_single(r0, v0, dt, mu)

    n_steps = int(np.ceil(abs(dt) / max_step_s))
    step    = dt / n_steps
    r, v    = r0.copy(), v0.copy()
    for _ in range(n_steps):
        r, v = _propagate_single(r, v, step, mu)
    return r, v


def test_propagator():
    """
    Validate propagator on two tests:
    1. Quarter-orbit circular Earth orbit — known exact position
    2. Full-orbit round-trip — checks sub-stepping handles multi-period
    """
    MU_SUN = 1.32712440018e11  # km^3/s^2
    a      = 149.597870700e6   # km
    v_circ = np.sqrt(MU_SUN / a)
    T      = 2.0 * np.pi * a / v_circ

    r0 = np.array([a, 0.0, 0.0])
    v0 = np.array([0.0, v_circ, 0.0])

    # Test 1: quarter orbit
    r1, v1 = propagate(r0, v0, T / 4.0, MU_SUN)
    expected_r = np.array([0.0, a, 0.0])
    expected_v = np.array([-v_circ, 0.0, 0.0])

    pos_err = np.linalg.norm(r1 - expected_r)
    vel_err = np.linalg.norm(v1 - expected_v)
    print(f"Quarter-orbit test:")
    print(f"  Position error : {pos_err:.3e} km  (target < 1 km)")
    print(f"  Velocity error : {vel_err:.2e} km/s  (target < 1e-4 km/s)")
    assert pos_err < 1.0,  f"FAIL pos_err={pos_err:.1f} km"
    assert vel_err < 1e-4, f"FAIL vel_err={vel_err:.2e} km/s"
    print("  PASS")

    # Test 2: full orbit round-trip via sub-stepping
    r2, v2 = propagate(r0, v0, T, MU_SUN)
    pos_err2 = np.linalg.norm(r2 - r0)
    vel_err2 = np.linalg.norm(v2 - v0)
    print(f"Full-orbit round-trip (sub-stepped):")
    print(f"  Position error : {pos_err2:.3e} km  (target < 10 km)")
    print(f"  Velocity error : {vel_err2:.2e} km/s  (target < 1e-4 km/s)")
    assert pos_err2 < 10.0, f"FAIL pos_err={pos_err2:.1f} km"
    assert vel_err2 < 1e-4, f"FAIL vel_err={vel_err2:.2e} km/s"
    print("  PASS")


if __name__ == "__main__":
    test_propagator()