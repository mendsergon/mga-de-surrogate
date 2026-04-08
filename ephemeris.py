"""
Analytical Planetary Ephemeris

Heliocentric ecliptic (J2000) state vectors for the eight planets from
JPL's approximate Keplerian elements with linear secular rates.

Source: Standish, E. M. (1992, updated 2022). Approximate Positions of the Planets.
        JPL Solar System Dynamics.

Valid 1800–2050 AD. Inner planets ±1 arcmin, outer ±15 arcmin.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# --- Constants ---

AU_KM = 149_597_870.7          # 1 AU in km
MU_SUN = 1.327_124_400_18e11   # μ☉ [km³/s²]
DAY_S = 86_400.0               # 1 day [s]
JD_J2000 = 2_451_545.0         # JD of J2000.0


# --- JPL approximate Keplerian elements (J2000 ecliptic) ---
# [a₀ AU, ȧ AU/cy, e₀, ė /cy, I₀ °, İ °/cy, L₀ °, L̇ °/cy, ω̃₀ °, ω̃̇ °/cy, Ω₀ °, Ω̇ °/cy]

_ELEMENTS = {
    "mercury": (
        0.38709927,  0.00000037,  0.20563593,  0.00001906,
        7.00497902, -0.00594749,  252.25032350, 149472.67411175,
        77.45779628,  0.16047689,  48.33076593, -0.12534081,
    ),
    "venus": (
        0.72333566,  0.00000390,  0.00677672, -0.00004107,
        3.39467605, -0.00078890,  181.97909950, 58517.81538729,
        131.60246718,  0.00268329,  76.67984255, -0.27769418,
    ),
    "earth": (
        1.00000261,  0.00000562,  0.01671123, -0.00004392,
       -0.00001531, -0.01294668,  100.46457166, 35999.37244981,
        102.93768193,  0.32327364,  0.0, 0.0,
    ),
    "earth-moon": (
        1.00000261,  0.00000562,  0.01671123, -0.00004392,
       -0.00001531, -0.01294668,  100.46457166, 35999.37244981,
        102.93768193,  0.32327364,  0.0, 0.0,
    ),
    "mars": (
        1.52371034,  0.00001847,  0.09339410,  0.00007882,
        1.84969142, -0.00813131, -4.55343205, 19140.30268499,
       -23.94362959,  0.44441088,  49.55953891, -0.29257343,
    ),
    "jupiter": (
        5.20288700, -0.00011607,  0.04838624, -0.00013253,
        1.30439695, -0.00183714,  34.39644051, 3034.74612775,
        14.72847983,  0.21252668,  100.47390909, 0.20469106,
    ),
    "saturn": (
        9.53667594, -0.00125060,  0.05386179, -0.00050991,
        2.48599187,  0.00193609,  49.95424423, 1222.49362201,
        92.59887831, -0.41897216,  113.66242448, -0.28867794,
    ),
}


# --- Kepler's equation (Newton–Raphson) ---

def _solve_kepler(M: float, e: float, tol: float = 1e-14) -> float:
    """Solve M = E − e sin E for eccentric anomaly E."""
    M = (M + np.pi) % (2.0 * np.pi) - np.pi
    E = M + 0.85 * np.sign(np.sin(M)) * e

    for _ in range(20):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break

    return E


# --- Rotation: orbital frame → ecliptic J2000 ---

def _rotation_matrix(omega: float, inc: float, Omega: float) -> np.ndarray:
    """3-1-3 Euler rotation Rz(−Ω) Rx(−I) Rz(−ω). All angles in radians."""
    co, so = np.cos(omega), np.sin(omega)
    ci, si = np.cos(inc),   np.sin(inc)
    cO, sO = np.cos(Omega), np.sin(Omega)

    return np.array([
        [ cO*co - sO*so*ci,  -cO*so - sO*co*ci,   sO*si],
        [ sO*co + cO*so*ci,  -sO*so + cO*co*ci,  -cO*si],
        [ so*si,              co*si,               ci   ],
    ])


# --- Public API ---

def state_vector(planet: str, jd: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heliocentric ecliptic J2000 state vector.

    Parameters
    ----------
    planet : planet name (lowercase)
    jd     : Julian Date

    Returns
    -------
    r_km  : position [km]
    v_kms : velocity [km/s]
    """
    planet = planet.lower().strip()
    if planet not in _ELEMENTS:
        raise ValueError(f"Unknown planet '{planet}'. Available: {list(_ELEMENTS.keys())}")

    elem = _ELEMENTS[planet]
    T = (jd - JD_J2000) / 36525.0  # Julian centuries past J2000

    # Evaluate elements at epoch
    a       = elem[0] + elem[1] * T          # AU
    e       = elem[2] + elem[3] * T
    inc     = np.radians(elem[4] + elem[5] * T)
    L       = elem[6] + elem[7] * T          # mean longitude [deg]
    lonperi = elem[8] + elem[9] * T          # longitude of perihelion [deg]
    Omega   = np.radians(elem[10] + elem[11] * T)

    omega = np.radians(lonperi) - Omega      # argument of perihelion [rad]
    M = np.radians(L - lonperi)              # mean anomaly [rad]

    # Solve Kepler's equation
    E = _solve_kepler(M, e)

    # Position in orbital frame [AU]
    cosE = np.cos(E)
    sinE = np.sin(E)
    r_orb = a * np.array([cosE - e, np.sqrt(1.0 - e * e) * sinE, 0.0])

    # Velocity in orbital frame [AU/day]
    mu_au = MU_SUN / (AU_KM**3) * (DAY_S**2)  # AU³/day²
    n = np.sqrt(mu_au / a**3)                  # mean motion [rad/day]
    r_mag = a * (1.0 - e * cosE)
    v_factor = n * a * a / r_mag               # ṙ = (n a² / r) · [−sin E, √(1−e²) cos E]

    v_orb = v_factor * np.array([-sinE, np.sqrt(1.0 - e * e) * cosE, 0.0])

    # Rotate to ecliptic J2000 and convert units
    R = _rotation_matrix(omega, inc, Omega)
    r_km = (R @ r_orb) * AU_KM
    v_kms = (R @ v_orb) * AU_KM / DAY_S

    return r_km, v_kms


# --- Date utilities ---

def calendar_to_jd(year: int, month: int, day: float) -> float:
    """Gregorian calendar → Julian Date (Meeus, Ch. 7)."""
    if month <= 2:
        year -= 1
        month += 12
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    return (int(365.25 * (year + 4716))
            + int(30.6001 * (month + 1))
            + day + B - 1524.5)


def jd_to_calendar(jd: float) -> Tuple[int, int, float]:
    """Julian Date → Gregorian calendar (year, month, day)."""
    jd += 0.5
    Z = int(jd)
    F = jd - Z

    if Z < 2299161:
        A = Z
    else:
        alpha = int((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - int(alpha / 4)

    B = A + 1524
    C = int((B - 122.1) / 365.25)
    D = int(365.25 * C)
    E = int((B - D) / 30.6001)

    day = B - D - int(30.6001 * E) + F
    month = E - 1 if E < 14 else E - 13
    year = C - 4716 if month > 2 else C - 4715

    return int(year), int(month), day
