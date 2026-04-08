"""
Multiple Gravity Assist (MGA) Trajectory Fitness Functions

Implements the Cassini1-style MGA benchmark from the ESA GTOP suite.

The decision vector encodes a departure date and a sequence of transfer
times. The fitness function solves Lambert's problem for each leg,
computes v_infinity at each planetary encounter, and accumulates the
Δv cost including gravity-assist feasibility constraints.

Reference problem:
    ESA GTOP Cassini1 — Earth-Venus-Venus-Earth-Jupiter-Saturn
    6 variables, best known ≈ 4.93 km/s (MGA-1DSM formulation)
    https://www.esa.int/gsp/ACT/projects/gtop/cassini1/

This is an unpowered-MGA formulation: no deep-space maneuvers, each
flyby must be geometrically feasible or incur a penalty. Results will
differ from the reference MGA-1DSM solution because that formulation
allows one deep-space impulse per leg.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from typing import List

from lambert import solve as lambert_solve
from ephemeris import state_vector, MU_SUN, DAY_S, jd_to_calendar


# --- Planet gravitational parameters [km³/s²] ---

MU_PLANET = {
    "mercury": 2.2032e4,
    "venus":   3.24859e5,
    "earth":   3.986004418e5,
    "mars":    4.282837e4,
    "jupiter": 1.26686534e8,
    "saturn":  3.7931187e7,
}

# --- Minimum flyby radius [km] = planet radius + safety altitude ---

R_MIN_FLYBY = {
    "mercury": 2440.0   + 200.0,
    "venus":   6051.8   + 300.0,
    "earth":   6378.137 + 300.0,
    "mars":    3389.5   + 200.0,
    "jupiter": 71492.0  + 6.0 * 71492.0,   # 6 R_J for radiation belts
    "saturn":  60268.0  + 1000.0,
}


# --- Date conversion ---

JD_MJD2000_OFFSET = 2451544.5


def mjd2000_to_jd(mjd: float) -> float:
    """Modified Julian Date 2000 → Julian Date."""
    return mjd + JD_MJD2000_OFFSET


# --- Flyby cost ---

def _flyby_cost(v_inf_in: np.ndarray, v_inf_out: np.ndarray,
                mu_planet: float, r_p_min: float) -> float:
    """
    Minimum Δv impulse for a gravity-assist encounter.

    The flyby rotates v_inf_in by at most δ_max around any axis, where
        sin(δ_max/2) = 1 / (1 + r_p_min · v_inf² / μ_planet)

    Let α be the angle between v_inf_in and v_inf_out:
        α ≤ δ_max :  cost = ||v_out| − |v_in||
        α  > δ_max :  cost = √(v_in² + v_out² − 2·v_in·v_out·cos(α−δ_max))

    The second case models the minimum impulse at periapsis that brings
    the incoming state onto the outgoing asymptote after maximum rotation.
    """
    v1 = norm(v_inf_in)
    v2 = norm(v_inf_out)

    if v1 < 1e-9 or v2 < 1e-9:
        return abs(v2 - v1) + 100.0

    cos_alpha = np.clip(np.dot(v_inf_in, v_inf_out) / (v1 * v2), -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    sin_half_max = 1.0 / (1.0 + r_p_min * v1**2 / mu_planet)
    sin_half_max = np.clip(sin_half_max, 0.0, 1.0)
    delta_max = 2.0 * np.arcsin(sin_half_max)

    if alpha <= delta_max:
        return abs(v2 - v1)

    residual = alpha - delta_max
    cost_sq = v1 * v1 + v2 * v2 - 2.0 * v1 * v2 * np.cos(residual)
    return float(np.sqrt(max(cost_sq, 0.0)))


# --- Cassini1 problem definition ---

CASSINI1_SEQUENCE = ["earth", "venus", "venus", "earth", "jupiter", "saturn"]

CASSINI1_BOUNDS = np.array([
    [-1000.0,    0.0],   # t0: departure MJD2000
    [   30.0,  400.0],   # T1: Earth → Venus [days]
    [  100.0,  470.0],   # T2: Venus → Venus
    [   30.0,  400.0],   # T3: Venus → Earth
    [  400.0, 2000.0],   # T4: Earth → Jupiter
    [ 1000.0, 6000.0],   # T5: Jupiter → Saturn
])

CASSINI1_REFERENCE_DV = 4.93   # km/s, best known MGA-1DSM


def cassini1_fitness(x: np.ndarray) -> float:
    """Cassini1 MGA total Δv [km/s]. x = [t0, T1, T2, T3, T4, T5]."""
    return mga_fitness(x, CASSINI1_SEQUENCE)


# --- General MGA fitness ---

def mga_fitness(x: np.ndarray, sequence: List[str]) -> float:
    """
    General MGA fitness.

    Parameters
    ----------
    x        : [t0_MJD2000, T1, T2, ..., T_{N-1}] in days
    sequence : list of N planet names (origin, flybys..., destination)

    Returns
    -------
    Total Δv [km/s] = launch + Σ flyby costs + rendezvous.
    Returns 1e6 on any failure.
    """
    N = len(sequence)
    if len(x) != N:
        return 1e6

    t0 = x[0]
    transfer_times = x[1:]

    jds = [mjd2000_to_jd(t0)]
    for T in transfer_times:
        if T <= 0:
            return 1e6
        jds.append(jds[-1] + T)

    try:
        states = [state_vector(p, jd) for p, jd in zip(sequence, jds)]
    except Exception:
        return 1e6

    # Solve Lambert for each leg
    leg_velocities = []
    for i in range(N - 1):
        r1, _ = states[i]
        r2, _ = states[i + 1]
        tof_sec = transfer_times[i] * DAY_S

        try:
            sols = lambert_solve(r1, r2, tof_sec, MU_SUN, prograde=True)
        except Exception:
            return 1e6

        if not sols:
            return 1e6

        v_dep, v_arr = sols[0]
        leg_velocities.append((v_dep, v_arr))

    # Launch Δv (hyperbolic excess at origin)
    _, v_planet_launch = states[0]
    v_dep_launch, _ = leg_velocities[0]
    cost = norm(v_dep_launch - v_planet_launch)

    # Intermediate flybys
    for k in range(1, N - 1):
        planet = sequence[k]
        _, v_planet = states[k]

        _, v_arr_in = leg_velocities[k - 1]
        v_dep_out, _ = leg_velocities[k]

        v_inf_in = v_arr_in - v_planet
        v_inf_out = v_dep_out - v_planet

        cost += _flyby_cost(
            v_inf_in, v_inf_out,
            MU_PLANET[planet], R_MIN_FLYBY[planet]
        )

    # Arrival Δv (rendezvous with destination)
    _, v_planet_arr = states[-1]
    _, v_arr_final = leg_velocities[-1]
    cost += norm(v_arr_final - v_planet_arr)

    return float(cost)


# --- Simpler benchmark: direct Earth → Mars ---

def earth_mars_direct_fitness(x: np.ndarray) -> float:
    """Direct Earth → Mars. x = [t0_MJD2000, T_days]."""
    return mga_fitness(x, ["earth", "mars"])


EARTH_MARS_BOUNDS = np.array([
    [ 9000.0, 10000.0],   # MJD2000 ~ 2024-2027
    [  150.0,   450.0],
])


# --- Earth → Venus → Earth → Jupiter (simplified 3-flyby) ---

EVEJ_SEQUENCE = ["earth", "venus", "earth", "jupiter"]

EVEJ_BOUNDS = np.array([
    [ 9000.0, 10500.0],
    [   80.0,   400.0],
    [  100.0,   600.0],
    [  400.0,  2000.0],
])


def evej_fitness(x: np.ndarray) -> float:
    """Earth → Venus → Earth → Jupiter. 4 variables."""
    return mga_fitness(x, EVEJ_SEQUENCE)


# --- Human-readable mission decoder ---

def decode_mission(x: np.ndarray, sequence: List[str]) -> dict:
    """Return a human-readable breakdown of an MGA solution."""
    t0 = x[0]
    transfer_times = x[1:]
    jds = [mjd2000_to_jd(t0)]
    for T in transfer_times:
        jds.append(jds[-1] + T)

    dates = [jd_to_calendar(jd) for jd in jds]

    return {
        "departure": f"{dates[0][0]:04d}-{dates[0][1]:02d}-{int(dates[0][2]):02d}",
        "arrival":   f"{dates[-1][0]:04d}-{dates[-1][1]:02d}-{int(dates[-1][2]):02d}",
        "total_days": float(np.sum(transfer_times)),
        "total_years": float(np.sum(transfer_times)) / 365.25,
        "legs": [
            {
                "from": sequence[i],
                "to":   sequence[i+1],
                "tof_days": float(transfer_times[i]),
                "arrival_date": f"{dates[i+1][0]:04d}-{dates[i+1][1]:02d}-{int(dates[i+1][2]):02d}",
            }
            for i in range(len(sequence) - 1)
        ],
        "total_dv": mga_fitness(x, sequence),
    }
