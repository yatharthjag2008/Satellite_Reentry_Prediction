"""Simulate how atmospheric drag shrinks an orbit.

The equations are derived in docs/derivation.md. Distances are in km and times
in days at the interface; internally the rates use SI units.
"""

import numpy as np
from scipy.special import dawsn

MU = 3.986004418e14       # m^3/s^2
R_EARTH = 6378.137        # km
SECONDS_PER_DAY = 86400.0


def circular_decay_rate(a_km, B, density):
    """da/dt in km/day for a circular orbit: da/dt = -B rho sqrt(mu a)."""
    rho = density(a_km - R_EARTH)
    return -B * rho * np.sqrt(MU * a_km * 1e3) * SECONDS_PER_DAY / 1e3


def exponential_lifetime(h_start, h_end, B, rho0, h0, scale_height):
    """Exact days to sink from h_start to h_end (km) in an exponential atmosphere.

    T = 2 sqrt(H) / (B rho0 sqrt(mu)) * [e^((ai-a0)/H) D(sqrt(ai/H)) - e^((af-a0)/H) D(sqrt(af/H))]
    with D the Dawson function (docs/derivation.md, section 3).
    """
    H = scale_height * 1e3
    a0, ai, af = [(R_EARTH + h) * 1e3 for h in (h0, h_start, h_end)]
    term = lambda a: np.exp((a - a0) / H) * dawsn(np.sqrt(a / H))
    return 2 * np.sqrt(H) / (B * rho0 * np.sqrt(MU)) * (term(ai) - term(af)) / SECONDS_PER_DAY


def orbit_averaged_rates(a_km, e, B, density, n_points=32):
    """Orbit-averaged (da/dt in km/day, de/dt per day) for drag on an elliptical orbit.

    Inputs can be arrays (one orbit per element). `density(h_km)` gives kg/m^3.
    Averages over the mean anomaly by sampling the eccentric anomaly E evenly and
    weighting by dM/dE = 1 - e cos E (docs/derivation.md, sections 4 and 5).
    """
    a_km, e, B = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in (a_km, e, B)])
    E = (np.arange(n_points) + 0.5) * 2 * np.pi / n_points
    a, ecc = a_km[..., None] * 1e3, e[..., None]
    one_minus = 1 - ecc * np.cos(E)
    r = a * one_minus
    v = np.sqrt(MU * (2 / r - 1 / a))
    cos_nu = (np.cos(E) - ecc) / one_minus
    rho = density(r / 1e3 - R_EARTH)
    weight = one_minus / n_points  # dM / (2 pi), with sum(dM) over the orbit = 2 pi
    da = -(B[..., None] * rho * v**3 * a**2 / MU * weight).sum(-1)
    de = -(B[..., None] * rho * v * (ecc + cos_nu) * weight).sum(-1)
    return da * SECONDS_PER_DAY / 1e3, de * SECONDS_PER_DAY


def points_needed(a_km, e, scale_height=40.0):
    """Sample points per orbit for the orbit average to converge.

    Density near perigee falls off like exp(-a e (1 - cos E) / H), a bump of width
    about sqrt(2H / (a e)) in E. Sampling a few points across that bump is enough,
    so near-circular orbits need 8 points and very stretched ones a few hundred.
    """
    z = np.asarray(a_km) * np.asarray(e) / scale_height
    z = np.where(np.isfinite(z) & (z > 0), z, 0.0)
    n = 6 * np.pi * np.sqrt(z / 2) + 1
    return np.clip(2 ** np.ceil(np.log2(n)), 8, 1024).astype(int)


def drag_rates(a_km, e, B, density):
    """Orbit-averaged rates, using as many sample points as each orbit needs.

    density(h_km, rows) gives density for the objects in positions `rows` of the inputs.
    """
    a_km, e, B = np.broadcast_arrays(*[np.atleast_1d(np.asarray(x, float)) for x in (a_km, e, B)])
    e = np.clip(e, 0.0, 0.99)  # an RK4 half-step can overshoot a nearly circular orbit below zero
    da, de = np.empty_like(a_km), np.empty_like(a_km)
    n = points_needed(a_km, e)
    for count in np.unique(n):
        rows = np.flatnonzero(n == count)
        da[rows], de[rows] = orbit_averaged_rates(a_km[rows], e[rows], B[rows],
                                                  lambda h: density(h, rows), n_points=count)
    return da, de


def propagate(a_km, e, B, density, max_days=3650.0, stop_altitude=120.0, max_step=30.0, step_km=3.0):
    """Days until each orbit decays, integrating the orbit-averaged rates with RK4.

    a_km, e, B: one value per object.
    density(h_km, idx, t_days): air density for objects `idx` (indices into the
        inputs, h_km of shape (len(idx), points)) at t_days after the start.
    Reentry is when the apogee sinks to stop_altitude (the whole orbit is then in
    dense air and the rest takes hours), or the perigee drops below 90 km.
    Each step lowers the perigee by at most about step_km, shrinks the orbit by at
    most 2%, and lasts at most max_step days. Objects still up after max_days get NaN.
    """
    a = np.array(a_km, dtype=float, ndmin=1)
    ecc = np.broadcast_to(np.asarray(e, float), a.shape).copy()
    B = np.broadcast_to(np.asarray(B, float), a.shape)
    t = np.zeros_like(a)
    reentry = np.full_like(a, np.nan)
    active = np.arange(len(a))

    def rates(idx, a_, e_, t_):
        return drag_rates(a_, e_, B[idx], lambda h, rows: density(h, idx[rows], t_[rows, None]))

    def fallen(a_, e_):
        # A step can overshoot into an impossible orbit (NaN) when an object is already falling fast.
        return ((a_ * (1 + e_) - R_EARTH <= stop_altitude) | (a_ * (1 - e_) - R_EARTH <= 90.0)
                | ~np.isfinite(a_) | ~np.isfinite(e_))

    done = fallen(a, ecc)
    reentry[done] = 0.0
    active = active[~done]
    while len(active):
        a0, e0, t0 = a[active], ecc[active], t[active]
        k1 = rates(active, a0, e0, t0)
        perigee_rate = (1 - e0) * k1[0] - a0 * k1[1]
        dt = np.minimum.reduce([np.full_like(a0, max_step),
                                step_km / np.maximum(-perigee_rate, 1e-12),
                                0.02 * a0 / np.maximum(-k1[0], 1e-12)])
        dt = np.minimum(dt, max_days - t0)
        k2 = rates(active, a0 + dt / 2 * k1[0], e0 + dt / 2 * k1[1], t0 + dt / 2)
        k3 = rates(active, a0 + dt / 2 * k2[0], e0 + dt / 2 * k2[1], t0 + dt / 2)
        k4 = rates(active, a0 + dt * k3[0], e0 + dt * k3[1], t0 + dt)
        a1 = a0 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        e1 = np.clip(e0 + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]), 0.0, 0.99)
        t1 = t0 + dt

        down = fallen(a1, e1)
        # Time of reentry within the last step, interpolating the apogee linearly.
        apo0, apo1 = a0 * (1 + e0) - R_EARTH, a1 * (1 + e1) - R_EARTH
        frac = np.clip((apo0 - stop_altitude) / np.maximum(apo0 - apo1, 1e-12), 0.0, 1.0)
        frac = np.where(np.isfinite(frac), frac, 1.0)
        reentry[active[down]] = (t0 + frac * dt)[down]
        a[active], ecc[active], t[active] = a1, e1, t1
        active = active[~down & (t1 < max_days)]
    return reentry
