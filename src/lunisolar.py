"""The Moon's and Sun's gravity and Earth's flattening, for stretched (transfer) orbits.

A transfer orbit reaches tens of thousands of kilometers, where the Moon and the
Sun pull on it enough to slowly raise or lower the perigee, and drag depends
exponentially on perigee height (docs/derivation.md, section 10). This module
adds those effects to the drag simulation:

- Orbit-averaged ("secular") rates for the eccentricity vector e (length e,
  pointing to perigee) and the angular momentum vector j = sqrt(1 - e^2) w
  (w the orbit's normal), from the quadrupole tide of the Sun and the Moon at
  their actual positions, and from Earth's J2 flattening, in the vector form of
  Milankovitch's equations (Tremaine, Touma & Namouni 2009). Averaged over
  the satellite's orbit, these leave the semi-major axis unchanged.
- Drag (orbit_decay.drag_rates) shrinks a and the length of e as before.
- Positions of the Sun and Moon from the Astronomical Almanac's low-precision
  formulas (about 0.01 degree and 0.3 degree), in Earth's equatorial frame.

All vectors are in kilometers, days and Earth's equatorial frame.

    from lunisolar import propagate_lunisolar
"""

import numpy as np

from orbit_decay import drag_rates

MU_EARTH = 398600.4418       # km^3/s^2
MU_SUN = 1.32712440018e11    # km^3/s^2
MU_MOON = 4902.800066        # km^3/s^2
R_EARTH = 6378.137           # km
J2 = 1.08262668e-3
AU = 149597870.7             # km
SECONDS_PER_DAY = 86400.0
JD_J2000 = 2451545.0


# ---------------------------------------------------------------- ephemerides

def _ecliptic_to_equatorial(lon, lat, dist, obliquity):
    """Positions (km) from ecliptic longitude, latitude (radians) and distance, rotated by the obliquity."""
    x = dist * np.cos(lat) * np.cos(lon)
    y = dist * np.cos(lat) * np.sin(lon)
    z = dist * np.sin(lat)
    c, s = np.cos(obliquity), np.sin(obliquity)
    return np.stack([x, c * y - s * z, s * y + c * z], axis=-1)


def sun_position(jd):
    """The Sun's position (km) at Julian dates jd, from the Astronomical Almanac's low-precision formula."""
    n = np.asarray(jd, float) - JD_J2000
    L = np.radians(280.460 + 0.9856474 * n)
    g = np.radians(357.528 + 0.9856003 * n)
    lon = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    dist = (1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)) * AU
    return _ecliptic_to_equatorial(lon, np.zeros_like(n), dist, np.radians(23.439 - 4e-7 * n))


def moon_position(jd):
    """The Moon's position (km) at Julian dates jd, from the Astronomical Almanac's low-precision formula."""
    T = (np.asarray(jd, float) - JD_J2000) / 36525
    d = lambda a, b: np.radians(a + b * T)
    lon = (np.radians(218.32 + 481267.881 * T) + np.radians(6.29) * np.sin(d(135.0, 477198.87))
           - np.radians(1.27) * np.sin(d(259.3, -413335.36)) + np.radians(0.66) * np.sin(d(235.7, 890534.22))
           + np.radians(0.21) * np.sin(d(269.9, 954397.74)) - np.radians(0.19) * np.sin(d(357.5, 35999.05))
           - np.radians(0.11) * np.sin(d(186.5, 966404.03)))
    lat = (np.radians(5.13) * np.sin(d(93.3, 483202.02)) + np.radians(0.28) * np.sin(d(228.2, 960400.89))
           - np.radians(0.28) * np.sin(d(318.3, 6003.15)) - np.radians(0.17) * np.sin(d(217.6, -407332.21)))
    parallax = np.radians(0.9508 + 0.0518 * np.cos(d(135.0, 477198.87)) + 0.0095 * np.cos(d(259.3, -413335.36))
                          + 0.0078 * np.cos(d(235.7, 890534.22)) + 0.0028 * np.cos(d(269.9, 954397.74)))
    obliquity = np.radians(23.439 - 4e-7 * (np.asarray(jd, float) - JD_J2000))
    return _ecliptic_to_equatorial(lon, lat, R_EARTH / np.sin(parallax), obliquity)


# ---------------------------------------------------------------- secular rates

def orbit_vectors(e, inclination, raan, arg_perigee):
    """Eccentricity vector and dimensionless angular momentum vector from elements (angles in degrees)."""
    i, O, w = (np.radians(np.asarray(x, float)) for x in (inclination, raan, arg_perigee))
    P = np.stack([np.cos(O) * np.cos(w) - np.sin(O) * np.sin(w) * np.cos(i),
                  np.sin(O) * np.cos(w) + np.cos(O) * np.sin(w) * np.cos(i),
                  np.sin(w) * np.sin(i)], axis=-1)
    W = np.stack([np.sin(O) * np.sin(i), -np.cos(O) * np.sin(i), np.cos(i)], axis=-1)
    e = np.asarray(e, float)[..., None]
    return e * P, np.sqrt(1 - e ** 2) * W


def _dot(x, y):
    return np.sum(x * y, axis=-1, keepdims=True)


def milankovitch(a, e_vec, j_vec, grad_e, grad_j):
    """de/dt and dj/dt (per day) from the gradients of an orbit-averaged potential (km^2/s^2)."""
    k = SECONDS_PER_DAY / np.sqrt(MU_EARTH * np.asarray(a, float))[..., None]
    de = -k * (np.cross(j_vec, grad_e) + np.cross(e_vec, grad_j))
    dj = -k * (np.cross(j_vec, grad_j) + np.cross(e_vec, grad_e))
    return de, dj


def j2_gradients(a, j_vec):
    """Gradients of the orbit-averaged J2 potential mu J2 R^2 (1 - 3 cos^2 i) / (4 a^3 (1 - e^2)^1.5)."""
    C = (MU_EARTH * J2 * R_EARTH ** 2 / (4 * np.asarray(a, float) ** 3))[..., None]
    z = np.array([0.0, 0.0, 1.0])
    jn = np.linalg.norm(j_vec, axis=-1, keepdims=True)
    jz = j_vec[..., 2:3]
    grad_j = C * (-3 * jn ** -5 * j_vec - 6 * jz * jn ** -5 * z + 15 * jz ** 2 * jn ** -7 * j_vec)
    return np.zeros_like(j_vec), grad_j


def tide_gradients(a, e_vec, j_vec, mu, r_body):
    """Gradients of the orbit-averaged quadrupole tide of a distant body at r_body (km):
    -mu a^2 / (4 R^3) (1 - 6 e^2 + 15 (e.u)^2 - 3 (j.u)^2), u the unit vector to the body."""
    R = np.linalg.norm(r_body, axis=-1, keepdims=True)
    u = r_body / R
    C = mu * np.asarray(a, float)[..., None] ** 2 / (4 * R ** 3)
    return C * (12 * e_vec - 30 * _dot(e_vec, u) * u), C * (6 * _dot(j_vec, u) * u)


def secular_rates(a, e_vec, j_vec, jd, sun=True, moon=True, j2=True):
    """de/dt and dj/dt (per day) from J2 and the tides of the Sun and Moon at Julian dates jd."""
    grad_e, grad_j = np.zeros_like(e_vec), np.zeros_like(j_vec)
    if j2:
        ge, gj = j2_gradients(a, j_vec)
        grad_e, grad_j = grad_e + ge, grad_j + gj
    for on, mu, position in ((sun, MU_SUN, sun_position), (moon, MU_MOON, moon_position)):
        if on:
            ge, gj = tide_gradients(a, e_vec, j_vec, mu, position(jd))
            grad_e, grad_j = grad_e + ge, grad_j + gj
    return milankovitch(a, e_vec, j_vec, grad_e, grad_j)


# ---------------------------------------------------------------- propagation

def propagate_lunisolar(a_km, e_vec, j_vec, B, density, jd0, max_days=3650.0, stop_altitude=120.0,
                        max_step=1.0, step_km=3.0, sun=True, moon=True, j2=True):
    """Days until each orbit decays, with drag and the secular effects of J2, the Sun and the Moon.

    Like orbit_decay.propagate, but the state is (a, e vector, j vector) and steps
    last at most max_step days, short enough to follow the Moon. jd0 is each
    object's starting Julian date. density(h_km, idx, t_days) as in propagate.
    """
    a = np.array(a_km, dtype=float, ndmin=1)
    ev, jv = np.array(e_vec, float).reshape(-1, 3).copy(), np.array(j_vec, float).reshape(-1, 3).copy()
    B = np.broadcast_to(np.asarray(B, float), a.shape)
    jd0 = np.broadcast_to(np.asarray(jd0, float), a.shape)
    t = np.zeros_like(a)
    reentry = np.full_like(a, np.nan)

    def rates(idx, a_, e_, j_, t_):
        e_len = np.linalg.norm(e_, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            da, de_len = drag_rates(a_, np.clip(e_len, 0, 0.99), B[idx],
                                    lambda h, rows: density(h, idx[rows], t_[rows, None]))
            e_hat = np.where(e_len[:, None] > 1e-9, e_ / np.maximum(e_len, 1e-12)[:, None], 0.0)
            j_hat = j_ / np.linalg.norm(j_, axis=1)[:, None]
            dj_len = -e_len * de_len / np.sqrt(np.maximum(1 - e_len ** 2, 1e-12))
        de, dj = secular_rates(a_, e_, j_, jd0[idx] + t_, sun=sun, moon=moon, j2=j2)
        return da, de + de_len[:, None] * e_hat, dj + dj_len[:, None] * j_hat

    def fallen(a_, e_):
        e_len = np.linalg.norm(e_, axis=1)
        return ((a_ * (1 + e_len) - R_EARTH <= stop_altitude) | (a_ * (1 - e_len) - R_EARTH <= 90.0)
                | ~np.isfinite(a_) | ~np.isfinite(e_len))

    done = fallen(a, ev)
    reentry[done] = 0.0
    active = np.flatnonzero(~done)
    while len(active):
        a0, e0, j0, t0 = a[active], ev[active], jv[active], t[active]
        k1 = rates(active, a0, e0, j0, t0)
        e_len = np.linalg.norm(e0, axis=1)
        perigee_rate = (1 - e_len) * k1[0] - a0 * _dot(e0 / np.maximum(e_len, 1e-12)[:, None], k1[1])[:, 0]
        dt = np.minimum.reduce([np.full_like(a0, max_step),
                                step_km / np.maximum(-perigee_rate, 1e-12),
                                0.02 * a0 / np.maximum(-k1[0], 1e-12), max_days - t0])
        h = dt[:, None]
        k2 = rates(active, a0 + dt / 2 * k1[0], e0 + h / 2 * k1[1], j0 + h / 2 * k1[2], t0 + dt / 2)
        k3 = rates(active, a0 + dt / 2 * k2[0], e0 + h / 2 * k2[1], j0 + h / 2 * k2[2], t0 + dt / 2)
        k4 = rates(active, a0 + dt * k3[0], e0 + h * k3[1], j0 + h * k3[2], t0 + dt)
        a1 = a0 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        e1 = e0 + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        j1 = j0 + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        t1 = t0 + dt

        down = fallen(a1, e1)
        reentry[active[down]] = t1[down]
        a[active], ev[active], jv[active], t[active] = a1, e1, j1, t1
        active = active[~down & (t1 < max_days)]
    return reentry
