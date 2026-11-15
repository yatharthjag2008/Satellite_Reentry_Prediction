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
