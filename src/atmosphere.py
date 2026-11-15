"""Air density models for the drag simulation.

- ExponentialAtmosphere: density falling off exponentially with height. Simple
  enough to have exact answers, so the tests use it to check the simulation.
"""

import numpy as np


class ExponentialAtmosphere:
    """rho(h) = rho0 * exp(-(h - h0) / H), ignoring solar activity."""

    def __init__(self, rho0=3.7e-12, h0=400.0, scale_height=60.0):
        self.rho0 = rho0              # kg/m^3 at h0
        self.h0 = h0                  # km
        self.scale_height = scale_height  # km

    def density(self, h_km, f107a=None, ap=None, doy=None):
        return self.rho0 * np.exp(-(np.asarray(h_km) - self.h0) / self.scale_height)
