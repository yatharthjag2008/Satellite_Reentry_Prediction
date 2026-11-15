"""Air density models for the drag simulation.

- ExponentialAtmosphere: density falling off exponentially with height. Simple
  enough to have exact answers, so the tests use it to check the simulation.
- MsisAtmosphere: the NRLMSIS 2.1 empirical model (via pymsis), averaged over the
  globe and pre-computed on a grid of altitude, solar flux, geomagnetic activity
  and time of year, because calling MSIS directly is far too slow for thousands
  of orbits.

    python src/atmosphere.py    # builds data/msis_density_table.npz (about 4 minutes)
"""

from pathlib import Path

import numpy as np

TABLE_FILE = Path(__file__).resolve().parent.parent / "data" / "msis_density_table.npz"

# Grid for the MSIS table. Ap is spaced unevenly because density changes fastest at low Ap.
ALTITUDES = np.arange(80.0, 1501.0, 5.0)           # km
F107S = np.arange(60.0, 301.0, 20.0)                # solar flux units, used as both daily and 81-day value
APS = np.array([0, 3, 5, 7, 10, 15, 22, 32, 48, 80, 130, 200, 300, 400], dtype=float)
DAYS_OF_YEAR = np.arange(12) * 365.25 / 12 + 15.0  # middle of each month
LATITUDES = np.arange(-80.0, 81.0, 20.0)
LONGITUDES = np.arange(-180.0, 180.0, 30.0)        # at a fixed time, longitude covers all local times


class ExponentialAtmosphere:
    """rho(h) = rho0 * exp(-(h - h0) / H), ignoring solar activity."""

    def __init__(self, rho0=3.7e-12, h0=400.0, scale_height=60.0):
        self.rho0 = rho0              # kg/m^3 at h0
        self.h0 = h0                  # km
        self.scale_height = scale_height  # km

    def density(self, h_km, f107a=None, ap=None, doy=None):
        return self.rho0 * np.exp(-(np.asarray(h_km) - self.h0) / self.scale_height)


def build_msis_table(path=TABLE_FILE):
    """Global-average log10 density from NRLMSIS 2.1 on the grid above."""
    import pymsis

    dates = np.datetime64("2023-01-01T12:00") + (DAYS_OF_YEAR - 1).astype("timedelta64[D]")
    weights = np.cos(np.radians(LATITUDES))  # area weighting
    weights /= weights.sum()
    log_rho = np.empty((len(ALTITUDES), len(F107S), len(APS), len(DAYS_OF_YEAR)), dtype=np.float32)
    for i, f107 in enumerate(F107S):
        for j, ap in enumerate(APS):
            n = len(dates)
            out = pymsis.calculate(dates, LONGITUDES, LATITUDES, ALTITUDES,
                                   np.full(n, f107), np.full(n, f107), np.full((n, 7), ap))
            rho = out[..., pymsis.Variable.MASS_DENSITY]       # (dates, lons, lats, alts)
            mean = np.einsum("dolh,l->hd", rho, weights) / len(LONGITUDES)
            log_rho[:, i, j, :] = np.log10(mean)
        print(f"F10.7 = {f107:.0f} done")
    np.savez(path, log_rho=log_rho, altitudes=ALTITUDES, f107s=F107S, aps=APS, days_of_year=DAYS_OF_YEAR)
    print(f"saved {path}")


if __name__ == "__main__":
    build_msis_table()
