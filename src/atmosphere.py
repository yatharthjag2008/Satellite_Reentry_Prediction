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



def _bracket(x, grid):
    """Index of the grid point at or below x, and the fraction of the way to the next one (x clipped to the grid)."""
    x = np.clip(x, grid[0], grid[-1])
    i = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, len(grid) - 2)
    return i, (x - grid[i]) / (grid[i + 1] - grid[i])


class MsisAtmosphere:
    """Global-average NRLMSIS 2.1 density, interpolated from the pre-computed table.

    Interpolates log10(density) linearly in altitude, solar flux, Ap and time of
    year (wrapping around from December to January). Inputs outside the grid are
    clipped to its edge. Without a day of year, uses the average over the year.
    """

    def __init__(self, path=TABLE_FILE):
        if not Path(path).exists():
            raise SystemExit(f"No density table at {path}. Run src/atmosphere.py first.")
        table = np.load(path)
        self.log_rho = table["log_rho"].astype(float)              # (alt, f107, ap, month)
        self.log_rho_year = np.log10((10 ** self.log_rho).mean(-1))  # (alt, f107, ap)
        self.altitudes, self.f107s, self.aps = table["altitudes"], table["f107s"], table["aps"]

    def _profile_weights(self, f107a, ap, doy):
        """Per-object weights for every table corner except altitude: list of (weight, f index, ap index, month)."""
        i_f, w_f = _bracket(np.asarray(f107a, float), self.f107s)
        i_a, w_a = _bracket(np.asarray(ap, float), self.aps)
        corners = []
        for df, wf in ((0, 1 - w_f), (1, w_f)):
            for da, wa in ((0, 1 - w_a), (1, w_a)):
                if doy is None:
                    corners.append((wf * wa, i_f + df, i_a + da, None))
                    continue
                pos = (np.asarray(doy, float) - DAYS_OF_YEAR[0]) / (365.25 / 12) % 12
                m0 = np.floor(pos).astype(int)
                w_m = pos - m0
                corners.append((wf * wa * (1 - w_m), i_f + df, i_a + da, m0 % 12))
                corners.append((wf * wa * w_m, i_f + df, i_a + da, (m0 + 1) % 12))
        return corners

    def density(self, h_km, f107a, ap, doy=None):
        """Density in kg/m^3; all inputs broadcast against each other."""
        h_km, f107a, ap = np.broadcast_arrays(np.asarray(h_km, float), f107a, ap)
        if doy is not None:
            doy = np.broadcast_to(doy, h_km.shape)
        i_h, w_h = _bracket(h_km, self.altitudes)
        log_rho = 0.0
        for w, i_f, i_a, m in self._profile_weights(f107a, ap, doy):
            table = self.log_rho_year if m is None else self.log_rho
            idx = (i_f, i_a) if m is None else (i_f, i_a, m)
            lo, hi = table[(i_h, *idx)], table[(i_h + 1, *idx)]
            log_rho = log_rho + w * (lo + w_h * (hi - lo))
        return 10 ** log_rho

    def profile(self, f107a, ap, doy=None):
        """Density as a function of altitude only, for space weather held fixed.

        f107a and ap are 1-D arrays (one value per object). Returns density(h_km) for
        h_km of shape (objects, ...), which is far faster than calling density() at
        every step of a simulation.
        """
        f107a, ap = np.atleast_1d(np.asarray(f107a, float)), np.atleast_1d(np.asarray(ap, float))
        doy = None if doy is None else np.broadcast_to(doy, f107a.shape)
        profiles = 0.0
        for w, i_f, i_a, m in self._profile_weights(f107a, ap, doy):
            rows = self.log_rho_year[:, i_f, i_a] if m is None else self.log_rho[:, i_f, i_a, m]
            profiles = profiles + w * rows                          # (alt, objects)
        profiles = profiles.T                                       # (objects, alt)
        alts = self.altitudes

        def density(h_km):
            h_km = np.asarray(h_km, float)
            i_h, w_h = _bracket(h_km, alts)
            obj = np.arange(len(profiles)).reshape((-1,) + (1,) * (h_km.ndim - 1))
            lo, hi = profiles[obj, i_h], profiles[obj, i_h + 1]
            return 10 ** (lo + w_h * (hi - lo))

        return density

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
