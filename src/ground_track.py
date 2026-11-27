"""Where an object can come down: its ground track during the 90% reentry window.

An object falls somewhere along its orbit, so the possible places are the ground
track (the path beneath it) between the earliest and latest likely reentry
times. Within a day or two of reentry that is a few strips across the map; with
a wider window the strips fill the whole band between latitudes ±inclination,
which is all that can be said.

The track comes from SGP4 (the sgp4 package) started from the latest orbit, with
the drag term set to zero: SGP4's own drag model is unreliable this low, and the
path the orbit traces over the ground hardly depends on it. What is uncertain
is *when* the object reaches each point, which the window already covers.
Latitudes are geocentric, which differ from geodetic ones by under 0.2°.

    from ground_track import ground_track
"""

from datetime import datetime

import numpy as np
import pandas as pd
from sgp4.api import WGS72, Satrec

EPOCH0 = datetime(1949, 12, 31)  # SGP4 counts epochs in days from here
MAX_WINDOW_DAYS = 2.0            # wider windows: just the latitude band
STEP_MINUTES = 2.0


def satellite(row):
    """An SGP4 satellite from one orbit (CelesTrak's OMM fields), without drag."""
    epoch = pd.Timestamp(row["EPOCH"]).to_pydatetime()
    sat = Satrec()
    sat.sgp4init(WGS72, "i", int(row["NORAD_CAT_ID"]) % 100000, (epoch - EPOCH0).total_seconds() / 86400,
                 0.0, 0.0, 0.0, float(row["ECCENTRICITY"]), np.radians(row["ARG_OF_PERICENTER"]),
                 np.radians(row["INCLINATION"]), np.radians(row["MEAN_ANOMALY"]),
                 float(row["MEAN_MOTION"]) * 2 * np.pi / 1440, np.radians(row["RA_OF_ASC_NODE"]))
    return sat


def gmst(jd_ut1):
    """Greenwich mean sidereal time (radians) for a UT1 Julian date (IAU 1982, as SGP4's TEME frame uses)."""
    t = (jd_ut1 - 2451545.0) / 36525.0
    seconds = 67310.54841 + (876600 * 3600 + 8640184.812866) * t + 0.093104 * t ** 2 - 6.2e-6 * t ** 3
    return np.radians((seconds % 86400) / 240.0)


def julian_date(times):
    return times.to_julian_date().to_numpy()


def subpoints(sat, times):
    """Latitude and longitude (degrees) beneath the satellite at each time (a DatetimeIndex, UTC)."""
    jd = julian_date(times)
    whole = np.floor(jd)
    err, r, _ = sat.sgp4_array(whole, jd - whole)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    lat = np.degrees(np.arctan2(z, np.hypot(x, y)))
    lon = np.degrees(np.arctan2(y, x) - gmst(jd))
    lon = (lon + 180) % 360 - 180
    ok = err == 0
    return lat[ok], lon[ok]


def ground_track(row, start, end, step_minutes=STEP_MINUTES):
    """Latitudes and longitudes along the track from start to end (Timestamps, UTC), or None if too wide."""
    if (end - start) / pd.Timedelta(days=1) > MAX_WINDOW_DAYS:
        return None
    times = pd.date_range(start, end, freq=pd.Timedelta(minutes=step_minutes))
    return subpoints(satellite(row), times)


def latitude_band(inclination):
    """The band an orbit covers: ±inclination, or ±(180° − inclination) for a retrograde orbit."""
    i = float(inclination)
    return min(i, 180.0 - i)
