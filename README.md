# Live reentry forecasts

Written every 6 hours by `src/live_forecasts.py` (workflow `forecasts.yml` on the
main branch) and read by the website. This branch is replaced on every run.

- `forecasts.json`: the latest physics forecast for every tracked object expected
  to reenter within a year, with its 90% range
- `scores.json`: how the forecasts did, once the objects reentered
- `tracks.json`: where objects with a 90% window under 2 days can come down: the
  ground track during the window (latitude and longitude, 0.1°)
- `log.csv.gz`: every forecast made, one row per object and orbit: the orbit's
  epoch, the days predicted from it, the 90% range in days and the fit used
- `satcat.csv`, `SW-All.csv`: the CelesTrak catalog and space weather used

The orbits themselves aren't published: they come from CelesTrak, which gets
them from Space-Track.
