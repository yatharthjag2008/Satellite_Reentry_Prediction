import numpy as np
import pandas as pd

from lifetime_model import MAX_MONTHS, error_by_band, months_from, object_groups, typical_ap, uses_lunisolar


def test_months_from_pads_a_short_series_with_its_last_value():
    series = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2040-01-01", periods=3, freq="MS"))
    path = months_from(series, pd.Timestamp("2040-02-01"))
    assert len(path) == MAX_MONTHS and path[0] == 2.0 and path[-1] == 3.0


def test_typical_ap_uses_only_the_past():
    index = pd.date_range("2000-01-01", "2030-01-01", freq="MS")
    ap = pd.Series(np.where(index < "2020-01-01", 10.0, 1000.0), index=index)
    assert typical_ap(ap, pd.Timestamp("2020-01-01")) == 10.0


def test_object_groups_split_by_type_and_orbit_shape():
    table = pd.DataFrame({"OBJECT_TYPE": ["R/B", "R/B", "PAY", "PAY"], "is_starlink": [0, 0, 0, 1],
                          "APOAPSIS": [600.0, 35000.0, 550.0, 560.0]})
    groups = object_groups(table)
    assert groups["rocket bodies, near-circular"].tolist() == [True, False, False, False]
    assert groups["rocket bodies, transfer orbits"].tolist() == [False, True, False, False]
    assert groups["satellites (not Starlink), near-circular"].tolist() == [False, False, True, False]
    assert groups["Starlink"].tolist() == [False, False, False, True]



def test_only_transfer_orbits_with_known_angles_get_the_sun_and_moon():
    table = pd.DataFrame({"APOAPSIS": [600.0, 35000.0, 35000.0], "RA_OF_ASC_NODE": [10.0, 20.0, np.nan],
                          "ARG_OF_PERICENTER": [30.0, 40.0, 50.0]})
    assert uses_lunisolar(table).tolist() == [False, True, False]
    assert uses_lunisolar(table.drop(columns="RA_OF_ASC_NODE")).tolist() == [False, False, False]


def test_error_by_band_counts_missing_predictions_as_thirty_years():
    table = pd.DataFrame({"days_left": [100.0, 1000.0]})
    result = error_by_band(table, {"model": np.array([100.0, np.nan])})
    assert result.loc["model", "under 1 year"] == 0.0
    assert result.loc["model", "1-3 years"] > 9
