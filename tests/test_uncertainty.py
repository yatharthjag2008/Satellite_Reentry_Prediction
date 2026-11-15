import numpy as np

from uncertainty import band_of, conformal_quantile, mondrian_widths


def test_conformal_quantile_uses_the_finite_sample_rank():
    # With 9 scores and 90% coverage, the rank is ceil(10 * 0.9) = 9: the largest score.
    assert conformal_quantile(np.arange(1, 10), 0.9) == 9
    # With 99 scores the rank is ceil(100 * 0.9) = 90.
    assert conformal_quantile(np.arange(1, 100), 0.9) == 90


def test_conformal_ranges_cover_new_data_from_the_same_distribution():
    rng = np.random.default_rng(0)
    truth = 10 ** rng.uniform(0, 2.5, 40_000)
    pred = truth * 10 ** rng.normal(0, 0.1 + 0.05 * np.log10(truth), truth.size)  # errors grow with horizon
    widths = mondrian_widths(pred[:20_000], truth[:20_000])
    w = widths[band_of(pred[20_000:])]
    inside = np.abs(np.log10(truth[20_000:] / pred[20_000:])) <= w
    assert 0.88 < inside.mean() < 0.92
    assert np.all(np.diff(widths) > 0)  # later bands are wider
