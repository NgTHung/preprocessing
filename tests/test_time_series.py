"""Check lag spacing, covariance normalization, gap masks, fills, and handoff."""

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import CubicSpline
from scipy.linalg import toeplitz
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import acf, pacf_yw

from preprocessing.time_series import (
    ar_information_criteria,
    compare_filled_dependence,
    fill_gaps,
    gap_aware_acf,
    gap_aware_ljung_box,
    gap_aware_pacf,
    pacf_lag_order,
    seeded_gap_mask,
)


def hourly(values):
    return pd.Series(
        values, index=pd.date_range("2013-03-01", periods=len(values), freq="h"),
        dtype=float,
    )


def test_gap_acf_matches_explicit_timestamp_pairs_without_compressing_time():
    series = hourly([2, 8, np.nan, 3, 9, 1, np.nan, 5, 4, 7])
    result = gap_aware_acf(series, 3)
    mean = series.mean()
    variance = ((series.dropna() - mean) ** 2).mean()
    for lag in range(4):
        pairs = pd.concat([series, series.shift(lag)], axis=1).dropna()
        covariance = ((pairs.iloc[:, 0] - mean) * (pairs.iloc[:, 1] - mean)).mean()
        assert result.loc[lag, "pair_counts"] == len(pairs)
        assert result.loc[lag, "acf"] == pytest.approx(covariance / variance)
        assert result.loc[lag, "band_half_width"] == pytest.approx(
            1.96 / np.sqrt(len(pairs))
        )
    assert result.loc[1, "pair_counts"] == 5
    # Removing the NaN timestamps would turn two-hour separations into lag 1.
    assert result.loc[1, "acf"] != pytest.approx(
        gap_aware_acf(hourly(series.dropna().values), 3).loc[1, "acf"]
    )


def test_gap_pacf_matches_independent_yule_walker_linear_solves():
    values = np.random.default_rng(42).normal(size=300)
    values[[4, 19, 20, 73, 102]] = np.nan
    result = gap_aware_pacf(hourly(values), 8)
    covariance = result["autocovariance"].to_numpy()
    for lag in range(1, 9):
        expected = np.linalg.solve(
            toeplitz(covariance[:lag]), covariance[1:lag + 1]
        )[-1]
        assert result.loc[lag, "pacf"] == pytest.approx(expected)


def test_complete_input_matches_adjusted_statsmodels_estimators():
    values = np.random.default_rng(19).normal(size=400)
    result = gap_aware_pacf(hourly(values), 12)
    np.testing.assert_allclose(
        result["acf"], acf(values, nlags=12, adjusted=True, fft=False)
    )
    np.testing.assert_allclose(
        result["pacf"], pacf_yw(values, nlags=12, method="adjusted")
    )
    np.testing.assert_array_equal(result["pair_counts"], 400 - np.arange(13))


def test_mcar_gaps_recover_known_ar1_dependence():
    rng = np.random.default_rng(42)
    noise = rng.normal(size=30_000)
    values = np.zeros_like(noise)
    for i in range(1, len(values)):
        values[i] = 0.7 * values[i - 1] + noise[i]
    values[rng.random(len(values)) < 0.1] = np.nan
    result = gap_aware_pacf(hourly(values), 6)
    np.testing.assert_allclose(result["acf"].iloc[1:], 0.7 ** np.arange(1, 7), atol=0.03)
    assert result.loc[1, "pacf"] == pytest.approx(0.7, abs=0.03)
    assert result["pacf"].iloc[2:].abs().max() < 0.03


@pytest.mark.parametrize("max_lag", [0, -1, 10, 2.5, True])
def test_acf_rejects_invalid_lag(max_lag):
    with pytest.raises(ValueError, match="max_lag"):
        gap_aware_acf(hourly(range(10)), max_lag)


@pytest.mark.parametrize(
    "series, message",
    [
        (hourly([np.nan] * 10), "pairs"),
        (hourly([2] * 10), "variance"),
        (hourly([1, 2, np.inf, 4, 5]), "infinite"),
        (hourly([1, np.nan, 3, np.nan, 5, np.nan]), "pairs"),
        (hourly(range(10)).drop(pd.Timestamp("2013-03-01 03:00")), "regular"),
        (hourly(range(10)).iloc[::-1], "ordered"),
        (hourly(range(10)).iloc[[0, 1, 1, 3]], "unique"),
    ],
)
def test_acf_rejects_invalid_series(series, message):
    with pytest.raises(ValueError, match=message):
        gap_aware_acf(series, 2)


def test_acf_requires_time_index():
    with pytest.raises(TypeError, match="DatetimeIndex"):
        gap_aware_acf(pd.Series(range(20)), 2)


def test_pacf_rejects_non_positive_definite_pairwise_covariance():
    with pytest.raises(ValueError, match="positive definite"):
        gap_aware_pacf(hourly([1, -1] * 30), 4)


def test_pacf_rejects_too_many_lags_for_observed_count():
    with pytest.raises(ValueError, match="half"):
        gap_aware_pacf(hourly(range(10)), 5)


def test_order_uses_each_lag_band_and_excludes_lag_zero():
    result = pd.DataFrame(
        {"pacf": [1, 0.3, 0.015, -0.04], "band_half_width": [0.01, 0.01, 0.02, 0.03]},
        index=[0, 1, 2, 3],
    )
    assert pacf_lag_order(result) == 3
    result.loc[1:, "pacf"] = 0
    assert pacf_lag_order(result) == 0


def test_handoff_reports_actual_filling_differences():
    series = hourly(np.random.default_rng(42).normal(size=200))
    series.iloc[40:44] = np.nan
    filled = series.interpolate(method="time", limit_area="inside")
    result = compare_filled_dependence(series, filled, 8)
    assert result["max_abs_acf_difference"] > 0
    assert result["max_abs_pacf_difference"] > 0
    assert result["p_gap"] == pacf_lag_order(gap_aware_pacf(series, 8))
    assert result["p_filled"] == pacf_lag_order(gap_aware_pacf(filled, 8))
    assert result["same_p"] == (result["p_gap"] == result["p_filled"])


@pytest.mark.parametrize("problem", ["grid", "unfilled", "changed_observed"])
def test_handoff_rejects_incomparable_series(problem):
    series = hourly(np.random.default_rng(42).normal(size=100))
    series.iloc[40] = np.nan
    filled = series.interpolate()
    if problem == "grid":
        filled = filled.iloc[1:]
    elif problem == "unfilled":
        filled = series.copy()
    else:
        filled.iloc[1] += 0.1
    with pytest.raises(ValueError):
        compare_filled_dependence(series, filled, 5)


def test_ljung_box_matches_statsmodels_on_complete_series():
    values = np.random.default_rng(7).normal(size=500)
    values[1:] += 0.3 * values[:-1]
    result = gap_aware_ljung_box(hourly(values), 10)
    expected = acorr_ljungbox(values, lags=10)
    np.testing.assert_allclose(result["lb_stat"], expected["lb_stat"])
    np.testing.assert_allclose(result["lb_pvalue"], expected["lb_pvalue"])


def test_ljung_box_uses_pair_counts_at_true_hour_spacing():
    values = np.random.default_rng(3).normal(size=200)
    values[np.random.default_rng(4).random(200) < 0.3] = np.nan
    series = hourly(values)
    result = gap_aware_ljung_box(series, 5)
    dependence = gap_aware_acf(series, 5)
    n = dependence.loc[0, "pair_counts"]
    terms = dependence["pair_counts"].iloc[1:] * dependence["acf"].iloc[1:] ** 2
    np.testing.assert_allclose(result["lb_stat"], (n + 2) / n * terms.cumsum())
    np.testing.assert_array_equal(result["pair_counts"], dependence["pair_counts"].iloc[1:])


def test_point_mask_hides_exact_share_of_observed_points_only():
    values = np.arange(1000, dtype=float)
    values[100:150] = np.nan
    series = hourly(values)
    mask = seeded_gap_mask(series, 0.05, 42)
    assert mask.sum() == round(0.05 * 950)
    assert series[mask].notna().all()
    assert mask.index.equals(series.index)


def test_mask_ignores_earlier_random_draws_and_values():
    series = hourly(np.random.default_rng(1).normal(size=500))
    first = seeded_gap_mask(series, 0.1, 42)
    np.random.default_rng(42).normal(size=10)
    np.testing.assert_array_equal(first, seeded_gap_mask(series, 0.1, 42))
    # The observed values do not enter the selection, only their presence.
    np.testing.assert_array_equal(first, seeded_gap_mask(series * 10 + 3, 0.1, 42))


def test_block_mask_places_whole_blocks_on_observed_points():
    values = np.ones(2020)
    values[500:520] = np.nan
    series = hourly(values)
    mask = seeded_gap_mask(series, 0.05, 42, block_lengths=[4])
    assert mask.sum() == 100
    assert series[mask].notna().all()
    blocks = (mask != mask.shift()).cumsum()[mask]
    run_lengths = mask[mask].groupby(blocks).size()
    # Adjacent blocks merge, so every run is a whole number of 4-hour blocks.
    assert (run_lengths % 4 == 0).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fraction": 0}, {"fraction": 1}, {"fraction": 0.0001},
        {"fraction": 0.1, "block_lengths": []},
        {"fraction": 0.1, "block_lengths": [0]},
        {"fraction": 0.1, "block_lengths": [2.5]},
        {"fraction": 0.5, "block_lengths": [30]},
    ],
)
def test_mask_rejects_invalid_settings(kwargs):
    series = hourly(np.ones(20))
    with pytest.raises(ValueError):
        seeded_gap_mask(series, seed=42, **kwargs)


@pytest.mark.parametrize(
    "method, kwargs",
    [("ffill", {}), ("linear", {}), ("polynomial", {"order": 2}),
     ("polynomial", {"order": 3}), ("cubic_spline", {})],
)
def test_fills_keep_observed_values_and_leave_unfillable_edges(method, kwargs):
    values = np.sin(np.arange(60) / 4)
    values[[0, 1, 10, 11, 30, 59]] = np.nan
    series = hourly(values)
    filled = fill_gaps(series, method, **kwargs)
    observed = series.notna()
    np.testing.assert_array_equal(filled[observed], series[observed])
    assert filled.iloc[:2].isna().all()
    if method != "ffill":
        assert np.isnan(filled.iloc[59])
    assert filled.iloc[[10, 11, 30]].notna().all()


def test_linear_and_local_polynomial_reproduce_their_own_degree():
    hours = np.arange(40, dtype=float)
    gaps = [5, 6, 20, 33]
    line = hourly(3 + 0.5 * hours)
    quadratic = hourly(1 - 0.2 * hours + 0.05 * hours**2)
    for truth, method, kwargs in [
        (line, "linear", {}),
        (quadratic, "polynomial", {"order": 2}),
        (quadratic, "polynomial", {"order": 2, "half_window": 4}),
        (quadratic, "cubic_spline", {}),
    ]:
        masked = truth.copy()
        masked.iloc[gaps] = np.nan
        np.testing.assert_allclose(fill_gaps(masked, method, **kwargs), truth)


def test_cubic_spline_matches_scipy_and_differs_from_local_cubic():
    values = np.random.default_rng(5).normal(size=200).cumsum()
    values[[20, 21, 50, 90, 91, 92, 150]] = np.nan
    series = hourly(values)
    observed = np.flatnonzero(~np.isnan(values))
    spline = fill_gaps(series, "cubic_spline")
    np.testing.assert_allclose(
        spline.iloc[[20, 50, 92]], CubicSpline(observed, values[observed])([20, 50, 92])
    )
    # pandas' order-3 "polynomial" equals its cubic spline; the local fit must not.
    local = fill_gaps(series, "polynomial", order=3)
    assert not np.allclose(local.iloc[[20, 50, 92]], spline.iloc[[20, 50, 92]])


def test_local_polynomial_needs_neighbours_on_both_sides():
    values = np.arange(12, dtype=float)
    values[[1, 9]] = np.nan
    filled = fill_gaps(hourly(values), "polynomial", order=2)
    assert np.isnan(filled.iloc[1])
    assert filled.iloc[9] == pytest.approx(9)


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("spline", {}), ("linear", {"order": 2}), ("polynomial", {}),
        ("polynomial", {"order": 0}), ("polynomial", {"order": 3, "half_window": 1}),
        ("polynomial", {"order": True}),
    ],
)
def test_fill_rejects_invalid_settings(method, kwargs):
    with pytest.raises(ValueError):
        fill_gaps(hourly(np.arange(10.0)), method, **kwargs)


def test_information_criteria_select_simulated_ar2_order_despite_gaps():
    rng = np.random.default_rng(42)
    noise = rng.normal(size=20_000)
    values = np.zeros_like(noise)
    for i in range(2, len(values)):
        values[i] = 0.6 * values[i - 1] - 0.3 * values[i - 2] + noise[i]
    values[rng.random(len(values)) < 0.02] = np.nan
    result = ar_information_criteria(hourly(values), 10)
    assert result["bic"].idxmin() == 2
    assert result["n_obs"].nunique() == 1


def test_information_criteria_use_common_complete_windows_at_true_spacing():
    values = np.random.default_rng(5).normal(size=300)
    values[[50, 51, 120]] = np.nan
    series = hourly(values)
    result = ar_information_criteria(series, 3)
    lags = pd.concat({k: series.shift(k) for k in range(4)}, axis=1).dropna()
    assert (result["n_obs"] == len(lags)).all()
    for p in range(4):
        design = add_constant(lags.iloc[:, 1:p + 1].to_numpy(), has_constant="add")
        fit = OLS(lags[0].to_numpy(), design).fit()
        assert result.loc[p, "rss"] == pytest.approx(fit.ssr)
    n = len(lags)
    expected_bic = n * np.log(result["rss"] / n) + (result.index + 1) * np.log(n)
    np.testing.assert_allclose(result["bic"], expected_bic)


@pytest.mark.parametrize("max_lag", [0, 2.5, True, 40])
def test_information_criteria_reject_invalid_or_unsupported_lag(max_lag):
    with pytest.raises(ValueError):
        ar_information_criteria(hourly(np.random.default_rng(1).normal(size=200)), max_lag)
