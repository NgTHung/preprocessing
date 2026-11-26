"""Gap-filling and dependence diagnostics that preserve a series' lag spacing."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import chi2
from statsmodels.tsa.stattools import acovf, levinson_durbin

FILL_METHODS = ("ffill", "linear", "polynomial", "cubic_spline")


def _grid_values(series: pd.Series) -> np.ndarray:
    """Validate a regular DatetimeIndex grid and return finite-or-NaN floats."""
    if not isinstance(series, pd.Series) or not isinstance(
        series.index, pd.DatetimeIndex
    ):
        raise TypeError("series must have a DatetimeIndex")
    index = series.index
    if (
        len(index) < 3
        or index.hasnans
        or not index.is_monotonic_increasing
        or not index.is_unique
    ):
        raise ValueError("series needs at least three ordered, unique grid points")
    steps = np.diff(index.asi8)
    if not np.all(steps == steps[0]):
        raise ValueError("index must be regular; reindex absent timestamps to NaN")
    values = series.to_numpy(dtype=float, na_value=np.nan)
    if np.isinf(values).any():
        raise ValueError("infinite observations are not supported")
    return values


def gap_aware_acf(series: pd.Series, max_lag: int) -> pd.DataFrame:
    """Estimate pair-adjusted ACF on a regular grid without filling NaNs.

    Args:
        series: Numeric values on a strictly increasing, complete, fixed-duration
            DatetimeIndex. NaNs represent unobserved grid points; infinities fail.
        max_lag: Largest lag in grid steps, from 1 to ``len(series) - 1``.

    Returns:
        A lag-indexed frame containing autocovariance, ACF, observed endpoint
        ``pair_counts``, and the pointwise screening half-width ``1.96/sqrt(n_k)``.
        A single observed-data mean is used at all lags. Interior missing values
        do not invalidate a pair whose two endpoints are observed. The band is a
        white-noise approximation, not calibrated inference for dependent data.

    Raises:
        ValueError: For an irregular grid, invalid lag, insufficient pairs at any
            requested lag, a constant series, or non-finite input other than NaN.
        TypeError: If the input is not a Series with a DatetimeIndex.
    """
    values = _grid_values(series)
    if (
        isinstance(max_lag, (bool, np.bool_))
        or not isinstance(max_lag, (int, np.integer))
        or not 1 <= max_lag < len(series)
    ):
        raise ValueError("max_lag must be an integer from 1 to len(series) - 1")
    observed = ~np.isnan(values)
    pair_counts = np.array(
        [
            observed.sum() if k == 0 else (observed[k:] & observed[:-k]).sum()
            for k in range(max_lag + 1)
        ],
        dtype=np.int64,
    )
    if pair_counts.min() < 2:
        raise ValueError("at least two observed endpoint pairs are needed at every lag")
    # Conservative missingness preserves elapsed time; adjusted=True divides
    # each cross-product sum by its own observed endpoint pair count.
    covariance = acovf(
        values, adjusted=True, demean=True, fft=False,
        missing="conservative", nlag=max_lag,
    )
    if not np.isfinite(covariance).all() or covariance[0] <= 0:
        raise ValueError("series must have finite, positive observed variance")
    return pd.DataFrame(
        {
            "autocovariance": covariance,
            "acf": covariance / covariance[0],
            "pair_counts": pair_counts,
            "band_half_width": 1.96 / np.sqrt(pair_counts),
        },
        index=pd.RangeIndex(max_lag + 1, name="lag"),
    )


def gap_aware_pacf(series: pd.Series, max_lag: int) -> pd.DataFrame:
    """Add a Yule–Walker PACF to :func:`gap_aware_acf` results.

    Levinson–Durbin recursion uses the pair-adjusted autocovariances. Lags are
    grid steps and the endpoint-count band is also only an approximation for
    PACF, which depends on the entire covariance sequence up to that lag.

    Raises:
        ValueError: In addition to ACF failures, if fewer than twice ``max_lag``
            observations remain, or the pairwise covariance sequence is not
            positive definite. No clipping or silent covariance repair is used.
    """
    result = gap_aware_acf(series, max_lag)
    if max_lag >= result.loc[0, "pair_counts"] // 2:
        raise ValueError("PACF max_lag must be below half the observed sample count")
    covariance = result["autocovariance"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        recursion = levinson_durbin(covariance, nlags=max_lag, isacov=True)
    # Pair-specific denominators do not guarantee a positive-definite Toeplitz
    # matrix. Non-positive innovation variance invalidates this PACF estimate.
    if (
        not np.isfinite(recursion.pacf).all()
        or not np.isfinite(recursion.sigma[1:]).all()
        or (recursion.sigma[1:] <= 0).any()
    ):
        raise ValueError("pair-adjusted covariance is not positive definite for PACF")
    result["pacf"] = recursion.pacf
    return result


def pacf_lag_order(dependence: pd.DataFrame) -> int:
    """Return the last positive lag outside its pointwise PACF band, or zero.

    ``dependence`` is the output of :func:`gap_aware_pacf`. This transparent
    screening rule supplies a candidate maximum lag, not a validated AR order.
    A last-lag crossing indicates that the search range may truncate dependence.
    """
    positive = dependence.loc[dependence.index > 0]
    outside = positive["pacf"].abs() > positive["band_half_width"]
    return int(positive.index[outside].max()) if outside.any() else 0


def ar_information_criteria(series: pd.Series, max_lag: int) -> pd.DataFrame:
    """Score AR(0) to AR(``max_lag``) by AIC and BIC on one common sample.

    Each order ``p`` is an ordinary least-squares regression of ``x_t`` on an
    intercept and ``x_{t-1}, ..., x_{t-p}``. Lags are grid steps, so gaps keep
    their elapsed time. Every order uses the same rows: the times ``t`` at which
    ``x_t`` and all ``max_lag`` lags are observed. Nothing is filled. With
    ``k = p + 1`` coefficients, ``aic = n log(rss / n) + 2k`` and
    ``bic = n log(rss / n) + k log(n)``; both omit constants that do not change
    the minimizing order.

    Args:
        series: Values on a regular DatetimeIndex grid; NaNs mark gaps.
        max_lag: Largest order, an integer of at least 1.

    Returns:
        A frame indexed by ``p`` with ``n_obs`` (constant), ``rss`` in squared
        series units, ``aic``, and ``bic``.

    Raises:
        ValueError: For an irregular grid, an invalid ``max_lag``, or fewer
            than ``10 * (max_lag + 1)`` complete lag windows.
        TypeError: If the input is not a Series with a DatetimeIndex.
    """
    values = _grid_values(series)
    if not _positive_int(max_lag) or max_lag >= len(values):
        raise ValueError("max_lag must be an integer from 1 to len(series) - 1")
    windows = np.lib.stride_tricks.sliding_window_view(values, max_lag + 1)[:, ::-1]
    windows = windows[~np.isnan(windows).any(axis=1)]
    n = len(windows)
    if n < 10 * (max_lag + 1):
        raise ValueError("too few complete lag windows for the requested max_lag")
    target, lags = windows[:, 0], windows[:, 1:]
    rows = []
    for p in range(max_lag + 1):
        design = np.column_stack([np.ones(n), lags[:, :p]])
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        rss = float(((target - design @ coefficients) ** 2).sum())
        k = p + 1
        fit = n * np.log(rss / n)
        rows.append({"n_obs": n, "rss": rss, "aic": fit + 2 * k, "bic": fit + k * np.log(n)})
    return pd.DataFrame(rows, index=pd.RangeIndex(max_lag + 1, name="p"))


def compare_filled_dependence(
    original: pd.Series, filled: pd.Series, max_lag: int
) -> dict:
    """Compare a completed training series against its gap-aware reference.

    Inputs must share an identical regular grid. ``filled`` must preserve every
    observed value and fill every NaN. Return maximum absolute ACF/PACF changes
    over lags 1 through ``max_lag``, both screened orders, and ``same_p``. A False
    result must be discussed before forecasting; it does not select a new order.

    Raises:
        ValueError: For mismatched grids, remaining gaps, changed observed values,
            or failures of :func:`gap_aware_pacf`.
    """
    if not original.index.equals(filled.index):
        raise ValueError("original and filled must have identical timestamp grids")
    if filled.isna().any():
        raise ValueError("filled series still contains missing values")
    observed = original.notna()
    if not np.array_equal(original.loc[observed], filled.loc[observed]):
        raise ValueError("gap filling must preserve observed values")
    reference = gap_aware_pacf(original, max_lag)
    complete = gap_aware_pacf(filled, max_lag)
    p_gap = pacf_lag_order(reference)
    p_filled = pacf_lag_order(complete)
    return {
        "max_abs_acf_difference": float(
            (reference["acf"] - complete["acf"]).abs().iloc[1:].max()
        ),
        "max_abs_pacf_difference": float(
            (reference["pacf"] - complete["pacf"]).abs().iloc[1:].max()
        ),
        "p_gap": p_gap,
        "p_filled": p_filled,
        "same_p": p_gap == p_filled,
    }


def gap_aware_ljung_box(series: pd.Series, max_lag: int) -> pd.DataFrame:
    """Ljung–Box statistics for a series with gaps, using observed lag pairs.

    The statistic at lag ``h`` is ``Q(h) = (n + 2) / n * sum_k n_k * r_k**2``
    for ``k = 1..h``, where ``n`` is the observed count, ``n_k`` the number of
    observed pairs exactly ``k`` grid steps apart, and ``r_k`` the pair-adjusted
    ACF from :func:`gap_aware_acf`. For a complete series this equals the
    classical Ljung–Box statistic. The p-value uses a chi-square distribution
    with ``h`` degrees of freedom under the null hypothesis of no
    autocorrelation at lags 1 to ``h``. The approximation needs many pairs at
    every lag; inspect ``pair_counts`` before interpreting a small p-value.

    Args:
        series: Values on a regular grid. NaN marks points without a value, such
            as points outside an artificial mask.
        max_lag: Largest lag in grid steps, as in :func:`gap_aware_acf`.

    Returns:
        A frame indexed by lag 1..``max_lag`` with ``lb_stat``, ``lb_pvalue``,
        ``acf``, and ``pair_counts``.

    Raises:
        ValueError, TypeError: As for :func:`gap_aware_acf`.
    """
    dependence = gap_aware_acf(series, max_lag)
    observed_count = int(dependence.loc[0, "pair_counts"])
    positive = dependence.iloc[1:]
    statistic = (observed_count + 2) / observed_count * (
        positive["pair_counts"] * positive["acf"] ** 2
    ).cumsum()
    return pd.DataFrame(
        {
            "lb_stat": statistic,
            "lb_pvalue": chi2.sf(statistic, df=positive.index.to_numpy()),
            "acf": positive["acf"],
            "pair_counts": positive["pair_counts"],
        }
    )


def seeded_gap_mask(
    series: pd.Series,
    fraction: float,
    seed: int,
    block_lengths: Sequence[int] | None = None,
) -> pd.Series:
    """Choose observed grid points to hide in a controlled gap-filling test.

    Selection ignores the observed values, so the artificial gaps are missing
    completely at random with respect to them. Native gaps are never selected.
    Each call creates ``np.random.default_rng(seed)``, so a mask does not depend
    on random draws made elsewhere in a notebook.

    Args:
        series: Values on a regular grid; NaN marks native gaps.
        fraction: Share of observed points to hide, strictly between 0 and 1.
            The mask hides exactly ``round(fraction * observed_count)`` points.
        seed: Seed for the mask's own generator.
        block_lengths: ``None`` hides single points drawn uniformly without
            replacement. Otherwise, positive gap lengths in grid steps, such as
            the native training gap lengths. Blocks are placed one at a time: a
            length is drawn uniformly from ``block_lengths``, then a start is
            drawn uniformly among positions where the entire block covers
            observed, not yet hidden points. A block may touch a native gap or
            an earlier block, which forms a longer combined gap. The last block
            is shortened so that the hidden count is exact. If the drawn length
            no longer fits anywhere, the other lengths are tried in random order.

    Returns:
        A boolean Series on ``series.index``; True marks a hidden point.

    Raises:
        ValueError: For an invalid grid, fraction, or block length, a fraction
            that hides no point, or when no block length fits any more.
    """
    values = _grid_values(series)
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between 0 and 1")
    available = ~np.isnan(values)
    target = round(fraction * int(available.sum()))
    if target < 1:
        raise ValueError("fraction hides no observed points")
    rng = np.random.default_rng(seed)
    hidden = np.zeros(len(values), dtype=bool)
    if block_lengths is None:
        hidden[rng.choice(np.flatnonzero(available), size=target, replace=False)] = True
        return pd.Series(hidden, index=series.index, name="hidden")

    lengths = np.asarray(block_lengths)
    if (
        lengths.ndim != 1
        or lengths.size == 0
        or not np.issubdtype(lengths.dtype, np.integer)
        or (lengths < 1).any()
    ):
        raise ValueError("block_lengths must be a non-empty sequence of positive integers")
    while (remaining := int(target - hidden.sum())) > 0:
        free = available & ~hidden
        # A start s fits a block of length L when free[s:s + L] are all True.
        cumulative = np.concatenate([[0], np.cumsum(free)])
        first = rng.choice(lengths)
        for length in [first, *rng.permutation(np.unique(lengths[lengths != first]))]:
            length = int(length)
            if length > len(free):
                continue
            starts = np.flatnonzero(cumulative[length:] - cumulative[:-length] == length)
            if starts.size:
                break
        else:
            raise ValueError("no block length fits the remaining observed points")
        start = int(rng.choice(starts))
        hidden[start:start + min(length, remaining)] = True
    return pd.Series(hidden, index=series.index, name="hidden")


def _local_polynomial_fill(
    values: np.ndarray, order: int, half_window: int
) -> np.ndarray:
    """Fill each interior NaN from a least-squares polynomial of nearby points."""
    filled = values.copy()
    observed = np.flatnonzero(~np.isnan(values))
    for position in np.flatnonzero(np.isnan(values)):
        after = np.searchsorted(observed, position)
        left = observed[max(after - half_window, 0):after]
        right = observed[after:after + half_window]
        if len(left) < half_window or len(right) < half_window:
            continue
        neighbours = np.concatenate([left, right])
        # Centring on the missing position keeps the fit well conditioned and
        # makes the intercept the fitted value at that position.
        coefficients = np.polyfit(neighbours - position, values[neighbours], order)
        filled[position] = coefficients[-1]
    return filled


def _positive_int(value) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, np.integer))
        and value >= 1
    )


def fill_gaps(
    series: pd.Series,
    method: str,
    *,
    order: int | None = None,
    half_window: int | None = None,
) -> pd.Series:
    """Fill NaNs on a regular grid with one of :data:`FILL_METHODS`.

    Observed values are returned unchanged. Points that a method cannot fill
    remain NaN, so callers can count them instead of scoring extrapolations.

    Methods:
        ``ffill``: Repeat the last observed value. Uses only past values; a
            leading gap stays NaN.
        ``linear``: Straight line between the nearest observed points on each
            side, with distance measured in grid steps.
        ``polynomial``: For each missing point, fit a least-squares polynomial
            of degree ``order`` to the ``half_window`` nearest observed points on
            each side and evaluate it there. ``half_window`` defaults to
            ``order``, giving ``2 * order`` points. A point without
            ``half_window`` observed points on both sides stays NaN. This local
            fit differs from pandas ``interpolate(method="polynomial")``, which
            builds a spline through all points and, for order 3, gives the same
            values as ``method="cubicspline"``.
        ``cubic_spline``: One not-a-knot cubic spline through all observed
            points (:class:`scipy.interpolate.CubicSpline`).

    ``linear``, ``polynomial``, and ``cubic_spline`` use observations after a
    gap and leave points before the first or after the last observation as NaN.
    They are retrospective and unsuitable for forecast inputs.

    Raises:
        ValueError: For an invalid grid, an unknown method, invalid polynomial
            settings, or fewer observed points than the method needs.
    """
    values = _grid_values(series)
    if method not in FILL_METHODS:
        raise ValueError(f"method must be one of {FILL_METHODS}")
    if method != "polynomial" and (order is not None or half_window is not None):
        raise ValueError("order and half_window apply only to the polynomial method")
    observed = np.flatnonzero(~np.isnan(values))
    if len(observed) < 2:
        raise ValueError("at least two observed points are needed")

    if method == "ffill":
        filled = pd.Series(values).ffill().to_numpy()
    elif method == "polynomial":
        if not _positive_int(order):
            raise ValueError("polynomial filling needs an integer order of at least 1")
        half_window = order if half_window is None else half_window
        if not _positive_int(half_window) or 2 * half_window < order + 1:
            raise ValueError("half_window must be an integer with 2 * half_window > order")
        filled = _local_polynomial_fill(values, int(order), int(half_window))
    else:
        filled = values.copy()
        inside = np.arange(observed[0], observed[-1] + 1)
        inside = inside[np.isnan(values[inside])]
        if method == "linear":
            filled[inside] = np.interp(inside, observed, values[observed])
        else:
            if len(observed) < 4:
                raise ValueError("cubic spline filling needs at least four observed points")
            filled[inside] = CubicSpline(observed, values[observed])(inside)
    return pd.Series(filled, index=series.index, name=series.name)
