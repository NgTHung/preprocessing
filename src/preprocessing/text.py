"""Descriptive text statistics shared by the text EDA and preprocessing chapters."""

import re
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Runs of Unicode letters or digits, optionally joined by internal apostrophes.
# Punctuation, "#", ";" and "&" split tokens, so broken entities such as "#39;s"
# stay visible as the separate tokens "39" and "s".
WORD_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


def word_tokens(text: object) -> list[str]:
    """Split raw text into case-folded word tokens for descriptive statistics.

    This is the EDA counting convention, not a preprocessing step: the input is
    not cleaned, so markup fragments become tokens.

    Args:
        text: A string; missing values (``None`` or NaN) count as empty.

    Returns:
        Case-folded tokens matching :data:`WORD_PATTERN`, in text order. Empty or
        punctuation-only text gives an empty list.
    """
    if not isinstance(text, str):
        return []
    return WORD_PATTERN.findall(text.casefold())


def rank_biserial(u_statistic: float, n_x: int, n_y: int) -> float:
    """Convert a Mann-Whitney U for sample ``x`` into the rank-biserial correlation.

    ``r = 2U / (n_x n_y) - 1`` equals ``P(X > Y) - P(X < Y)`` with ties counted as
    half, so it lies in [-1, 1]; positive values mean ``x`` tends to be larger.
    """
    return 2.0 * u_statistic / (n_x * n_y) - 1.0


def pairwise_mann_whitney(
    values: pd.Series,
    groups: pd.Series,
    order: Sequence | None = None,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Run Mann-Whitney U tests for every pair of groups with Holm correction.

    Args:
        values: Numeric observations.
        groups: Group label for each observation, aligned with ``values``.
        order: Group order; pairs follow ``itertools.combinations(order, 2)``.
            Defaults to sorted unique labels.
        alternative: Passed to :func:`scipy.stats.mannwhitneyu`.

    Returns:
        One row per pair with sample sizes, medians, ``u_statistic`` (for
        ``group_a``), ``p_value``, Holm-adjusted ``p_holm`` across all pairs, and
        ``rank_biserial`` (positive when ``group_a`` tends to be larger).

    Raises:
        ValueError: If fewer than two groups are present.
    """
    order = list(order) if order is not None else sorted(groups.unique())
    if len(order) < 2:
        raise ValueError("need at least two groups")
    samples = {group: values[groups == group].to_numpy() for group in order}
    rows = []
    for group_a, group_b in combinations(order, 2):
        x, y = samples[group_a], samples[group_b]
        result = mannwhitneyu(x, y, alternative=alternative)
        rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "n_a": len(x),
                "n_b": len(y),
                "median_a": float(np.median(x)),
                "median_b": float(np.median(y)),
                "u_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "rank_biserial": rank_biserial(result.statistic, len(x), len(y)),
            }
        )
    table = pd.DataFrame(rows)
    table.insert(
        table.columns.get_loc("p_value") + 1,
        "p_holm",
        multipletests(table["p_value"], method="holm")[1],
    )
    return table


def type_token_ratio(tokens: Sequence[str]) -> float:
    """Return ``V / N``: distinct token types over total tokens.

    Raises:
        ValueError: If ``tokens`` is empty, where the ratio is undefined.
    """
    if len(tokens) == 0:
        raise ValueError("TTR is undefined for zero tokens")
    return len(set(tokens)) / len(tokens)


def sampled_ttr(
    documents: Sequence[Sequence[str]],
    budget: int,
    rng: np.random.Generator,
    draws: int = 20,
) -> np.ndarray:
    """Estimate TTR at a fixed token budget by sampling whole documents.

    TTR falls as the token count grows, so comparing groups of different sizes
    requires a common ``N``. Each draw shuffles document order, concatenates
    documents until ``budget`` tokens are reached, and truncates the last
    document so exactly ``budget`` tokens are counted.

    Args:
        documents: Token lists, one per document; empty lists contribute nothing.
        budget: Tokens per draw.
        rng: Seeded generator controlling document order.
        draws: Number of independent shuffles.

    Returns:
        Array of ``draws`` TTR values.

    Raises:
        ValueError: If ``budget`` is not positive or exceeds the total token count.
    """
    total = sum(len(doc) for doc in documents)
    if budget <= 0 or budget > total:
        raise ValueError(f"budget must be in [1, {total}]")
    ratios = np.empty(draws)
    for draw in range(draws):
        seen: set[str] = set()
        remaining = budget
        for position in rng.permutation(len(documents)):
            doc = documents[position][:remaining]
            seen.update(doc)
            remaining -= len(doc)
            if remaining == 0:
                break
        ratios[draw] = len(seen) / budget
    return ratios


def rank_frequency(counts: Mapping[str, int] | Iterable[tuple[str, int]]) -> pd.DataFrame:
    """Order token counts for a Zipf plot.

    Tokens are sorted by descending frequency, ties alphabetically, and receive
    distinct ordinal ranks 1..V, so tied tokens occupy consecutive ranks.

    Returns:
        Frame with ``token``, ``frequency``, and ``rank`` columns.
    """
    items = counts.items() if isinstance(counts, Mapping) else counts
    table = pd.DataFrame(list(items), columns=["token", "frequency"])
    table = table.sort_values(["frequency", "token"], ascending=[False, True])
    table = table.reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def fit_zipf(
    table: pd.DataFrame, min_rank: int, max_rank: int, log_points: int | None = None
) -> pd.Series:
    """Fit ``log10 f = a + b log10 r`` by least squares over a closed rank interval.

    Zipf's law predicts ``f ∝ r^(-s)`` with ``s`` near 1, so the slope ``b``
    estimates ``-s``. The R² describes linearity on the log-log scale within the
    chosen interval; it is not a goodness-of-fit test for a power law.

    Ranks are evenly spaced on a linear scale, so with every rank included most
    points lie in the highest decade and dominate the slope. ``log_points``
    instead keeps only the ranks nearest to that many log-spaced positions,
    weighting each decade of rank about equally.

    Args:
        table: Output of :func:`rank_frequency` (positive frequencies).
        min_rank, max_rank: Inclusive rank interval used for the fit.
        log_points: If given, the number of log-spaced target ranks; duplicates
            after rounding are dropped, so fewer points may remain.

    Returns:
        Series with ``min_rank``, ``max_rank``, ``n_points``, ``slope``,
        ``exponent`` (``-slope``), ``intercept``, and ``r_squared``.

    Raises:
        ValueError: If fewer than two ranks fall in the interval.
    """
    window = table[(table["rank"] >= min_rank) & (table["rank"] <= max_rank)]
    if log_points is not None and len(window) >= 2:
        targets = np.unique(
            np.rint(np.logspace(np.log10(min_rank), np.log10(max_rank), log_points))
        )
        window = window[window["rank"].isin(targets)]
    if len(window) < 2:
        raise ValueError("need at least two ranks in the fit interval")
    x = np.log10(window["rank"].to_numpy(dtype=float))
    y = np.log10(window["frequency"].to_numpy(dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope * x)
    total = y - y.mean()
    r_squared = 1.0 - residual @ residual / (total @ total) if total.any() else np.nan
    return pd.Series(
        {
            "min_rank": int(min_rank),
            "max_rank": int(max_rank),
            "n_points": len(window),
            "slope": float(slope),
            "exponent": float(-slope),
            "intercept": float(intercept),
            "r_squared": float(r_squared),
        }
    )
