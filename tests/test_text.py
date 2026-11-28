import numpy as np
import pandas as pd
import pytest

from preprocessing.text import (
    fit_zipf,
    pairwise_mann_whitney,
    rank_biserial,
    rank_frequency,
    sampled_ttr,
    type_token_ratio,
    word_tokens,
)


def test_word_tokens_casefold_and_keep_internal_apostrophes():
    assert word_tokens("Bush's  U.S. plan, 2004!") == ["bush's", "u", "s", "plan", "2004"]


def test_word_tokens_expose_broken_entities_in_raw_text():
    assert word_tokens("Haiti #39;s straits &lt;b&gt;") == [
        "haiti", "39", "s", "straits", "lt", "b", "gt"
    ]


def test_word_tokens_handle_empty_and_missing_text():
    assert word_tokens("") == []
    assert word_tokens("  -- ") == []
    assert word_tokens(None) == []
    assert word_tokens(np.nan) == []


def test_rank_biserial_spans_complete_dominance():
    assert rank_biserial(0, 3, 4) == -1.0
    assert rank_biserial(12, 3, 4) == 1.0
    assert rank_biserial(6, 3, 4) == 0.0


def test_pairwise_mann_whitney_reports_direction_and_holm_adjustment():
    values = pd.Series([1, 2, 3, 4, 10, 11, 12, 13, 5, 6, 7, 8])
    groups = pd.Series(["a"] * 4 + ["b"] * 4 + ["c"] * 4)

    table = pairwise_mann_whitney(values, groups, order=["a", "b", "c"])

    assert list(zip(table["group_a"], table["group_b"])) == [("a", "b"), ("a", "c"), ("b", "c")]
    a_vs_b = table.iloc[0]
    assert a_vs_b["n_a"] == 4 and a_vs_b["n_b"] == 4
    assert a_vs_b["rank_biserial"] == -1.0  # every "a" value is below every "b" value
    assert table.loc[table["group_a"] == "b", "rank_biserial"].item() == 1.0
    assert (table["p_holm"] >= table["p_value"]).all()
    assert table["p_holm"].le(1).all()


def test_pairwise_mann_whitney_rejects_single_group():
    with pytest.raises(ValueError):
        pairwise_mann_whitney(pd.Series([1, 2]), pd.Series(["a", "a"]))


def test_type_token_ratio():
    assert type_token_ratio(["a", "b", "a", "c"]) == 0.75
    with pytest.raises(ValueError):
        type_token_ratio([])


def test_sampled_ttr_uses_exact_budget_and_is_seeded():
    docs = [["a", "b"], ["a", "c", "d"], ["e"], ["a", "a", "a"]]

    first = sampled_ttr(docs, budget=4, rng=np.random.default_rng(0), draws=5)
    second = sampled_ttr(docs, budget=4, rng=np.random.default_rng(0), draws=5)

    assert first.shape == (5,)
    np.testing.assert_array_equal(first, second)
    # Every draw keeps exactly four tokens, so TTR is a multiple of 1/4 in (0, 1].
    assert np.all(np.isin(first * 4, [1, 2, 3, 4]))


def test_sampled_ttr_rejects_budget_above_corpus_size():
    with pytest.raises(ValueError):
        sampled_ttr([["a"], []], budget=2, rng=np.random.default_rng(0))


def test_rank_frequency_breaks_ties_alphabetically():
    table = rank_frequency({"b": 2, "a": 2, "c": 5, "d": 1})

    assert table["token"].tolist() == ["c", "a", "b", "d"]
    assert table["rank"].tolist() == [1, 2, 3, 4]


def test_fit_zipf_recovers_exact_power_law():
    ranks = np.arange(1, 1001)
    table = pd.DataFrame({"rank": ranks, "frequency": 1e6 * ranks**-1.2})

    fit = fit_zipf(table, min_rank=10, max_rank=500)

    assert fit["exponent"] == pytest.approx(1.2)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert fit["n_points"] == 491


def test_fit_zipf_needs_two_ranks():
    table = pd.DataFrame({"rank": [1, 2], "frequency": [4, 2]})
    with pytest.raises(ValueError):
        fit_zipf(table, min_rank=2, max_rank=2)


def test_fit_zipf_log_spaced_points_balance_head_and_tail():
    # Slope -1 up to rank 100, then -2: many more raw points lie in the tail.
    ranks = np.arange(1, 10_001)
    frequency = np.where(ranks <= 100, 1e6 / ranks, 1e8 / ranks**2)
    table = pd.DataFrame({"rank": ranks, "frequency": frequency})

    all_points = fit_zipf(table, 1, 10_000)
    log_points = fit_zipf(table, 1, 10_000, log_points=200)

    assert log_points["n_points"] < 200
    # Two decades at slope 1 and two at slope 2: equal decade weights give about 1.5,
    # while the all-rank fit is pulled towards the tail slope.
    assert abs(log_points["exponent"] - 1.5) < abs(all_points["exponent"] - 1.5)
    assert all_points["exponent"] > 1.9


def test_fit_zipf_log_points_recover_exact_power_law():
    ranks = np.arange(1, 5001)
    table = pd.DataFrame({"rank": ranks, "frequency": 1e6 * ranks**-0.9})

    fit = fit_zipf(table, 1, 5000, log_points=100)

    assert fit["exponent"] == pytest.approx(0.9)
