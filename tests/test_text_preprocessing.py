import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.metrics import mutual_info_score

from preprocessing.text_preprocessing import (
    NORMALIZATION_STEPS,
    MeanWordVectors,
    collapse_whitespace,
    collision_rates,
    lowercase,
    normalize,
    oov_rate,
    presence_mutual_information,
    remove_digits_and_symbols,
    remove_urls,
    remove_wire_tags,
    repair_entities,
    replace_backslashes,
    sample_label_pairs,
    sparsity,
    stratified_sample,
    strip_html,
    strip_mentions_hashtags,
    vocabulary_change,
)


def test_repair_entities_restores_lost_ampersands_and_nested_escapes():
    raw = "Haiti #39;s  quot;x quot; AT amp;T &amp;lt;b&amp;gt; at  #36;5"
    assert repair_entities(raw) == 'Haiti\'s "x" AT&T <b> at $5'


def test_repair_entities_leaves_words_ending_in_entity_names_alone():
    assert repair_entities("xquot; slamp; &#39;ok") == "xquot; slamp; 'ok"


def test_strip_html_keeps_link_text():
    assert collapse_whitespace(strip_html('see <A HREF="x">Ford</A> now')) == "see Ford now"


def test_strip_html_without_tags_returns_text_unchanged():
    assert strip_html("profit < forecast") == "profit < forecast"


def test_replace_backslashes_splits_joined_words():
    assert replace_backslashes("Friday\\but") == "Friday but"


def test_remove_urls_drops_scheme_and_www_addresses():
    assert collapse_whitespace(remove_urls("go http://a.com/x?y and www.b.org.")) == "go and"


def test_strip_mentions_hashtags_drops_mentions_and_keeps_hashtag_words():
    assert strip_mentions_hashtags("@bob loves #Tech") == " loves Tech"


def test_lowercase_casefolds():
    assert lowercase("Straße AND U.S.") == "strasse and u.s."


def test_remove_digits_and_symbols_keeps_alphanumerics_and_internal_marks():
    text = "in 2004, mp3 sales (3g) rose 5% -- it's well-known 'blades 1.98"
    assert collapse_whitespace(remove_digits_and_symbols(text)) == (
        "in mp3 sales 3g rose it's well-known blades"
    )


def test_remove_wire_tags_removes_only_the_three_agencies():
    text = "LONDON (Reuters) - x (AP) y (AFP) z (CP)"
    assert collapse_whitespace(remove_wire_tags(text)) == "LONDON - x y z (CP)"


def test_normalize_runs_steps_in_order():
    raw = "Ford #39;s &lt;A HREF=\"http://x\"&gt;F.N&lt;/A&gt; (Reuters) Reuters - Sales\\rose 5%"
    assert normalize(raw) == "ford's f n reuters reuters sales rose"


def test_normalize_markup_only_text_becomes_empty():
    assert normalize("&lt;p&gt; 123 &lt;/p&gt;") == ""
    assert normalize("") == ""


def test_normalization_steps_are_named_and_ordered():
    assert [name for name, _ in NORMALIZATION_STEPS] == [
        "repair entities",
        "strip HTML",
        "split backslashes",
        "remove URLs",
        "mentions and hashtags",
        "lowercase",
        "digits and symbols",
        "collapse whitespace",
    ]


def test_vocabulary_change_uses_previous_vocabulary_as_denominator():
    change = vocabulary_change({"a", "b", "c", "d"}, {"a", "b", "e"})
    assert change["types_before"] == 4 and change["types_after"] == 3
    assert change["removed_share"] == 0.5
    assert change["new_share"] == 0.25


def test_oov_rate_counts_tokens_and_types():
    rates = oov_rate(["a", "b"], [["a", "c"], ["c", "b", "d"]])
    assert rates["test_units"] == 5 and rates["oov_units"] == 3
    assert rates["token_rate"] == pytest.approx(3 / 5)
    assert rates["test_types"] == 4 and rates["oov_types"] == 2
    assert rates["type_rate"] == pytest.approx(2 / 4)


def test_oov_rate_rejects_empty_test_set():
    with pytest.raises(ValueError):
        oov_rate(["a"], [[], []])


def test_collision_rates_count_types_sharing_a_root():
    counts = {"run": 3, "runs": 1, "ran": 1, "cat": 5}
    roots = {"run": "run", "runs": "run", "ran": "ran", "cat": "cat"}
    rates = collision_rates(counts, roots.get)
    assert rates["types"] == 4 and rates["roots"] == 3
    assert rates["collision_rate"] == 0.5
    assert rates["reduction"] == 0.25
    assert rates["token_collision_rate"] == pytest.approx(0.4)


def test_presence_mutual_information_matches_sklearn_per_column():
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 3, size=(50, 6)) * (rng.random((50, 6)) < 0.5)
    labels = rng.integers(0, 3, size=50)
    expected = [mutual_info_score(counts[:, j] > 0, labels) for j in range(6)]
    result = presence_mutual_information(csr_matrix(counts), labels)
    assert result == pytest.approx(expected, abs=1e-12)


def test_presence_mutual_information_is_zero_for_constant_columns():
    X = csr_matrix(np.array([[1, 0], [1, 0], [1, 0], [1, 0]]))
    assert presence_mutual_information(X, np.array([0, 1, 0, 1])) == pytest.approx([0, 0])


def test_sparsity_of_sparse_and_dense_matrices():
    assert sparsity(csr_matrix(np.array([[1, 0], [0, 0]]))) == 0.75
    assert sparsity(np.array([[1.0, 2.0], [0.0, 3.0]])) == 0.25


def test_sparsity_rejects_empty_matrix():
    with pytest.raises(ValueError):
        sparsity(csr_matrix((0, 3)))


def test_sample_label_pairs_respect_label_constraint():
    labels = np.repeat([0, 1, 2, 3], 10)
    rng = np.random.default_rng(1)
    same = sample_label_pairs(labels, 200, same=True, rng=rng)
    different = sample_label_pairs(labels, 200, same=False, rng=rng)
    assert same.shape == (200, 2) and different.shape == (200, 2)
    assert (labels[same[:, 0]] == labels[same[:, 1]]).all()
    assert (same[:, 0] != same[:, 1]).all()
    assert (labels[different[:, 0]] != labels[different[:, 1]]).all()


def test_stratified_sample_is_balanced_and_excludes_index():
    frame = pd.DataFrame({"label": np.repeat([0, 1], 10)}, index=np.arange(100, 120))
    first = stratified_sample(frame, "label", 3, np.random.default_rng(2))
    second = stratified_sample(frame, "label", 3, np.random.default_rng(2), exclude=first.index)
    assert first["label"].value_counts().to_dict() == {0: 3, 1: 3}
    assert first.index.is_monotonic_increasing
    assert first.index.intersection(second.index).empty
    assert second["label"].value_counts().to_dict() == {0: 3, 1: 3}


def test_stratified_sample_rejects_too_small_label():
    frame = pd.DataFrame({"label": [0, 0, 1]})
    with pytest.raises(ValueError):
        stratified_sample(frame, "label", 2, np.random.default_rng(0))


DOCS = [["oil", "price", "rise"], ["oil", "stocks", "fall"], ["team", "wins", "game"]] * 20


def test_mean_word_vectors_zero_vector_for_all_oov_documents():
    model = MeanWordVectors(vector_size=8, min_count=1, epochs=2).fit(DOCS)
    vectors = model.transform([["zzz"], ["oil", "zzz"], []])
    assert vectors.shape == (3, 8) and vectors.dtype == np.float32
    assert not vectors[0].any() and not vectors[2].any()
    assert np.allclose(vectors[1], model.model_.wv["oil"])
    assert model.n_all_oov_ == 2


def test_mean_word_vectors_are_reproducible():
    first = MeanWordVectors(vector_size=8, min_count=1, epochs=2).fit(DOCS).transform(DOCS[:3])
    second = MeanWordVectors(vector_size=8, min_count=1, epochs=2).fit(DOCS).transform(DOCS[:3])
    assert np.array_equal(first, second)


def test_mean_word_vectors_do_not_depend_on_the_process_hash_seed():
    # Built-in str hashing is salted per process; notebook reruns must still
    # produce the same Word2Vec vectors.
    import os
    import subprocess
    import sys

    script = (
        "from preprocessing.text_preprocessing import MeanWordVectors\n"
        "docs = [['oil', 'price', 'rise'], ['team', 'wins', 'game']] * 20\n"
        "model = MeanWordVectors(vector_size=8, min_count=1, epochs=2).fit(docs)\n"
        "print(model.transform(docs[:2]).tobytes().hex())\n"
    )
    outputs = []
    for hash_seed in ["1", "2"]:
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
