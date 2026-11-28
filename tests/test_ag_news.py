import numpy as np
import pandas as pd

from preprocessing.ag_news import (
    audit_text_split,
    count_overlap,
    nearest_train_similarity,
)


def frame(rows):
    return pd.DataFrame(rows, columns=["text", "label"])


def test_audit_counts_empty_duplicate_and_conflicting_texts():
    split = frame(
        [
            ("Stocks rally", 2),
            ("Stocks rally", 2),  # duplicate, same label
            ("Cup final", 1),
            ("Cup final", 3),  # duplicate, conflicting label
            ("   ", 0),
            (np.nan, 0),
        ]
    )

    audit = audit_text_split(split)

    assert audit["samples"] == 6
    assert audit["empty_texts"] == 2
    assert audit["duplicate_texts"] == 2
    assert audit["conflicting_label_texts"] == 1


def test_audit_reports_zero_for_clean_split():
    audit = audit_text_split(frame([("a", 0), ("b", 1)]))

    assert audit[["empty_texts", "duplicate_texts", "conflicting_label_texts"]].sum() == 0


def test_overlap_counts_test_rows_found_in_train():
    train = frame([("Oil prices rise", 2), ("Mars rover lands", 3)])
    test = frame([("Oil prices rise", 2), ("oil  PRICES rise", 2), ("New text", 0)])

    assert count_overlap(train, test) == 1
    assert count_overlap(train, test, normalize=True) == 2


def test_overlap_ignores_empty_texts():
    train = frame([("", 0), (np.nan, 1)])
    test = frame([("", 0), (np.nan, 1), (" ", 2)])

    assert count_overlap(train, test, normalize=True) == 0


def test_nearest_train_similarity_finds_reworded_copy():
    train = frame(
        [
            ("Oil prices climb as Gulf supply fears grow", 2),
            ("Mars rover sends new images of crater", 3),
            ("Champions league final ends in penalties", 1),
        ]
    )
    test = frame(
        [
            ("Oil prices climb as Gulf supply fears grow again", 2),
            ("Completely unrelated gardening advice", 0),
        ]
    )

    nearest = nearest_train_similarity(train, test, batch_size=1)

    assert nearest["train_position"].tolist()[0] == 0
    assert nearest["cosine"].iloc[0] > 0.9
    assert nearest["cosine"].iloc[1] == 0.0
    assert nearest["train_position"].iloc[1] == -1
    assert nearest.index.equals(test.index)


def test_nearest_train_similarity_handles_empty_test_text():
    train = frame([("Stocks rally on earnings", 2)])
    test = frame([("", 2), (np.nan, 1)])

    nearest = nearest_train_similarity(train, test)

    assert nearest["cosine"].tolist() == [0.0, 0.0]
    assert nearest["train_position"].tolist() == [-1, -1]
