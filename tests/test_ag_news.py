import numpy as np
import pandas as pd

from preprocessing.ag_news import audit_text_split, count_overlap


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
