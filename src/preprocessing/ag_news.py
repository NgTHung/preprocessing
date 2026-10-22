"""Acquisition and label-quality audit for the AG News topic dataset."""

from pathlib import Path

import pandas as pd

from preprocessing import RAW_DATA_DIR

AG_NEWS_DATASET = "fancyzhx/ag_news"
AG_NEWS_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"


def load_ag_news(
    cache_dir: Path = RAW_DATA_DIR / "huggingface",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load the labeled AG News splits at the pinned Hugging Face revision.

    Requires network access on first use; later calls read the cache.

    Args:
        cache_dir: Hugging Face ``datasets`` cache directory.

    Returns:
        ``(train, test, label_names)``. Each frame has the original ``text`` and
        integer ``label`` columns; ``label_names[label]`` gives the topic name.
        The published splits hold 120,000 training and 7,600 test articles.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        AG_NEWS_DATASET, revision=AG_NEWS_REVISION, cache_dir=str(cache_dir)
    )
    label_names = dataset["train"].features["label"].names
    return dataset["train"].to_pandas(), dataset["test"].to_pandas(), label_names


def _empty(text: pd.Series) -> pd.Series:
    return text.isna() | text.fillna("").str.strip().eq("")


def _normalized(text: pd.Series) -> pd.Series:
    return text.map(lambda value: " ".join(value.casefold().split()))


def audit_text_split(
    split: pd.DataFrame, text_col: str = "text", label_col: str = "label"
) -> pd.Series:
    """Count empty, duplicated, and conflicting-label texts in one split.

    Duplicates use exact string equality and exclude empty texts.

    Returns:
        Series with ``samples`` (rows), ``empty_texts`` (missing or
        whitespace-only), ``duplicate_texts`` (rows repeating an earlier
        non-empty text), and ``conflicting_label_texts`` (distinct non-empty
        texts that appear with more than one label).
    """
    empty = _empty(split[text_col])
    texts = split.loc[~empty]
    return pd.Series(
        {
            "samples": len(split),
            "empty_texts": int(empty.sum()),
            "duplicate_texts": int(texts[text_col].duplicated().sum()),
            "conflicting_label_texts": int(
                texts.groupby(text_col)[label_col].nunique().gt(1).sum()
            ),
        }
    )


def count_overlap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    text_col: str = "text",
    normalize: bool = False,
) -> int:
    """Count test rows whose text also occurs in the training split.

    Args:
        train, test: Frames with a text column.
        text_col: Name of that column.
        normalize: If true, compare case-folded text with collapsed whitespace.
            This catches trivial near-duplicates only; it is not a similarity
            measure.

    Returns:
        Number of non-empty test rows with a matching non-empty training text.
    """
    train_text = train.loc[~_empty(train[text_col]), text_col]
    test_text = test.loc[~_empty(test[text_col]), text_col]
    if normalize:
        train_text, test_text = _normalized(train_text), _normalized(test_text)
    return int(test_text.isin(set(train_text)).sum())
