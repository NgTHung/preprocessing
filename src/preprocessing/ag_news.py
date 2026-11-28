"""Acquisition and label-quality audit for the AG News topic dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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


def nearest_train_similarity(
    train: pd.DataFrame,
    test: pd.DataFrame,
    text_col: str = "text",
    batch_size: int = 256,
) -> pd.DataFrame:
    """Find each test text's most similar training text by TF-IDF cosine.

    The vectorizer is fitted on training text only, with scikit-learn's English
    stop-word list removed so that shared function words do not dominate the
    similarity. Test words absent from the training vocabulary are ignored.
    The scan is exhaustive (no approximate index) and processes test rows in
    batches to bound memory.

    Args:
        train, test: Frames with a text column; missing text counts as empty.
        text_col: Name of that column.
        batch_size: Test rows per sparse matrix product.

    Returns:
        Frame indexed like ``test`` with ``train_position`` (positional index
        into ``train``) and ``cosine`` in [0, 1]. Test rows with no in-vocabulary
        word get ``train_position = -1`` and ``cosine = 0``.
    """

    vectorizer = TfidfVectorizer(stop_words="english", dtype=np.float32)
    train_matrix = vectorizer.fit_transform(train[text_col].fillna(""))
    test_matrix = vectorizer.transform(test[text_col].fillna(""))
    train_transposed = train_matrix.T.tocsr()

    positions = np.full(test_matrix.shape[0], -1, dtype=np.int64)
    cosine = np.zeros(test_matrix.shape[0], dtype=np.float64)
    for start in range(0, test_matrix.shape[0], batch_size):
        # Rows are L2-normalized, so the dot product is the cosine similarity.
        scores = (test_matrix[start : start + batch_size] @ train_transposed).toarray()
        best = scores.argmax(axis=1)
        best_score = scores[np.arange(len(best)), best]
        found = best_score > 0
        positions[start : start + len(best)][found] = best[found]
        cosine[start : start + len(best)] = np.minimum(best_score, 1.0)
    return pd.DataFrame(
        {"train_position": positions, "cosine": cosine}, index=test.index
    )
