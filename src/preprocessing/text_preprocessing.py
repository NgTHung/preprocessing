"""Normalization steps and measurement helpers for the text preprocessing chapter.

Every normalization step is a pure ``str -> str`` function with no fitted state,
so applying it before cross-validation cannot leak information between folds.
"""

import html
import re
import warnings
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from preprocessing import RANDOM_STATE

# AG News lost the "&" of many entities and left a space in its place
# ("Haiti #39;s", "AT amp;T"). The optional space is consumed so the repaired
# entity rejoins its neighbor. The second look-behind skips words that merely
# end in an entity name, such as "slamp;", and the "lt;" inside "&amp;lt;".
_BROKEN_ENTITY = re.compile(r"(?<!&) ?(?<![\w&;])(quot|lt|gt|amp|#\d+|#[xX][0-9A-Fa-f]+);")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"(?<![\w.])@\w+")
_HASHTAG = re.compile(r"(?<![\w&])#([^\W\d_]\w*)")
_SYMBOL = re.compile(r"[^\w'’-]|_")
# Apostrophes and hyphens survive only between two letters or digits.
_LOOSE_MARK = re.compile(r"(?<![^\W_])['’-]|['’-](?![^\W_])")
_NUMBER_TOKEN = re.compile(r"(?<!\S)[\d'’-]+(?!\S)")
# "(Reuters)" tags and "Reuters - " datelines; "Reuters-owned" is left alone.
_WIRE_TAG = re.compile(r"\((?:Reuters|AP|AFP)\)|(?<![\w-])(?:Reuters|AP|AFP)\s+-(?=\s)")


def repair_entities(text: str) -> str:
    """Restore entities that lost their ampersand, then decode all HTML entities.

    ``#39;``, ``#36;``, and bare ``quot;``, ``lt;``, ``gt;``, ``amp;`` regain
    their ``&`` (absorbing the single space that replaced it). Decoding repeats
    until the text stops changing, so double escapes such as ``&amp;lt;`` become
    ``<``. Escaped markup therefore turns into real tags for :func:`strip_html`.
    """
    text = _BROKEN_ENTITY.sub(r"&\1;", text)
    while (decoded := html.unescape(text)) != text:
        text = decoded
    return text


def strip_html(text: str) -> str:
    """Replace HTML tags with spaces and keep their text content.

    Text without ``<`` is returned unchanged, which also keeps a literal
    comparison such as ``profit < forecast`` intact.
    """
    if "<" not in text:
        return text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        return BeautifulSoup(text, "html.parser").get_text(" ")


def replace_backslashes(text: str) -> str:
    """Replace literal backslashes, which stand for lost line breaks, with spaces."""
    return text.replace("\\", " ")


def remove_urls(text: str) -> str:
    """Delete ``http(s)://`` and ``www.`` addresses up to the next whitespace."""
    return _URL.sub(" ", text)


def strip_mentions_hashtags(text: str) -> str:
    """Delete ``@mentions`` and keep the word of each ``#hashtag`` without ``#``."""
    return _HASHTAG.sub(r"\1", _MENTION.sub("", text))


def lowercase(text: str) -> str:
    """Case-fold the text (Unicode-aware lowercasing, so ``ß`` becomes ``ss``)."""
    return text.casefold()


def remove_digits_and_symbols(text: str) -> str:
    """Keep letters, digits inside words, and internal apostrophes and hyphens.

    Every other character becomes a space. Apostrophes and hyphens that are not
    between two alphanumeric characters also become spaces. Tokens made only of
    digits (and leftover marks) are then deleted, so ``2004`` and ``5%`` vanish
    while ``mp3``, ``3g``, ``it's``, and ``well-known`` stay. Abbreviations with
    periods split into letters: ``u.s.`` becomes ``u s``.
    """
    text = _LOOSE_MARK.sub(" ", _SYMBOL.sub(" ", text))
    return _NUMBER_TOKEN.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    """Join whitespace-separated pieces with single spaces and trim the ends."""
    return " ".join(text.split())


def remove_wire_tags(text: str) -> str:
    """Delete the agency markers of Reuters, AP, and AFP (case-sensitive).

    Both forms are removed: the tag ``(Reuters)`` and the dateline ``Reuters -``
    that often follows it. This is an optional step evaluated separately: the
    markers are content, but their frequency depends on topic. Other agencies,
    place datelines, and agency names inside links or words are kept.
    """
    return _WIRE_TAG.sub(" ", text)


NORMALIZATION_STEPS: list[tuple[str, Callable[[str], str]]] = [
    ("repair entities", repair_entities),
    ("strip HTML", strip_html),
    ("split backslashes", replace_backslashes),
    ("remove URLs", remove_urls),
    ("mentions and hashtags", strip_mentions_hashtags),
    ("lowercase", lowercase),
    ("digits and symbols", remove_digits_and_symbols),
    ("collapse whitespace", collapse_whitespace),
]


def normalize(
    text: str, steps: list[tuple[str, Callable[[str], str]]] = NORMALIZATION_STEPS
) -> str:
    """Apply ``steps`` in order; the default is the full normalization pipeline."""
    for _, step in steps:
        text = step(text)
    return text


def stratified_sample(
    frame: pd.DataFrame,
    labels: str,
    per_label: int,
    rng: np.random.Generator,
    exclude: pd.Index | None = None,
) -> pd.DataFrame:
    """Draw ``per_label`` rows of every label without replacement.

    Args:
        frame: Rows to sample from.
        labels: Name of the label column.
        per_label: Rows to draw for each label.
        rng: Seeded generator; the same seed and inputs give the same rows.
        exclude: Index labels that may not be drawn, for example an earlier sample.

    Returns:
        The sampled rows in their original index order.

    Raises:
        ValueError: If a label has fewer than ``per_label`` eligible rows.
    """
    eligible = frame if exclude is None else frame.loc[~frame.index.isin(exclude)]
    chosen = []
    for label, group in eligible.groupby(labels, sort=True, observed=True):
        if len(group) < per_label:
            raise ValueError(f"label {label!r} has {len(group)} rows, need {per_label}")
        chosen.append(group.index[rng.choice(len(group), size=per_label, replace=False)])
    return frame.loc[np.sort(np.concatenate(chosen))]


def vocabulary_change(before: set[str], after: set[str]) -> pd.Series:
    """Compare two vocabularies (sets of token types).

    Returns:
        Series with ``types_before``, ``types_after``, ``removed_share`` (types of
        ``before`` missing from ``after``) and ``new_share`` (types of ``after``
        missing from ``before``). Both shares divide by ``types_before`` and are
        NaN when ``before`` is empty.
    """
    size = len(before)
    return pd.Series(
        {
            "types_before": size,
            "types_after": len(after),
            "removed_share": len(before - after) / size if size else np.nan,
            "new_share": len(after - before) / size if size else np.nan,
        }
    )


def oov_rate(
    train_units: Iterable[Hashable], test_docs: Iterable[Sequence[Hashable]]
) -> pd.Series:
    """Measure how many test units are missing from the training vocabulary.

    Args:
        train_units: Units seen in training; their distinct values form the
            vocabulary.
        test_docs: Tokenized test documents.

    Returns:
        Series with ``test_units`` (all test tokens), ``oov_units``, ``token_rate``
        (``oov_units / test_units``), ``test_types`` (distinct test units),
        ``oov_types``, and ``type_rate``.

    Raises:
        ValueError: If the test documents contain no units.
    """
    vocabulary = set(train_units)
    counts = Counter(unit for doc in test_docs for unit in doc)
    if not counts:
        raise ValueError("test documents contain no units")
    oov = {unit: count for unit, count in counts.items() if unit not in vocabulary}
    total = sum(counts.values())
    return pd.Series(
        {
            "test_units": total,
            "oov_units": sum(oov.values()),
            "token_rate": sum(oov.values()) / total,
            "test_types": len(counts),
            "oov_types": len(oov),
            "type_rate": len(oov) / len(counts),
        }
    )


def collision_rates(types: Mapping[str, int], mapping: Callable[[str], str]) -> pd.Series:
    """Measure how often a stemmer or lemmatizer merges distinct word types.

    Args:
        types: Token count of every distinct original type.
        mapping: Function from a type to its root (stem or lemma).

    Returns:
        Series with ``types``, ``roots`` (distinct roots), ``collision_rate``
        (share of types whose root is shared with at least one other type),
        ``reduction`` (``1 - roots / types``), and ``token_collision_rate`` (the
        share of tokens whose type belongs to such a shared group).
    """
    roots = {word: mapping(word) for word in types}
    group_size = Counter(roots.values())
    collided = [word for word, root in roots.items() if group_size[root] > 1]
    total_tokens = sum(types.values())
    return pd.Series(
        {
            "types": len(roots),
            "roots": len(group_size),
            "collision_rate": len(collided) / len(roots),
            "reduction": 1 - len(group_size) / len(roots),
            "token_collision_rate": sum(types[word] for word in collided) / total_tokens,
        }
    )


def presence_mutual_information(X, y: np.ndarray) -> np.ndarray:
    """Mutual information in nats between each term's presence and the label.

    Presence is ``X > 0``, so term counts are ignored. For term ``j`` with
    presence ``P_j`` and label ``C``,
    ``I(P_j; C) = sum_{p, c} P(p, c) log(P(p, c) / (P(p) P(c)))`` with
    ``0 log 0 = 0``, computed for all columns at once from sparse contingency
    counts.

    Args:
        X: Document-term matrix (sparse or dense), one row per document.
        y: Label of each row.

    Returns:
        Array with one non-negative value per column of ``X``.
    """
    present = sparse.csr_matrix(X > 0, dtype=np.float64)
    classes, codes = np.unique(np.asarray(y), return_inverse=True)
    n = present.shape[0]
    onehot = sparse.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, len(classes)))
    joint_present = np.asarray((present.T @ onehot).todense()) / n
    class_share = np.bincount(codes, minlength=len(classes)) / n
    joint_absent = class_share - joint_present
    term_share = joint_present.sum(axis=1, keepdims=True)
    information = np.zeros(present.shape[1])
    for joint, marginal in [(joint_present, term_share), (joint_absent, 1 - term_share)]:
        expected = marginal * class_share
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(joint > 0, joint * np.log(joint / expected), 0.0)
        information += terms.sum(axis=1)
    return np.maximum(information, 0.0)


def sparsity(X) -> float:
    """Return ``1 - nnz / (n d)``, the share of zero entries in an ``n x d`` matrix.

    Raises:
        ValueError: If the matrix has no entries.
    """
    n, d = X.shape
    if n * d == 0:
        raise ValueError("sparsity is undefined for an empty matrix")
    nonzero = X.count_nonzero() if sparse.issparse(X) else np.count_nonzero(X)
    return 1.0 - nonzero / (n * d)


def sample_label_pairs(
    labels: np.ndarray, n_pairs: int, same: bool, rng: np.random.Generator
) -> np.ndarray:
    """Draw random pairs of distinct positions with equal or different labels.

    Pairs are drawn uniformly among the qualifying ordered pairs by rejection
    sampling, so a pair may repeat.

    Returns:
        Integer array of shape ``(n_pairs, 2)`` holding positions into ``labels``.
    """
    labels = np.asarray(labels)
    pairs = np.empty((0, 2), dtype=np.int64)
    while len(pairs) < n_pairs:
        candidates = rng.integers(0, len(labels), size=(2 * n_pairs, 2))
        equal = labels[candidates[:, 0]] == labels[candidates[:, 1]]
        keep = (candidates[:, 0] != candidates[:, 1]) & (equal if same else ~equal)
        pairs = np.vstack([pairs, candidates[keep]])
    return pairs[:n_pairs]


class MeanWordVectors(BaseEstimator, TransformerMixin):
    """Train Word2Vec on tokenized documents and average word vectors per document.

    ``fit`` trains gensim Word2Vec with one worker thread and a fixed seed, which
    makes training deterministic; with several workers the update order, and so
    the vectors, would vary between runs. ``transform`` returns the mean of a document's
    in-vocabulary word vectors as float32. A document with no in-vocabulary
    token, or no tokens, gets the zero vector; the number of such documents in
    the last ``transform`` call is stored as ``n_all_oov_``.

    Inputs are lists of token lists, so the transformer can sit in a
    scikit-learn pipeline and is refitted on each training fold.
    """

    def __init__(
        self,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 5,
        sg: int = 1,
        seed: int = RANDOM_STATE,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.sg = sg
        self.seed = seed

    def fit(self, docs: Sequence[Sequence[str]], y=None) -> "MeanWordVectors":
        from gensim.models import Word2Vec

        self.model_ = Word2Vec(
            sentences=[list(doc) for doc in docs],
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            sg=self.sg,
            seed=self.seed,
            workers=1,
        )
        return self

    def transform(self, docs: Sequence[Sequence[str]]) -> np.ndarray:
        vectors = self.model_.wv
        output = np.zeros((len(docs), self.vector_size), dtype=np.float32)
        all_oov = 0
        for row, doc in enumerate(docs):
            known = [token for token in doc if token in vectors.key_to_index]
            if known:
                output[row] = vectors[known].mean(axis=0)
            else:
                all_oov += 1
        self.n_all_oov_ = all_oov
        return output
