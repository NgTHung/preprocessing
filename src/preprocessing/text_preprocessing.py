"""Normalization steps and measurement helpers for the text preprocessing chapter.

Every normalization step is a pure ``str -> str`` function with no fitted state,
so applying it before cross-validation cannot leak information between folds.
"""

import html
import re
import warnings
from collections.abc import Callable

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

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
_WIRE_TAG = re.compile(r"\((?:Reuters|AP|AFP)\)")


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
    """Delete the case-sensitive source tags ``(Reuters)``, ``(AP)``, and ``(AFP)``.

    This is an optional step evaluated separately: the tags are content, but
    their frequency depends on topic. Other datelines are kept.
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
