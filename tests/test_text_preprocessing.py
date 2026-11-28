from preprocessing.text_preprocessing import (
    NORMALIZATION_STEPS,
    collapse_whitespace,
    lowercase,
    normalize,
    remove_digits_and_symbols,
    remove_urls,
    remove_wire_tags,
    repair_entities,
    replace_backslashes,
    strip_html,
    strip_mentions_hashtags,
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
