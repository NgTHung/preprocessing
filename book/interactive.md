# Interactive preprocessing

Rolling windows and text normalization change which details remain visible in
data. These examples let you explore each effect with synthetic values and sample
text. Change a control to rerun the Python cells that depend on it.

The page shows initial output before the controls load. Controls appear after the
browser finishes loading Python. The first visit needs internet access to download
the browser runtime. Text entered here is processed in your browser.

```{marimo-config}
:external-env: true
:error: false
```

```{marimo} python
:include: false

import marimo as mo
```

## Part (a): Rolling mean

The chart uses 120 synthetic daily observations with a trend and periodic
components of 7 and 30 days. Increasing the trailing window smooths short-term
fluctuations but also delays the response to changes. Only full windows are shown;
values have arbitrary units.

```{marimo} python
:server-output: false

window = mo.ui.slider(
    start=1, stop=30, step=1, value=7,
    label="Rolling window (days)", show_value=True,
)
window
```

For observations $x_t$, the trailing mean is
$\bar{x}_t = w^{-1}\sum_{i=0}^{w-1}x_{t-i}$.
Here $t$ indexes the day, $w$ is the selected window in days, $i$ indexes the lag,
and $\bar{x}_t$ is the mean of the $w$ observations ending at $t$.
This display includes the current observation. When creating a forecasting feature,
shift it to match the forecast origin and use only observations available then.

```{marimo} python
:include: false

import math

daily_values = [
    40 + 0.04 * day + 3 * math.sin(2 * math.pi * day / 7)
    + 1.2 * math.cos(2 * math.pi * day / 30)
    for day in range(120)
]

def series_svg(values, averages):
    low, high = min(values) - 1, max(values) + 1

    def points(sequence):
        return " ".join(
            f"{36 + 628 * index / (len(values) - 1):.2f},"
            f"{210 - 180 * (value - low) / (high - low):.2f}"
            for index, value in enumerate(sequence) if value is not None
        )

    return mo.Html(
        '<svg viewBox="0 0 700 250" role="img" '
        'aria-label="Synthetic daily series and trailing rolling mean" '
        'style="width:100%;background:#fff;border-radius:8px">'
        '<path d="M36 20 V210 H664" fill="none" stroke="#64748b"/>'
        f'<polyline points="{points(values)}" fill="none" stroke="#94a3b8" '
        'stroke-width="1.5"/>'
        f'<polyline points="{points(averages)}" fill="none" stroke="#2563eb" '
        'stroke-width="3"/>'
        '<text x="36" y="240" fill="#475569" font-size="13">Day 1</text>'
        '<text x="612" y="240" fill="#475569" font-size="13">Day 120</text>'
        '<text x="45" y="18" fill="#2563eb" font-size="13">'
        'Blue: rolling mean | Gray: observations</text></svg>'
    )
```

```{marimo} python
rolling_mean = [
    sum(daily_values[index - window.value + 1:index + 1]) / window.value
    if index >= window.value - 1 else None
    for index in range(len(daily_values))
]
mo.vstack([
    mo.md(
        f"Window: {window.value} days. "
        f"Latest rolling mean: {rolling_mean[-1]:.3f} (arbitrary units)."
    ),
    series_svg(daily_values, rolling_mean),
])
```

Figure 1. Gray shows the synthetic observations; blue shows the trailing mean.
Compare their peaks as you change the window to examine smoothing and delay.

## Part (b): Text normalization

Edit the review or toggle either step. The normalized text, tokens, and counts
update together. The example uses whitespace tokenization to make each change
visible. It does not implement the assignment's full normalization pipeline.

```{marimo} python
:server-output: false

review = mo.ui.text_area(
    value="A GREAT movie! I would watch it again in 2026.",
    label="Sample review", full_width=True,
)
lowercase = mo.ui.checkbox(value=True, label="Lowercase")
remove_punctuation = mo.ui.checkbox(value=True, label="Remove punctuation")
mo.vstack([review, mo.hstack([lowercase, remove_punctuation])])
```

Digits and negation words are retained. Punctuation is replaced with spaces, which
can change token boundaries; for example, `can't` becomes two tokens. Inspect
these choices before using them in a sentiment classifier.

```{marimo} python
import re

normalized_text = review.value.lower() if lowercase.value else review.value
if remove_punctuation.value:
    normalized_text = re.sub(r"[^\w\s]", " ", normalized_text)
normalized_text = " ".join(normalized_text.split())
tokens = normalized_text.split()
mo.vstack([
    mo.md("Normalized text:"),
    mo.as_html(normalized_text),
    mo.md(
        f"Token count: {len(tokens)}. "
        f"Unique tokens: {len(set(tokens))}. "
        f"Characters: {len(normalized_text)}."
    ),
    mo.as_html(tokens),
])
```

The counts describe whitespace-delimited tokens and characters in the displayed
normalized text. Compare the output with the original review to identify which
boundaries or distinctions each step removes. Dataset comparisons belong in the
[text preprocessing notebook](04_text_preprocessing.ipynb).
