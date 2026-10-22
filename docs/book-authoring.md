# Book authoring

The book combines notebook experiments with reactive demonstrations so readers can
inspect measured results and explore individual preprocessing choices. Use analysis
notebooks for dataset experiments and MyST pages for navigation, discussion, and
small browser examples. The [README](../README.md) owns build and preview commands;
the [writing guide](WRITING_GUIDE.md) owns narrative and evidence conventions.

## Organize chapters

Register book pages in [book/myst.yml](../book/myst.yml). Each analysis notebook
must run in a fresh kernel and load its inputs explicitly. Shared helpers belong
in `src/preprocessing/`; keep the experiment's theory, calls, outputs, and
interpretation visible in the notebook.

Use relative links to connect chapters and MyST labels for references to specific
results. Define symbols near equations, label plot axes and units, and give figures
and tables descriptive captions. Check links and layout in the rendered site.

## Add reactive cells

[Interactive preprocessing](../book/interactive.md) demonstrates a rolling mean and
text normalization on synthetic values and sample text. These examples illustrate
individual effects; they do not supply evidence for dataset-level comparisons.

`jupyter-book-marimo` executes cells during the build and exports browser-reactive
output. The Node adapter in [marimo.mjs](../book/plugins/marimo.mjs) invokes the uv
environment's executable on Windows and Linux. Keep that adapter registered in
`book/myst.yml`; direct Python-launcher registration fails with `spawn EFTYPE` on
Windows.

Use the README's UTF-8 environment setting when building on Windows so the
plugin preserves Unicode in the document tree, including punctuation and math.

Use the following structure in a MyST page:

````markdown
```{marimo-config}
:external-env: true
:error: false
```

```{marimo} python
:server-output: false

import marimo as mo
size = mo.ui.slider(start=1, stop=30, value=7, label="Window")
size
```

```{marimo} python
mo.md(f"Selected window: {size.value}")
```
````

Cells on the same page share a reactive dependency graph. Define each shared
variable in one cell and read a control's `.value` in a dependent cell.
`:external-env: true` uses the build's uv environment. `:error: false` makes
execution errors fail the build. Editable input cells use `:server-output: false`
so controls appear after Python starts and early edits are not reset by hydration.

## Verify browser behavior

Pyodide runs reactive Python in the reader's browser. The first visit needs
internet access to fetch runtime assets, and browser packages must be compatible
with that environment. Keep examples small and retain full-dataset training in
the analysis notebooks.

After building, wait for the runtime to load, change each affected control, and
check that dependent output updates. Verify that the first edit persists and that
the page explains its inputs and limitations. A static build alone does not prove
browser reactivity.

Generated `.jupyter-book-marimo` assets, marimo caches, and built HTML are ignored
and regenerated during the build. See the
[upstream integration documentation](https://docs.marimo.io/guides/exporting/jupyter_book/)
for directive options.
