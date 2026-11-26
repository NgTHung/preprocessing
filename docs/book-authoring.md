# Book authoring

The book combines executed notebook experiments with MyST pages so readers can
inspect measured results alongside their interpretation. Use analysis notebooks for
dataset experiments and MyST pages for navigation and discussion. The [README](../README.md) owns build and preview commands;
the [writing guide](WRITING_GUIDE.md) owns narrative and evidence conventions.

## Organize chapters

Register book pages in [book/myst.yml](../book/myst.yml). Each analysis notebook
must run in a fresh kernel and load its inputs explicitly. Shared helpers belong
in `src/preprocessing/`; keep the experiment's theory, calls, outputs, and
interpretation visible in the notebook.

Use relative links to connect chapters and MyST labels for references to specific
results. Define symbols near equations, label plot axes and units, and give figures
and tables descriptive captions. Check links and layout in the rendered site.
