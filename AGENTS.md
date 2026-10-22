# Data preprocessing

This repository builds a Jupyter Book that compares preprocessing methods for time series and text in the HCMUS Data Mining course. Part (a) uses time series; part (b) uses text. Read the original Vietnamese [assignment](docs/preprocessing.md) before changing scope, datasets, deliverables, or report content, and use the [README](README.md) for setup and authoring commands.

The four analysis notebooks are starter scaffolds. A successful build proves that the scaffold executes; it does not establish that the required experiments or interpretations are complete. Do not describe proposed datasets or unexecuted analyses as validated results.

## Priorities

1. Correctness. Prevent data leakage, use appropriate statistical assumptions, and compare methods on the same observations and splits.
2. Evidence. Every required technique needs theory, implementation, measured output, and interpretation in the book.
3. Reproducibility. Results must regenerate from the locked environment, documented data acquisition, and seeded experiments.

Choose correctness and reproducibility over shorter runtimes or more favorable scores. Optimize only after checking that the result and evaluation protocol remain valid.

## Scope and sources

- `docs/preprocessing.md` owns the course requirements. Preserve it as the assignment reference rather than rewriting it to match an implementation.
- `book/01_time_series_eda.ipynb` and `book/02_time_series_preprocessing.ipynb` contain part (a).
- `book/03_text_eda.ipynb` and `book/04_text_preprocessing.ipynb` contain part (b).
- `book/report.md` compares results and discusses limitations. It links to notebook evidence rather than inventing or manually reproducing scores.
- `book/interactive.md` contains small reactive teaching examples. Synthetic demonstrations do not replace dataset experiments.
- `src/preprocessing/` contains shared Python helpers. Its `__init__.py` owns project paths and `RANDOM_STATE`.
- `book/myst.yml` owns book metadata, navigation, and plugin registration; `.github/workflows/pages.yml` owns CI and deployment.

The selected datasets are Beijing Multi-Site Air Quality (UCI dataset 501) for part (a) and AG News (Hugging Face `fancyzhx/ag_news`, pinned revision) for part (b). The README records their sources, revisions, citations, and licenses. Time series must cover at least two years at daily or hourly frequency; text must contain at least 10,000 labeled samples and at least two labels. Recheck these constraints after any filtering or subsampling.

Granger causality and Sentence Transformer comparisons are marked advanced in the brief but also appear in the marking rubric. Keep that distinction visible in planning and submission checks; do not promise full rubric coverage without addressing it.

## Academic integrity

- Use Python and the libraries permitted by the assignment. This project allows library implementations; there is no requirement to reimplement every algorithm from a paper.
- Attribute datasets, methods, borrowed material, and external resources. Never fabricate observations, scores, p-values, citations, or completed reviews.
- The student must understand and be able to explain every submitted method and line of code at the oral exam. Record AI assistance in the report's reproducibility discussion.
- Keep changes attributable with small, focused commits. Do not infer permission to collaborate on another student's submission.

## Environment and commands

Run from the repository root. `.python-version` selects Python 3.12; `pyproject.toml` constrains it to the 3.12 series. Use uv and the repository's `.venv`. CI uses Node.js 24 for Jupyter Book 2/MyST tooling.

```powershell
$env:PYTHONUTF8 = "1"
uv sync --locked
uv run --locked ruff check src tests
uv run --locked pytest
node --input-type=module -e "await import('./book/plugins/marimo.mjs')"
uv run --locked --directory book jupyter book build --html --execute --strict --execute-parallel 1 --ci
```

The README owns the full setup, preview, cache-cleaning, and dependency-export procedures. If data, imported helpers, or dependencies change, clear the notebook execution cache before validating affected results. Cached notebook output is not proof that changed helper code executes correctly.

Manage dependencies through `pyproject.toml` and uv. Commit `uv.lock` and the regenerated `requirements.txt` with dependency changes. Do not hand-edit resolved versions. Sentence Transformers use the optional `embeddings` extra; experiments requiring it must also be reproducible in the build environment. CI's default sync does not install that extra.

## Analysis rules

### Shared protocol

- Each notebook must run top to bottom in a fresh kernel without another notebook's variables. Put acquisition code in the notebook or import an explicit helper; document credentials or downloads needed for a fresh environment.
- Use `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, and `OUTPUT_DIR` instead of machine-specific paths. Preserve raw data and write derived artifacts separately.
- Record dataset revision, filtering, split definitions, sample counts, and relevant package/model versions. Explain subsampling and use identical subsets across comparisons.
- Use `RANDOM_STATE` for estimators and seeded generators such as `np.random.default_rng(RANDOM_STATE)` for stochastic operations. Record settings that affect determinism.
- Keep every score, test statistic, and p-value in a named Python variable and display it. Report sample counts, metric definitions, and units; do not hard-code results in prose.
- Explain null hypotheses, assumptions, significance thresholds, and limitations. Distinguish statistical significance from practical effect and predictive association from causation.
- Compare preprocessing alternatives under the same evaluation protocol. Keep the final test set out of preprocessing and model selection.

### Time series

- Audit timestamp ordering, duplicates, timezone/daylight-saving conventions, gaps, frequency, and units before regularizing the index. Convert day-based rolling windows to observation counts only after validating frequency.
- Split chronologically before fitting or selecting preprocessing. Use temporal validation rather than random folds for forecasting.
- For controlled imputation, use the same seeded mask of observed training points for every method. Score only hidden values with known ground truth and report unfillable points and evaluation counts.
- Offline interpolation and two-sided full-series decomposition can use future observations. Keep retrospective comparisons separate from causal forecast features.
- Align features and the one-step target explicitly. Shift target-derived rolling features, choose lags from training data, and ensure each forecast uses only information available at its prediction time.
- Fit transformation parameters and anomaly thresholds on training data. State seasonal periods, positivity constraints, residual definitions, and test lag settings.
- Report forecast MAE/RMSE in original units against a persistence baseline. Explain whether test predictions use observed history or recursive predictions.

### Text

- Preserve raw text and audit labels, empty samples, duplicates, and cross-split overlap. For AG News, keep the published 7,600-article test split for final evaluation and check for near-duplicates of training articles.
- Fit BPE, vocabularies, vectorizers, corpus-trained Word2Vec, and other learned transforms on training data only, including within each cross-validation fold. Use pipelines or equivalent fold-local fitting.
- Define normalization policies for HTML, URLs, mentions, hashtags, punctuation, digits, and negation. Measure the effect of each step rather than assuming removal improves topic classification.
- Define token boundaries, OOV denominators, collision rates, pooling, all-OOV handling, and sparsity before comparing representations.
- Keep document-term matrices sparse. Use a documented seeded subset for expensive visualizations; compute silhouette in the stated feature space rather than treating a t-SNE layout as proof of separation.
- Document pretrained model revisions, truncation, resource downloads, and the evaluation split for embedding comparisons.

## Implementation and validation

Keep shared acquisition, transformations, and evaluation logic in focused modules under `src/preprocessing/` when multiple notebooks need it. Keep notebook cells readable and retain the visible theory, calls, outputs, and interpretations needed for grading. Avoid abstractions that hide the experiment protocol.

For substantive helper changes, add focused checks for behavior and relevant edge cases, such as timestamp alignment, train/test separation, or empty text. Add regression coverage for bugs. Do not require a new test framework for prose edits; validate links and commands instead.

Run checks appropriate to the change:

- Python helpers: Ruff and focused behavior checks, followed by affected notebooks with fresh execution when imported logic changes.
- Notebook or book content: strict executed HTML build; inspect equations, figures, tables, references, and outputs in the rendered pages.
- Plugin or reactive cells: plugin import check, strict build, and browser interaction checks for each changed control after the runtime loads.
- Submission: fresh environment and full book execution with the actual datasets, followed by a check of the published pages. Record any unmet requirement explicitly.

Do not equate lint success, a cached build, or executable TODO cells with completed analysis. Report what was checked and any limits of the evidence.

## MyST and reactive examples

Keep `book/plugins/marimo.mjs` registered in `book/myst.yml`. Its Node adapter invokes the uv environment's platform-specific executable on Windows and Linux. Registering the Python launcher directly causes `spawn EFTYPE` in the Windows workflow.

Follow [Book authoring](docs/book-authoring.md) for reactive-cell structure, directive options, and browser verification. Preserve the documented safeguards against execution errors and startup hydration resetting edits.

Browser execution uses Pyodide and needs internet access for runtime assets. Check package compatibility and actual recomputation in a browser; a successful static build alone does not validate interactivity. Keep full-dataset training in the analysis notebooks.

## Documentation and delivery

Read [docs/WRITING_GUIDE.md](docs/WRITING_GUIDE.md) before editing Markdown or notebook narrative. Explain why, keep claims tied to evidence, and describe implemented behavior accurately. Public Python helpers need concise docstrings covering inputs, outputs, units, edge cases, and failures. Comments explain non-obvious choices, especially leakage prevention; they do not narrate obvious code or refactor history.

Keep task status in the tracker explicitly used for the work, and history in commits or dated records. Do not assume a particular external tracker or issue-based branch convention. Use focused Conventional Commit messages when committing; do not push or deploy merely to validate a local documentation edit.

Keep raw/processed datasets, model weights, caches, credentials, `.venv`, `outputs/`, and generated book/marimo assets out of Git. The submission is one public Jupyter Book on GitHub Pages from the student's personal repository. Before submission, verify student name/MSSV, dataset and resource links, setup instructions, and the published URL.

Pushes to `main` and manual Pages workflow runs deploy after a successful build; pull requests validate without deploying. Preserve the workflow's project-site `BASE_URL` handling when changing navigation or hosting configuration.
