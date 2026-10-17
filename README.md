# Data preprocessing

uv-managed Python project for **part (a): time series** and **part (b): text**.
This is an initialized project with runnable starter notebooks, not a completed
analysis. No datasets or model weights have been downloaded.

The original assignment is saved in [docs/preprocessing.md](docs/preprocessing.md).

**Student name:** TODO

**Student ID (MSSV):** TODO

**Published book / GitHub Pages URL:** https://ngthung.github.io/preprocessing/

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from
the repository root:

```powershell
uv sync --locked
uv run jupyter lab
```

Python 3.12 is selected in `.python-version`. uv manages `.venv` and the exact
dependency versions in `uv.lock`; commit the lockfile. JupyterLab launched through
uv uses this environment's Python kernel. If another editor needs a named kernel:

```powershell
uv run python -m ipykernel install --user --name preprocessing --display-name "Preprocessing (uv)"
```

Install [Node.js 24 LTS](https://nodejs.org/) for the Jupyter Book 2 site tools.
The first HTML build downloads and installs the book theme's web dependencies.

For the optional Sentence Transformer experiment (downloads PyTorch dependencies):

```powershell
uv sync --locked --extra embeddings
```

The first embedding model load downloads its weights separately. NLP resources
such as WordNet and NLTK tokenization tables are also separate downloads; the text
preprocessing notebook contains the preparation commands.

## Layout

```text
book/
  index.md
  01_time_series_eda.ipynb
  02_time_series_preprocessing.ipynb
  03_text_eda.ipynb
  04_text_preprocessing.ipynb
  report.md
  myst.yml
.github/workflows/pages.yml  # execute, build, and deploy to GitHub Pages
src/preprocessing/       # shared project paths and seed
data/raw/               # original data, ignored by Git
data/processed/         # derived data, ignored by Git
outputs/                # experiment exports, ignored by Git
pyproject.toml
uv.lock
requirements.txt        # exported exact versions, excluding the local package
```

## Proposed datasets

- **Time series:** [PJM hourly energy consumption on Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption),
  initially `PJME_hourly.csv`. Verify at least two years of hourly observations.
  Inspect timestamp ordering, duplicates, timezone/daylight-saving conventions,
  and gaps before constructing the regular time index. Additional series can be
  considered for the advanced Granger extension if their timestamps align.
- **Text:** [IMDb on Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb),
  using the labeled train/test splits for binary sentiment classification. Verify
  at least 10,000 labeled samples; exclude the unsupervised split from evaluation.

Download examples live in the EDA notebooks, as required by the assignment.
Uncomment and run them when you are ready to acquire the data; Kaggle access may
require authentication. Record dataset versions, provenance, and any access steps
in the notebooks. Raw datasets and caches must stay out of Git.

## Notebook workflow

1. Complete time-series EDA: time plot, ACF/PACF, rolling 7/30/90-day statistics.
2. Implement imputation comparisons, stationarity tests, decomposition, temporal
   features, anomaly detection, and a Random Forest forecast in part (a).
3. Complete text EDA: lengths and Mann-Whitney U, class word clouds/top-50 words,
   type-token ratio, and Zipf plots.
4. Implement normalization, four tokenizers, stop-word ablations, stemming versus
   lemmatization, and BoW/TF-IDF/Word2Vec comparisons in part (b).
5. Add advanced Granger and Sentence Transformer comparisons if pursuing the
   extensions. These are also mentioned in the marking rubric.
6. Replace all TODOs with theory/formulas, implementations, and interpretations.
   Store and print numerical scores and p-values in Python variables.

For forecasting, split chronologically before fitting transforms, shift rolling
features to use only known history, and avoid full-series STL or interpolation
that uses future test observations in forecast features. For text, fit vocabularies,
BPE, vectorizers, and Word2Vec only on training data, including within CV folds.
Keep the final test set untouched during model and preprocessing selection.

## Build the book

This project uses [Jupyter Book 2](https://jupyterbook.org/stable/) and the MyST
Document Engine. `book/myst.yml` contains the book metadata, table of contents,
and theme configuration. Run these commands from the repository root:

```powershell
uv run --locked --directory book jupyter book build --html --execute --strict --execute-parallel 1
```

The HTML entry point is `book/_build/html/index.html`. The `--execute` flag runs
notebook code and `--strict` fails the build on reported warnings/errors. MyST
caches execution outputs, so unchanged notebooks may reuse cached results. To
force a fresh execution after changing data, imported helpers, or dependencies:

```powershell
uv run --locked --directory book jupyter book clean --execute --yes
uv run --locked --directory book jupyter book build --html --execute --strict --execute-parallel 1
```

The initial scaffold executes
without dataset downloads; a successful starter build does not mean the assignment
analyses are finished. After implementation, rebuild the entire book from a fresh
environment and verify every chapter before submission.

For an interactive local preview:

```powershell
uv run --locked --directory book jupyter book start --execute
```

To serve the static HTML after building with no `BASE_URL` set:

```powershell
uv run python -m http.server 8000 --directory book/_build/html
```

Open <http://localhost:8000>.

## GitHub Pages

Repository: https://github.com/NgTHung/preprocessing

Book: https://ngthung.github.io/preprocessing/

The workflow in `.github/workflows/pages.yml` installs the locked uv environment,
executes the notebooks, builds strict static HTML, and uploads the Pages artifact.
Pushes to `main` and manual workflow runs deploy the book after a successful build.
Pull requests build the book for validation without deploying it. CI uses a fresh
checkout and does not restore notebook execution caches.

GitHub Pages must use **GitHub Actions** as its source in repository Settings →
Pages. The workflow sets `BASE_URL` to `/preprocessing` dynamically from the
repository name so links and assets work under the project site's subdirectory.
If moving this workflow to a `<username>.github.io` repository or a custom domain
served at its root, change `BASE_URL` to an empty string.

To trigger another deployment or inspect recent builds with the GitHub CLI:

```powershell
gh workflow run pages.yml
gh run list --workflow pages.yml
```

The default workflow does not install the optional embeddings extra. When you
enable the Sentence Transformer experiment, update the workflow's sync command
to `uv sync --locked --extra embeddings`. Add any required dataset/NLTK downloads
to the notebooks so a fresh CI runner can reproduce the analysis.

## Environment maintenance

```powershell
uv run ruff check src
uv export --locked --no-hashes --no-emit-project --output-file requirements.txt
```

The export includes the default development group (JupyterLab and book tools).
`uv sync --locked` remains the recommended installation method because it also
installs the local package. To include optional embeddings in an export, add
`--extra embeddings` to the export command.

Core dependencies cover numerical analysis, ADF/KPSS and Phillips-Perron tests
(`statsmodels` and `arch`), calendar features (`holidays`), classical ML, NLTK,
Hugging Face BPE tokenizers, Word2Vec (`gensim`), word clouds, and data acquisition.
