# Data preprocessing

This Python project provides a Jupyter Book for comparing time-series and
text preprocessing methods in the HCMUS Data Mining course. The notebooks organize
data acquisition, statistical analysis, and evaluation so readers can examine the
evidence behind each preprocessing choice.

The time-series chapters contain executed analyses: timestamp and missingness
audits, gap-aware ACF/PACF, gap-filling comparisons, stationarity tests,
decompositions, calendar and rolling features, anomaly detectors, Granger tests, and
a one-step Random Forest forecast against persistence. The text EDA chapter is
executed: a label, overlap, and near-duplicate audit of AG News, markup prevalence,
length tests, frequency tables, type-token ratios, and a Zipf analysis. The text
preprocessing chapter is executed: a normalization pipeline, four tokenizers with test
OOV rates, stop-word, stemming, and lemmatization ablations, BoW, TF-IDF, and Word2Vec
representations, and the advanced Sentence Transformer comparison, all under one
cross-validation protocol, followed by one test-split evaluation. Building the book
downloads the Beijing archive, the pinned AG News revision, NLTK resources, and the
pinned `all-MiniLM-L6-v2` model on first use.

Student name: TODO. Student ID (MSSV): TODO.

[Published book](https://ngthung.github.io/preprocessing/) ·
[Source repository](https://github.com/NgTHung/preprocessing) ·
[Course assignment](docs/preprocessing.md)

## Install the environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Node.js](https://nodejs.org/) 24, which CI uses for the Jupyter Book 2 site tools.
Run all commands below from the repository root:

```powershell
uv sync --locked --extra embeddings
uv run --locked jupyter lab
```

`.python-version` selects Python 3.12. uv installs the local `preprocessing`
package and dependencies into `.venv`, using the exact versions in `uv.lock`.
JupyterLab launched through uv uses that environment's Python kernel. If your
editor requires a named kernel, register it with:

```powershell
uv run --locked python -m ipykernel install --user --name preprocessing --display-name "Preprocessing (uv)"
```

The `embeddings` extra installs `sentence-transformers` and CPU-only PyTorch,
which the advanced Sentence Transformer section of the
[text preprocessing notebook](book/04_text_preprocessing.ipynb) requires. uv takes
`torch` from the PyTorch CPU index configured in `pyproject.toml`; installing
`requirements.txt` with pip needs
`--extra-index-url https://download.pytorch.org/whl/cpu` for the same wheel.

The text preprocessing notebook downloads its NLTK resources into
`data/raw/nltk_data` and the
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
model (Apache-2.0, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, about
90 MB) into `data/raw/huggingface` on first run. Both stay outside version control.

## Prepare data

The book uses the datasets below. Record the source revision, access steps,
filtering, and split definitions in the EDA notebooks.

| Part | Dataset | Contents | Validation |
|---|---|---|---|
| (a): Time series | [Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data), UCI repository, DOI [10.24432/C5RK5G](https://doi.org/10.24432/C5RK5G), CC BY 4.0 | Hourly readings of six pollutants and six weather variables at 12 stations, 2013-03-01 to 2017-02-28 | Audit timestamp conventions, duplicates, gaps, and native missing values; state the station and target variable for each analysis |
| (b): Text | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news), Hugging Face revision `eb185aade064a813bc0b7f42de02595523103ca4` | News titles and descriptions in four balanced topic classes; 120,000 training and 7,600 test articles | Audit HTML-entity residue, duplicates, conflicting labels, and train/test overlap |

Cite the Beijing data as Zhang et al., "Cautionary tales on air-quality improvement
in Beijing", Proceedings of the Royal Society A, 2017. Cite AG News as Zhang, Zhao,
and LeCun, "Character-level Convolutional Networks for Text Classification", NeurIPS
2015. The Hugging Face card states no license; the source
[AG corpus](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html) is
distributed for research and non-commercial use.

The time-series notebooks load and checksum-verify the UCI archive automatically;
the text EDA notebook downloads AG News at the pinned revision through
`preprocessing.ag_news.load_ag_news` and caches it under `data/raw/huggingface`. The UCI archive
also contains unrelated files (`data.csv`, `test.csv`, and an image); the loader
reads only the station archive inside it. Keep raw data,
processed data, model weights, and caches out of Git. Store experiment exports in
`outputs/`.

## Find the analysis

| Source | Purpose |
|---|---|
| [Book introduction](book/index.md) | Analytical questions and chapter navigation |
| [Time-series EDA](book/01_time_series_eda.ipynb) | Timestamp audit, temporal patterns, ACF/PACF, rolling statistics |
| [Time-series preprocessing](book/02_time_series_preprocessing.ipynb) | Gap filling, stationarity, decomposition, anomalies, forecasting, Granger extension |
| [Text EDA](book/03_text_eda.ipynb) | Label audit, document lengths, vocabulary, Zipf distribution |
| [Text preprocessing](book/04_text_preprocessing.ipynb) | Normalization, tokenization, ablations, representations, embedding extension |
| [Discussion](book/report.md) | Evidence-based comparisons and limitations |
| [Shared Python package](src/preprocessing/) | Project paths and `RANDOM_STATE`; dataset download, audit, and split helpers, tested in `tests/` |

Follow [AGENTS.md](AGENTS.md) for development and evaluation rules and the
[writing guide](docs/WRITING_GUIDE.md) for documentation. The assignment marks
Granger and Sentence Transformer comparisons as advanced but also includes them in
the rubric; account for both references when assessing submission coverage.

## Build and preview

[Jupyter Book 2](https://jupyterbook.org/stable/) builds the site from
`book/myst.yml`. The first HTML build installs the theme's web dependencies.

```powershell
$env:PYTHONUTF8 = "1"
uv run --locked --directory book jupyter book build --html --execute --strict --execute-parallel 1 --ci
```

In PowerShell, `PYTHONUTF8` makes Python use UTF-8 for file and console I/O on
Windows, so punctuation and equation symbols in notebook output are not corrupted.
Keep it set for builds and previews in that shell.

HTML is written to `book/_build/html/`. `--execute` runs notebook code and
`--strict` fails on reported warnings or errors. MyST can reuse execution caches.
After changing data, imported helpers, or dependencies, clear the cache before
running the build command again:

```powershell
uv run --locked --directory book jupyter book clean --execute --yes
```

For a live preview:

```powershell
uv run --locked --directory book jupyter book start --execute
```

To inspect the built static site, build with `BASE_URL` unset, then serve it:

```powershell
uv run --locked python -m http.server 8000 --directory book/_build/html
```

Open <http://localhost:8000>. Check equations, references, tables, and displayed
outputs. Before submission, execute the full book with the actual datasets from a
fresh environment. A passing scaffold build does not establish completed analysis.

See [Book authoring](docs/book-authoring.md) for navigation and chapter rules.

## Publish on GitHub Pages

[pages.yml](.github/workflows/pages.yml) installs the locked environment, lints and
tests shared Python code, executes notebooks, and builds strict HTML. Pull requests
validate the build. Pushes to `main` and manual workflow runs deploy after a
successful build. CI uses a fresh checkout without execution caches.

Select GitHub Actions as the source in repository Settings → Pages. The workflow
derives `BASE_URL` from the repository name for project-site links and assets. Use
an empty `BASE_URL` if hosting at a domain's root instead.

The workflow syncs the `embeddings` extra so the Sentence Transformer section
executes in CI. Dataset, NLTK, and model downloads happen inside the notebooks, so
a fresh runner reproduces the analysis without manual steps.

Inspect runs or request a deployment with:

```powershell
gh run list --workflow pages.yml
gh workflow run pages.yml
```

## Maintain the environment

```powershell
uv run --locked ruff check src tests
uv run --locked pytest
uv export --locked --no-hashes --no-emit-project --extra embeddings --output-file requirements.txt
```

Commit `pyproject.toml`, `uv.lock`, and the regenerated `requirements.txt` together
when dependencies change. The export includes development tools and excludes the
local package; uv remains the installation path that also installs that package.
