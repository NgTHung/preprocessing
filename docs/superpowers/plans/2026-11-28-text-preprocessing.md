# Text Preprocessing Chapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete `book/04_text_preprocessing.ipynb` with every part (b) text technique, measured under one protocol, and executed in CI. The advanced Sentence Transformer section is included.

**Architecture:** A new `src/preprocessing/text_preprocessing.py` holds the stateless normalization steps, the metric helpers, and a fold-local Word2Vec transformer, each with tests. The notebook keeps the protocol visible (subsets, folds, pipelines) and imports only the mechanics. CI installs the `embeddings` extra with CPU-only torch.

**Tech Stack:** Python 3.12, uv, pandas, scikit-learn, NLTK, HF `tokenizers`, gensim, sentence-transformers (CPU torch), Jupyter Book 2.

**Spec:** `docs/superpowers/specs/2026-11-28-text-preprocessing-design.md`

## Global Constraints

- `RANDOM_STATE` comes from `preprocessing`. Paths come from `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, and `OUTPUT_DIR`.
- `cv_subset` = 30,000 stratified training articles (7,500 per topic). `probe_sample` = 4,000 stratified training articles disjoint from `cv_subset`. `folds` = `StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)`.
- Test (7,600) is used only for tokenizer OOV and the final evaluation.
- Macro-F1 is the primary metric. Every score is kept in a named variable and displayed.
- Notebook 04 targets ≤ 10 minutes on CI (4 vCPU).
- Commits use Conventional Commit messages.

## Review Focus

- Text that contains only markup or digits must normalize to `""` without crashing. Tested in Task 2.
- Nested escaping such as `&amp;lt;` must decode fully, and `#36;` must become `$`. Tested in Task 2.
- A document whose tokens are all out of vocabulary must give the zero vector from `MeanWordVectors`, not NaN. Tested in Task 3.
- `sparsity` on an empty matrix must raise `ValueError`. Tested in Task 3.
- Two concurrent downloads of the air-quality archive must both succeed. Tested in Task 1.

---

### Task 1: Fix the air-quality download race

**Files:** Modify `src/preprocessing/air_quality.py` (`download_air_quality`). Test: `tests/test_air_quality.py`.

- [ ] Test `test_download_tolerates_concurrent_writer`. Two threads call `download_air_quality` against the same `file://` source. Both must return the verified path, and no `*.part` files may remain.
- [ ] Run the test and confirm it fails or is flaky.
- [ ] Implement: download to `tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False)`, verify the checksum, `os.replace` into place, and unlink the temporary file in `finally`.
- [ ] Run `uv run --locked pytest tests/test_air_quality.py` and confirm it passes.
- [ ] Commit `fix: make air-quality download safe under parallel notebook execution`.

### Task 2: Normalization steps

**Files:** Create `src/preprocessing/text_preprocessing.py`. Test: `tests/test_text_preprocessing.py`.

**Produces:**

- `repair_entities(text: str) -> str`
- `strip_html(text: str) -> str`
- `replace_backslashes(text: str) -> str`
- `remove_urls(text: str) -> str`
- `strip_mentions_hashtags(text: str) -> str`
- `lowercase(text: str) -> str`
- `remove_digits_and_symbols(text: str) -> str`
- `collapse_whitespace(text: str) -> str`
- `remove_wire_tags(text: str) -> str`
- `NORMALIZATION_STEPS: list[tuple[str, Callable[[str], str]]]`, in the spec order
- `normalize(text: str, steps=NORMALIZATION_STEPS) -> str`

**Test values:**

- `repair_entities("Haiti #39;s quot;x quot; &amp;lt;b&amp;gt; #36;5")` returns `"Haiti's \"x \" <b> $5"`.
- `strip_html("see <A HREF=\"x\">Ford</A> now")` returns `"see Ford now"` after whitespace collapse.
- `replace_backslashes("Friday\\but")` returns `"Friday but"`.
- `remove_urls("go http://a.com/x?y and www.b.org.")` gives `"go  and ."` once collapsed.
- `strip_mentions_hashtags("@bob loves #Tech")` returns `" loves Tech"`.
- `remove_digits_and_symbols("in 2004, mp3 sales (3g) rose 5% -- it's well-known")` returns `"in  mp3 sales 3g rose  it's well-known"`, modulo whitespace.
- `remove_wire_tags("LONDON (Reuters) - x (AP) y (AFP)")` collapses to `"LONDON - x y"`.
- `normalize("&lt;p&gt; 123 &lt;/p&gt;")` returns `""`.

**Steps:**

- [ ] Write the tests, then implement.
- [ ] Run the pytest file and ruff.
- [ ] Commit `feat: add text normalization steps for AG News`.

### Task 3: Metric helpers and the Word2Vec transformer

**Files:** Same module and test file as Task 2.

**Produces:**

- `stratified_sample(frame, labels: str, per_label: int, rng: np.random.Generator, exclude: pd.Index | None = None) -> pd.DataFrame`
- `vocabulary_change(before: set[str], after: set[str]) -> pd.Series` with `types_before`, `types_after`, `removed_share`, and `new_share`. Shares use the denominator `types_before`.
- `oov_rate(train_units: Iterable[Hashable], test_docs: Iterable[Sequence[Hashable]]) -> pd.Series` with `test_units`, `oov_units`, `token_rate`, `test_types`, `oov_types`, and `type_rate`.
- `collision_rates(types: Mapping[str, int], mapping: Callable[[str], str]) -> pd.Series` with `types`, `roots`, `collision_rate` (share of types whose root is shared), `reduction` (`1 - roots/types`), and `token_collision_rate`.
- `presence_mutual_information(X: sparse, y: np.ndarray) -> np.ndarray`, in nats, one value per column. Presence is `X > 0`.
- `sparsity(X) -> float` computed as `1 - nnz/(n*d)`. Raises `ValueError` if `n*d == 0`.
- `sample_label_pairs(labels: np.ndarray, n_pairs: int, same: bool, rng) -> np.ndarray` of shape `(n_pairs, 2)`, with no self-pairs.
- `MeanWordVectors(BaseEstimator, TransformerMixin)` with parameters `vector_size=100, window=5, min_count=2, epochs=5, sg=1, seed=RANDOM_STATE`. `fit(docs: list[list[str]])` trains gensim with `workers=1` and a `hashfxn` based on crc32. `transform` returns a dense float32 mean vector per document, or zero when every token is OOV. The `n_all_oov_` attribute is set by the last `transform` call.

**Test values:**

- `vocabulary_change({"a","b","c","d"}, {"a","b","e"})` gives `removed_share == 0.5` and `new_share == 0.25`.
- `oov_rate(["a","b"], [["a","c"],["c","b","d"]])` gives `token_rate == 3/5` and `type_rate == 2/4`.
- `collision_rates({"run":3,"runs":1,"ran":1,"cat":5}, {"run":"run","runs":"run","ran":"ran","cat":"cat"}.get)` gives `collision_rate == 0.5`, `reduction == 0.25`, and `token_collision_rate == 0.4`.
- `presence_mutual_information` matches `sklearn.metrics.mutual_info_score` per column on a random 50×6 binary matrix to within 1e-12.
- `sparsity(csr_matrix((0, 3)))` raises.
- For `sample_label_pairs`, every same pair has equal labels and every different pair has unequal labels.
- `MeanWordVectors` gives a zero row for `[["zzz"]]`, `n_all_oov_ == 1`, and identical output from two fits.

**Steps:**

- [ ] Write the tests, then implement.
- [ ] Run pytest and ruff.
- [ ] Commit `feat: add text preprocessing metric helpers`.

### Task 4: Dependencies and CI

**Files:** `pyproject.toml`, `uv.lock`, `requirements.txt`, `.github/workflows/pages.yml`, `README.md`.

- [ ] Add a `[[tool.uv.index]]` named `pytorch-cpu` at `https://download.pytorch.org/whl/cpu` with `explicit = true`, and `[tool.uv.sources] torch = { index = "pytorch-cpu" }`.
- [ ] Run `uv lock`. Confirm the lock has no `nvidia-*` packages, and that `uv sync --locked --extra embeddings` imports `sentence_transformers`.
- [ ] Regenerate `requirements.txt` with the README export command plus `--extra embeddings`.
- [ ] Change the CI sync step to `uv sync --locked --extra embeddings`. Update the README setup section.
- [ ] Commit `build: install CPU-only embeddings extra in CI`.

### Task 5: Notebook sections (setup, data, normalization, tokenization)

**Files:** `book/04_text_preprocessing.ipynb`.

- [ ] Setup: package versions table, NLTK downloads into `RAW_DATA_DIR / "nltk_data"`, `load_ag_news`, `cv_subset`, `probe_sample`, and `folds`. Assert the sample sizes and disjointness.
- [ ] Normalization: per-step table on full train (whitespace-split, case-sensitive), length table and plot, examples. CV ablation (`TfidfVectorizer` + `LogisticRegression(max_iter=1000)`) on raw, normalized, and normalized without wire tags, with paired differences.
- [ ] Tokenization: a table for the four strategies plus a word variant with min frequency 2, and BPE fertility.
- [ ] Execute the notebook with `jupyter nbconvert --execute` and inspect the outputs. Write the interpretations from those outputs, with numbers via `{eval}`.

### Task 6: Notebook sections (stop words, stemming and lemmatization)

- [ ] Stop words: the NLTK list and its negation words. Vocabulary sizes, a mean MI table (all terms, df ≥ 5, stop words), and NB CV paired differences.
- [ ] Stemming and lemmatization: a collision table, the largest merge groups, and LR CV paired differences against "none".
- [ ] Execute, then interpret.

### Task 7: Notebook sections (representations, Sentence Transformer, final test)

- [ ] Representations: a dimension and sparsity table, the cosine pair table with rank-biserial, silhouette (cosine), t-SNE figures, and LR CV for each representation.
- [ ] Sentence Transformer: pinned model revision, a truncation share, K-Means silhouette and ARI, LinearSVC CV against the best TF-IDF, and disagreement examples.
- [ ] Final test: three pipelines and the near-duplicate sensitivity row.
- [ ] Execute, interpret, and measure the notebook runtime.
- [ ] Commit `feat: execute AG News text preprocessing chapter`.

### Task 8: Report, README, validation, PR

- [ ] Rewrite the `book/report.md` text section so it links to the notebook evidence, and add the AI-assistance line. Update the README status paragraph.
- [ ] Run ruff and pytest, then the strict executed book build with the cache cleared.
- [ ] Commit `docs: describe executed text preprocessing chapter`.
- [ ] Push and open the PR, then link it.
