# Text preprocessing chapter design (2026-11-28)

## Intent

Complete `book/04_text_preprocessing.ipynb`, part (b) of the assignment, for AG News. Every technique the brief lists for text gets theory, an implementation, measured output in named variables, and an interpretation grounded in the [text EDA](../../../book/03_text_eda.ipynb). That includes the advanced Sentence Transformer comparison, which the rubric also names. The chapter has to execute in CI within the workflow's 30-minute job limit, alongside the other three notebooks.

## Goals

- Cover each required item from `docs/preprocessing.md` § "Dữ liệu văn bản": the normalization pipeline, four tokenizers with test OOV, stop-word removal (vocabulary, MI, Naive Bayes), Porter/Snowball/WordNet with collision rates and Logistic Regression 5-fold CV, BoW/TF-IDF(n ≤ 1, 2, 3)/Word2Vec with dimension, sparsity, cosine, t-SNE and silhouette, and Sentence Transformer vs TF-IDF using K-Means silhouette and Linear SVM.
- Run every preprocessing alternative under one shared protocol: the same subset, the same folds, and fold-local fitting of learned transforms.
- Use the published test split once, for the pipelines fixed before it is opened.
- Update `book/report.md` (text section), README, and CI to match.

## Non-goals

- Tuning classifiers beyond defaults plus fixed, documented settings. The chapter compares preprocessing, not models.
- Deep-learning classifiers, or fine-tuning the Sentence Transformer.
- Removing near-duplicate test articles. The test split stays intact, matching the EDA policy.

## Anti-goals

- Test data that influences any choice: vocabulary, BPE merges, Word2Vec, thresholds, or the selected pipeline.
- Scores typed into prose. All numbers come from `{eval}` or displayed variables.
- An extension that executes only locally. If CI does not run the Sentence Transformer section, the rubric row is not met.

## Constraints (from the repository)

- Dataset: `fancyzhx/ag_news` at revision `eb185aa…`, through `load_ag_news`. Train has 120,000 articles and test 7,600, with 4 balanced topics.
- EDA findings this chapter acts on: about 25% of articles contain entities without their `&` (`#39;`, `#36;`, `quot;`), about 10% contain literal backslashes, about 1–2% contain escaped `<A HREF>` links, and URLs appear in under 2%. Mentions and hashtags are almost absent. Wire tags (`(Reuters)`, `(AP)`, `(AFP)`) depend on topic. About 11% of test articles have a training neighbour at TF-IDF cosine ≥ 0.9, and some of those pairs carry conflicting labels.
- Allowed libraries: those in `pyproject.toml`. `sentence-transformers` is in the `embeddings` extra.
- CI: `ubuntu-latest` (4 vCPU), 30-minute job limit. The last green run took about 8–11 minutes.
- Measured locally (20 cores): TF-IDF + LR 5-fold CV takes 4 s on 30k articles and 14 s on 120k. The trigram TF-IDF + LR takes 18 s on 30k and 126 s on 120k. Word2Vec on 120k with 1 worker takes 38 s. BPE training takes 3 s. Porter on 67,670 types takes 1 s.

## Approaches considered (Sentence Transformer)

- **A. Run it in the book (chosen by the student).** CI installs the `embeddings` extra with CPU-only torch and pins the model revision. Embeddings are computed for seeded subsets. Cost: dependency and CI changes, plus a model download of about 90 MB per run.
- B. Guard the cell so it runs only when the extra is installed. Rejected: the published book would show "not evaluated".
- C. Skip it and record it as not evaluated. Rejected: this gives up a rubric row.

## Shared protocol

| Set | Definition | Used for |
|---|---|---|
| `cv_subset` | Seeded stratified sample of 30,000 training articles (7,500 per topic), `RANDOM_STATE` | All classifier comparisons except the embedding one; fitting representations for descriptive stats |
| `folds` | `StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)` over `cv_subset`; one object is reused | Every CV score |
| `probe_sample` | Seeded stratified 2,000 training articles **disjoint from `cv_subset`** (500 per topic) | Cosine pairs, t-SNE, silhouette, K-Means |
| `st_subset` | Seeded stratified 8,000 articles from `cv_subset` (2,000 per topic), with its own 5 folds | Sentence Transformer vs TF-IDF `LinearSVC` CV; CPU encoding of all 30k would exceed the CI budget |
| full train | 120,000 articles | Normalization statistics, tokenizer vocabularies, BPE, collision rates |
| test | Published 7,600 articles | Tokenizer OOV (required by the brief), plus one final evaluation |

- The 30k subset is a runtime choice. The notebook states it and shows that the full-train baseline score is within about 1.5 points (measured locally: 0.917 vs 0.903 LR accuracy). The comparison is between preprocessing variants on identical folds, so the absolute level matters less than paired differences.
- Metrics: macro-F1 is primary and accuracy is also reported. Classes are balanced, so the two track closely. Results are reported per fold, as a mean ± standard deviation, and as the **paired per-fold difference** against the stated baseline.
- Deterministic, stateless transforms (normalization, stemming, lemmatization) are applied once, before CV. They learn nothing from the data, so this does not leak. Learned transforms (vectorizers, BPE, Word2Vec, SVD) are fitted inside the pipeline or loop for each fold.
- The descriptive statistics in the vectorization and embedding sections fit on `cv_subset` and are measured on `probe_sample`. Those statistics are out-of-sample without opening the test split.
- NLTK resources (`stopwords`, `wordnet`, `omw-1.4`, `punkt_tab`, `averaged_perceptron_tagger_eng`) are downloaded into `RAW_DATA_DIR / "nltk_data"`, which is prepended to `nltk.data.path`. The notebook prints the NLTK version and the list of resources.

## Sections

### 1. Normalization pipeline

The pipeline has ordered cumulative steps, and each is a pure function in the helper module:

1. **Repair entities.** Restore the missing `&` (`#39;` → `&#39;`, bare `quot;`/`lt;`/`gt;`/`amp;`), then `html.unescape` until the text stops changing.
2. **Strip HTML.** BeautifulSoup `get_text(" ")`. After step 1, escaped `<A HREF…>` links have become real tags.
3. **Replace backslashes** with a space, so `Friday\but` splits into two words.
4. **Remove URLs** (`https?://…`, `www.…`).
5. **Mentions and hashtags.** Mentions are dropped. Hashtags keep the word without `#`. The step is measured even though the effect is expected to be about zero.
6. **Lowercase.**
7. **Digits and special characters.** Standalone digit runs are removed. Alphanumeric tokens such as `mp3` and `3g` are kept, and apostrophes and hyphens inside words are kept. Every other non-letter, non-digit character becomes a space.
8. **Collapse whitespace.**

A separate, optional step removes wire tags, both the `(Reuters)` form and the `Reuters -` dateline that usually follows it. It is not in the default pipeline and is evaluated as its own ablation, because the EDA flags these tags as a label shortcut.

Measurements per step use a fixed, case-sensitive whitespace-split counting convention on full train:

- vocabulary size $V$;
- the share of the previous step's types that disappear;
- the share of new types;
- documents changed;
- the change in mean and median word and character length;
- quantile shifts and a before/after length plot.

The section also shows representative before/after examples and a CV ablation: TF-IDF + LR on raw text, on normalized text, and on normalized text without wire tags.

The writeup discusses negation and punctuation. For topic classification, negation carries little label signal, and the CV result tests that claim rather than assuming it.

### 2. Tokenization (fit on full train, OOV on test)

All four tokenizers run on cleaned text (steps 1–5 plus whitespace collapse), because sentence splitting needs the case and punctuation that steps 6–7 remove.

| Strategy | Unit and vocabulary | OOV definition |
|---|---|---|
| Word | NLTK `word_tokenize`; vocabulary = train types | Test tokens not in the train vocabulary / test tokens |
| Sentence | NLTK `sent_tokenize`; vocabulary = distinct train sentences | Test sentences not in train / test sentences |
| Character | Unicode code points; vocabulary = train characters | Unseen code points / test characters |
| Subword BPE | HF `tokenizers` BPE, vocabulary 30,000, `[UNK]`, `Whitespace` pre-tokenizer, trained on train | `[UNK]` tokens / test tokens |

Reported for each strategy: vocabulary size, mean tokens per document (train and test), test OOV token rate, and test OOV type rate. A word-level variant keeps only types with frequency ≥ 2 in train, to show how a frequency cutoff moves OOV. BPE fertility is also reported (mean pieces per word).

The prose explains why sentence "OOV" measures repeated sentences, not unknown units, so it is not comparable with the others. Empty documents are counted and handled: none are expected after normalization, and an assertion checks this.

### 3. Stop-word removal

- Stop-word list: NLTK English. The section lists the negation words it contains.
- Vocabulary: `CountVectorizer` with the same token pattern in both conditions, fitted on `cv_subset`, with and without stop words.
- MI: binary term presence against the topic label, computed in a vectorized way from the sparse contingency counts (helper, tested against `sklearn.metrics.mutual_info_score` on small inputs), in nats. Reported for each condition: mean MI over all terms, the mean over terms with document frequency ≥ 5, and the MI of the stop words themselves. This separates "removing low-MI terms raises the average" from "the remaining terms carry more information".
- Classifier: `MultinomialNB` on counts, 5-fold on `folds`, with and without stop words, reported as paired differences. Logistic regression on TF-IDF is also reported, because the final pipeline's selection uses LR.

### 4. Stemming and lemmatization

- Variants: none, Porter, Snowball (English), and WordNet lemmatizer.
- WordNet POS policy: in-context `nltk.pos_tag` mapped to WordNet POS, defaulting to noun. For type-level collision, each type's lemma comes from its most frequent tag in train.
- Collision rate: over the distinct lowercase alphabetic word types $T$ in normalized full train, with map $f$.
  - Primary: $|\{t \in T : |f^{-1}(f(t))| > 1\}| / |T|$, the share of types that share a root with another type.
  - Secondary: $1 - |f(T)|/|T|$, the vocabulary reduction.
  - Both are also given token-weighted.
- The section shows the largest merge groups and example false merges.
- Classifier: TF-IDF + `LogisticRegression` 5-fold on `folds`, with paired differences against "none". POS tagging runs only on `cv_subset` and train types, not on all 120k documents, to stay within the time budget.

### 5. BoW, TF-IDF, and Word2Vec

- Representations:
  - BoW unigram counts;
  - TF-IDF with cumulative n-gram ranges (1,1), (1,2), (1,3) and `min_df=2`;
  - Word2Vec trained on the training text, using skip-gram, 100 dimensions, window 5, `min_count=2`, 5 epochs, `workers=1`, `seed=RANDOM_STATE`. gensim 4 seeds vectors with its own generator, so `workers=1` and the seed make training deterministic without a custom hash function.
  - The document vector is the mean of in-vocabulary word vectors. Documents with no in-vocabulary word get the zero vector and are counted.
- Dimension and sparsity: $1 - \mathrm{nnz}/(nd)$ on `probe_sample`. An empty matrix raises an error. Dense Word2Vec is reported as about 0 sparsity, with zero vectors noted.
- Cosine similarity: a seeded 2,000 same-topic and 2,000 different-topic pairs from `probe_sample`. Reported: mean, median, the gap, and a Mann-Whitney rank-biserial effect between the two pair groups.
- Silhouette: cosine metric in the **original feature space** on 2,000 stratified `probe_sample` articles, using topic labels.
- t-SNE: 2D, the same 2,000 articles, seeded. Sparse inputs first go through `TruncatedSVD(50)` fitted on `cv_subset`. The figures are illustrative only.
- Classification: LR 5-fold on `folds` for each representation. Word2Vec is retrained inside each fold through an sklearn-compatible transformer in the helper module.

### 6. Advanced: Sentence Transformer vs TF-IDF

- Model: `sentence-transformers/all-MiniLM-L6-v2` at a pinned commit hash, cached under `RAW_DATA_DIR / "huggingface"`.
  - Inputs are the cleaned text used for tokenization, with case and punctuation kept, because the model was trained on natural text. `max_seq_length` is the model default (256 word pieces), and the share of inputs truncated is reported.
  - `normalize_embeddings=True`, mean pooling as packaged, CPU, batch size 64.
  - Encoding has no fitted state, so `st_subset`, `probe_sample`, and test are encoded once.
- K-Means (k = 4, `n_init=10`, seeded) on `probe_sample`, for each representation. Reported: silhouette of the cluster assignment (cosine), adjusted Rand index against the topics, and the topic-label silhouette.
- Classification: `LinearSVC` 5-fold on the `st_subset` folds, TF-IDF (the best n-gram range from section 5 by CV) vs embeddings, reported as paired differences.
- Discussion: show example pairs where the two similarities disagree.

### 7. Final test evaluation

The pipelines are fixed before this cell:

- the raw TF-IDF + LR baseline;
- the CV-selected preprocessing + TF-IDF + LR. The selection is greedy across sections 1, 3, 4 and 5. Each section takes the variant with the highest mean CV macro-F1, except that a variant whose paired gain over the simpler option is smaller than the fold standard deviation of that difference keeps the simpler option.
- the Sentence Transformer + LinearSVC.

Each is trained on `cv_subset` and scored once on all 7,600 test articles. Reported: accuracy, macro-F1, and per-class F1. A sensitivity row repeats the scores on test articles without a training neighbour at cosine ≥ 0.9, using `nearest_train_similarity` from the EDA. The test split is never used to choose among these pipelines.

## Code layout

- `src/preprocessing/text_preprocessing.py` (new) holds:
  - the normalization step functions and the ordered `NORMALIZATION_STEPS`;
  - `remove_wire_tags`, `vocabulary_change`, `oov_rate`, `collision_rates`, `presence_mutual_information`, `sparsity`;
  - `MeanWordVectors` (an sklearn transformer wrapping gensim);
  - `sample_label_pairs` and `stratified_sample`.
  - Every public function has a docstring.
- `tests/test_text_preprocessing.py` covers each helper's behaviour and edge cases: broken entities, escaped tags, empty text, all-OOV documents, collision counting, MI against sklearn, sparsity on an empty matrix, and pair sampling that stays within label constraints.
- The notebook keeps the protocol visible (subsets, folds, pipelines, metrics) and imports only the mechanics.

## Dependencies and CI

- `pyproject.toml`:
  - torch comes from the PyTorch CPU index through `[tool.uv.sources]` with `[[tool.uv.index]] explicit = true`, which avoids about 2 GB of CUDA wheels on Linux.
  - Re-lock, then regenerate `requirements.txt` including the `embeddings` extra.
- `pages.yml`: `uv sync --locked --extra embeddings`.
- README: the setup command, NLTK and model downloads, and the model revision.
- Prerequisite fix, in its own commit: `download_air_quality` writes to a unique temporary file in the target directory and handles another process finishing first. A regression test covers this. This is the cause of the current red `main` build: notebooks 01 and 02 race on `uci_501.zip.part`.
- Runtime target: notebook 04 at most about 10 minutes on CI. If it exceeds that, the first lever is the Word2Vec epochs or the probe sizes, not the protocol.

## Report and attribution

- The `book/report.md` text section is rewritten to compare the techniques and link to the notebook tables and figures that hold the numbers. It contains no hand-copied scores.
- The AI-assistance note covers this chapter.
- Citations: Porter (1980), Snowball, WordNet, Sennrich et al. (2016) for BPE, Mikolov et al. (2013) for Word2Vec, Reimers & Gurevych (2019) for Sentence-BERT, and the `all-MiniLM-L6-v2` model card.

## Calls made by Claude (check these)

- A 30k CV subset instead of the full 120k.
- A disjoint training-derived probe sample for the descriptive statistics, instead of the test split.
- Wire-tag removal as a separate ablation, not a default step.
- Standalone numbers are removed and alphanumerics are kept.
- WordNet uses in-context POS, and type collision uses the most frequent tag.
- Word2Vec hyperparameters, BPE vocabulary of 30k, and `min_df=2` for TF-IDF.
- Macro-F1 is the primary metric.
- The final pipeline is selected greedily, and the simpler option wins unless the gain exceeds the fold noise.
- The CI download-race fix comes first, as a separate commit.

## Left to the builder

- Plot styling, table layout, and the exact example articles shown.
- Batch sizes and minor performance tuning that does not change results.
