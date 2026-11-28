# Results and discussion

Preprocessing choices need evidence from comparable experiments. This chapter
connects the time-series and text results to those choices and states the limits of
each conclusion. All four analysis chapters are executed. Scores live in the notebooks, and this page
links to the tables that own them.

## Time series

The [time-series EDA](01_time_series_eda.ipynb) chose Wanshouxigong by gap length,
kept native gaps on the hourly grid, and took the candidate AR order from BIC because
the pointwise PACF band flags coefficients too small to matter at this sample size.
The station choice depends on the rule (`rule_comparison`), so every fitted parameter
in the preprocessing chapter belongs to this station only.

The [preprocessing chapter](02_time_series_preprocessing.ipynb) supports these
choices:

- Gap filling. Linear interpolation wins the pre-declared rank rule (`fill_selection`).
  On isolated artificial gaps several methods tie; on block gaps shaped like the native
  ones, linear interpolation is clearly best and higher-order fits overshoot
  (`gap_fill_comparison`). Reconstruction errors are autocorrelated for almost every
  method. Linear interpolation reads later observations, so it completes the training
  series for the statistical tests only; forecast inputs use a causal forward fill.
  Filling leaves the targets' screened and BIC lag orders unchanged
  (`chosen_fill_dependence_check`).
- Stationarity. ADF, KPSS, and Phillips-Perron agree that both targets are
  level-stationary as observed, while every covariate needs one difference
  (`stationarity_choice`). The tests do not address the changing variance seen in the
  EDA, and the Box-Cox fit for pressure is numerically degenerate.
- Decomposition. The lowest residual variance ratio belongs to a non-robust STL whose
  seasonal component overfits (strongly negative residual ACF at 24 hours). Robust STL
  keeps episodes in the residual and is the basis for the anomaly section; the weekly
  MSTL component absorbs episodes rather than a weekly cycle (`decomposition_comparison`).
- Anomalies. The three detectors flag different hours with low Jaccard overlap
  (`anomaly_overlap`). Episodes confirmed by two detectors include the nights of Lunar
  New Year and the Lantern Festival, which fits a fireworks hypothesis the data cannot
  confirm (`anomaly_episodes`).
- Granger tests. Nearly every pair is significant after Benjamini-Hochberg correction,
  so the chapter reports effect sizes and draws only edges that remove at least 1% of
  the residual sum of squares (`granger_results`). The edges are predictive, linear,
  and may reflect shared drivers such as the daily cycle.
- Forecasting. On the test interval the Random Forest improves on persistence by a
  few percent for PM2.5 and by about one percent for PM10 (`forecast_skill`). On
  validation the selected PM10 configuration is worse than persistence, and the
  ablation shows that the long rolling windows hurt both targets while the covariates
  help most (`forecast_ablation`). The test interval was scored once with the declared
  protocol, and the model was not adjusted afterwards.

Open questions remain. A feature-group selection step declared in advance, a test of
whether the forecast differences are significant (for example Diebold-Mariano), and a
second station would each strengthen or weaken these conclusions. The Granger section
is advanced in the brief and is evaluated here with the limits above.

## Text

The [text EDA](03_text_eda.ipynb) describes the raw published text, so its counts
include encoding residue. It supports these points for the preprocessing chapter:

- Data quality. The splits are balanced and free of empty, repeated, or exactly shared
  texts, but about one test article in nine has a training article at TF-IDF cosine
  0.9 or above, and roughly a tenth of those pairs carry different topics, mostly
  Business against Sci/Tech (`agreement_by_similarity`, `conflict_pairs`). Test
  scores therefore include some near-copies of training text, and label noise caps
  attainable accuracy. Near-duplicates within the training split were not searched.
- Markup. About a quarter of articles contain entities that lost their `&`, a tenth
  contain literal backslashes, and escaped Reuters links contribute frequent link
  fragments; mentions and hashtags are essentially absent (`markup_prevalence`). Wire
  tags differ strongly by topic and could act as a source shortcut.
- Length. All topic pairs differ significantly, but every rank-biserial effect is
  small, so length is a weak topic signal (`length_tests`).
- Vocabulary. Function words and residue tokens fill the raw top-50 tables, while
  content tokens separate the topics clearly. At fixed token budgets, Sci/Tech has the
  highest type-token ratio and Business the lowest (`budget_ttr`).
- Zipf. With log-spaced ranks the exponent is close to 1, but an all-rank fit gives a
  much larger value and the curve bends away from a single power law at both ends
  (`zipf_fit_comparison`, `zipf_by_rank_range`).

The [preprocessing chapter](04_text_preprocessing.ipynb) scores every variant on the
same 30,000 training articles and five folds, with learned transforms fitted inside the
folds, and reads the test split only for tokenizer OOV and one final evaluation. It
supports these conclusions:

- Normalization. Lowercasing and removing digits and symbols shrink the vocabulary to
  under half its raw size and give the only clear classification gain
  (`normalization_effects`, `normalization_cv`). Repairing entities, stripping HTML,
  and splitting backslashes clean about a third of articles but slightly lower
  macro-F1 on their own, because escaped stock-quote links were topic cues for
  Business and Sci/Tech. Removing wire tags and datelines lowers macro-F1 on all five
  folds, so the source shortcut found in the EDA is real, though smaller than the gain
  from normalization. Scores with the markers kept overstate accuracy on text from
  other sources.
- Tokenization. Word-level OOV on the test split is small by tokens and large by
  types, as Zipf's law predicts; characters and a 30,000-entry BPE vocabulary have no
  unknown units, and BPE splits few words. The sentence-level rate measures repeated
  sentences, not coverage, and is not comparable with the others
  (`tokenization_comparison`).
- Stop words. Removal shortens articles a lot and the vocabulary very little, leaves
  mean MI almost unchanged, and helps Naive Bayes but not logistic regression beyond
  fold noise. Several stop words (`its`, `his`, `he`) rank among the most informative
  terms (`stop_word_summary`, `stop_word_mi`, `stop_word_cv`).
- Stemming and lemmatization. Porter and Snowball merge about half of the word types,
  including unrelated words such as `general` and `generous`; WordNet merges fewer and
  more accurately. No method changes classification beyond fold noise at this
  training size (`collision_table`, `stemming_cv`).
- Representations. Silhouette and same-topic cosine favor averaged Word2Vec vectors,
  while sparse unigram TF-IDF classifies best; bigrams and trigrams add features faster
  than evidence. Separation measured in a feature space and accuracy of a linear model
  are different properties (`representation_summary`, `representation_cv`).
- Sentence Transformer (advanced). Pretrained `all-MiniLM-L6-v2` embeddings, pinned to
  a model revision, cluster by topic far better than TF-IDF and classify better with
  `LinearSVC` when both are trained on the same 8,000 articles (`clustering_comparison`,
  `embedding_cv`). No test article was truncated.
- Final evaluation. The pipeline selected by the pre-declared rule (full normalization,
  unigram TF-IDF, no stop-word removal or stemming) beats raw text on the test split in
  the same direction as cross-validation, and excluding test articles with a near-copy
  in training barely changes any score (`final_results`, `final_per_class_f1`).

Limits of these conclusions: the comparisons use a 30,000-article subset for runtime,
and the Sentence Transformer comparison an 8,000-article subset because embeddings are
computed on CPU in CI, so the ranking of TF-IDF and embeddings may change with more
labeled data. Choices were made greedily one decision at a time, so interactions
between steps were not searched. Each test score comes from one fit on one split.
Part-of-speech tags for the WordNet lemmatizer come from lowercase text without
punctuation, which lowers their accuracy.

## Limitations and reproducibility

Link to the notebook cells or rendered tables that own the evidence rather than
copying scores into another table. Identify the dataset revisions, split definitions,
seed configuration, environment, and any subsampling that bounds the conclusions.
Report measured runtimes with their hardware and timing boundaries when relevant.

Cite dataset cards and method references. Record AI assistance and describe how
the implementation and interpretations were checked. List any unexecuted section,
unmet requirement, or reproduction step that needs external access.

AI assisted the time-series EDA and preprocessing implementations and narratives,
including the gap-mask, gap-filling, Ljung-Box, rolling-window, and wind-component
helpers. Its evidence remains in executable notebook cells, with focused checks for
lag spacing, pair counts, covariance normalization, mask selection, fill behavior,
agreement with the statsmodels Ljung-Box statistic, trailing-window alignment, and the preprocessing handoff.
AI also assisted the text EDA implementation and narrative, including the
descriptive tokenizer, pairwise Mann-Whitney and rank-biserial helper, fixed-budget
TTR sampler, Zipf fit, and the near-duplicate search, each with focused tests.
AI also assisted the text preprocessing implementation and narrative, including the
normalization steps, OOV, collision-rate, mutual-information, sparsity, and pair
sampling helpers, the fold-local Word2Vec transformer, and the CPU-only embedding
setup. Their tests cover entity repair on AG News patterns, agreement with the
scikit-learn mutual information score, empty and all-OOV inputs, label constraints on
sampled pairs, and Word2Vec reproducibility across processes. This
does not establish
student understanding or completed human review; the student must be able to
explain the submitted code and statistical limitations at the oral exam.
