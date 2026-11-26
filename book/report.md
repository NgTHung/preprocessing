# Results and discussion

Preprocessing choices need evidence from comparable experiments. This chapter
connects the time-series and text results to those choices and states the limits of
each conclusion. Both time-series chapters are executed; the text experiments remain
unfinished, so the text section below is still a scaffold. Scores live in the
notebooks, and this page links to the tables that own them.

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

Use the [text audit](03_text_eda.ipynb) to interpret document lengths, vocabulary,
and the Zipf plot. Account for counting conventions and sample-size effects before
attributing differences to labels.

Use the [preprocessing experiments](04_text_preprocessing.ipynb) to address:

- Normalization: explain measured vocabulary and length changes, including any
  topic cues removed by digit, punctuation, or HTML-entity handling.
- Tokenization: compare vocabulary size, token lengths, and OOV rates under explicit
  unit and denominator definitions.
- Ablations: interpret the effect of stop-word removal, stemming, and lemmatization
  under the same validation protocol.
- Representations: compare dimensions, sparsity, cosine similarities, and silhouette
  scores. Distinguish visual patterns in t-SNE from evidence in the feature space.

The advanced Sentence Transformer section is also named in the rubric. If evaluated,
compare its K-Means and Linear SVM results with TF-IDF under the same splits, and
discuss truncation and model revision. If omitted, state the coverage limitation.

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
This does not establish
student understanding or completed human review; the student must be able to
explain the submitted code and statistical limitations at the oral exam.
