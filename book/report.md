# Results and discussion

Preprocessing choices need evidence from comparable experiments. This chapter
provides prompts for connecting the time-series and text results to those choices
and explaining the limits of each conclusion. Time-series EDA has executed
evidence, while preprocessing comparisons, forecasting, and text experiments remain
unfinished. The cross-method discussion below remains a scaffold for those results.

## Time series

The [time-series EDA](01_time_series_eda.ipynb) retains native gaps on the hourly
grid and reports observed endpoint pairs for each ACF lag. Its linear-interpolation
sensitivity comparison shows why agreement in a screened lag order does not imply
agreement in every PACF coefficient. At this sample size the pointwise PACF band
flags coefficients too small to matter, so the chapter takes the candidate AR order
from BIC on one common sample of complete lag windows and keeps the band rule as a
diagnostic. Candidate orders require another comparison
after notebook 02 selects its actual filling method; that check remains pending.
STL, Ljung–Box, and Granger inputs will still need a documented complete-series
treatment. Their results cannot be attributed to observed data alone after filling.

Use the [time-series audit](01_time_series_eda.ipynb) to justify the dataset,
sampling frequency, timestamp convention, and seasonal periods. Identify which
patterns are observed and which explanations remain assumptions. Explain the
station-selection rule, whether an alternative rule chooses the same station, and
the limits of applying one station's results to the other 11.

Use the [preprocessing experiments](02_time_series_preprocessing.ipynb) to address:

- Gap filling: compare MAE/RMSE on the shared artificial mask and interpret residual
  autocorrelation. State which methods are suitable only for retrospective use.
- Stationarity: interpret ADF, KPSS, and Phillips-Perron results before and after
  transformations, accounting for their different null hypotheses.
- Decomposition: compare residual variance ratios over the same valid interval and
  explain the chosen seasonal period.
- Anomalies: discuss detection rates and overlap, separating flagged observations
  from verified explanations for them.
- Forecasting: compare Random Forest with persistence for PM2.5 and PM10 on the
  untouched chronological test set. Explain feature availability, target alignment,
  covariate choice, and the use of test history, and discuss where the targets differ.

The advanced Granger section is also named in the rubric. If evaluated, discuss lag
orders, stationarity, multiple comparisons, effect sizes at large sample sizes,
pairwise versus conditional tests, shared weather drivers, and the distinction
between predictive relationships and real-world causation. If omitted, state the
coverage limitation.

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

AI assisted the time-series EDA implementation and narrative, and the time-series
preprocessing protocol, gap-mask, gap-filling, Ljung–Box, and wind-component
helpers. Its evidence remains in executable notebook cells, with focused checks for
lag spacing, pair counts, covariance normalization, mask selection, fill behavior,
agreement with the statsmodels Ljung–Box statistic, and the preprocessing handoff.
This does not establish
student understanding or completed human review; the student must be able to
explain the submitted code and statistical limitations at the oral exam.
