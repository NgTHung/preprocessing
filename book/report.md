# Results and discussion

This report is a planning template. Replace the prompts with measured results and
your own reasoning after completing the notebooks.

## Part (a): Time series

- Justify the dataset, sampling frequency, timestamp convention, and seasonal periods.
- Compare imputation MAE/RMSE and residual autocorrelation.
- Interpret ADF, KPSS, and Phillips-Perron results before and after transformations.
- Compare additive, multiplicative, and STL residual variance ratios.
- Discuss anomaly overlap and plausible causes, without claiming unverified causes.
- Compare Random Forest forecast MAE/RMSE with a persistence baseline on an untouched
  chronological test set. Explain how each transformation avoids using future data.
- If implementing the advanced extension, interpret Granger tests and their limitations.

## Part (b): Text

- Interpret length differences, lexical richness, and the Zipf plot.
- Compare normalization effects, vocabulary sizes, OOV rates, and token lengths.
- Discuss stop-word removal and stemming/lemmatization using measured ablations.
- Compare representation dimensions, sparsity, cosine similarities, and silhouette scores.
- If implementing the advanced extension, compare Sentence Transformer embeddings
  against TF-IDF using K-Means and Linear SVM.

## Limitations and reproducibility

Record dataset versions, split definitions, seeds, hardware, runtimes, and remaining
limitations. Add references to dataset cards and the methods used. Record any AI
assistance and verify that you can explain the implementation during the oral exam.
