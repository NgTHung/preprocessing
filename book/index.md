# Data Preprocessing: Time Series and Text

**Student:** TODO: name and student ID.

This book is a starter for the data preprocessing assignment. Part (a) uses time
series and part (b) uses text. The notebooks contain environment setup, data
acquisition examples, and analysis prompts. Experiments and conclusions are still
to be implemented; there are no measured results yet.

## Data selection

- **Time series:** proposed [PJM hourly energy consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).
  Start with `PJME_hourly.csv`; verify its time span covers at least two years.
  Document the timestamp convention, duplicates, and gaps before regularizing it.
- **Text:** proposed [IMDb movie reviews](https://huggingface.co/datasets/stanfordnlp/imdb).
  Use the labeled training and test splits, with at least 10,000 labeled reviews
  and both sentiment labels. Exclude the unlabeled split from supervised metrics.

## Reproducibility

Try the [interactive preprocessing examples](interactive.md) to explore rolling
windows and text normalization with reactive controls.

Use the environment locked in `uv.lock`, a random seed of 42, and the commands in
the repository README. Keep raw data outside version control. Data acquisition
examples are initially commented out so the starter book builds without downloads.
Once you choose the datasets, enable those cells and implement the analyses before
submission.

For each experiment, record the theory and formula, implementation, numeric
results in Python variables, and an interpretation of the observed results.
