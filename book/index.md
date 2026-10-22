# Data preprocessing: Time series and text

Preprocessing changes both the information available to a model and the conclusions
an analyst can draw. This book provides experiment scaffolds for examining those
changes in time-series forecasting and text classification. Its chapters organize
the questions, comparisons, and evidence required by the course assignment.

Student name and student ID: TODO.

The dataset analyses are incomplete. Acquisition examples are commented out,
analysis cells are prompts, and no dataset experiment results are included.

## Questions and chapters

| Question | Chapters |
|---|---|
| What structure and data-quality problems affect the series? | [Time-series EDA](01_time_series_eda.ipynb) |
| How do gap filling, transformations, and features affect forecasting? | [Time-series preprocessing](02_time_series_preprocessing.ipynb) |
| How do document lengths and vocabulary differ across labels? | [Text EDA](03_text_eda.ipynb) |
| What information do normalization, tokenization, and representations retain? | [Text preprocessing](04_text_preprocessing.ipynb) |
| Which choices are supported by the experiments? | [Results and discussion](report.md) |

Part (a) uses hourly Beijing Multi-Site Air Quality readings, and part (b) uses AG
News topic-labeled articles.
The repository README lists their public sources and setup instructions. The EDA
chapters contain acquisition examples and prompts for recording provenance,
checking constraints, and defining evaluation splits.

## Explore a single preprocessing choice

The [interactive examples](interactive.md) show how a rolling window changes a
synthetic series and how normalization changes sample text. Their controls help
explain individual operations; they do not replace experiments on the chosen data.

## Read the evidence

Each completed technique needs theory and formulas, executable code, displayed
measurements, and an interpretation. Forecasting comparisons use chronological
splits and information available at prediction time. Text comparisons fit learned
preprocessing on training data, including within validation folds. The discussion
chapter provides prompts for comparing results and identifying limitations.
