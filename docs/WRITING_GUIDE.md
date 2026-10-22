# Writing guide

This guide covers project documentation, notebook narrative, the book's discussion chapter, Python docstrings, and code comments. It helps readers understand why a preprocessing choice was made and how the evidence supports it. The [assignment](preprocessing.md) owns course requirements; [AGENTS.md](../AGENTS.md) owns development and evaluation rules.

## Start with the reader's question

Decide which question a page answers before writing it. Open with the problem and purpose in two to four sentences, then give the context needed to use or assess the work. Put task progress and change history in the tracker, commits, or dated records.

Weak opening:

> This is the setup for the text pipeline. The remaining cells will be completed later.

Useful opening:

> Text normalization changes which distinctions a sentiment classifier can learn. This chapter compares normalization and tokenization choices on labeled movie reviews, using a shared evaluation protocol to measure their effect.

Use that wording only when the chapter actually contains the comparison. If a page is a scaffold, label it clearly; do not turn intended behavior into a claim of completed work.

## Choose the document's job

| Type | Reader's question | Content |
|---|---|---|
| Root README | What is the project and how do I reproduce it? | Purpose, student name/MSSV, dataset descriptions and links, layout, environment/build commands, published book and external resources |
| Book introduction (`book/index.md`) | What analysis will I read and why? | Selected domains, dataset context, analytical questions, navigation |
| Analysis notebooks (`book/*.ipynb`) | Why use this method and what does the experiment show? | Acquisition, theory, formulas, implementation, displayed measurements, interpretation |
| Discussion (`book/report.md`) | Which choices does the evidence support? | Cross-method comparisons, links to notebook evidence, limitations, reproducibility, references |
| Interactive examples (`book/interactive.md`) | How does this parameter affect the demonstration? | Labeled synthetic/sample inputs, controls, expected behavior, explanation and limitations |
| Agent guide (`AGENTS.md`) | How should changes be made and validated? | Repository conventions, evaluation safeguards, verification and delivery rules |
| Docstring | How do I call this helper and what does it guarantee? | Inputs, outputs, units, edge cases, errors, small examples where useful |
| Dated record or task | What happened or what remains? | Date, scope, environment, commands, findings, limitations, status |

Keep setup instructions in the README, authoring mechanics in [Book authoring](book-authoring.md), and analytical detail in the book. A dated record preserves what was observed on its date; add another dated entry rather than silently updating it to imply a different historical result.

## Describe behavior accurately

Use present tense for implemented behavior. Avoid change-history phrases such as "now supports", "no longer uses", and "was removed" in reference material. State limitations directly when they affect reproduction or interpretation.

| Instead of | Write, when supported |
|---|---|
| The notebook now uses a chronological split. | The notebook reserves the final time interval for testing. |
| We have added better tokenization. | The experiment compares four tokenization strategies using the same training corpus. |
| The pipeline is fast and robust. | Report measured runtime and failure cases, or omit the claim. |

Do not hide missing functionality or missing results to make a page read as complete. Identify incomplete analysis honestly and keep plans separate from measured findings.

## Build each technique around evidence

Each preprocessing technique follows the assignment's sequence:

1. A Markdown cell explains the question, theory, assumptions, and formula where applicable. Define every symbol and unit.
2. Code implements the comparison, stores numerical results in named Python variables, and displays them.
3. A Markdown cell interprets the actual results, compares alternatives, and explains limitations.

Explain why a choice matters before listing implementation details. For example, explain that centered rolling features include future observations before showing how a shifted window prevents forecast leakage.

Report the dataset revision, split, sample count, method parameters, and seed, or link to the cell that defines them. For runtime and memory claims, also identify the environment and measurement boundary. Never fill example scores or p-values into prose as though they came from an executed experiment.

Keep measurements in executable output. Narrative may discuss a measured value when it helps the argument, but verify it after rerunning the experiment. The discussion chapter should link to the owning table or figure rather than maintain a second handwritten results table.

## Interpret statistics and comparisons

- Name each test's null hypothesis, significance threshold, assumptions, statistic, and p-value. Interpret rejection or failure to reject; do not claim that a p-value proves a hypothesis.
- State metric units and direction, split definitions, and evaluation counts. Compare methods on the same masks, folds, or holdout observations.
- Distinguish offline gap reconstruction from forecasting with information available at prediction time. Explain the feature/target alignment and treatment of test history.
- Define tokenization and normalization conventions before reporting vocabulary size, OOV, TTR, or sparsity. Discuss negation and other task-relevant information lost during preprocessing.
- Describe plots as observations. A t-SNE view does not establish classification quality, and a Granger test does not establish real-world causation.
- Discuss effect sizes, uncertainty, confounding choices, and practical relevance alongside statistical significance. Tie conclusions to the evaluated data and protocol.

Mark Granger and Sentence Transformer sections as advanced where appropriate, while acknowledging that the rubric also names them. Do not state that either has been completed without its experimental evidence.

## Give each fact a home

- Course requirements: [docs/preprocessing.md](preprocessing.md).
- Installation, environment maintenance, build, preview, and deployment instructions: [README.md](../README.md).
- Navigation and reactive-cell authoring mechanics: [docs/book-authoring.md](book-authoring.md).
- Paths and the random seed: [src/preprocessing/__init__.py](../src/preprocessing/__init__.py).
- Data provenance, acquisition, split definitions, method parameters, and measurements: the relevant analysis notebook.
- Cross-method conclusions and limitations: [book/report.md](../book/report.md), with links to evidence.
- Book navigation and metadata: [book/myst.yml](../book/myst.yml); CI execution and deployment: [pages.yml](../.github/workflows/pages.yml).

Search before adding a fact. Link to its owner instead of copying it into another document. Keep required identity, dataset, and submission information visible where the assignment requires it.

## Structure and rendering

Use short headings that match the reader's task, such as "Audit timestamps" or "Compare tokenizers". Use prose for reasoning, lists for discrete steps or parallel items, and tables for comparisons with shared attributes.

Use MyST math and cross-references in book pages. Give figures and tables descriptive captions, label axes and units, and refer to them in the interpretation. Explain every formula variable and keep notation consistent across theory and implementation.

Use relative links in repository documentation. In notebooks, check links and anchors in rendered HTML as well as the source. Add pages to `book/myst.yml` when they belong in the book navigation. Prefer diagrams supported by the existing renderer and verify the rendered result; avoid ASCII art.

Code examples should be the smallest working example. Shell commands run from the repository root unless their working directory is stated. Match the project's Python, uv, PowerShell, and Jupyter Book 2 workflow and verify examples against the checked-in configuration.

Use the language of the surrounding page and keep technical terminology consistent. The brief is Vietnamese and the book scaffold is English; the brief does not specify a separate report language. When introducing a translated term, give its English equivalent once if that helps the reader.

Keep the README focused on reproduction and required submission information. Move detailed explanations to the book or `docs/` and link to them. Analysis chapters may be longer because they must retain the theory, code, output, and interpretation together.

## Sentences and comments

Use short, active sentences. Address the reader as "you" in instructions. Let measurements and concrete examples support claims.

Avoid em dashes, filler transitions, metaphors, unmeasured praise, and phrases such as "it is worth noting" or "in conclusion". Avoid promotional words such as "powerful", "seamless", "revolutionary", and "groundbreaking". Do not use bold or italic to add emphasis in running prose; reserve formatting for structure and technical notation.

Docstrings state a helper's contract. Comments explain constraints and non-obvious decisions, such as fitting a vectorizer inside each fold. Keep theory and extended reasoning in notebook Markdown rather than long comments. Do not put task IDs, refactor history, obvious narration, or decorative section dividers in code.

## Review checklist

1. Does the opening identify the reader's question and accurately describe the page's contents?
2. Does the page serve its document type without mixing plans with results?
3. Does each technique include theory, executable code, displayed variables, and interpretation?
4. Do claims link to measured evidence with enough dataset, split, parameter, and metric context to assess them?
5. Are leakage safeguards, statistical assumptions, and limitations explained where relevant?
6. Are repeated facts linked to their owner, and are examples and paths valid for this repository?
7. Do equations, captions, references, notebook outputs, and changed controls work in the rendered book?
8. Are incomplete work and the distinction between scaffold execution and completed analysis stated honestly?
