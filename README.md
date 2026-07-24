# AI Trustworthiness in Educational Evaluation: Code and Data

This repository contains the analysis code and anonymized grading data used in the paper submitted to ACM AISec 2026.

## Repository Contents

**analysis.py**
Calculates Cohen's Kappa, Spearman correlation, mean absolute error (MAE), and exact agreement for each grader compared with the human baseline.

**generate_figures.py**
Generates the four figures reported in the paper and saves them as PNG files.

**grading_data.csv**
Contains the anonymized grading dataset for all 32 free-response submissions. The file includes the question identifier, student response, model-generated scores, and human grader scores.

## Requirements

Python 3.8 or later is required.

Install the required dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

## Running the Analysis

Run the statistical analysis:

```bash
python analysis.py
```

Generate the figures:

```bash
python generate_figures.py
```

The generated figures will be saved as PNG files in the current directory.

## Dataset Description

The dataset contains 32 free-response submissions collected from two game modes:

- **Story Mode:** 8 submissions scored using a 0 to 3 partial credit scale.
- **Level Select Mode:** 24 submissions independently scored by both human graders using the same 0 to 3 scale.

The dataset includes responses from Level 3 of the CyberAI game, which focuses on privacy concepts including personally identifiable information, data aggregation risks, data minimization, and privacy controls.

Two responses are excluded from Figure 1:

- `SM-Q3`
- `LS-Q22`

Responses are included in anonymized form for reproducibility. All personally identifying information has been removed before release.

## Privacy and Anonymization

The dataset does not contain any personally identifiable information. Responses have been de-identified prior to release and are provided solely to support reproducibility of the reported analyses.
