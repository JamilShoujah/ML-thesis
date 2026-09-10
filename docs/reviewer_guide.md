# Reviewer Guide

This repository is best reviewed as a portfolio reconstruction of a private thesis-support project.

## Fast Path

1. Read `README.md` for scope and confidentiality context.
2. Read `docs/architecture.md` for the pipeline design.
3. Open `notebooks/financial_aid_cleaning.ipynb` to see the cleaning, parsing, QA, and model-safe export logic.
4. Open `notebooks/test_financial_aid_cleaning.ipynb` to see the validation and regression-test mindset.
5. Open `notebooks/financial_aid_eligibility_model.ipynb` and `notebooks/financial_aid_percentage_model.ipynb` for the modeling workflows.
6. Use `scripts/extracted_notebook_code/` if GitHub notebook rendering is slow or if you prefer reading plain Python.

## What To Look For

- Evidence of defensive data engineering around messy spreadsheet exports.
- Explicit leakage prevention before modeling.
- Separation between model features, audit fields, and QA signals.
- Tests for parser edge cases, schema drift, and output contracts.
- Translation of model output into review-aware decision recommendations.
- Fairness, calibration, and interpretability considerations in a high-stakes workflow.

## What Not To Expect

- Private raw data.
- Reproduction of original thesis metrics.
- Trained model bundles.
- Original private repository history.
- One-command end-to-end execution.

Those pieces are excluded either because they were confidential or because the available archive contained only the delivered notebook files.

