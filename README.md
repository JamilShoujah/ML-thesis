# Financial Aid Thesis Support Portfolio

This repository is a public portfolio reconstruction of financial-aid analytics work I built for a graduate thesis project. The original working repository, private data, generated artifacts, and thesis submission package are not included here.

The public version focuses on the technical work: data cleaning, schema validation, model-safe feature engineering, reproducibility tests, eligibility modeling, aid-percentage modeling, purchasing-power sensitivity analysis, and decision-support framing.

## Current Status

This repo is ready to publish as a sanitized portfolio snapshot.

- Notebook logic and narrative structure are preserved.
- Notebook outputs and execution counts are stripped.
- Private source data, trained models, generated predictions, and thesis submission materials are excluded.
- A tiny synthetic sample dataset is included only to document the expected input shape.
- Full end-to-end reruns require reconstructing the original source modules or adapting the notebooks to the synthetic schema.

## Project Context

This project began as software and analytical notebook development for a graduate thesis collaboration. My role focused on building the data-cleaning pipeline, validation logic, model-safe feature engineering, reproducibility tests, and financial-aid modeling workflows.

The original dataset and generated outputs are not included because they contained confidential financial-aid records and row-level information. The original materials were delivered as notebooks for thesis submission rather than preserved as a complete public software repository.

This public repository reconstructs the parts suitable for portfolio review while excluding private data, trained artifacts, generated outputs, and thesis submission materials. The goal is to show the engineering and analytical approach, not to reproduce or redistribute the original thesis package.

## Repository Contents

- `notebooks/financial_aid_cleaning.ipynb` - data cleaning, parsing, QA flags, exports, and data dictionary logic.
- `notebooks/test_financial_aid_cleaning.ipynb` - notebook version of the cleaner test suite.
- `notebooks/financial_aid_decision_analysis.ipynb` - decision analysis, leakage controls, model comparison, threshold policy, error analysis, fairness checks, and interpretability.
- `notebooks/financial_aid_eligibility_model.ipynb` - governed eligibility classification workflow.
- `notebooks/financial_aid_percentage_model.ipynb` - aid-percentage modeling workflow.
- `notebooks/purchasing_power_experiment.ipynb` - sensitivity experiment around purchasing-power adjustment.
- `scripts/extracted_notebook_code/` - code-cell exports from the sanitized notebooks for easier GitHub review.
- `docs/architecture.md` - high-level pipeline and governance architecture.
- `docs/reviewer_guide.md` - suggested reading path for portfolio reviewers.
- `data/sample/synthetic_financial_aid_applications.csv` - fake, schema-shaped rows for context only.
- `data/README.md` - explains why raw/private data is intentionally excluded.
- `REPRODUCIBILITY.md` - documents what is needed to make the archive fully runnable again.
- `NOTICE.md` - publication and privacy notes.

## Important Scope Note

This is not the original full project repository. The available archive contained notebooks only, so this repo should be read as a portfolio snapshot unless the missing source modules, scripts, private data contracts, and synthetic demo data are reconstructed.

Notebook outputs were stripped before publication to avoid exposing private or row-level information from the original data environment.

## What This Demonstrates

- Robust parsing of messy operational spreadsheet exports.
- Explicit separation of raw, parsed, inferred, QA, audit-only, and model-safe fields.
- Data dictionary generation and schema validation.
- Unit-style tests for parsers, workbook edge cases, exports, and policy flags.
- Leakage-aware model design for financial-aid decision support.
- Comparison of baseline and tree-based models.
- Calibration, threshold policy, review-band logic, and error analysis.
- Fairness and governance framing for high-stakes decision workflows.
- SHAP or fallback interpretability for model behavior review.

## Quick Review Path

For a fast technical review, start with:

1. `docs/architecture.md`
2. `notebooks/financial_aid_cleaning.ipynb`
3. `notebooks/test_financial_aid_cleaning.ipynb`
4. `notebooks/financial_aid_eligibility_model.ipynb`
5. `notebooks/financial_aid_percentage_model.ipynb`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The notebooks may not execute end to end from this public archive alone because the original source package, raw workbook, cleaned exports, and generated artifacts are not included. See `REPRODUCIBILITY.md`.

## Suggested GitHub Description

Financial-aid thesis support portfolio: data cleaning, model-safe feature engineering, tests, eligibility modeling, aid-percentage modeling, fairness checks, and decision-support analysis.
