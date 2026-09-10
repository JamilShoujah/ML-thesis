# Reproducibility Notes

The local archive used to create this repository contained only Jupyter notebooks. Several notebooks reference project files and generated artifacts that were not present in the archive.

## Missing From The Archive

- Original source package under `faid_models/`.
- Original data-cleaning package under `financial_aid_datacleaning/`.
- Original rebuild scripts under model-specific `scripts/` directories.
- Raw financial-aid workbook.
- Cleaned CSV exports.
- Model artifacts, reports, metadata, and generated notebook artifacts.
- Original dependency lockfile or environment file.

## Current Status

The notebooks are useful as a portfolio artifact, but they should not be represented as a fully reproducible repository yet.

The sanitized notebooks:

- have execution outputs stripped,
- have execution counts reset,
- neutralize old local filesystem paths,
- neutralize person/project-specific labels,
- preserve the code and markdown structure of the available archive.

This repository now also includes a tiny synthetic CSV under `data/sample/`. It is useful for understanding the input shape, but it is not a drop-in replacement for the original private workbook and does not reproduce the thesis results.

## To Make This Fully Runnable

1. Reconstruct or add the source modules referenced by the notebooks.
2. Expand the synthetic sample data into a complete public demo fixture.
3. Update notebooks to load sample data by default.
4. Convert the test notebook into a normal test module under `tests/`.
5. Add a reproducible command such as `pytest` and a notebook execution check.
6. Add CI once the sample-data path runs cleanly.

## Suggested Smoke Checks Later

```bash
python -m pytest
jupyter nbconvert --to notebook --execute --inplace notebooks/financial_aid_cleaning.ipynb
```

Those commands are intentionally not promised to work in this public archive until the missing source and sample data are reconstructed.
