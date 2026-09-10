# Extracted notebook code for portfolio review.
# Generated from a sanitized notebook copy; outputs and execution state are intentionally excluded.


# %% [notebook cell 3]
from __future__ import annotations

from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from sklearn.base import clone
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd()  # portfolio version: run from repository root
OUTPUT_DIR = PROJECT_ROOT / "cleaned data"

summary = json.loads((OUTPUT_DIR / "faid_cleaned_summary.json").read_text())
data_dictionary = json.loads((OUTPUT_DIR / "faid_cleaned_data_dictionary.json").read_text())
cleaned = pd.read_csv(OUTPUT_DIR / "faid_cleaned.csv")
model_safe = pd.read_csv(OUTPUT_DIR / "faid_cleaned_model_safe.csv")
review_required = pd.read_csv(OUTPUT_DIR / "faid_cleaned_review_required.csv")

print("cleaned shape:", cleaned.shape)
print("model_safe shape:", model_safe.shape)
print("review_required shape:", review_required.shape)

# %% [notebook cell 5]
award_like = {"awarded", "awarded_affidavit_of_promise", "usaid"}
deny_like = {"denied"}

award_rate = cleaned["parsed_decision"].isin(award_like).mean()
deny_rate = cleaned["parsed_decision"].isin(deny_like).mean()
review_share = cleaned["qa_requires_review"].fillna(0).astype(int).mean()
school_missing_share = cleaned["qa_school_was_missing"].fillna(0).astype(float).mean()
undergraduate_share = (cleaned["parsed_level"] == "Undergraduate").mean()
quality_mean = cleaned["qa_quality_score"].mean()
quality_p25 = cleaned["qa_quality_score"].quantile(0.25)
top_issue = max(summary["issue_counts"].items(), key=lambda item: item[1])

display(
    Markdown(
        f"""
### What this dataset shows

This dataset shows a historically **award-like rate of {award_rate:.1%}** versus a **denial rate of {deny_rate:.1%}**. It also shows that **{review_share:.1%}** of cleaned rows still require manual review, which is large enough to make QA governance part of the operational workflow rather than an afterthought.

The average row quality score is **{quality_mean:.1f}/100**, but the lower quartile falls to **{quality_p25:.1f}/100**, meaning trust is not evenly distributed. Undergraduate applications account for **{undergraduate_share:.1%}** of the portfolio, and **{school_missing_share:.1%}** of rows carry missing-school risk. The most frequent logged issue is **`{top_issue[0]}`** with **{top_issue[1]}** occurrences.
"""
    )
)

# %% [notebook cell 7]
import pandas as pd
from IPython.display import display

if "dictionary_frame" not in globals():
    dictionary_frame = pd.DataFrame(data_dictionary["tables"]["cleaned"]["columns"])


def justify_feature(export_field_name: str) -> tuple[str, str, str, str]:
    name = export_field_name.lower()
    if "income" in name or "gross" in name or "net_" in name or "salary" in name:
        return (
            "Household earning capacity",
            "Income-related features are retained because aid decisions ultimately estimate ability to pay.",
            "They help distinguish structural need from temporary cash-flow issues.",
            "These values are noisy in the raw workbook, so the confidence columns should always be read beside them.",
        )
    if "loan" in name:
        return (
            "Debt burden",
            "Loan features capture existing financial obligations that reduce disposable household capacity.",
            "They support need assessment and review triage for highly leveraged households.",
            "Free-text loan narratives can be ambiguous, so remaining-balance fields require especially careful review.",
        )
    if "property" in name or "car" in name or "investment" in name:
        return (
            "Asset position",
            "Asset features are retained because liquidity and wealth proxies affect how the committee interprets need.",
            "They help distinguish high-need applicants from households with hidden asset strength.",
            "Asset ownership does not always equal liquid wealth, so these fields should inform but not dominate decisions.",
        )
    if "dependent" in name or "sibling" in name or "reside" in name:
        return (
            "Household obligations",
            "Family-burden features explain why equal incomes can imply very different financial stress.",
            "They contextualize the denominator of household resources and expected support burden.",
            "These fields are sensitive to wording differences in the source text and should be audited for missingness.",
        )
    if "school" in name or "level" in name or "application" in name or "term" in name or "merit" in name:
        return (
            "Program and application context",
            "Application-context features are retained because committee policy differs by level, school, and application pathway.",
            "They help separate policy logic from financial need logic and make downstream analysis interpretable.",
            "Some of these fields are inferred from other columns, so analysts should disclose that provenance explicitly.",
        )
    if "confidence" in name or name.startswith("qa_") or "issue" in name:
        return (
            "Trust and governance",
            "QA features are retained to show whether a value is reliable enough for modeling or only suitable for human review.",
            "They convert cleaning uncertainty into an explicit operational decision signal.",
            "These fields should gate decisions, not stand in for applicant merit or need.",
        )
    if name.startswith("raw_"):
        return (
            "Audit trail",
            "Raw-preserved fields remain available so every parsed value can be traced back to its original text.",
            "They support manual verification, dispute resolution, and thesis defensibility.",
            "Raw fields are not model-safe by default because they often contain messy, high-variance text.",
        )
    return (
        "Operational context",
        "The feature was retained because it adds either explanatory context or traceability to committee workflows.",
        "It helps analysts reconstruct the case rather than rely on a single derived number.",
        "If the business role is unclear, the feature should be reviewed before it is used in a final model.",
    )


catalog = dictionary_frame.copy()
justifications = catalog["export_field_name"].map(justify_feature)
catalog[["business_role", "why_retained", "decision_use", "caution"]] = pd.DataFrame(
    justifications.tolist(), index=catalog.index
)

model_safe_catalog = catalog[catalog["model_safe_included"]].copy()
display(
    model_safe_catalog[
        [
            "export_field_name",
            "group",
            "business_role",
            "why_retained",
            "decision_use",
            "caution",
        ]
    ].head(25)
)

# %% [notebook cell 9]
dictionary_frame = pd.DataFrame(data_dictionary["tables"]["cleaned"]["columns"])

decision_frame = cleaned[
    [
        "raw_source_row_number",
        "parsed_decision",
        "parsed_level",
        "parsed_school",
        "parsed_applicant_citizenship",
        "parsed_nationality",
        "qa_issue_count",
        "qa_max_severity",
        "qa_quality_score",
        "qa_requires_review",
    ]
].copy()
decision_frame["decision_target"] = np.where(
    decision_frame["parsed_decision"].isin(award_like),
    1,
    np.where(decision_frame["parsed_decision"].isin(deny_like), 0, np.nan),
)

analysis_frame = model_safe.merge(
    decision_frame[["raw_source_row_number", "parsed_decision", "decision_target"]],
    on="raw_source_row_number",
    how="inner",
)
analysis_frame = analysis_frame[analysis_frame["decision_target"].notna()].copy()
analysis_frame["decision_target"] = analysis_frame["decision_target"].astype(int)
analysis_frame = analysis_frame.sort_values("raw_source_row_number").reset_index(drop=True)

audit_frame = decision_frame[decision_frame["decision_target"].notna()].copy()
audit_frame = audit_frame.sort_values("raw_source_row_number").reset_index(drop=True)

assert analysis_frame["raw_source_row_number"].equals(audit_frame["raw_source_row_number"])

identifier_columns = ["raw_source_row_number", "raw_source_sheet_name"]
governance_columns = [column for column in analysis_frame.columns if column.startswith("qa_")]
audit_only_columns = [
    column
    for column in ["parsed_nationality", "parsed_applicant_citizenship", "parsed_school"]
    if column in analysis_frame.columns
]

explicit_target_columns = ["parsed_decision", "decision_target"]
leakage_patterns = ["decision", "bin", "award", "deny", "scholarship"]
potential_leakage_columns = [
    column
    for column in analysis_frame.columns
    if any(pattern in column.lower() for pattern in leakage_patterns)
]

feature_columns = [
    column
    for column in analysis_frame.columns
    if column not in identifier_columns + governance_columns + audit_only_columns + explicit_target_columns
]

assert "parsed_decision" not in feature_columns
assert not any(column.startswith("qa_") for column in feature_columns)
assert not any(any(pattern in column.lower() for pattern in leakage_patterns) for column in feature_columns)

X = analysis_frame[feature_columns].copy()
y = analysis_frame["decision_target"].copy()

leakage_report = pd.DataFrame(
    {
        "category": [
            "identifier columns excluded",
            "QA governance columns excluded",
            "sensitive audit columns excluded",
            "potential leakage columns discovered",
            "modeling feature columns retained",
        ],
        "count": [
            len([column for column in identifier_columns if column in analysis_frame.columns]),
            len(governance_columns),
            len(audit_only_columns),
            len(potential_leakage_columns),
            len(feature_columns),
        ],
        "detail": [
            ", ".join(column for column in identifier_columns if column in analysis_frame.columns),
            ", ".join(governance_columns[:10]),
            ", ".join(audit_only_columns),
            ", ".join(potential_leakage_columns),
            ", ".join(feature_columns[:10]),
        ],
    }
)
display(leakage_report)

X_train_raw, X_test_raw, y_train, y_test, audit_train, audit_test = train_test_split(
    X,
    y,
    audit_frame,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

split_overview = pd.DataFrame(
    {
        "split": ["train", "test"],
        "rows": [len(X_train_raw), len(X_test_raw)],
        "award_rate": [y_train.mean(), y_test.mean()],
    }
)
display(split_overview)

# %% [notebook cell 11]
def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical_features = [column for column in X.columns if column not in numeric_features]

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", make_one_hot_encoder()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ],
    remainder="drop",
)

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)
feature_names = preprocessor.get_feature_names_out()

candidate_models = {
    "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

comparison_rows = []
fitted_models = {}

for name, estimator in candidate_models.items():
    model = clone(estimator)
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    fitted_models[name] = model
    comparison_rows.append(
        {
            "model": name,
            "roc_auc": roc_auc_score(y_test, probability),
            "average_precision": average_precision_score(y_test, probability),
            "brier_score": brier_score_loss(y_test, probability),
            "precision_at_0_5": precision_score(y_test, prediction, zero_division=0),
            "recall_at_0_5": recall_score(y_test, prediction, zero_division=0),
        }
    )

comparison_results = pd.DataFrame(comparison_rows).sort_values(
    ["roc_auc", "average_precision"],
    ascending=[False, False],
).reset_index(drop=True)
display(comparison_results)

best_auc = comparison_results["roc_auc"].max()
candidate_names = comparison_results.loc[
    comparison_results["roc_auc"] >= (best_auc - 0.02),
    "model",
].tolist()
if "LogisticRegression" in candidate_names:
    selected_model_name = "LogisticRegression"
else:
    selected_model_name = comparison_results.iloc[0]["model"]

selected_model = fitted_models[selected_model_name]
selected_prob_award = selected_model.predict_proba(X_test)[:, 1]
selected_pred_award = (selected_prob_award >= 0.5).astype(int)

display(
    Markdown(
        f"""
### Model-selection decision

The final model selected for the thesis workflow is **{selected_model_name}**. The selection rule is intentionally conservative: if logistic regression performs within 0.02 ROC AUC of the best model, it is preferred because calibration and explainability are easier to defend academically. Otherwise the top-ranked model by discrimination is selected.
"""
    )
)

display(pd.DataFrame(classification_report(y_test, selected_pred_award, output_dict=True)).T)

# %% [notebook cell 13]
calibration_bins = pd.qcut(
    pd.Series(selected_prob_award),
    q=min(10, len(np.unique(selected_prob_award))),
    duplicates="drop",
)

calibration_table = (
    pd.DataFrame(
        {
            "predicted_probability": selected_prob_award,
            "actual_outcome": y_test.to_numpy(),
            "probability_bin": calibration_bins,
        }
    )
    .groupby("probability_bin", dropna=False)
    .agg(
        avg_predicted_probability=("predicted_probability", "mean"),
        actual_award_rate=("actual_outcome", "mean"),
        cases=("actual_outcome", "size"),
    )
    .reset_index()
)
calibration_table["absolute_gap"] = (
    calibration_table["avg_predicted_probability"] - calibration_table["actual_award_rate"]
).abs()
expected_calibration_error = np.average(
    calibration_table["absolute_gap"],
    weights=calibration_table["cases"] / calibration_table["cases"].sum(),
)

display(calibration_table)
print("Expected calibration error:", round(float(expected_calibration_error), 4))
print("Brier score:", round(float(brier_score_loss(y_test, selected_prob_award)), 4))

fig, ax = plt.subplots(figsize=(6, 6))
CalibrationDisplay.from_predictions(
    y_test,
    selected_prob_award,
    n_bins=10,
    strategy="quantile",
    ax=ax,
)
ax.set_title(f"Calibration Curve for {selected_model_name}")
plt.show()

mean_gap = float(calibration_table["avg_predicted_probability"].mean() - calibration_table["actual_award_rate"].mean())
if mean_gap > 0.03:
    calibration_read = "over-confident"
elif mean_gap < -0.03:
    calibration_read = "under-confident"
else:
    calibration_read = "reasonably well calibrated on average"

display(
    Markdown(
        f"""
### Calibration interpretation

The selected model appears **{calibration_read}** on the holdout sample. The expected calibration error is **{expected_calibration_error:.3f}**, which should be discussed together with discrimination metrics because a highly ranked model is still risky if its probabilities are poorly calibrated.
"""
    )
)

# %% [notebook cell 15]
priority_scenarios = {
    "need_sensitive_default": {
        "benefit_true_positive": 1.0,
        "cost_false_positive_review": 1.0,
        "cost_false_negative_missed_need": 3.0,
        "policy_rationale": "Missing a deserving student is treated as materially worse than sending an extra borderline case to committee review.",
    },
    "review_capacity_constrained": {
        "benefit_true_positive": 1.0,
        "cost_false_positive_review": 2.0,
        "cost_false_negative_missed_need": 3.0,
        "policy_rationale": "The committee still prioritizes deserving students, but additional review load becomes more costly when operational capacity is tight.",
    },
}

display(
    pd.DataFrame(
        [
            {
                "scenario": name,
                "benefit_true_positive": config["benefit_true_positive"],
                "cost_false_positive_review": config["cost_false_positive_review"],
                "cost_false_negative_missed_need": config["cost_false_negative_missed_need"],
                "policy_rationale": config["policy_rationale"],
            }
            for name, config in priority_scenarios.items()
        ]
    )
)

threshold_rows = []
for threshold in [0.50, 0.60, 0.70, 0.80]:
    prediction = (selected_prob_award >= threshold).astype(int)
    tp = int(((y_test == 1) & (prediction == 1)).sum())
    fp = int(((y_test == 0) & (prediction == 1)).sum())
    fn = int(((y_test == 1) & (prediction == 0)).sum())
    row = {
        "threshold": threshold,
        "students_shortlisted_for_aid": int(prediction.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
    }
    for scenario_name, config in priority_scenarios.items():
        row[f"{scenario_name}_priority_score"] = (
            tp * config["benefit_true_positive"]
            - fp * config["cost_false_positive_review"]
            - fn * config["cost_false_negative_missed_need"]
        )
    threshold_rows.append(row)

selected_policy_name = "need_sensitive_default"
threshold_table = pd.DataFrame(threshold_rows)
threshold_table["impact_score"] = threshold_table[
    f"{selected_policy_name}_priority_score"
]
display(threshold_table)

best_threshold_row = threshold_table.sort_values(
    [f"{selected_policy_name}_priority_score", "precision", "recall"],
    ascending=[False, False, False],
).iloc[0]

selected_model_full_probability = selected_model.predict_proba(preprocessor.transform(X))[:, 1]
full_portfolio_table = []
for threshold in [0.60, 0.80, float(best_threshold_row["threshold"])]:
    shortlist_count = int((selected_model_full_probability >= threshold).sum())
    full_portfolio_table.append(
        {
            "threshold": threshold,
            "estimated_full_portfolio_shortlist": shortlist_count,
            "share_of_modeled_portfolio": shortlist_count / len(selected_model_full_probability),
        }
    )
full_portfolio_table = (
    pd.DataFrame(full_portfolio_table)
    .drop_duplicates(subset=["threshold"])
    .sort_values("threshold")
    .reset_index(drop=True)
)
display(full_portfolio_table)

display(
    Markdown(
        f"""
### Business-impact interpretation

The impact simulation is expressed in **committee-priority units**, not fake currency. The default scenario reflects a FAID-style priority: **missing a deserving student is worse than reviewing an extra borderline case**. Under that policy, the strongest threshold in the holdout simulation is **{best_threshold_row['threshold']:.2f}**.

This makes the tradeoff more realistic for financial-aid operations:

- lower thresholds increase coverage and reduce missed deserving students,
- higher thresholds reduce review burden but risk excluding students who should have been shortlisted,
- and the committee can swap to the capacity-constrained scenario if review resources become scarce.
"""
    )
)

# %% [notebook cell 17]
results = audit_test.copy()
results["y_true"] = y_test.to_numpy()
results["pred_prob_award"] = selected_prob_award
results["y_pred"] = selected_pred_award
results["error_type"] = np.select(
    [
        (results["y_true"] == 1) & (results["y_pred"] == 1),
        (results["y_true"] == 0) & (results["y_pred"] == 0),
        (results["y_true"] == 0) & (results["y_pred"] == 1),
        (results["y_true"] == 1) & (results["y_pred"] == 0),
    ],
    [
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ],
    default="unknown",
)
results["is_error"] = results["y_true"] != results["y_pred"]
results["quality_band"] = pd.qcut(
    results["qa_quality_score"].fillna(results["qa_quality_score"].median()),
    q=4,
    duplicates="drop",
)

display(results["error_type"].value_counts(dropna=False).rename_axis("error_type").reset_index(name="rows"))

severity_error_view = (
    results.groupby("qa_max_severity", dropna=False)
    .agg(
        rows=("raw_source_row_number", "count"),
        error_rate=("is_error", "mean"),
        avg_issue_count=("qa_issue_count", "mean"),
        avg_quality_score=("qa_quality_score", "mean"),
        false_positive_rate=("error_type", lambda values: (values == "false_positive").mean()),
        false_negative_rate=("error_type", lambda values: (values == "false_negative").mean()),
    )
    .reset_index()
)
display(severity_error_view)

quality_error_view = (
    results.groupby("quality_band", dropna=False)
    .agg(
        rows=("raw_source_row_number", "count"),
        error_rate=("is_error", "mean"),
        false_positive_rate=("error_type", lambda values: (values == "false_positive").mean()),
        false_negative_rate=("error_type", lambda values: (values == "false_negative").mean()),
    )
    .reset_index()
)
display(quality_error_view)

review_error_view = (
    results.groupby("qa_requires_review", dropna=False)
    .agg(
        rows=("raw_source_row_number", "count"),
        error_rate=("is_error", "mean"),
        avg_quality_score=("qa_quality_score", "mean"),
    )
    .reset_index()
)
display(review_error_view)

level_error_view = (
    results.groupby("parsed_level", dropna=False)
    .agg(
        rows=("raw_source_row_number", "count"),
        error_rate=("is_error", "mean"),
    )
    .reset_index()
    .sort_values("rows", ascending=False)
)
display(level_error_view)

false_positive_examples = results[results["error_type"] == "false_positive"].sort_values(
    ["pred_prob_award", "qa_quality_score"],
    ascending=[False, True],
)
false_negative_examples = results[results["error_type"] == "false_negative"].sort_values(
    ["pred_prob_award", "qa_quality_score"],
    ascending=[True, True],
)
display(false_positive_examples.head(15))
display(false_negative_examples.head(15))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, selected_pred_award, ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix for {selected_model_name}")
plt.show()

# %% [notebook cell 19]
try:
    import shap
except ImportError:
    shap = None

if shap is None:
    display(
        Markdown(
            f"""
### Fallback interpretability

SHAP is not installed in this environment, so the notebook falls back to model-native interpretability for **{selected_model_name}**.
"""
        )
    )

    if selected_model_name == "LogisticRegression":
        coefficient_frame = (
            pd.DataFrame(
                {
                    "feature_name": feature_names,
                    "coefficient": selected_model.coef_.ravel(),
                }
            )
            .assign(
                absolute_coefficient=lambda frame: frame["coefficient"].abs(),
                direction=lambda frame: np.where(frame["coefficient"] >= 0, "pushes toward award", "pushes toward denial"),
            )
            .sort_values("absolute_coefficient", ascending=False)
            .reset_index(drop=True)
        )
        display(coefficient_frame.head(20))
    elif hasattr(selected_model, "feature_importances_"):
        importance_frame = (
            pd.DataFrame(
                {
                    "feature_name": feature_names,
                    "feature_importance": selected_model.feature_importances_,
                }
            )
            .sort_values("feature_importance", ascending=False)
            .reset_index(drop=True)
        )
        display(importance_frame.head(20))
    else:
        print("No native feature-importance fallback is available for this model.")
else:
    background_size = min(300, X_train.shape[0])
    explain_size = min(200, X_test.shape[0])

    X_background = X_train[:background_size]
    X_explain = X_test[:explain_size]

    explainer = shap.Explainer(selected_model, X_background, feature_names=feature_names)
    shap_values = explainer(X_explain)

    shap.plots.beeswarm(shap_values, max_display=20)
    plt.show()

    example_index = 0
    shap.plots.waterfall(shap_values[example_index], max_display=15)
    plt.show()

# %% [notebook cell 21]
def fairness_table(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    audited = frame.copy()
    audited[group_column] = audited[group_column].fillna("Missing").astype(str)

    return (
        audited.groupby(group_column, dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "cases": len(group),
                    "predicted_award_rate": group["y_pred"].mean(),
                    "actual_award_rate": group["y_true"].mean(),
                    "avg_predicted_probability": group["pred_prob_award"].mean(),
                    "false_positive_rate": (
                        ((group["y_true"] == 0) & (group["y_pred"] == 1)).sum()
                        / max((group["y_true"] == 0).sum(), 1)
                    ),
                    "false_negative_rate": (
                        ((group["y_true"] == 1) & (group["y_pred"] == 0)).sum()
                        / max((group["y_true"] == 1).sum(), 1)
                    ),
                }
            )
        )
        .sort_values("cases", ascending=False)
        .reset_index()
    )


fairness_results = {}
for audit_column in ["parsed_level", "parsed_applicant_citizenship", "parsed_nationality", "parsed_school"]:
    fairness_results[audit_column] = fairness_table(results, audit_column)
    print("\nFairness view for:", audit_column)
    display(fairness_results[audit_column].head(15))


def largest_gap(metric: str) -> tuple[str, float]:
    gaps = {}
    for audit_column, fairness_frame in fairness_results.items():
        if len(fairness_frame) == 0:
            continue
        gaps[audit_column] = float(fairness_frame[metric].max() - fairness_frame[metric].min())
    return max(gaps.items(), key=lambda item: item[1])


largest_fpr_gap = largest_gap("false_positive_rate")
largest_fnr_gap = largest_gap("false_negative_rate")
overall_false_positive_rate = float((results["error_type"] == "false_positive").mean())
overall_false_negative_rate = float((results["error_type"] == "false_negative").mean())
fairness_action_threshold = 0.10
fairness_min_cases = 30

fairness_action_rows = []
for audit_column, fairness_frame in fairness_results.items():
    for record in fairness_frame.to_dict(orient="records"):
        fn_gap = float(record["false_negative_rate"] - overall_false_negative_rate)
        fp_gap = float(record["false_positive_rate"] - overall_false_positive_rate)
        if record["cases"] < fairness_min_cases:
            action = "monitor_only_small_sample"
            reason = "Sample too small for threshold intervention."
        elif fn_gap >= fairness_action_threshold:
            action = "lower_threshold_and_force_fairness_review"
            reason = "This subgroup is missing too many positive cases relative to the portfolio baseline."
        elif fp_gap >= fairness_action_threshold:
            action = "force_review_before_shortlist_release"
            reason = "This subgroup is generating too many false positives relative to the portfolio baseline."
        else:
            action = "monitor_only"
            reason = "No intervention trigger exceeded the governance threshold."
        fairness_action_rows.append(
            {
                "audit_dimension": audit_column,
                "subgroup": record[audit_column],
                "cases": record["cases"],
                "false_positive_rate": record["false_positive_rate"],
                "false_negative_rate": record["false_negative_rate"],
                "false_positive_gap_vs_overall": fp_gap,
                "false_negative_gap_vs_overall": fn_gap,
                "recommended_action": action,
                "reason": reason,
            }
        )

fairness_actions = (
    pd.DataFrame(fairness_action_rows)
    .sort_values(
        ["recommended_action", "false_negative_gap_vs_overall", "false_positive_gap_vs_overall", "cases"],
        ascending=[True, False, False, False],
    )
    .reset_index(drop=True)
)
display(fairness_actions.head(30))

display(
    Markdown(
        f"""
### Fairness audit interpretation

The largest observed false-positive-rate gap appears in **{largest_fpr_gap[0]}** with a spread of **{largest_fpr_gap[1]:.3f}**, and the largest false-negative-rate gap appears in **{largest_fnr_gap[0]}** with a spread of **{largest_fnr_gap[1]:.3f}**.

These are not proof of unfair treatment by themselves, but they are enough to justify action rules before any operational deployment. In this notebook:

- if a subgroup's false-negative gap exceeds **{fairness_action_threshold:.2f}** with at least **{fairness_min_cases}** cases, the recommended action is **lower_threshold_and_force_fairness_review**,
- if a subgroup's false-positive gap exceeds the same threshold, the recommendation is **force_review_before_shortlist_release**,
- otherwise the subgroup stays in **monitor_only** status.
"""
    )
)

# %% [notebook cell 23]
selected_policy_name = globals().get("selected_policy_name", "need_sensitive_default")
policy_score_column = (
    f"{selected_policy_name}_priority_score"
    if f"{selected_policy_name}_priority_score" in threshold_table.columns
    else "impact_score"
)
best_threshold_row = threshold_table.sort_values(
    [policy_score_column, "precision", "recall"],
    ascending=[False, False, False],
).iloc[0]
best_model_row = comparison_results.loc[comparison_results["model"] == selected_model_name].iloc[0]
error_rate = float(results["is_error"].mean())
review_error_rate = float(
    results.loc[results["qa_requires_review"].fillna(0).astype(int) == 1, "is_error"].mean()
) if (results["qa_requires_review"].fillna(0).astype(int) == 1).any() else float("nan")
clean_error_rate = float(
    results.loc[results["qa_requires_review"].fillna(0).astype(int) == 0, "is_error"].mean()
) if (results["qa_requires_review"].fillna(0).astype(int) == 0).any() else float("nan")

display(
    Markdown(
        f"""
## Executive Summary

### Key findings

- The cleaned FAID portfolio contains **{summary['cleaned_row_count']}** applications, and **{summary['review_output_row_count']}** of them still require review. This means the operational challenge is not just prediction quality; it is deciding which rows are reliable enough to move faster.
- The final thesis model is **{selected_model_name}**, with **ROC AUC = {best_model_row['roc_auc']:.3f}**, **average precision = {best_model_row['average_precision']:.3f}**, and **Brier score = {best_model_row['brier_score']:.3f}** on the holdout split.
- The model's overall holdout error rate is **{error_rate:.1%}**. Rows already flagged for manual review show an error rate of **{review_error_rate:.1%}**, compared with **{clean_error_rate:.1%}** on non-review rows, which supports the value of the QA layer.
- Calibration analysis indicates the model is suitable for prioritization only when its probabilities are read together with QA status and threshold policy.
- The impact simulation is grounded in a FAID-style policy preference: **missing a deserving student is worse than reviewing an extra borderline case**. Threshold selection is therefore justified in committee-priority units rather than arbitrary business value claims.

### Recommended actions

1. Use the model for **triage and prioritization**, not automatic award decisions.
2. Keep **QA severity and review flags as hard gates**: medium/high-severity rows should remain in manual review even when predicted award probability is high.
3. Use **threshold {best_threshold_row['threshold']:.2f}** as the starting operational threshold for shortlist simulation, then tune it with committee preferences for false-positive versus false-negative cost.
4. Apply fairness governance actions directly: use **force review** for groups with elevated false-positive gaps and consider **lower-threshold safeguards** for groups with elevated false-negative gaps.

### Expected impact

- At threshold **{best_threshold_row['threshold']:.2f}**, the holdout simulation shortlists **{int(best_threshold_row['students_shortlisted_for_aid'])}** students with precision **{best_threshold_row['precision']:.1%}** and recall **{best_threshold_row['recall']:.1%}**.
- On the modeled portfolio, the shortlist size shrinks materially when the threshold moves from **0.60** to **0.80**, showing a clear risk-versus-reward tradeoff between coverage and precision.
- The thesis contribution is therefore not “a perfect model,” but a **governed decision-support pipeline** that connects raw data quality, model trust, and committee action.
"""
    )
)
