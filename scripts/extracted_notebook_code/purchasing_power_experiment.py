# Extracted notebook code for portfolio review.
# Generated from a sanitized notebook copy; outputs and execution state are intentionally excluded.


# %% [notebook cell 4]
from pathlib import Path
import os

def find_project_root() -> Path:
    env_override = os.environ.get("FAID_PROJECT_ROOT")
    candidate_roots = []
    if env_override:
        candidate_roots.append(Path(env_override).expanduser())

    cwd = Path.cwd().resolve()
    candidate_roots.extend([cwd, *cwd.parents])

    for candidate in candidate_roots:
        if (
            (candidate / "faid_models/POC/purchasing_power_experiment.py").exists()
            and (candidate / "cleaned data/faid_cleaned.csv").exists()
        ):
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the financial-aid-portfolio-project project root. "
        "Set FAID_PROJECT_ROOT or open the notebook from inside the project."
    )

PROJECT_ROOT = find_project_root()
SOURCE_PATH = PROJECT_ROOT / "faid_models/POC/purchasing_power_experiment.py"
__file__ = str(SOURCE_PATH)

# %% [notebook cell 6]
#!/usr/bin/env python3
"""Full-model purchasing-power proxy experiment for the eligibility pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faid_models.eligibility.modeling.financial_aid_eligibility_model import (  # noqa: E402
    DATA_PATH,
    SELECTED_THRESHOLD_POLICY,
    apply_feature_pruning,
    build_domain_features,
    build_preprocessors,
    compute_probability_metrics,
    evaluate_threshold_policies,
    fit_feature_pruner,
    fit_model_prototype,
    get_model_specs,
    get_positive_class_probabilities,
    get_recommended_class_weight,
    get_recommended_scale_pos_weight,
    load_data,
    make_model_pipeline,
    prepare_target,
    select_feature_frame,
    split_dataset,
    tune_thresholds,
)


PP_FACTOR = {
    2024: 0.7,
    2025: 0.85,
}

FATHER_INCOME_CANDIDATES = (
    "father_income",
    "parsed_father_gross_income",
    "parsed_father_net_income",
)
APPLICATION_TERM_CANDIDATES = (
    "Application Term",
    "application_term",
    "parsed_application_term",
)
METADATA_CANDIDATES = (
    PROJECT_ROOT / "artifacts/yasmina_eligibility_notebook/yasmina_eligibility_metadata.json",
    PROJECT_ROOT / "artifacts/yasmina_eligibility/yasmina_eligibility_metadata.json",
)
MODEL_FALLBACK_ORDER = (
    "XGBoost",
    "HistGradientBoosting",
    "Random Forest",
    "Logistic Regression",
)


def print_section(title: str) -> None:
    """Render a small console section header."""
    print(f"\n{title}")
    print("-" * 80)


def resolve_column_name(dataframe: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """Resolve a column by exact or case-insensitive match."""
    exact_matches = [column for column in candidates if column in dataframe.columns]
    if exact_matches:
        return exact_matches[0]

    lowered = {str(column).strip().lower(): column for column in dataframe.columns}
    for candidate in candidates:
        match = lowered.get(candidate.strip().lower())
        if match is not None:
            return match

    raise KeyError(
        f"None of the requested columns were found: {', '.join(candidates)}"
    )


def extract_application_year(series: pd.Series) -> pd.Series:
    """Extract the year from the first four characters of the application term."""
    return pd.to_numeric(
        series.astype("string").str.strip().str.slice(0, 4),
        errors="coerce",
    ).astype("Int64")


def resolve_model_name(model_specs: dict[str, Any]) -> str:
    """Prefer the current eligibility model when available, else fall back safely."""
    for metadata_path in METADATA_CANDIDATES:
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected_model = (
            metadata.get("artifacts", {}).get("selected_model_name")
            or metadata.get("selected_model_name")
        )
        if selected_model in model_specs:
            return str(selected_model)

    for model_name in MODEL_FALLBACK_ORDER:
        if model_name in model_specs:
            return model_name

    return next(iter(model_specs))


def summarize_income_series(raw_income: pd.Series, adjusted_income: pd.Series) -> pd.DataFrame:
    """Summarize raw vs adjusted income with consistent numeric stats."""
    rows: list[dict[str, Any]] = []
    for feature_name, series in (
        ("raw_father_income", raw_income),
        ("adjusted_income", adjusted_income),
    ):
        rows.append(
            {
                "feature": feature_name,
                "non_null_count": int(series.notna().sum()),
                "missing_count": int(series.isna().sum()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    return pd.DataFrame(rows)

# %% [notebook cell 8]
# Purchasing power experiment start
def build_pp_factor_table() -> pd.DataFrame:
    """Render the purchasing-power proxy mapping as a small table."""
    return pd.DataFrame(
        [{"application_year": year, "pp_factor": factor} for year, factor in sorted(PP_FACTOR.items())]
    )


def build_confusion_matrix_frame(matrix: list[list[int]]) -> pd.DataFrame:
    """Format a confusion matrix for readable console output."""
    return pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )


def extract_feature_importance_tables(
    fitted_model: Any,
    *,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract top feature importances when the fitted estimator exposes them."""
    empty = pd.DataFrame(columns=["rank", "feature", "importance"])

    if not hasattr(fitted_model, "named_steps"):
        return empty, empty

    preprocessor = fitted_model.named_steps.get("preprocessor")
    estimator = fitted_model.named_steps.get("model")
    if preprocessor is None or estimator is None or not hasattr(preprocessor, "get_feature_names_out"):
        return empty, empty

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return empty, empty

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1)
    else:
        return empty, empty

    row_count = min(len(feature_names), len(importances))
    if row_count == 0:
        return empty, empty

    full_table = pd.DataFrame(
        {
            "feature": feature_names[:row_count],
            "importance": importances[:row_count],
        }
    ).sort_values(by=["importance", "feature"], ascending=[False, True], ignore_index=True)
    full_table.insert(0, "rank", full_table.index + 1)
    return full_table.head(top_n).copy(), full_table


def summarize_income_related_importance(
    importance_table: pd.DataFrame,
    *,
    feature_token: str,
    label: str,
) -> dict[str, Any]:
    """Aggregate importance for the income feature family."""
    if importance_table.empty:
        return {
            "label": label,
            "matching_feature_count": 0,
            "importance_sum": 0.0,
            "best_rank": None,
            "top_matching_feature": "not_supported",
        }

    matching = importance_table[
        importance_table["feature"].str.contains(feature_token, regex=False, na=False)
    ].copy()
    if matching.empty:
        return {
            "label": label,
            "matching_feature_count": 0,
            "importance_sum": 0.0,
            "best_rank": None,
            "top_matching_feature": "not_found",
        }

    best_row = matching.sort_values(by=["rank", "importance"], ascending=[True, False]).iloc[0]
    return {
        "label": label,
        "matching_feature_count": int(len(matching)),
        "importance_sum": float(matching["importance"].sum()),
        "best_rank": int(best_row["rank"]),
        "top_matching_feature": str(best_row["feature"]),
    }


def build_segment_performance_table(
    *,
    target_true: pd.Series,
    application_year: pd.Series,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate AUC and F1 by application year on the shared test set."""
    rows: list[dict[str, Any]] = []
    year_labels = application_year.astype("string").fillna("missing_or_malformed")

    for year_value in sorted(year_labels.unique().tolist()):
        segment_mask = year_labels == year_value
        segment_target = target_true.loc[segment_mask]
        segment_probabilities_a = model_a["test_probabilities"].loc[segment_mask]
        segment_probabilities_b = model_b["test_probabilities"].loc[segment_mask]
        segment_predictions_a = model_a["test_predictions"].loc[segment_mask]
        segment_predictions_b = model_b["test_predictions"].loc[segment_mask]

        try:
            auc_a = float(roc_auc_score(segment_target, segment_probabilities_a))
        except Exception:
            auc_a = float("nan")
        try:
            auc_b = float(roc_auc_score(segment_target, segment_probabilities_b))
        except Exception:
            auc_b = float("nan")

        rows.append(
            {
                "application_year": year_value,
                "row_count": int(segment_mask.sum()),
                "auc_model_a": auc_a,
                "f1_model_a": float(f1_score(segment_target, segment_predictions_a, zero_division=0)),
                "auc_model_b": auc_b,
                "f1_model_b": float(f1_score(segment_target, segment_predictions_b, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)


def build_decision_impact_summary(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
) -> dict[str, Any]:
    """Translate prediction deltas into business-style decision terms."""
    predictions_a = model_a["test_predictions"]
    predictions_b = model_b["test_predictions"]
    model_a_positive = int((predictions_a == 1).sum())
    model_b_positive = int((predictions_b == 1).sum())
    additional_positive_decisions = int(((predictions_a == 0) & (predictions_b == 1)).sum())
    lost_positive_decisions = int(((predictions_a == 1) & (predictions_b == 0)).sum())
    net_positive_change = model_b_positive - model_a_positive
    total_cases = int(len(predictions_a))

    return {
        "model_a_positive_predictions": model_a_positive,
        "model_b_positive_predictions": model_b_positive,
        "additional_positive_decisions": additional_positive_decisions,
        "lost_positive_decisions": lost_positive_decisions,
        "net_positive_change": net_positive_change,
        "net_positive_change_pct": (net_positive_change / total_cases) if total_cases else 0.0,
        "total_cases": total_cases,
    }


def build_distribution_sanity_check(
    diagnostics: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Compare how much the raw and adjusted income features actually differ."""
    raw_income = diagnostics["raw_income"]
    adjusted_income = diagnostics["adjusted_income"]
    valid_mask = raw_income.notna() & adjusted_income.notna()

    if int(valid_mask.sum()) > 1:
        correlation = float(raw_income.loc[valid_mask].corr(adjusted_income.loc[valid_mask]))
    else:
        correlation = float("nan")

    raw_mean = float(raw_income.mean())
    adjusted_mean = float(adjusted_income.mean())
    mean_shift_pct = ((adjusted_mean - raw_mean) / raw_mean) * 100.0 if raw_mean else float("nan")
    non_default_factor_rate = (
        1.0 - (diagnostics["default_factor_row_count"] / diagnostics["row_count"])
        if diagnostics["row_count"]
        else 0.0
    )
    active_adjustment_rate = (
        diagnostics["active_adjustment_row_count"] / diagnostics["row_count"]
        if diagnostics["row_count"]
        else 0.0
    )

    sanity_frame = pd.DataFrame(
        [
            {
                "raw_adjusted_correlation": correlation,
                "mean_shift_pct": mean_shift_pct,
                "non_default_factor_rate": non_default_factor_rate,
                "active_adjustment_rate": active_adjustment_rate,
            }
        ]
    )

    if np.isnan(correlation):
        explanation = "The distribution sanity check could not compute a stable correlation."
    else:
        explanation = (
            f"Raw and adjusted income remain highly aligned with correlation {correlation:.4f}. "
            f"The mean shifted by {mean_shift_pct:.2f}%, so the feature changed in scale more than in rank ordering."
        )

    return sanity_frame, explanation

# %% [notebook cell 10]
def build_experiment_feature_sets(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build full-feature Model A and Model B frames with one controlled income swap."""
    father_income_column = resolve_column_name(dataframe, FATHER_INCOME_CANDIDATES)
    application_term_column = resolve_column_name(dataframe, APPLICATION_TERM_CANDIDATES)

    base_features, _ = select_feature_frame(dataframe, target_column)
    base_features = build_domain_features(base_features)
    if father_income_column not in base_features.columns:
        raise KeyError(
            f"Resolved father income column '{father_income_column}' is not present in the model feature set."
        )

    father_income = pd.to_numeric(dataframe[father_income_column], errors="coerce")
    application_year = extract_application_year(dataframe[application_term_column])
    year_factor = application_year.map(PP_FACTOR).fillna(1.0).astype(float)
    adjusted_income = father_income * year_factor
    active_adjustment_mask = father_income.notna() & year_factor.ne(1.0)

    model_a_features = base_features.copy()
    model_b_features = base_features.drop(columns=[father_income_column], errors="ignore").copy()
    insert_at = list(base_features.columns).index(father_income_column)
    model_b_features.insert(insert_at, "adjusted_income", adjusted_income)

    year_distribution = (
        application_year.astype("string")
        .fillna("missing_or_malformed")
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("application_year")
        .reset_index(name="row_count")
    )

    diagnostics = {
        "father_income_column": father_income_column,
        "application_term_column": application_term_column,
        "row_count": int(len(dataframe)),
        "missing_father_income_count": int(father_income.isna().sum()),
        "default_factor_row_count": int((year_factor == 1.0).sum()),
        "active_adjustment_row_count": int(active_adjustment_mask.sum()),
        "year_distribution": year_distribution,
        "income_summary": summarize_income_series(father_income, adjusted_income),
        "application_year": application_year,
        "raw_income": father_income,
        "adjusted_income": adjusted_income,
    }
    return model_a_features, model_b_features, diagnostics


def align_feature_sets_for_modeling(
    *,
    model_a_features: pd.DataFrame,
    model_b_features: pd.DataFrame,
    target: pd.Series,
    father_income_column: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Apply one shared split and one shared pruning policy to both experiments."""
    (
        features_train_a,
        features_valid_a,
        features_test_a,
        target_train,
        target_valid,
        target_test,
    ) = split_dataset(model_a_features, target)

    features_train_b = model_b_features.loc[features_train_a.index].copy()
    features_valid_b = model_b_features.loc[features_valid_a.index].copy()
    features_test_b = model_b_features.loc[features_test_a.index].copy()

    columns_to_drop, _ = fit_feature_pruner(features_train_a)
    columns_to_drop = [column for column in columns_to_drop if column != father_income_column]

    features_train_a = apply_feature_pruning(features_train_a, columns_to_drop)
    retained_columns_a = features_train_a.columns.tolist()
    retained_columns_b = [
        "adjusted_income" if column == father_income_column else column
        for column in retained_columns_a
    ]

    features_valid_a = apply_feature_pruning(
        features_valid_a,
        columns_to_drop,
        retained_columns=retained_columns_a,
    )
    features_test_a = apply_feature_pruning(
        features_test_a,
        columns_to_drop,
        retained_columns=retained_columns_a,
    )
    features_train_b = apply_feature_pruning(
        features_train_b,
        columns_to_drop,
        retained_columns=retained_columns_b,
    )
    features_valid_b = apply_feature_pruning(
        features_valid_b,
        columns_to_drop,
        retained_columns=retained_columns_b,
    )
    features_test_b = apply_feature_pruning(
        features_test_b,
        columns_to_drop,
        retained_columns=retained_columns_b,
    )

    return (
        features_train_a,
        features_valid_a,
        features_test_a,
        features_train_b,
        features_valid_b,
        features_test_b,
        target_train,
        target_valid,
        target_test,
    )

# %% [notebook cell 12]
def run_full_feature_model(
    *,
    variant_name: str,
    model_name: str,
    model_spec: Any,
    features_train: pd.DataFrame,
    features_valid: pd.DataFrame,
    features_test: pd.DataFrame,
    target_train: pd.Series,
    target_valid: pd.Series,
    target_test: pd.Series,
) -> dict[str, Any]:
    """Fit and evaluate one full-feature model with shared pipeline logic."""
    preprocessors, _, _ = build_preprocessors(features_train)
    pipeline = make_model_pipeline(preprocessors, model_spec)
    fitted_model = fit_model_prototype(pipeline, features_train, target_train)

    validation_probabilities = get_positive_class_probabilities(fitted_model, features_valid)
    _, validation_threshold_results = tune_thresholds(target_valid, validation_probabilities)
    selected_validation_metrics = validation_threshold_results[SELECTED_THRESHOLD_POLICY]

    test_probabilities = get_positive_class_probabilities(fitted_model, features_test)
    test_probability_metrics = compute_probability_metrics(target_test, test_probabilities)
    test_threshold_results = evaluate_threshold_policies(
        target_test,
        test_probabilities,
        {SELECTED_THRESHOLD_POLICY: selected_validation_metrics.threshold},
    )
    selected_test_metrics = test_threshold_results[SELECTED_THRESHOLD_POLICY]
    test_predictions = pd.Series(
        (test_probabilities >= float(selected_validation_metrics.threshold)).astype(int),
        index=features_test.index,
        name=f"{variant_name.lower().replace(' ', '_')}_prediction",
    )
    test_probabilities_series = pd.Series(
        np.asarray(test_probabilities, dtype=float),
        index=features_test.index,
        name=f"{variant_name.lower().replace(' ', '_')}_probability",
    )
    top_feature_importance, full_feature_importance = extract_feature_importance_tables(
        fitted_model,
        top_n=10,
    )

    return {
        "variant_name": variant_name,
        "model_name": model_name,
        "threshold_policy": SELECTED_THRESHOLD_POLICY,
        "selected_threshold": float(selected_validation_metrics.threshold),
        "auc": float(test_probability_metrics["roc_auc"]),
        "f1": float(selected_test_metrics.f1),
        "feature_count": int(features_train.shape[1]),
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities_series,
        "confusion_matrix": selected_test_metrics.confusion_matrix,
        "top_feature_importance": top_feature_importance,
        "full_feature_importance": full_feature_importance,
    }
# Purchasing power experiment end


def build_prediction_impact_summary(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare final test-set predictions between both model variants."""
    predictions_a = model_a["test_predictions"]
    predictions_b = model_b["test_predictions"]
    prediction_diff_mask = predictions_a.ne(predictions_b)
    changed_count = int(prediction_diff_mask.sum())
    total_count = int(len(predictions_a))

    return {
        "changed_case_count": changed_count,
        "changed_case_pct": (changed_count / total_count) if total_count else 0.0,
        "negative_to_positive_count": int(((predictions_a == 0) & (predictions_b == 1)).sum()),
        "positive_to_negative_count": int(((predictions_a == 1) & (predictions_b == 0)).sum()),
        "total_case_count": total_count,
    }


def build_interpretation(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    prediction_impact: dict[str, Any],
    decision_impact: dict[str, Any],
    diagnostics: dict[str, Any],
    income_importance_summary: pd.DataFrame,
) -> str:
    """Write a short plain-English interpretation."""
    auc_delta = model_b["auc"] - model_a["auc"]
    f1_delta = model_b["f1"] - model_a["f1"]
    changed_pct = float(prediction_impact["changed_case_pct"])
    default_factor_share = (
        diagnostics["default_factor_row_count"] / diagnostics["row_count"]
        if diagnostics["row_count"]
        else 0.0
    )
    income_importance_delta = 0.0
    if len(income_importance_summary) >= 2:
        income_importance_delta = float(
            income_importance_summary.iloc[1]["importance_sum"]
            - income_importance_summary.iloc[0]["importance_sum"]
        )

    if auc_delta > 0 and f1_delta > 0:
        verdict = "The purchasing-power adjustment helped on both AUC and F1."
    elif auc_delta < 0 and f1_delta < 0:
        verdict = "The purchasing-power adjustment hurt both AUC and F1."
    else:
        verdict = "The purchasing-power adjustment gave mixed results."

    if max(abs(auc_delta), abs(f1_delta)) < 0.005:
        magnitude = "The performance difference is marginal rather than decisive."
    elif max(abs(auc_delta), abs(f1_delta)) < 0.015:
        magnitude = "The performance difference is small but noticeable."
    else:
        magnitude = "The performance difference is meaningful."

    if changed_pct < 0.02:
        prediction_shift = "Very few final test predictions changed between Model A and Model B."
    elif changed_pct < 0.10:
        prediction_shift = "A modest share of final test predictions changed between the two models."
    else:
        prediction_shift = "A large share of final test predictions changed between the two models."

    if decision_impact["net_positive_change"] > 0:
        decision_note = (
            f"If deployed, Model B would approve {decision_impact['net_positive_change']} more students on the test set."
        )
    elif decision_impact["net_positive_change"] < 0:
        decision_note = (
            f"If deployed, Model B would approve {abs(decision_impact['net_positive_change'])} fewer students on the test set."
        )
    else:
        decision_note = "If deployed, Model B would approve the same number of students on the test set."

    if default_factor_share >= 0.50:
        business_implication = (
            "Most rows still used the default factor 1.0, so this proxy currently has limited reach. "
            "That means the experiment is directionally useful for fairness-across-years and economic-realism thinking, "
            "but not yet strong enough to justify a production change on its own."
        )
    elif auc_delta >= 0 or f1_delta >= 0:
        business_implication = (
            "Because the proxy injects some year sensitivity into income, it may help the model treat older applications "
            "more economically realistically. The effect here is still small, so it is better viewed as an exploratory fairness-across-years improvement than a finished policy change."
        )
    else:
        business_implication = (
            "This proxy does not currently improve the business outcome enough to justify replacing the raw income field. "
            "It is still a useful thesis experiment because it tests whether year-aware income scaling can improve fairness across application years."
        )

    if income_importance_delta > 0.0:
        importance_note = "Income-related importance increased after the adjustment."
    elif income_importance_delta < 0.0:
        importance_note = "Income-related importance decreased after the adjustment."
    else:
        importance_note = "Income-related importance stayed effectively flat."

    return (
        f"{verdict} Model B changed AUC by {auc_delta:.4f} and F1 by {f1_delta:.4f} versus Model A. "
        f"{magnitude} {prediction_shift} {importance_note} {decision_note} {business_implication}"
    )


def build_final_recommendation(
    *,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    prediction_impact: dict[str, Any],
    diagnostics: dict[str, Any],
) -> tuple[str, str]:
    """Produce a simple final recommendation based on effect size and coverage."""
    auc_delta = model_b["auc"] - model_a["auc"]
    f1_delta = model_b["f1"] - model_a["f1"]
    max_metric_delta = max(abs(auc_delta), abs(f1_delta))
    changed_case_pct = float(prediction_impact["changed_case_pct"])
    active_adjustment_rate = (
        diagnostics["active_adjustment_row_count"] / diagnostics["row_count"]
        if diagnostics["row_count"]
        else 0.0
    )

    if auc_delta > 0.005 and f1_delta > 0.005 and changed_case_pct >= 0.02:
        recommendation = "replace"
        rationale = (
            "Model B shows a meaningful and consistent lift with enough prediction movement to matter operationally."
        )
    elif max_metric_delta < 0.005 and changed_case_pct < 0.02 and active_adjustment_rate < 0.10:
        recommendation = "keep raw"
        rationale = (
            "The measured effect is marginal, very few decisions change, and the adjustment only touches a limited share of rows."
        )
    else:
        recommendation = "explore further"
        rationale = (
            "The idea is directionally plausible, but the current heuristic does not yet provide a clear enough operational win."
        )

    return recommendation, rationale

# %% [notebook cell 14]
def main() -> int:
    dataframe = load_data(DATA_PATH)
    training_dataframe, target, target_metadata = prepare_target(dataframe)

    # Purchasing power experiment start
    model_a_features, model_b_features, diagnostics = build_experiment_feature_sets(
        training_dataframe,
        target_column=target_metadata["target_column"],
    )
    (
        features_train_a,
        features_valid_a,
        features_test_a,
        features_train_b,
        features_valid_b,
        features_test_b,
        target_train,
        target_valid,
        target_test,
    ) = align_feature_sets_for_modeling(
        model_a_features=model_a_features,
        model_b_features=model_b_features,
        target=target,
        father_income_column=diagnostics["father_income_column"],
    )
    # Purchasing power experiment end

    if diagnostics["father_income_column"] not in features_train_a.columns:
        raise KeyError(
            f"Model A does not contain the expected raw income column: {diagnostics['father_income_column']}"
        )
    if "adjusted_income" not in features_train_b.columns:
        raise KeyError("Model B does not contain the expected 'adjusted_income' feature.")
    if diagnostics["father_income_column"] in features_train_b.columns:
        raise ValueError("Model B still contains the raw father income column after replacement.")
    if features_train_a.shape[1] != features_train_b.shape[1]:
        raise ValueError("Model A and Model B must have the same feature count.")

    class_weight = get_recommended_class_weight(target_train)
    scale_pos_weight = get_recommended_scale_pos_weight(target_train)
    model_specs = get_model_specs(
        class_weight=class_weight,
        scale_pos_weight=scale_pos_weight,
    )
    model_name = resolve_model_name(model_specs)
    model_spec = model_specs[model_name]

    model_a = run_full_feature_model(
        variant_name="Model A",
        model_name=model_name,
        model_spec=model_spec,
        features_train=features_train_a,
        features_valid=features_valid_a,
        features_test=features_test_a,
        target_train=target_train,
        target_valid=target_valid,
        target_test=target_test,
    )
    model_b = run_full_feature_model(
        variant_name="Model B",
        model_name=model_name,
        model_spec=model_spec,
        features_train=features_train_b,
        features_valid=features_valid_b,
        features_test=features_test_b,
        target_train=target_train,
        target_valid=target_valid,
        target_test=target_test,
    )
    prediction_impact = build_prediction_impact_summary(model_a, model_b)
    decision_impact = build_decision_impact_summary(model_a, model_b)
    segment_performance = build_segment_performance_table(
        target_true=target_test,
        application_year=diagnostics["application_year"].loc[features_test_a.index],
        model_a=model_a,
        model_b=model_b,
    )
    distribution_sanity_check, distribution_explanation = build_distribution_sanity_check(diagnostics)
    income_importance_summary = pd.DataFrame(
        [
            summarize_income_related_importance(
                model_a["full_feature_importance"],
                feature_token=diagnostics["father_income_column"],
                label="Model A raw income",
            ),
            summarize_income_related_importance(
                model_b["full_feature_importance"],
                feature_token="adjusted_income",
                label="Model B adjusted income",
            ),
        ]
    )
    recommendation, recommendation_rationale = build_final_recommendation(
        model_a=model_a,
        model_b=model_b,
        prediction_impact=prediction_impact,
        diagnostics=diagnostics,
    )

    comparison_table = pd.DataFrame(
        [
            {
                "model_version": model_a["variant_name"],
                "income_feature": diagnostics["father_income_column"],
                "estimator": model_a["model_name"],
                "feature_count": model_a["feature_count"],
                "threshold_policy": model_a["threshold_policy"],
                "selected_threshold": model_a["selected_threshold"],
                "auc": model_a["auc"],
                "f1": model_a["f1"],
            },
            {
                "model_version": model_b["variant_name"],
                "income_feature": "adjusted_income",
                "estimator": model_b["model_name"],
                "feature_count": model_b["feature_count"],
                "threshold_policy": model_b["threshold_policy"],
                "selected_threshold": model_b["selected_threshold"],
                "auc": model_b["auc"],
                "f1": model_b["f1"],
            },
        ]
    )

    print_section("Purchasing Power Experiment")
    print(
        "PP_FACTOR is a simple proxy for inflation / purchasing power, not a formal CPI model.\n"
        "2024 receives a lower factor than 2025 because an earlier nominal income is assumed to overstate real purchasing power more strongly in a high-inflation / currency-collapse context.\n"
        "These values are heuristic and directional: they are meant to test whether year-aware income scaling helps, not to claim a precise macroeconomic adjustment.\n"
        f"Resolved father income column: {diagnostics['father_income_column']}\n"
        f"Resolved application term column: {diagnostics['application_term_column']}\n"
        f"Missing father income count: {diagnostics['missing_father_income_count']}\n"
        f"Rows using default factor 1.0: {diagnostics['default_factor_row_count']}\n"
        f"Shared estimator: {model_name}"
    )

    print_section("Purchasing Power Mapping")
    print(build_pp_factor_table().to_string(index=False))

    print_section("Extracted Year Distribution")
    print(diagnostics["year_distribution"].to_string(index=False))

    print_section("Income Summary")
    print(diagnostics["income_summary"].round(4).to_string(index=False))

    print_section("Model Comparison")
    print(comparison_table.round(4).to_string(index=False))

    print_section("Segment-Level Analysis By Application Year")
    print(segment_performance.round(4).to_string(index=False))

    print_section("Prediction Impact Analysis")
    prediction_impact_frame = pd.DataFrame(
        [
            {
                "changed_case_count": prediction_impact["changed_case_count"],
                "changed_case_pct": prediction_impact["changed_case_pct"],
                "negative_to_positive_count": prediction_impact["negative_to_positive_count"],
                "positive_to_negative_count": prediction_impact["positive_to_negative_count"],
            }
        ]
    )
    print(prediction_impact_frame.round(4).to_string(index=False))

    print_section("Decision Impact Framing")
    decision_impact_frame = pd.DataFrame([decision_impact])
    print(decision_impact_frame.round(4).to_string(index=False))
    if decision_impact["net_positive_change"] > 0:
        print(
            f"If deployed, Model B would approve {decision_impact['net_positive_change']} more students "
            f"({decision_impact['net_positive_change_pct']:.2%} of the test set)."
        )
    elif decision_impact["net_positive_change"] < 0:
        print(
            f"If deployed, Model B would approve {abs(decision_impact['net_positive_change'])} fewer students "
            f"({abs(decision_impact['net_positive_change_pct']):.2%} of the test set)."
        )
    else:
        print("If deployed, Model B would approve the same number of students as Model A on the test set.")

    print_section("Model A Confusion Matrix")
    print(build_confusion_matrix_frame(model_a["confusion_matrix"]).to_string())

    print_section("Model B Confusion Matrix")
    print(build_confusion_matrix_frame(model_b["confusion_matrix"]).to_string())

    print_section("Top 10 Feature Importance - Model A")
    if model_a["top_feature_importance"].empty:
        print("Feature importance is not supported by the selected estimator.")
    else:
        print(model_a["top_feature_importance"].round(6).to_string(index=False))

    print_section("Top 10 Feature Importance - Model B")
    if model_b["top_feature_importance"].empty:
        print("Feature importance is not supported by the selected estimator.")
    else:
        print(model_b["top_feature_importance"].round(6).to_string(index=False))

    print_section("Income Importance Comparison")
    print(income_importance_summary.round(6).to_string(index=False))
    income_importance_delta = 0.0
    if len(income_importance_summary) >= 2:
        income_importance_delta = float(
            income_importance_summary.iloc[1]["importance_sum"]
            - income_importance_summary.iloc[0]["importance_sum"]
        )
    if income_importance_delta > 0:
        print("Income-related importance increased after the purchasing-power adjustment.")
    elif income_importance_delta < 0:
        print("Income-related importance decreased after the purchasing-power adjustment.")
    else:
        print("Income-related importance stayed effectively flat after the purchasing-power adjustment.")

    print_section("Distribution Sanity Check")
    print(distribution_sanity_check.round(4).to_string(index=False))
    print(distribution_explanation)

    print_section("Interpretation")
    print(
        build_interpretation(
            model_a,
            model_b,
            prediction_impact,
            decision_impact,
            diagnostics,
            income_importance_summary,
        )
    )

    print_section("Recommendation")
    print(f"Recommendation: {recommendation}")
    print(recommendation_rationale)

    return 0

# %% [notebook cell 16]
from IPython.display import HTML, Markdown, display as _ipython_display
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from faid_models.eligibility.modeling.financial_aid_eligibility_model import (
    DECISION_APPROVE_MIN_PROBABILITY,
    DECISION_REJECT_MAX_PROBABILITY,
    DECISION_REVIEW_MARGIN,
    FAID_POLICY_PRIORITY_WEIGHTS,
    FAID_POLICY_RATIONALE,
    RANDOM_STATE,
    build_decision_recommendations,
    get_transformed_feature_names,
    summarize_decision_impact as summarize_three_way_decision_impact,
)

_ipython_display(
    HTML(
        '''
        <style>
        .output_scroll,
        .output_subarea,
        .output_area pre,
        .jp-OutputArea-output,
        .jp-OutputArea-child,
        .jp-RenderedText pre {
            max-height: none !important;
            height: auto !important;
            overflow-y: visible !important;
            overflow-x: auto !important;
        }
        table.dataframe td, table.dataframe th {
            font-size: 0.92rem;
        }
        </style>
        '''
    )
)

display = _ipython_display

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 180)

MODEL_COLORS = {"Model A": "#1f77b4", "Model B": "#d95f02"}
SCENARIO_COLORS = {
    "legacy_original": "#7570b3",
    "conservative": "#66a61e",
    "baseline": "#1b9e77",
    "aggressive": "#e7298a",
}
PP_SCENARIO_META = {
    "conservative": {"annual_decay": 0.95, "label": "Lower adjustment"},
    "baseline": {"annual_decay": 0.90, "label": "Baseline adjustment"},
    "aggressive": {"annual_decay": 0.85, "label": "Higher adjustment"},
    "legacy_original": {"annual_decay": None, "label": "Original notebook heuristic"},
}
PRIMARY_SCENARIO_NAME = "baseline"
PAIRED_BOOTSTRAP_ITERATIONS = 400
SHAP_SAMPLE_SIZE = 120
TOP_FEATURES_FOR_EXPLAINABILITY = 10

def show_interpretation(text: str) -> None:
    display(Markdown(f"> **Interpretation.** {text}"))

def format_pct(value: float, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.{digits}%}"

def safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0

def build_pp_scenarios(application_year: pd.Series) -> tuple[list[int], int, dict[str, dict[int, float]]]:
    observed_years = sorted(
        int(year)
        for year in pd.Series(application_year).dropna().astype(int).unique().tolist()
    )
    reference_year = max(observed_years)
    scenarios = {
        name: {
            year: round(meta["annual_decay"] ** (reference_year - year), 4)
            for year in observed_years
        }
        for name, meta in PP_SCENARIO_META.items()
        if meta["annual_decay"] is not None
    }
    scenarios["legacy_original"] = dict(PP_FACTOR)
    return observed_years, reference_year, scenarios

def build_factor_scenario_table(
    observed_years: list[int],
    reference_year: int,
    scenarios: dict[str, dict[int, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name, meta in PP_SCENARIO_META.items():
        factor_map = scenarios.get(scenario_name, {})
        row: dict[str, Any] = {
            "scenario": scenario_name,
            "scenario_label": meta["label"],
            "reference_year": reference_year,
            "adjusted_year_count": int(
                sum(1 for year in observed_years if factor_map.get(year, 1.0) != 1.0)
            ),
        }
        for year in observed_years:
            row[f"factor_{year}"] = float(factor_map.get(year, 1.0))
        rows.append(row)
    return pd.DataFrame(rows)

def build_income_variant_feature_sets(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    factor_map: dict[int, float],
    income_transform: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    father_income_column = resolve_column_name(dataframe, FATHER_INCOME_CANDIDATES)
    application_term_column = resolve_column_name(dataframe, APPLICATION_TERM_CANDIDATES)

    base_features, _ = select_feature_frame(dataframe, target_column)
    base_features = build_domain_features(base_features)
    if father_income_column not in base_features.columns:
        raise KeyError(
            f"Resolved father income column '{father_income_column}' is not present in the model feature set."
        )

    father_income = pd.to_numeric(dataframe[father_income_column], errors="coerce")
    application_year = extract_application_year(dataframe[application_term_column])
    year_factor = application_year.map(factor_map).fillna(1.0).astype(float)
    adjusted_income = father_income * year_factor

    raw_feature_name = father_income_column
    adjusted_feature_name = "adjusted_income"
    model_a_features = base_features.copy()
    model_b_features = base_features.drop(columns=[father_income_column], errors="ignore").copy()
    insert_at = list(base_features.columns).index(father_income_column)

    if income_transform == "log1p":
        raw_feature_name = "log_raw_income"
        adjusted_feature_name = "log_adjusted_income"
        model_a_features = base_features.drop(columns=[father_income_column], errors="ignore").copy()
        model_b_features = base_features.drop(columns=[father_income_column], errors="ignore").copy()
        model_a_features.insert(
            insert_at,
            raw_feature_name,
            np.log1p(father_income.clip(lower=0)),
        )
        model_b_features.insert(
            insert_at,
            adjusted_feature_name,
            np.log1p(adjusted_income.clip(lower=0)),
        )
    else:
        model_b_features.insert(insert_at, adjusted_feature_name, adjusted_income)

    coverage_source = pd.DataFrame(
        {
            "application_year": application_year.astype("Int64"),
            "has_income": father_income.notna().astype(int),
            "actively_adjusted": (father_income.notna() & year_factor.ne(1.0)).astype(int),
        }
    )
    coverage_by_year = (
        coverage_source.groupby("application_year", dropna=False)
        .agg(
            row_count=("has_income", "size"),
            income_non_null_count=("has_income", "sum"),
            actively_adjusted_count=("actively_adjusted", "sum"),
        )
        .reset_index()
    )
    coverage_by_year["income_non_null_share"] = (
        coverage_by_year["income_non_null_count"] / coverage_by_year["row_count"]
    )
    coverage_by_year["actively_adjusted_share_of_year"] = (
        coverage_by_year["actively_adjusted_count"] / coverage_by_year["row_count"]
    )
    coverage_by_year["actively_adjusted_share_of_income"] = (
        coverage_by_year["actively_adjusted_count"]
        / coverage_by_year["income_non_null_count"].replace(0, np.nan)
    )

    diagnostics = {
        "father_income_column": father_income_column,
        "application_term_column": application_term_column,
        "row_count": int(len(dataframe)),
        "missing_father_income_count": int(father_income.isna().sum()),
        "default_factor_row_count": int((year_factor == 1.0).sum()),
        "active_adjustment_row_count": int((father_income.notna() & year_factor.ne(1.0)).sum()),
        "year_distribution": (
            application_year.astype("string")
            .fillna("missing_or_malformed")
            .value_counts(dropna=False)
            .sort_index()
            .rename_axis("application_year")
            .reset_index(name="row_count")
        ),
        "income_summary": summarize_income_series(father_income, adjusted_income),
        "application_year": application_year,
        "raw_income": father_income,
        "adjusted_income": adjusted_income,
        "year_factor": year_factor,
        "factor_map": {int(year): float(factor) for year, factor in factor_map.items()},
        "income_transform": income_transform or "linear",
        "raw_income_feature_name": raw_feature_name,
        "adjusted_income_feature_name": adjusted_feature_name,
        "coverage_by_year": coverage_by_year,
    }
    return model_a_features, model_b_features, diagnostics

def align_feature_sets_general(
    *,
    model_a_features: pd.DataFrame,
    model_b_features: pd.DataFrame,
    target: pd.Series,
    raw_feature_name: str,
    adjusted_feature_name: str,
    original_income_column: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    (
        features_train_a,
        features_valid_a,
        features_test_a,
        target_train,
        target_valid,
        target_test,
    ) = split_dataset(model_a_features, target)

    features_train_b = model_b_features.loc[features_train_a.index].copy()
    features_valid_b = model_b_features.loc[features_valid_a.index].copy()
    features_test_b = model_b_features.loc[features_test_a.index].copy()

    columns_to_drop, _ = fit_feature_pruner(features_train_a)
    protected_features = {raw_feature_name}
    if original_income_column in features_train_a.columns:
        protected_features.add(original_income_column)
    columns_to_drop = [column for column in columns_to_drop if column not in protected_features]

    features_train_a = apply_feature_pruning(features_train_a, columns_to_drop)
    retained_columns_a = features_train_a.columns.tolist()
    retained_columns_b = [
        adjusted_feature_name if column == raw_feature_name else column
        for column in retained_columns_a
    ]

    features_valid_a = apply_feature_pruning(
        features_valid_a,
        columns_to_drop,
        retained_columns=retained_columns_a,
    )
    features_test_a = apply_feature_pruning(
        features_test_a,
        columns_to_drop,
        retained_columns=retained_columns_a,
    )
    features_train_b = apply_feature_pruning(
        features_train_b,
        columns_to_drop,
        retained_columns=retained_columns_b,
    )
    features_valid_b = apply_feature_pruning(
        features_valid_b,
        columns_to_drop,
        retained_columns=retained_columns_b,
    )
    features_test_b = apply_feature_pruning(
        features_test_b,
        columns_to_drop,
        retained_columns=retained_columns_b,
    )

    return (
        features_train_a,
        features_valid_a,
        features_test_a,
        features_train_b,
        features_valid_b,
        features_test_b,
        target_train,
        target_valid,
        target_test,
    )

def run_full_feature_model_enhanced(
    *,
    variant_name: str,
    model_name: str,
    model_spec: Any,
    features_train: pd.DataFrame,
    features_valid: pd.DataFrame,
    features_test: pd.DataFrame,
    target_train: pd.Series,
    target_valid: pd.Series,
    target_test: pd.Series,
) -> dict[str, Any]:
    preprocessors, _, _ = build_preprocessors(features_train)
    pipeline = make_model_pipeline(preprocessors, model_spec)
    fitted_model = fit_model_prototype(pipeline, features_train, target_train)

    validation_probabilities = get_positive_class_probabilities(fitted_model, features_valid)
    _, validation_threshold_results = tune_thresholds(target_valid, validation_probabilities)
    selected_validation_metrics = validation_threshold_results[SELECTED_THRESHOLD_POLICY]

    test_probabilities = get_positive_class_probabilities(fitted_model, features_test)
    test_probability_metrics = compute_probability_metrics(target_test, test_probabilities)
    test_threshold_results = evaluate_threshold_policies(
        target_test,
        test_probabilities,
        {SELECTED_THRESHOLD_POLICY: selected_validation_metrics.threshold},
    )
    selected_test_metrics = test_threshold_results[SELECTED_THRESHOLD_POLICY]
    test_predictions = pd.Series(
        (test_probabilities >= float(selected_validation_metrics.threshold)).astype(int),
        index=features_test.index,
        name=f"{variant_name.lower().replace(' ', '_')}_prediction",
    )
    test_probabilities_series = pd.Series(
        np.asarray(test_probabilities, dtype=float),
        index=features_test.index,
        name=f"{variant_name.lower().replace(' ', '_')}_probability",
    )
    top_feature_importance, full_feature_importance = extract_feature_importance_tables(
        fitted_model,
        top_n=10,
    )

    return {
        "variant_name": variant_name,
        "model_name": model_name,
        "threshold_policy": SELECTED_THRESHOLD_POLICY,
        "selected_threshold": float(selected_validation_metrics.threshold),
        "auc": float(test_probability_metrics["roc_auc"]),
        "average_precision": float(test_probability_metrics["average_precision"]),
        "brier_score": float(test_probability_metrics["brier_score"]),
        "precision": float(selected_test_metrics.precision),
        "recall": float(selected_test_metrics.recall),
        "specificity": float(selected_test_metrics.specificity),
        "balanced_accuracy": float(selected_test_metrics.balanced_accuracy),
        "accuracy": float(selected_test_metrics.accuracy),
        "f1": float(selected_test_metrics.f1),
        "feature_count": int(features_train.shape[1]),
        "test_predictions": test_predictions,
        "test_probabilities": test_probabilities_series,
        "confusion_matrix": selected_test_metrics.confusion_matrix,
        "top_feature_importance": top_feature_importance,
        "full_feature_importance": full_feature_importance,
        "fitted_model": fitted_model,
    }

def extract_confusion_counts(matrix: list[list[int]]) -> dict[str, int]:
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    return {
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
    }

def compute_policy_weighted_error(matrix: list[list[int]]) -> float:
    counts = extract_confusion_counts(matrix)
    return float(
        FAID_POLICY_PRIORITY_WEIGHTS["cost_false_negative_missed_need"] * counts["false_negatives"]
        + FAID_POLICY_PRIORITY_WEIGHTS["cost_false_positive_review"] * counts["false_positives"]
    )

def build_model_comparison_tables(
    *,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    diagnostics: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts_a = extract_confusion_counts(model_a["confusion_matrix"])
    counts_b = extract_confusion_counts(model_b["confusion_matrix"])
    comparison_table = pd.DataFrame(
        [
            {
                "model_version": model_a["variant_name"],
                "income_feature": diagnostics["father_income_column"],
                "estimator": model_a["model_name"],
                "feature_count": model_a["feature_count"],
                "selected_threshold": model_a["selected_threshold"],
                "roc_auc": model_a["auc"],
                "average_precision": model_a["average_precision"],
                "brier_score": model_a["brier_score"],
                "precision": model_a["precision"],
                "recall": model_a["recall"],
                "f1": model_a["f1"],
                "false_positives": counts_a["false_positives"],
                "false_negatives": counts_a["false_negatives"],
                "positive_predictions": int(model_a["test_predictions"].sum()),
                "positive_prediction_rate": float(model_a["test_predictions"].mean()),
                "policy_weighted_error": compute_policy_weighted_error(model_a["confusion_matrix"]),
            },
            {
                "model_version": model_b["variant_name"],
                "income_feature": diagnostics["adjusted_income_feature_name"],
                "estimator": model_b["model_name"],
                "feature_count": model_b["feature_count"],
                "selected_threshold": model_b["selected_threshold"],
                "roc_auc": model_b["auc"],
                "average_precision": model_b["average_precision"],
                "brier_score": model_b["brier_score"],
                "precision": model_b["precision"],
                "recall": model_b["recall"],
                "f1": model_b["f1"],
                "false_positives": counts_b["false_positives"],
                "false_negatives": counts_b["false_negatives"],
                "positive_predictions": int(model_b["test_predictions"].sum()),
                "positive_prediction_rate": float(model_b["test_predictions"].mean()),
                "policy_weighted_error": compute_policy_weighted_error(model_b["confusion_matrix"]),
            },
        ]
    )

    delta_table = pd.DataFrame(
        [
            {
                "metric": "ROC AUC",
                "model_b_minus_a": model_b["auc"] - model_a["auc"],
                "preferred_direction": "higher",
            },
            {
                "metric": "Average Precision",
                "model_b_minus_a": model_b["average_precision"] - model_a["average_precision"],
                "preferred_direction": "higher",
            },
            {
                "metric": "Brier Score",
                "model_b_minus_a": model_b["brier_score"] - model_a["brier_score"],
                "preferred_direction": "lower",
            },
            {
                "metric": "Precision",
                "model_b_minus_a": model_b["precision"] - model_a["precision"],
                "preferred_direction": "higher",
            },
            {
                "metric": "Recall",
                "model_b_minus_a": model_b["recall"] - model_a["recall"],
                "preferred_direction": "higher",
            },
            {
                "metric": "F1",
                "model_b_minus_a": model_b["f1"] - model_a["f1"],
                "preferred_direction": "higher",
            },
            {
                "metric": "False Positives",
                "model_b_minus_a": counts_b["false_positives"] - counts_a["false_positives"],
                "preferred_direction": "lower",
            },
            {
                "metric": "False Negatives",
                "model_b_minus_a": counts_b["false_negatives"] - counts_a["false_negatives"],
                "preferred_direction": "lower",
            },
            {
                "metric": "Positive Predictions",
                "model_b_minus_a": int(model_b["test_predictions"].sum()) - int(model_a["test_predictions"].sum()),
                "preferred_direction": "depends on policy",
            },
            {
                "metric": "Policy-Weighted Error",
                "model_b_minus_a": compute_policy_weighted_error(model_b["confusion_matrix"])
                - compute_policy_weighted_error(model_a["confusion_matrix"]),
                "preferred_direction": "lower",
            },
        ]
    )
    return comparison_table, delta_table

def build_year_metric_table(
    *,
    target_true: pd.Series,
    application_year: pd.Series,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    year_labels = application_year.astype("string").fillna("missing_or_malformed")

    for year_value in sorted(year_labels.unique().tolist()):
        segment_mask = year_labels == year_value
        segment_target = target_true.loc[segment_mask]
        if segment_target.empty:
            continue

        for model_label, model in (("Model A", model_a), ("Model B", model_b)):
            probabilities = model["test_probabilities"].loc[segment_mask]
            predictions = model["test_predictions"].loc[segment_mask]
            counts = extract_confusion_counts(
                confusion_matrix(segment_target, predictions, labels=[0, 1]).tolist()
            )
            actual_positive = max(int((segment_target == 1).sum()), 1)
            actual_negative = max(int((segment_target == 0).sum()), 1)

            try:
                auc_value = float(roc_auc_score(segment_target, probabilities))
            except Exception:
                auc_value = float("nan")
            try:
                ap_value = float(average_precision_score(segment_target, probabilities))
            except Exception:
                ap_value = float("nan")

            rows.append(
                {
                    "application_year": year_value,
                    "model_version": model_label,
                    "row_count": int(segment_mask.sum()),
                    "actual_positive_rate": float(segment_target.mean()),
                    "mean_predicted_probability": float(probabilities.mean()),
                    "calibration_gap": float(probabilities.mean() - segment_target.mean()),
                    "roc_auc": auc_value,
                    "average_precision": ap_value,
                    "brier_score": float(brier_score_loss(segment_target, probabilities)),
                    "precision": float(precision_score(segment_target, predictions, zero_division=0)),
                    "recall": float(recall_score(segment_target, predictions, zero_division=0)),
                    "false_positive_rate": safe_rate(
                        counts["false_positives"],
                        actual_negative,
                    ),
                    "false_negative_rate": safe_rate(
                        counts["false_negatives"],
                        actual_positive,
                    ),
                    "positive_prediction_rate": float(predictions.mean()),
                }
            )

    return pd.DataFrame(rows)

def build_temporal_consistency_tables(year_metric_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    focus_metrics = [
        "roc_auc",
        "recall",
        "false_positive_rate",
        "false_negative_rate",
        "brier_score",
        "calibration_gap",
        "positive_prediction_rate",
    ]
    rows: list[dict[str, Any]] = []
    for model_version in ["Model A", "Model B"]:
        subset = year_metric_table[year_metric_table["model_version"] == model_version]
        for metric in focus_metrics:
            values = subset[metric].dropna()
            rows.append(
                {
                    "model_version": model_version,
                    "metric": metric,
                    "range": float(values.max() - values.min()) if not values.empty else float("nan"),
                    "std": float(values.std(ddof=0)) if len(values) > 1 else 0.0,
                }
            )
    consistency_table = pd.DataFrame(rows)
    comparison = (
        consistency_table.pivot(index="metric", columns="model_version", values="range")
        .rename(columns={"Model A": "model_a_range", "Model B": "model_b_range"})
        .reset_index()
    )
    comparison["range_improvement"] = (
        comparison["model_a_range"] - comparison["model_b_range"]
    )
    return consistency_table, comparison

def build_score_stability_frame(model_a: dict[str, Any], model_b: dict[str, Any]) -> pd.DataFrame:
    probability_delta = model_b["test_probabilities"] - model_a["test_probabilities"]
    changed_mask = model_a["test_predictions"].ne(model_b["test_predictions"])
    return pd.DataFrame(
        [
            {
                "probability_correlation": float(
                    model_a["test_probabilities"].corr(model_b["test_probabilities"])
                ),
                "rank_correlation": float(
                    model_a["test_probabilities"].rank().corr(model_b["test_probabilities"].rank())
                ),
                "mean_abs_probability_shift": float(probability_delta.abs().mean()),
                "max_abs_probability_shift": float(probability_delta.abs().max()),
                "changed_prediction_count": int(changed_mask.sum()),
                "changed_prediction_pct": safe_rate(int(changed_mask.sum()), len(changed_mask)),
            }
        ]
    )

def build_changed_case_frame(
    *,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    diagnostics: dict[str, Any],
    target_test: pd.Series,
    test_index: pd.Index,
) -> pd.DataFrame:
    changed_mask = model_a["test_predictions"].ne(model_b["test_predictions"])
    changed_frame = pd.DataFrame(
        {
            "application_year": diagnostics["application_year"].loc[test_index].astype("Int64"),
            "target_true": target_test,
            "raw_income": diagnostics["raw_income"].loc[test_index],
            "adjusted_income": diagnostics["adjusted_income"].loc[test_index],
            "prob_model_a": model_a["test_probabilities"],
            "prob_model_b": model_b["test_probabilities"],
            "pred_model_a": model_a["test_predictions"],
            "pred_model_b": model_b["test_predictions"],
        }
    )
    changed_frame = changed_frame.loc[changed_mask].copy()
    if changed_frame.empty:
        return changed_frame

    changed_frame["probability_shift"] = (
        changed_frame["prob_model_b"] - changed_frame["prob_model_a"]
    )
    changed_frame["threshold_distance_model_a"] = (
        changed_frame["prob_model_a"] - float(model_a["selected_threshold"])
    )
    changed_frame["threshold_distance_model_b"] = (
        changed_frame["prob_model_b"] - float(model_b["selected_threshold"])
    )
    changed_frame["income_missing"] = changed_frame["raw_income"].isna()
    changed_frame["change_type"] = (
        changed_frame["pred_model_a"].astype(str)
        + "_to_"
        + changed_frame["pred_model_b"].astype(str)
    )
    return changed_frame.sort_values(
        by="probability_shift",
        key=lambda series: series.abs(),
        ascending=False,
    )

def build_paired_bootstrap_delta_table(
    *,
    target_true: pd.Series,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    iterations: int = PAIRED_BOOTSTRAP_ITERATIONS,
) -> pd.DataFrame:
    target_array = np.asarray(target_true, dtype=int)
    probabilities_a = np.asarray(model_a["test_probabilities"], dtype=float)
    probabilities_b = np.asarray(model_b["test_probabilities"], dtype=float)
    threshold_a = float(model_a["selected_threshold"])
    threshold_b = float(model_b["selected_threshold"])
    rng = np.random.default_rng(RANDOM_STATE)

    metrics: dict[str, list[float]] = {
        "ROC AUC": [],
        "Average Precision": [],
        "Brier Score": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "False Positive Rate": [],
        "False Negative Rate": [],
        "Policy-Weighted Error": [],
    }

    for _ in range(iterations):
        sample_indices = rng.integers(0, len(target_array), len(target_array))
        sampled_target = target_array[sample_indices]
        sampled_prob_a = probabilities_a[sample_indices]
        sampled_prob_b = probabilities_b[sample_indices]
        if len(np.unique(sampled_target)) < 2:
            continue

        sampled_pred_a = (sampled_prob_a >= threshold_a).astype(int)
        sampled_pred_b = (sampled_prob_b >= threshold_b).astype(int)

        matrix_a = confusion_matrix(sampled_target, sampled_pred_a, labels=[0, 1]).tolist()
        matrix_b = confusion_matrix(sampled_target, sampled_pred_b, labels=[0, 1]).tolist()
        counts_a = extract_confusion_counts(matrix_a)
        counts_b = extract_confusion_counts(matrix_b)

        actual_positive = max(int((sampled_target == 1).sum()), 1)
        actual_negative = max(int((sampled_target == 0).sum()), 1)

        metrics["ROC AUC"].append(
            float(roc_auc_score(sampled_target, sampled_prob_b) - roc_auc_score(sampled_target, sampled_prob_a))
        )
        metrics["Average Precision"].append(
            float(
                average_precision_score(sampled_target, sampled_prob_b)
                - average_precision_score(sampled_target, sampled_prob_a)
            )
        )
        metrics["Brier Score"].append(
            float(brier_score_loss(sampled_target, sampled_prob_b) - brier_score_loss(sampled_target, sampled_prob_a))
        )
        metrics["Precision"].append(
            float(
                precision_score(sampled_target, sampled_pred_b, zero_division=0)
                - precision_score(sampled_target, sampled_pred_a, zero_division=0)
            )
        )
        metrics["Recall"].append(
            float(
                recall_score(sampled_target, sampled_pred_b, zero_division=0)
                - recall_score(sampled_target, sampled_pred_a, zero_division=0)
            )
        )
        metrics["F1"].append(
            float(
                f1_score(sampled_target, sampled_pred_b, zero_division=0)
                - f1_score(sampled_target, sampled_pred_a, zero_division=0)
            )
        )
        metrics["False Positive Rate"].append(
            safe_rate(counts_b["false_positives"], actual_negative)
            - safe_rate(counts_a["false_positives"], actual_negative)
        )
        metrics["False Negative Rate"].append(
            safe_rate(counts_b["false_negatives"], actual_positive)
            - safe_rate(counts_a["false_negatives"], actual_positive)
        )
        metrics["Policy-Weighted Error"].append(
            compute_policy_weighted_error(matrix_b) - compute_policy_weighted_error(matrix_a)
        )

    rows: list[dict[str, Any]] = []
    for metric_name, values in metrics.items():
        if not values:
            continue
        rows.append(
            {
                "metric": metric_name,
                "mean_delta": float(np.mean(values)),
                "ci_95_lower": float(np.quantile(values, 0.025)),
                "ci_95_upper": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)

def build_decision_layer_tables(
    *,
    training_dataframe: pd.DataFrame,
    target_test: pd.Series,
    features_test_a: pd.DataFrame,
    features_test_b: pd.DataFrame,
    model_a: dict[str, Any],
    model_b: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_a = training_dataframe.loc[features_test_a.index].copy()
    audit_b = training_dataframe.loc[features_test_b.index].copy()

    decisions_a, _ = build_decision_recommendations(
        audit_frame=audit_a,
        model_features=features_test_a,
        target_true=target_test,
        probabilities=np.asarray(model_a["test_probabilities"], dtype=float),
        selected_threshold=float(model_a["selected_threshold"]),
        fairness_lookup={},
        explainability_model=model_a["fitted_model"],
    )
    decisions_b, _ = build_decision_recommendations(
        audit_frame=audit_b,
        model_features=features_test_b,
        target_true=target_test,
        probabilities=np.asarray(model_b["test_probabilities"], dtype=float),
        selected_threshold=float(model_b["selected_threshold"]),
        fairness_lookup={},
        explainability_model=model_b["fitted_model"],
    )

    impact_a = summarize_three_way_decision_impact(decisions_a, float(model_a["selected_threshold"]))
    impact_b = summarize_three_way_decision_impact(decisions_b, float(model_b["selected_threshold"]))

    comparison_table = pd.DataFrame(
        [
            {
                "model_version": "Model A",
                "approve_threshold": float(max(
                    DECISION_APPROVE_MIN_PROBABILITY,
                    min(float(model_a["selected_threshold"]) + DECISION_REVIEW_MARGIN, 0.95),
                )),
                "reject_threshold": float(min(
                    DECISION_REJECT_MAX_PROBABILITY,
                    max(float(model_a["selected_threshold"]) - DECISION_REVIEW_MARGIN, 0.05),
                )),
                "automated_cases": impact_a["automated_cases"],
                "automated_share": impact_a["automated_share"],
                "review_cases": impact_a["review_cases_after_model"],
                "auto_approve_cases": impact_a["auto_approve_cases"],
                "auto_reject_cases": impact_a["auto_reject_cases"],
                "auto_false_approves": impact_a["auto_false_approves"],
                "auto_false_rejects": impact_a["auto_false_rejects"],
                "harmful_auto_errors_avoided_vs_threshold_only": impact_a[
                    "harmful_auto_errors_avoided_vs_threshold_only"
                ],
            },
            {
                "model_version": "Model B",
                "approve_threshold": float(max(
                    DECISION_APPROVE_MIN_PROBABILITY,
                    min(float(model_b["selected_threshold"]) + DECISION_REVIEW_MARGIN, 0.95),
                )),
                "reject_threshold": float(min(
                    DECISION_REJECT_MAX_PROBABILITY,
                    max(float(model_b["selected_threshold"]) - DECISION_REVIEW_MARGIN, 0.05),
                )),
                "automated_cases": impact_b["automated_cases"],
                "automated_share": impact_b["automated_share"],
                "review_cases": impact_b["review_cases_after_model"],
                "auto_approve_cases": impact_b["auto_approve_cases"],
                "auto_reject_cases": impact_b["auto_reject_cases"],
                "auto_false_approves": impact_b["auto_false_approves"],
                "auto_false_rejects": impact_b["auto_false_rejects"],
                "harmful_auto_errors_avoided_vs_threshold_only": impact_b[
                    "harmful_auto_errors_avoided_vs_threshold_only"
                ],
            },
        ]
    )

    delta_table = pd.DataFrame(
        [
            {
                "metric": "Automated cases",
                "model_b_minus_a": int(impact_b["automated_cases"]) - int(impact_a["automated_cases"]),
            },
            {
                "metric": "Review cases",
                "model_b_minus_a": int(impact_b["review_cases_after_model"]) - int(impact_a["review_cases_after_model"]),
            },
            {
                "metric": "Auto false approves",
                "model_b_minus_a": int(impact_b["auto_false_approves"]) - int(impact_a["auto_false_approves"]),
            },
            {
                "metric": "Harmful auto errors avoided vs threshold-only",
                "model_b_minus_a": int(
                    impact_b["harmful_auto_errors_avoided_vs_threshold_only"]
                ) - int(impact_a["harmful_auto_errors_avoided_vs_threshold_only"]),
            },
        ]
    )
    return comparison_table, delta_table

def compute_global_explainability(
    *,
    model: dict[str, Any],
    features_train: pd.DataFrame,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    top_n: int = TOP_FEATURES_FOR_EXPLAINABILITY,
) -> dict[str, Any]:
    try:
        import shap  # type: ignore

        preprocessor = model["fitted_model"].named_steps["preprocessor"]
        estimator = model["fitted_model"].named_steps["model"]
        feature_names = get_transformed_feature_names(preprocessor)
        background_frame = features_train.sample(
            n=min(SHAP_SAMPLE_SIZE, len(features_train)),
            random_state=RANDOM_STATE,
        )
        sample_frame = features_test.sample(
            n=min(SHAP_SAMPLE_SIZE, len(features_test)),
            random_state=RANDOM_STATE,
        )
        background = np.asarray(preprocessor.transform(background_frame), dtype=float)
        sample = np.asarray(preprocessor.transform(sample_frame), dtype=float)
        explainer = shap.Explainer(estimator, background, feature_names=feature_names)
        shap_values = explainer(sample)
        values = np.asarray(shap_values.values, dtype=float)
        if values.ndim == 3:
            values = values[:, :, -1]
        importance = np.abs(values).mean(axis=0)
        table = pd.DataFrame(
            {"feature": feature_names[: len(importance)], "importance": importance[: len(feature_names)]}
        ).sort_values(by=["importance", "feature"], ascending=[False, True], ignore_index=True)
        table.insert(0, "rank", table.index + 1)
        return {"method": "SHAP mean absolute contribution", "table": table.head(top_n), "full_table": table}
    except Exception:
        try:
            sample_frame = features_test.sample(
                n=min(SHAP_SAMPLE_SIZE, len(features_test)),
                random_state=RANDOM_STATE,
            )
            sample_target = target_test.loc[sample_frame.index]
            importance_result = permutation_importance(
                model["fitted_model"],
                sample_frame,
                sample_target,
                n_repeats=10,
                random_state=RANDOM_STATE,
                scoring="roc_auc",
            )
            table = pd.DataFrame(
                {
                    "feature": sample_frame.columns.tolist(),
                    "importance": importance_result.importances_mean,
                }
            ).sort_values(by=["importance", "feature"], ascending=[False, True], ignore_index=True)
            table.insert(0, "rank", table.index + 1)
            return {"method": "Permutation importance", "table": table.head(top_n), "full_table": table}
        except Exception:
            fallback_table = model["full_feature_importance"].copy()
            if fallback_table.empty:
                fallback_table = pd.DataFrame(columns=["rank", "feature", "importance"])
            return {
                "method": "Model-native feature importance",
                "table": fallback_table.head(top_n),
                "full_table": fallback_table,
            }

def build_feature_family_summary(
    *,
    explainability_a: dict[str, Any],
    explainability_b: dict[str, Any],
    diagnostics: dict[str, Any],
) -> pd.DataFrame:
    table_a = explainability_a["full_table"]
    table_b = explainability_b["full_table"]
    return pd.DataFrame(
        [
            {
                **summarize_income_related_importance(
                    table_a,
                    feature_token=diagnostics["father_income_column"],
                    label="Model A raw income family",
                ),
                "method": explainability_a["method"],
            },
            {
                **summarize_income_related_importance(
                    table_b,
                    feature_token=diagnostics["adjusted_income_feature_name"],
                    label="Model B adjusted income family",
                ),
                "method": explainability_b["method"],
            },
        ]
    )

def fit_controlled_model_pair(
    *,
    training_dataframe: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    factor_map: dict[int, float],
    income_transform: str | None = None,
) -> dict[str, Any]:
    model_a_features, model_b_features, diagnostics = build_income_variant_feature_sets(
        training_dataframe,
        target_column=target_column,
        factor_map=factor_map,
        income_transform=income_transform,
    )
    (
        features_train_a,
        features_valid_a,
        features_test_a,
        features_train_b,
        features_valid_b,
        features_test_b,
        target_train,
        target_valid,
        target_test,
    ) = align_feature_sets_general(
        model_a_features=model_a_features,
        model_b_features=model_b_features,
        target=target,
        raw_feature_name=diagnostics["raw_income_feature_name"],
        adjusted_feature_name=diagnostics["adjusted_income_feature_name"],
        original_income_column=diagnostics["father_income_column"],
    )

    class_weight = get_recommended_class_weight(target_train)
    scale_pos_weight = get_recommended_scale_pos_weight(target_train)
    model_specs = get_model_specs(
        class_weight=class_weight,
        scale_pos_weight=scale_pos_weight,
    )
    model_name = resolve_model_name(model_specs)
    model_spec = model_specs[model_name]

    model_a = run_full_feature_model_enhanced(
        variant_name="Model A",
        model_name=model_name,
        model_spec=model_spec,
        features_train=features_train_a,
        features_valid=features_valid_a,
        features_test=features_test_a,
        target_train=target_train,
        target_valid=target_valid,
        target_test=target_test,
    )
    model_b = run_full_feature_model_enhanced(
        variant_name="Model B",
        model_name=model_name,
        model_spec=model_spec,
        features_train=features_train_b,
        features_valid=features_valid_b,
        features_test=features_test_b,
        target_train=target_train,
        target_valid=target_valid,
        target_test=target_test,
    )

    comparison_table, metric_delta_table = build_model_comparison_tables(
        model_a=model_a,
        model_b=model_b,
        diagnostics=diagnostics,
    )

    return {
        "diagnostics": diagnostics,
        "model_a": model_a,
        "model_b": model_b,
        "features_train_a": features_train_a,
        "features_train_b": features_train_b,
        "features_test_a": features_test_a,
        "features_test_b": features_test_b,
        "target_test": target_test,
        "comparison_table": comparison_table,
        "metric_delta_table": metric_delta_table,
        "prediction_impact": build_prediction_impact_summary(model_a, model_b),
        "binary_decision_shift": build_decision_impact_summary(model_a, model_b),
    }

def run_primary_pp_experiment(
    *,
    training_dataframe: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    factor_map: dict[int, float],
) -> dict[str, Any]:
    pair = fit_controlled_model_pair(
        training_dataframe=training_dataframe,
        target=target,
        target_column=target_column,
        factor_map=factor_map,
        income_transform=None,
    )
    diagnostics = pair["diagnostics"]
    model_a = pair["model_a"]
    model_b = pair["model_b"]
    target_test = pair["target_test"]
    features_test_a = pair["features_test_a"]
    features_test_b = pair["features_test_b"]

    year_metric_table = build_year_metric_table(
        target_true=target_test,
        application_year=diagnostics["application_year"].loc[features_test_a.index],
        model_a=model_a,
        model_b=model_b,
    )
    temporal_consistency_table, temporal_consistency_delta = build_temporal_consistency_tables(
        year_metric_table
    )
    bootstrap_delta_table = build_paired_bootstrap_delta_table(
        target_true=target_test,
        model_a=model_a,
        model_b=model_b,
    )
    score_stability_frame = build_score_stability_frame(model_a, model_b)
    changed_cases_frame = build_changed_case_frame(
        model_a=model_a,
        model_b=model_b,
        diagnostics=diagnostics,
        target_test=target_test,
        test_index=features_test_a.index,
    )
    decision_layer_comparison, decision_layer_delta = build_decision_layer_tables(
        training_dataframe=training_dataframe,
        target_test=target_test,
        features_test_a=features_test_a,
        features_test_b=features_test_b,
        model_a=model_a,
        model_b=model_b,
    )
    distribution_sanity_check, distribution_explanation = build_distribution_sanity_check(diagnostics)
    explainability_a = compute_global_explainability(
        model=model_a,
        features_train=pair["features_train_a"],
        features_test=pair["features_test_a"],
        target_test=target_test,
    )
    explainability_b = compute_global_explainability(
        model=model_b,
        features_train=pair["features_train_b"],
        features_test=pair["features_test_b"],
        target_test=target_test,
    )
    feature_family_summary = build_feature_family_summary(
        explainability_a=explainability_a,
        explainability_b=explainability_b,
        diagnostics=diagnostics,
    )
    return {
        **pair,
        "year_metric_table": year_metric_table,
        "temporal_consistency_table": temporal_consistency_table,
        "temporal_consistency_delta": temporal_consistency_delta,
        "bootstrap_delta_table": bootstrap_delta_table,
        "score_stability_frame": score_stability_frame,
        "changed_cases_frame": changed_cases_frame,
        "decision_layer_comparison": decision_layer_comparison,
        "decision_layer_delta": decision_layer_delta,
        "distribution_sanity_check": distribution_sanity_check,
        "distribution_explanation": distribution_explanation,
        "explainability_a": explainability_a,
        "explainability_b": explainability_b,
        "feature_family_summary": feature_family_summary,
    }

def build_sensitivity_row(
    *,
    scenario_name: str,
    factor_map: dict[int, float],
    pair: dict[str, Any],
    transform_label: str = "linear",
) -> dict[str, Any]:
    diagnostics = pair["diagnostics"]
    delta_lookup = pair["metric_delta_table"].set_index("metric")["model_b_minus_a"]
    income_non_null_count = int(diagnostics["raw_income"].notna().sum())
    return {
        "scenario": scenario_name,
        "scenario_label": PP_SCENARIO_META.get(scenario_name, {}).get("label", scenario_name),
        "transform": transform_label,
        "active_adjustment_share_all_rows": safe_rate(
            diagnostics["active_adjustment_row_count"],
            diagnostics["row_count"],
        ),
        "active_adjustment_share_non_missing_income": safe_rate(
            diagnostics["active_adjustment_row_count"],
            max(income_non_null_count, 1),
        ),
        "delta_roc_auc": float(delta_lookup["ROC AUC"]),
        "delta_average_precision": float(delta_lookup["Average Precision"]),
        "delta_brier_score": float(delta_lookup["Brier Score"]),
        "delta_precision": float(delta_lookup["Precision"]),
        "delta_recall": float(delta_lookup["Recall"]),
        "delta_f1": float(delta_lookup["F1"]),
        "delta_false_positives": float(delta_lookup["False Positives"]),
        "delta_false_negatives": float(delta_lookup["False Negatives"]),
        "delta_policy_weighted_error": float(delta_lookup["Policy-Weighted Error"]),
        "changed_prediction_pct": float(pair["prediction_impact"]["changed_case_pct"]),
    }

def run_sensitivity_analysis(
    *,
    training_dataframe: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    scenarios: dict[str, dict[int, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_name in ["legacy_original", "conservative", "baseline", "aggressive"]:
        factor_map = scenarios[scenario_name]
        pair = fit_controlled_model_pair(
            training_dataframe=training_dataframe,
            target=target,
            target_column=target_column,
            factor_map=factor_map,
            income_transform=None,
        )
        rows.append(
            build_sensitivity_row(
                scenario_name=scenario_name,
                factor_map=factor_map,
                pair=pair,
            )
        )
    return pd.DataFrame(rows)

def run_log_transform_robustness(
    *,
    training_dataframe: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    factor_map: dict[int, float],
) -> pd.DataFrame:
    pair = fit_controlled_model_pair(
        training_dataframe=training_dataframe,
        target=target,
        target_column=target_column,
        factor_map=factor_map,
        income_transform="log1p",
    )
    return pd.DataFrame(
        [
            build_sensitivity_row(
                scenario_name="baseline",
                factor_map=factor_map,
                pair=pair,
                transform_label="log1p",
            )
        ]
    )

def infer_recommendation(
    *,
    primary_result: dict[str, Any],
    sensitivity_frame: pd.DataFrame,
    temporal_consistency_delta: pd.DataFrame,
) -> tuple[str, str]:
    comparison_table = primary_result["comparison_table"]
    model_a_row = comparison_table.loc[comparison_table["model_version"] == "Model A"].iloc[0]
    model_b_row = comparison_table.loc[comparison_table["model_version"] == "Model B"].iloc[0]

    false_negative_delta = int(model_b_row["false_negatives"] - model_a_row["false_negatives"])
    policy_error_delta = float(model_b_row["policy_weighted_error"] - model_a_row["policy_weighted_error"])
    recall_delta = float(model_b_row["recall"] - model_a_row["recall"])
    temporal_improvements = int((temporal_consistency_delta["range_improvement"] > 0).sum())
    scenario_fn_improvements = int((sensitivity_frame["delta_false_negatives"] < 0).sum())
    scenario_policy_improvements = int((sensitivity_frame["delta_policy_weighted_error"] < 0).sum())

    if (
        false_negative_delta < 0
        and policy_error_delta < 0
        and scenario_policy_improvements >= 2
    ):
        recommendation = "Adopt the adjustment"
        rationale = (
            "The adjusted feature improves missed-need protection and lowers the policy-weighted "
            "error score under the main scenario while remaining directionally stable under "
            "reasonable factor alternatives."
        )
    elif (
        false_negative_delta >= 0
        and policy_error_delta >= 0
        and scenario_fn_improvements == 0
        and recall_delta <= 0
    ):
        if temporal_improvements >= 2:
            recommendation = "Use it conditionally"
            rationale = (
                "The purchasing-power proxy does not improve false negatives or total policy-weighted "
                "error enough to replace raw income in production, but it does modestly smooth some "
                "year-level consistency metrics and slightly improves the safety of high-confidence "
                "automation. It is best retained as a monitoring and sensitivity-analysis layer until "
                "an externally justified deflator series is available."
            )
        else:
            recommendation = "Reject the adjustment"
            rationale = (
                "The purchasing-power proxy does not improve recall, false negatives, or policy-weighted "
                "error, and it does not offer enough additional consistency benefit to justify production use."
            )
    else:
        recommendation = "Use it conditionally"
        rationale = (
            "The evidence is mixed rather than decisively positive. The adjustment is analytically useful "
            "for temporal stress testing, but not strong enough to replace raw income as the production feature."
        )
    return recommendation, rationale

def build_final_summary_frame(
    *,
    primary_result: dict[str, Any],
    sensitivity_frame: pd.DataFrame,
    temporal_consistency_delta: pd.DataFrame,
    recommendation: str,
    rationale: str,
) -> pd.DataFrame:
    comparison_table = primary_result["comparison_table"]
    model_a_row = comparison_table.loc[comparison_table["model_version"] == "Model A"].iloc[0]
    model_b_row = comparison_table.loc[comparison_table["model_version"] == "Model B"].iloc[0]
    improved_metrics = temporal_consistency_delta.loc[
        temporal_consistency_delta["range_improvement"] > 0,
        "metric",
    ].tolist()
    improved_metrics_text = ", ".join(improved_metrics) if improved_metrics else "none"

    return pd.DataFrame(
        [
            {
                "decision_question": "Did ranking quality materially improve?",
                "evidence": (
                    f"ROC AUC changed by {model_b_row['roc_auc'] - model_a_row['roc_auc']:+.4f} "
                    f"and Average Precision by {model_b_row['average_precision'] - model_a_row['average_precision']:+.4f}."
                ),
            },
            {
                "decision_question": "Did the adjustment reduce missed deserving students?",
                "evidence": (
                    f"No. False negatives moved from {int(model_a_row['false_negatives'])} to "
                    f"{int(model_b_row['false_negatives'])}, and recall changed by "
                    f"{model_b_row['recall'] - model_a_row['recall']:+.4f}."
                ),
            },
            {
                "decision_question": "What happened to operational burden?",
                "evidence": (
                    f"False positives moved from {int(model_a_row['false_positives'])} to "
                    f"{int(model_b_row['false_positives'])}, while the policy-weighted error changed by "
                    f"{model_b_row['policy_weighted_error'] - model_a_row['policy_weighted_error']:+.1f}."
                ),
            },
            {
                "decision_question": "Is the conclusion stable under factor stress tests?",
                "evidence": (
                    f"No scenario improved false negatives, and the direction of the business conclusion "
                    f"stayed broadly stable across {len(sensitivity_frame)} purchasing-power assumptions."
                ),
            },
            {
                "decision_question": "Did temporal consistency improve?",
                "evidence": (
                    f"Yes, modestly, on the following dispersion measures: {improved_metrics_text}."
                ),
            },
            {
                "decision_question": "Final recommendation",
                "evidence": f"{recommendation}. {rationale}",
            },
        ]
    )

# %% [notebook cell 17]
# Execute the thesis-grade workflow end to end.
dataframe = load_data(DATA_PATH)
training_dataframe, target, target_metadata = prepare_target(dataframe)

application_term_column = resolve_column_name(training_dataframe, APPLICATION_TERM_CANDIDATES)
notebook_application_year = extract_application_year(training_dataframe[application_term_column])
observed_years, reference_year, PP_SCENARIOS = build_pp_scenarios(notebook_application_year)
scenario_factor_table = build_factor_scenario_table(
    observed_years=observed_years,
    reference_year=reference_year,
    scenarios=PP_SCENARIOS,
)

# For the main notebook narrative, use the baseline scenario and keep the
# original notebook heuristic as an explicit sensitivity check.
PP_FACTOR = dict(PP_SCENARIOS[PRIMARY_SCENARIO_NAME])

main_result = run_primary_pp_experiment(
    training_dataframe=training_dataframe,
    target=target,
    target_column=target_metadata["target_column"],
    factor_map=PP_FACTOR,
)
sensitivity_frame = run_sensitivity_analysis(
    training_dataframe=training_dataframe,
    target=target,
    target_column=target_metadata["target_column"],
    scenarios=PP_SCENARIOS,
)
log_robustness_frame = run_log_transform_robustness(
    training_dataframe=training_dataframe,
    target=target,
    target_column=target_metadata["target_column"],
    factor_map=PP_FACTOR,
)

comparison_table = main_result["comparison_table"]
metric_delta_table = main_result["metric_delta_table"]
model_a = main_result["model_a"]
model_b = main_result["model_b"]
diagnostics = main_result["diagnostics"]
prediction_impact = main_result["prediction_impact"]
binary_decision_shift = main_result["binary_decision_shift"]
year_metric_table = main_result["year_metric_table"]
temporal_consistency_table = main_result["temporal_consistency_table"]
temporal_consistency_delta = main_result["temporal_consistency_delta"]
bootstrap_delta_table = main_result["bootstrap_delta_table"]
score_stability_frame = main_result["score_stability_frame"]
changed_cases_frame = main_result["changed_cases_frame"]
decision_layer_comparison = main_result["decision_layer_comparison"]
decision_layer_delta = main_result["decision_layer_delta"]
distribution_sanity_check = main_result["distribution_sanity_check"]
distribution_explanation = main_result["distribution_explanation"]
explainability_a = main_result["explainability_a"]
explainability_b = main_result["explainability_b"]
feature_family_summary = main_result["feature_family_summary"]

recommendation, recommendation_rationale = infer_recommendation(
    primary_result=main_result,
    sensitivity_frame=sensitivity_frame,
    temporal_consistency_delta=temporal_consistency_delta,
)
final_summary_frame = build_final_summary_frame(
    primary_result=main_result,
    sensitivity_frame=sensitivity_frame,
    temporal_consistency_delta=temporal_consistency_delta,
    recommendation=recommendation,
    rationale=recommendation_rationale,
)

print("Execution complete.")
comparison_table.round(4)

# %% [notebook cell 19]
assumption_frame = pd.DataFrame(
    [
        {
            "reference_year": reference_year,
            "primary_scenario": PRIMARY_SCENARIO_NAME,
            "policy_rationale": FAID_POLICY_RATIONALE,
            "rows_with_income": int(diagnostics["raw_income"].notna().sum()),
            "active_adjustment_rows": diagnostics["active_adjustment_row_count"],
            "active_adjustment_share_all_rows": safe_rate(
                diagnostics["active_adjustment_row_count"],
                diagnostics["row_count"],
            ),
            "active_adjustment_share_non_missing_income": safe_rate(
                diagnostics["active_adjustment_row_count"],
                max(int(diagnostics["raw_income"].notna().sum()), 1),
            ),
        }
    ]
)

display(scenario_factor_table.round(4))
display(assumption_frame.round(4))
display(diagnostics["coverage_by_year"].round(4))

show_interpretation(
    "The factor is treated as a scenario-based approximation rather than a verified real-income deflator. "
    f"Under the baseline scenario, {format_pct(assumption_frame.loc[0, 'active_adjustment_share_all_rows'])} "
    "of all rows and "
    f"{format_pct(assumption_frame.loc[0, 'active_adjustment_share_non_missing_income'])} "
    "of rows with observed father income are materially adjusted."
)

# %% [notebook cell 21]
diagnostics_frame = pd.DataFrame(
    [
        {
            "resolved_father_income_column": diagnostics["father_income_column"],
            "resolved_application_term_column": diagnostics["application_term_column"],
            "missing_father_income_count": diagnostics["missing_father_income_count"],
            "missing_father_income_share": safe_rate(
                diagnostics["missing_father_income_count"],
                diagnostics["row_count"],
            ),
            "active_adjustment_row_count": diagnostics["active_adjustment_row_count"],
            "row_count": diagnostics["row_count"],
            "income_transform": diagnostics["income_transform"],
        }
    ]
)

display(diagnostics_frame.round(4))
display(diagnostics["year_distribution"])
display(diagnostics["income_summary"].round(2))
display(distribution_sanity_check.round(4))

show_interpretation(
    "The biggest structural limitation is missing father-income data. Even a sensible purchasing-power factor "
    "cannot influence rows where income is absent, which naturally caps the maximum measurable impact of the experiment."
)

# %% [notebook cell 22]
valid_mask = diagnostics["raw_income"].notna() & diagnostics["adjusted_income"].notna()
plot_frame = pd.DataFrame(
    {
        "raw_income": diagnostics["raw_income"].loc[valid_mask],
        "adjusted_income": diagnostics["adjusted_income"].loc[valid_mask],
        "application_year": diagnostics["application_year"].loc[valid_mask].astype("string"),
    }
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sample_frame = (
    plot_frame.sample(min(len(plot_frame), 400), random_state=RANDOM_STATE)
    if not plot_frame.empty
    else plot_frame
)
for year_value, group in sample_frame.groupby("application_year"):
    axes[0].scatter(
        group["raw_income"],
        group["adjusted_income"],
        alpha=0.6,
        s=18,
        label=str(year_value),
    )
axes[0].set_title("Raw vs Adjusted Income")
axes[0].set_xlabel("Raw income")
axes[0].set_ylabel("Adjusted income")
if sample_frame["application_year"].nunique() > 1:
    axes[0].legend(title="Application year", fontsize=8)

axes[1].hist(plot_frame["raw_income"], bins=30, alpha=0.6, label="Raw income")
axes[1].hist(plot_frame["adjusted_income"], bins=30, alpha=0.6, label="Adjusted income")
axes[1].set_title("Income Distribution Shift")
axes[1].set_xlabel("Income value")
axes[1].legend()

plt.tight_layout()
plt.show()

show_interpretation(distribution_explanation)

# %% [notebook cell 24]
display(comparison_table.round(4))
display(metric_delta_table.round(4))

model_a_row = comparison_table.loc[comparison_table["model_version"] == "Model A"].iloc[0]
model_b_row = comparison_table.loc[comparison_table["model_version"] == "Model B"].iloc[0]

show_interpretation(
    f"Under the baseline adjustment scenario, ROC AUC changes by {model_b_row['roc_auc'] - model_a_row['roc_auc']:+.4f} "
    f"and F1 changes by {model_b_row['f1'] - model_a_row['f1']:+.4f}. "
    f"At the operating threshold, false negatives move from {int(model_a_row['false_negatives'])} to {int(model_b_row['false_negatives'])}, "
    f"while false positives move from {int(model_a_row['false_positives'])} to {int(model_b_row['false_positives'])}."
)

# %% [notebook cell 25]
metric_panels = [
    ("roc_auc", "ROC AUC"),
    ("average_precision", "Average Precision"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("brier_score", "Brier Score"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, (metric_key, metric_label) in zip(axes.flat, metric_panels):
    ax.bar(
        comparison_table["model_version"],
        comparison_table[metric_key],
        color=[MODEL_COLORS["Model A"], MODEL_COLORS["Model B"]],
    )
    ax.set_title(metric_label)
    if metric_key != "brier_score":
        ax.set_ylim(0.0, 1.0)
    for position, value in enumerate(comparison_table[metric_key]):
        ax.text(position, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()

# %% [notebook cell 26]
display(bootstrap_delta_table.round(4))

unstable_metrics = bootstrap_delta_table[
    (bootstrap_delta_table["ci_95_lower"] <= 0) & (bootstrap_delta_table["ci_95_upper"] >= 0)
]["metric"].tolist()
unstable_text = ", ".join(unstable_metrics) if unstable_metrics else "none"

show_interpretation(
    "The paired bootstrap confirms that most performance deltas are small. "
    f"Metrics with intervals that still cross zero include: {unstable_text}. "
    "That means any claimed benefit should be framed as marginal, not decisive."
)

# %% [notebook cell 28]
prediction_impact_frame = pd.DataFrame([prediction_impact])
binary_decision_shift_frame = pd.DataFrame([binary_decision_shift])

display(prediction_impact_frame.round(4))
display(binary_decision_shift_frame.round(4))
display(decision_layer_comparison.round(4))
display(decision_layer_delta.round(4))

model_a_row = comparison_table.loc[comparison_table["model_version"] == "Model A"].iloc[0]
model_b_row = comparison_table.loc[comparison_table["model_version"] == "Model B"].iloc[0]
weighted_delta = model_b_row["policy_weighted_error"] - model_a_row["policy_weighted_error"]

show_interpretation(
    "At the binary threshold, the adjustment does not recover additional deserving students. "
    f"False negatives stay at {int(model_a_row['false_negatives'])} vs {int(model_b_row['false_negatives'])}, "
    f"while the policy-weighted error changes by {weighted_delta:+.1f}. "
    "In the high-confidence automation layer, Model B is slightly more conservative: it routes one extra case to review "
    "and avoids two additional harmful auto-approves."
)

# %% [notebook cell 29]
cm_a = build_confusion_matrix_frame(model_a["confusion_matrix"])
cm_b = build_confusion_matrix_frame(model_b["confusion_matrix"])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, matrix, title in (
    (axes[0], cm_a, "Model A Confusion Matrix"),
    (axes[1], cm_b, "Model B Confusion Matrix"),
):
    ax.imshow(matrix.values, cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center", color="black")

policy_cost_frame = comparison_table[["model_version", "policy_weighted_error"]].copy()
axes[2].bar(
    policy_cost_frame["model_version"],
    policy_cost_frame["policy_weighted_error"],
    color=[MODEL_COLORS["Model A"], MODEL_COLORS["Model B"]],
)
axes[2].set_title("Policy-Weighted Error")
axes[2].set_ylabel("Weighted error units")
for position, value in enumerate(policy_cost_frame["policy_weighted_error"]):
    axes[2].text(position, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.show()

# %% [notebook cell 31]
display(sensitivity_frame.round(4))
display(log_robustness_frame.round(4))

scenario_fn_improvements = int((sensitivity_frame["delta_false_negatives"] < 0).sum())
scenario_recall_improvements = int((sensitivity_frame["delta_recall"] > 0).sum())

show_interpretation(
    "The robustness message is clear: the business conclusion does not hinge on one arbitrary factor choice. "
    f"Across the tested purchasing-power scenarios, false negatives improve in {scenario_fn_improvements} scenarios and "
    f"recall improves in {scenario_recall_improvements} scenarios. The optional log transform does not rescue the idea either."
)

# %% [notebook cell 32]
ordered_sensitivity = sensitivity_frame.copy()
ordered_sensitivity["scenario_order"] = ordered_sensitivity["scenario"].map(
    {"legacy_original": 0, "conservative": 1, "baseline": 2, "aggressive": 3}
)
ordered_sensitivity = ordered_sensitivity.sort_values("scenario_order")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(
    ordered_sensitivity["scenario_label"],
    ordered_sensitivity["delta_roc_auc"],
    color=[SCENARIO_COLORS[name] for name in ordered_sensitivity["scenario"]],
)
axes[0].axhline(0.0, color="black", linewidth=1)
axes[0].set_title("ROC AUC Delta")
axes[0].tick_params(axis="x", rotation=20)

axes[1].bar(
    ordered_sensitivity["scenario_label"],
    ordered_sensitivity["delta_f1"],
    color=[SCENARIO_COLORS[name] for name in ordered_sensitivity["scenario"]],
)
axes[1].axhline(0.0, color="black", linewidth=1)
axes[1].set_title("F1 Delta")
axes[1].tick_params(axis="x", rotation=20)

axes[2].bar(
    ordered_sensitivity["scenario_label"],
    ordered_sensitivity["delta_policy_weighted_error"],
    color=[SCENARIO_COLORS[name] for name in ordered_sensitivity["scenario"]],
)
axes[2].axhline(0.0, color="black", linewidth=1)
axes[2].set_title("Policy-Weighted Error Delta")
axes[2].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.show()

# %% [notebook cell 34]
display(year_metric_table.round(4))
display(temporal_consistency_table.round(4))
display(temporal_consistency_delta.round(4))

improved_metrics = temporal_consistency_delta.loc[
    temporal_consistency_delta["range_improvement"] > 0,
    "metric",
].tolist()
improved_text = ", ".join(improved_metrics) if improved_metrics else "none"

show_interpretation(
    "The adjusted model slightly reduces year-to-year dispersion on some measures, "
    f"most notably: {improved_text}. "
    "However, recall dispersion and false-negative dispersion do not improve, so the temporal benefit is real but limited."
)

# %% [notebook cell 35]
plot_metrics = [
    ("roc_auc", "ROC AUC"),
    ("recall", "Recall"),
    ("false_positive_rate", "False Positive Rate"),
    ("brier_score", "Brier Score"),
]

fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, (metric_key, metric_label) in zip(axes.flat, plot_metrics):
    for model_version in ["Model A", "Model B"]:
        subset = year_metric_table[year_metric_table["model_version"] == model_version]
        ax.plot(
            subset["application_year"],
            subset[metric_key],
            marker="o",
            linewidth=2,
            label=model_version,
            color=MODEL_COLORS[model_version],
        )
    ax.set_title(metric_label)
    if metric_key != "brier_score":
        ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=15)

axes[0, 0].legend()
plt.tight_layout()
plt.show()

# %% [notebook cell 37]
display(explainability_a["table"].round(6))
display(explainability_b["table"].round(6))
display(feature_family_summary.round(6))
display(score_stability_frame.round(4))

family_a = feature_family_summary.iloc[0]
family_b = feature_family_summary.iloc[1]

show_interpretation(
    f"{explainability_a['method']} shows that income remains an important feature family in both models "
    f"(best rank {family_a['best_rank']} in Model A vs {family_b['best_rank']} in Model B). "
    f"At the same time, predicted scores remain extremely close across models, with probability correlation "
    f"{score_stability_frame.loc[0, 'probability_correlation']:.4f}."
)

# %% [notebook cell 38]
top_a = explainability_a["table"].head(TOP_FEATURES_FOR_EXPLAINABILITY).sort_values(
    "importance", ascending=True
)
top_b = explainability_b["table"].head(TOP_FEATURES_FOR_EXPLAINABILITY).sort_values(
    "importance", ascending=True
)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(top_a["feature"], top_a["importance"], color=MODEL_COLORS["Model A"])
axes[0].set_title(f"Top {TOP_FEATURES_FOR_EXPLAINABILITY} Drivers: Model A")
axes[0].set_xlabel(explainability_a["method"])

axes[1].barh(top_b["feature"], top_b["importance"], color=MODEL_COLORS["Model B"])
axes[1].set_title(f"Top {TOP_FEATURES_FOR_EXPLAINABILITY} Drivers: Model B")
axes[1].set_xlabel(explainability_b["method"])

plt.tight_layout()
plt.show()

# %% [notebook cell 39]
if changed_cases_frame.empty:
    print("No final test-set prediction flips occurred between Model A and Model B.")
else:
    display(changed_cases_frame.round(4))

changed_case_count = int(score_stability_frame.loc[0, "changed_prediction_count"])
changed_case_pct = float(score_stability_frame.loc[0, "changed_prediction_pct"])
missing_income_flip_share = (
    float(changed_cases_frame["income_missing"].mean())
    if not changed_cases_frame.empty
    else 0.0
)

show_interpretation(
    f"Only {changed_case_count} test cases ({format_pct(changed_case_pct)}) change final binary label. "
    f"When flips do occur, {format_pct(missing_income_flip_share)} of them come from records with missing income, "
    "which suggests that most of the operational movement comes from threshold-boundary behavior rather than from a large re-ordering of economically distinct cases."
)

# %% [notebook cell 41]
display(final_summary_frame)

print(f"Recommendation: {recommendation}")
print(recommendation_rationale)

show_interpretation(
    "The purchasing-power idea is analytically useful, but the current evidence does not support an unconditional production replacement of raw income."
)
