# Extracted notebook code for portfolio review.
# Generated from a sanitized notebook copy; outputs and execution state are intentionally excluded.


# %% [notebook cell 2]
from pathlib import Path
import importlib.util
import json
import os
import re
import subprocess
import sys
import warnings

REQUIRED_NOTEBOOK_PACKAGES = {
    "joblib": "joblib",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
}

missing_packages = [
    pip_name
    for module_name, pip_name in REQUIRED_NOTEBOOK_PACKAGES.items()
    if importlib.util.find_spec(module_name) is None
]

if missing_packages:
    print("Installing missing notebook dependencies:", ", ".join(missing_packages))
    install_commands = [
        [sys.executable, "-m", "pip", "install", *missing_packages],
        [sys.executable, "-m", "pip", "install", "--user", *missing_packages],
    ]
    last_error = None
    for command in install_commands:
        try:
            subprocess.check_call(command)
            last_error = None
            break
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise RuntimeError(
            "Missing notebook dependencies could not be installed automatically. "
            f"Please install them in the active Jupyter kernel environment: {', '.join(missing_packages)}"
        ) from last_error

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from html import escape
from IPython.display import HTML, display as _ipython_display
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score, recall_score

try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = None
except Exception as exc:
    shap = None
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# Compatibility layer: some notebook viewers show Markdown output as raw source.
# Convert dynamic Markdown blocks into clean plain text without changing the analysis flow.
def _markdown_to_plain_text(text) -> str:
    rendered_lines = []
    for raw_line in str(text).strip().splitlines():
        line = raw_line.strip()
        if not line:
            rendered_lines.append("")
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)
        rendered_lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(rendered_lines)).strip()

class _PlainTextMarkdown(str):
    pass

def Markdown(text):
    return _PlainTextMarkdown(_markdown_to_plain_text(text))

def display(*objs, **kwargs):
    for obj in objs:
        if isinstance(obj, _PlainTextMarkdown):
            _ipython_display(
                HTML(
                    "<div style='white-space: pre-wrap; line-height: 1.5; "
                    "font-family: inherit; max-height: none; overflow: visible;'>"
                    f"{escape(str(obj))}"
                    "</div>"
                )
            )
        else:
            _ipython_display(obj, **kwargs)

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
        </style>
        '''
    )
)

def find_project_root() -> Path:
    env_override = os.environ.get("FAID_PROJECT_ROOT")
    candidate_roots = []
    if env_override:
        candidate_roots.append(Path(env_override).expanduser())

    cwd = Path.cwd().resolve()
    candidate_roots.extend([cwd, *cwd.parents])

    for candidate in candidate_roots:
        if (
            (candidate / "cleaned data/faid_cleaned.csv").exists()
            and (candidate / "faid_models/eligibility/modeling/financial_aid_eligibility_model.py").exists()
        ):
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the financial-aid-portfolio-project project root. "
        "Set FAID_PROJECT_ROOT or run the notebook from inside the repository."
    )


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def relative_display_path(path: Path) -> str:
    resolved_path = path.expanduser().resolve()
    try:
        relative = resolved_path.relative_to(PROJECT_ROOT)
        return "." if str(relative) == "." else str(relative)
    except ValueError:
        return str(resolved_path)

def extract_confusion_counts(confusion_matrix):
    if isinstance(confusion_matrix, dict):
        return {
            "tn": int(confusion_matrix.get("tn", 0)),
            "fp": int(confusion_matrix.get("fp", 0)),
            "fn": int(confusion_matrix.get("fn", 0)),
            "tp": int(confusion_matrix.get("tp", 0)),
        }

    matrix = np.asarray(confusion_matrix, dtype=float)
    if matrix.shape == (2, 2):
        return {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        }

    return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}

from faid_models.eligibility.modeling.financial_aid_eligibility_model import (
    DATA_PATH,
    DECISION_OUTPUT_PATH,
    DEFAULT_WORKFLOW_EXCLUSION_COLUMNS,
    EXCLUDED_TARGET_LABELS,
    FAID_POLICY_PRIORITY_WEIGHTS,
    FAID_POLICY_WEIGHT_SENSITIVITY_RATIOS,
    FAIRNESS_GAP_THRESHOLD,
    FAIRNESS_MIN_GROUP_SIZE,
    METADATA_OUTPUT_PATH,
    MIN_PRECISION_FLOOR,
    MODEL_OUTPUT_PATH,
    NEGATIVE_TARGET_LABELS,
    POSITIVE_TARGET_LABELS,
    REPORT_OUTPUT_PATH,
    TARGET_COLUMN,
    apply_feature_pruning,
    assess_temporal_split_feasibility,
    build_future_deployment_validation_plan,
    build_priority_weight_sensitivity,
    build_audit_frame,
    build_domain_features,
    clean_feature_name,
    evaluate_decision_boundary_scenarios,
    evaluate_holdout_summary,
    evaluate_threshold_policies,
    fit_calibrated_model,
    fit_feature_pruner,
    fit_model_prototype,
    get_positive_class_probabilities,
    load_data,
    prepare_target,
    resolve_model_exclusion_summary,
    resolve_target_column,
    resolve_temporal_validation_split,
    select_feature_frame,
    split_dataset,
    summarize_decision_impact,
    tune_thresholds,
)

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")

NOTEBOOK_ARTIFACT_DIR = PROJECT_ROOT / "artifacts/yasmina_eligibility_notebook"
FULL_THESIS_ARTIFACT_DIR = Path(
    os.environ.get("FAID_ELIGIBILITY_ARTIFACT_DIR", str(MODEL_OUTPUT_PATH.parent))
).expanduser()
ACTIVE_ARTIFACT_DIR = NOTEBOOK_ARTIFACT_DIR if NOTEBOOK_ARTIFACT_DIR.exists() else FULL_THESIS_ARTIFACT_DIR
ARTIFACT_MODE = (
    "fast_refresh_convenience_build"
    if ACTIVE_ARTIFACT_DIR.resolve() == NOTEBOOK_ARTIFACT_DIR.resolve()
    else "full_thesis_artifacts"
)

BUNDLE_PATH = ACTIVE_ARTIFACT_DIR / MODEL_OUTPUT_PATH.name
DECISION_OUTPUT_PATH = ACTIVE_ARTIFACT_DIR / DECISION_OUTPUT_PATH.name
METADATA_OUTPUT_PATH = ACTIVE_ARTIFACT_DIR / METADATA_OUTPUT_PATH.name
REPORT_OUTPUT_PATH = ACTIVE_ARTIFACT_DIR / REPORT_OUTPUT_PATH.name

bundle_load_error = None
bundle = None
if BUNDLE_PATH.exists():
    try:
        bundle = joblib.load(BUNDLE_PATH)
    except ModuleNotFoundError as exc:
        bundle_load_error = (
            f"{type(exc).__name__}: {exc}. "
            "The saved model bundle depends on an optional modeling package in the active notebook kernel."
        )
        print("Warning:", bundle_load_error)
    except Exception as exc:
        bundle_load_error = f"{type(exc).__name__}: {exc}"
        print("Warning: saved model bundle could not be loaded.", bundle_load_error)
decision_df = pd.read_csv(DECISION_OUTPUT_PATH) if DECISION_OUTPUT_PATH.exists() else pd.DataFrame()
metadata = json.loads(METADATA_OUTPUT_PATH.read_text()) if METADATA_OUTPUT_PATH.exists() else {}
report_text = REPORT_OUTPUT_PATH.read_text(encoding="utf-8") if REPORT_OUTPUT_PATH.exists() else ""
normal_workflow = metadata.get("normal_workflow", {})

availability = pd.DataFrame(
    [
        {"setting": "project_root", "value": "."},
        {"setting": "artifact_mode", "value": ARTIFACT_MODE},
        {"setting": "active_artifact_dir", "value": relative_display_path(ACTIVE_ARTIFACT_DIR)},
        {"setting": "data_path", "value": relative_display_path(DATA_PATH)},
        {"setting": "model_bundle_available", "value": BUNDLE_PATH.exists()},
        {"setting": "decision_csv_available", "value": DECISION_OUTPUT_PATH.exists()},
        {"setting": "metadata_json_available", "value": METADATA_OUTPUT_PATH.exists()},
        {"setting": "report_txt_available", "value": REPORT_OUTPUT_PATH.exists()},
        {"setting": "bundle_load_error", "value": bundle_load_error or ""},
    ]
)
display(availability)

# %% [notebook cell 4]
overview_rows = []

if bundle is not None:
    overview_rows.extend(
        [
            {"metric": "selected model", "value": bundle.get("selected_model_name")},
            {"metric": "selected model family", "value": bundle.get("selected_model_family")},
            {"metric": "selected threshold policy", "value": bundle.get("selected_threshold_policy")},
            {"metric": "selected threshold", "value": round(float(bundle.get("selected_threshold", 0.0)), 3)},
        ]
    )

if not decision_df.empty:
    action_counts = decision_df["recommended_action"].value_counts()
    total_cases = int(len(decision_df))
    review_cases = int(action_counts.get("review", 0))
    automated_cases = total_cases - review_cases
    overview_rows.extend(
        [
            {"metric": "decision cases", "value": total_cases},
            {"metric": "approve recommendations", "value": int(action_counts.get("approve", 0))},
            {"metric": "reject recommendations", "value": int(action_counts.get("reject", 0))},
            {"metric": "review recommendations", "value": review_cases},
            {"metric": "automated share", "value": f"{automated_cases / max(total_cases, 1):.1%}"},
        ]
    )

display(pd.DataFrame(overview_rows))

# %% [notebook cell 6]
if decision_df.empty:
    print("The executive summary needs the decision CSV.")
else:
    executive_threshold = (
        float(bundle.get("selected_threshold", 0.0))
        if bundle is not None and bundle.get("selected_threshold") is not None
        else float(decision_df["selected_threshold"].iloc[0])
    )
    impact_summary = (
        normal_workflow.get("business_impact_summary", {})
        if normal_workflow
        else summarize_decision_impact(decision_df, executive_threshold)
    )
    total_cases = int(impact_summary.get("total_cases", len(decision_df)))
    automated_cases = int(impact_summary.get("automated_cases", 0))
    review_cases = int(impact_summary.get("review_cases_after_model", 0))
    fairness_risk_cases = int(decision_df["fairness_risk_flag"].fillna(False).astype(bool).sum())
    missed_need_weight = int(float(FAID_POLICY_PRIORITY_WEIGHTS["cost_false_negative_missed_need"]))
    review_weight = int(float(FAID_POLICY_PRIORITY_WEIGHTS["cost_false_positive_review"]))
    selected_policy_metrics = (
        normal_workflow.get("test_summary", {}).get("selected_policy_metrics", {})
        if normal_workflow
        else {}
    )
    selected_confusion = extract_confusion_counts(
        selected_policy_metrics.get("confusion_matrix", {})
    )

    executive_summary = pd.DataFrame(
        [
            {
                "headline": "System recommendation",
                "value": "Deploy as governed triage, not full automation",
                "why_it_matters": "The workflow supports high-confidence approvals while keeping borderline, fairness-risk, and uncertain cases in human review.",
            },
            {
                "headline": "Selected model",
                "value": bundle.get("selected_model_name") if bundle is not None else "saved model",
                "why_it_matters": "This is the model that underpins the governed routing policy.",
            },
            {
                "headline": "Selected policy",
                "value": f"{bundle.get('selected_threshold_policy') if bundle is not None else 'saved policy'} at threshold {executive_threshold:.3f}; FN:review cost = {missed_need_weight}:{review_weight}",
                "why_it_matters": "The thesis policy makes missed deserving students materially more costly than one extra manual review.",
            },
            {
                "headline": "Automation capacity",
                "value": f"{automated_cases}/{total_cases} ({automated_cases / max(total_cases, 1):.1%})",
                "why_it_matters": "This is the share of cases the workflow can route automatically under the governed decision layer.",
            },
            {
                "headline": "Manual review workload",
                "value": f"{review_cases}/{total_cases}",
                "why_it_matters": "Review is preserved for the cases most likely to need judgment or safeguards.",
            },
            {
                "headline": "Missed deserving students",
                "value": int(selected_confusion["fn"]),
                "why_it_matters": "This is the thesis-critical error type the need-sensitive policy is designed to minimize.",
            },
            {
                "headline": "Fairness-risk overrides",
                "value": fairness_risk_cases,
                "why_it_matters": "These cases are explicitly pulled out of automation because subgroup safeguards were triggered.",
            },
            {
                "headline": "Automation risk avoided",
                "value": impact_summary.get("harmful_auto_errors_avoided_vs_threshold_only", 0),
                "why_it_matters": "This measures how many harmful automatic errors are avoided versus forcing threshold-only decisions on everyone.",
            },
        ]
    )
    display(executive_summary)

    display(
        Markdown(
            f'''
            ### Executive recommendation

            Use the model as a **governed decision system**, not as a raw auto-approval/auto-rejection engine. On the saved run, the workflow auto-routes **{automated_cases} of {total_cases}** cases, keeps **{review_cases}** in committee review, and pulls **{fairness_risk_cases}** cases out of automation because safeguards were triggered.
            '''
        )
    )

# %% [notebook cell 8]
raw_dataframe = load_data(DATA_PATH)
target_column = resolve_target_column(raw_dataframe, TARGET_COLUMN)
labeled_dataframe, binary_target, data_metadata = prepare_target(
    dataframe=raw_dataframe,
    target_column=target_column,
    positive_labels=POSITIVE_TARGET_LABELS,
    negative_labels=NEGATIVE_TARGET_LABELS,
    excluded_labels=EXCLUDED_TARGET_LABELS,
)

exclusion_summary = resolve_model_exclusion_summary(
    dataframe=labeled_dataframe,
    target_column=target_column,
)

data_summary = pd.DataFrame(
    [
        {"metric": "raw rows", "value": len(raw_dataframe)},
        {"metric": "labeled rows used", "value": len(labeled_dataframe)},
        {"metric": "rows dropped missing target", "value": data_metadata["rows_dropped_missing_target"]},
        {"metric": "rows dropped non-binary target", "value": data_metadata["rows_dropped_non_binary_target"]},
        {"metric": "positive rate", "value": f"{binary_target.mean():.1%}"},
        {"metric": "post-decision columns excluded", "value": len(exclusion_summary["post_decision_columns"])},
    ]
)
display(data_summary)

excluded_preview = pd.DataFrame(
    [
        {"category": "target column", "examples": ", ".join(exclusion_summary["target_columns"][:5]) or "None"},
        {"category": "post-decision columns", "examples": ", ".join(exclusion_summary["post_decision_columns"][:8]) or "None"},
        {"category": "identifier columns", "examples": ", ".join(exclusion_summary["identifier_columns"][:5]) or "None"},
        {"category": "audit-only columns", "examples": ", ".join(exclusion_summary["audit_only_columns"][:5]) or "None"},
    ]
)
display(excluded_preview)

feature_frame, _ = select_feature_frame(labeled_dataframe, target_column)
workflow_exclusions = sorted(
    column for column in feature_frame.columns if column in DEFAULT_WORKFLOW_EXCLUSION_COLUMNS
)
feature_frame = feature_frame.drop(columns=workflow_exclusions, errors="ignore")
audit_frame = build_audit_frame(labeled_dataframe)

(
    base_train_features,
    base_valid_features,
    base_test_features,
    target_train,
    target_valid,
    target_test,
) = split_dataset(feature_frame, binary_target)

audit_test = audit_frame.loc[base_test_features.index].copy()

train_features_engineered = build_domain_features(base_train_features)
test_features_engineered = build_domain_features(base_test_features)
pruning_columns_to_drop, pruning_summary = fit_feature_pruner(train_features_engineered)
retained_columns = [
    column for column in train_features_engineered.columns if column not in pruning_columns_to_drop
]
test_model_features = apply_feature_pruning(
    test_features_engineered,
    pruning_columns_to_drop,
    retained_columns=retained_columns,
)

pipeline = bundle.get("explainability_estimator") or bundle.get("estimator") if bundle is not None else None
preprocessor = pipeline.named_steps["preprocessor"] if pipeline is not None else None
estimator = pipeline.named_steps["model"] if pipeline is not None else None
feature_names = [clean_feature_name(name) for name in preprocessor.get_feature_names_out()] if preprocessor is not None else []

display(
    pd.DataFrame(
        [
            {"metric": "workflow exclusions applied", "value": len(workflow_exclusions)},
            {"metric": "engineered feature columns retained", "value": len(retained_columns)},
            {"metric": "test rows for analysis", "value": len(test_model_features)},
        ]
    )
)

# %% [notebook cell 10]
if not normal_workflow:
    print("Metadata is missing, so only the saved bundle summary can be shown.")
else:
    probability_metrics = normal_workflow["test_summary"]["probability_metrics"]
    calibration_summary = normal_workflow["calibration_summary"]
    model_summary = pd.DataFrame(
        [
            {"metric": "selected candidate", "value": normal_workflow["selected_candidate"]},
            {"metric": "model family", "value": normal_workflow["selected_model_family"]},
            {"metric": "ROC AUC", "value": round(probability_metrics["roc_auc"], 4)},
            {"metric": "average precision", "value": round(probability_metrics["average_precision"], 4)},
            {"metric": "Brier score", "value": round(probability_metrics["brier_score"], 4)},
            {"metric": "calibration mode", "value": calibration_summary["selected_mode"]},
        ]
    )
    display(model_summary)

    validation_table = pd.DataFrame(normal_workflow.get("validation_comparison", []))
    if not validation_table.empty:
        cols = [
            column
            for column in [
                "model",
                "family",
                "validation_roc_auc",
                "validation_brier_score",
                "validation_average_precision",
                "selection_rank",
            ]
            if column in validation_table.columns
        ]
        display(validation_table[cols].head(8))

    display(
        Markdown(
            f'''
            ### Model reading

            The selected model is **{normal_workflow["selected_candidate"]}**. The probabilities are used as decision confidence, so calibration matters directly for threshold design and safe automation.
            '''
        )
    )

# %% [notebook cell 12]
if not normal_workflow:
    print("Metadata is missing, so the baseline-comparison section cannot be reconstructed.")
else:
    validation_table = pd.DataFrame(normal_workflow.get("validation_comparison", []))
    focus_models = ["Logistic Regression", "Gradient Boosting", "Random Forest", "XGBoost"]
    baseline_table = (
        validation_table[validation_table["model"].isin(focus_models)]
        .copy()
        .sort_values("validation_roc_auc", ascending=False)
    )

    if baseline_table.empty:
        print("No saved baseline comparison rows were found.")
    else:
        baseline_table["governance_value"] = baseline_table["model"].map(
            {
                "Logistic Regression": "Most transparent baseline, but materially weaker discrimination and worse Brier score.",
                "Gradient Boosting": "Strong tree baseline, but slightly weaker validation ranking than the selected model.",
                "Random Forest": "Competitive AUC, but larger overfit gap makes governance less comfortable.",
                "XGBoost": "Best validation ROC AUC among the main contenders with a better overfit trade-off than Random Forest.",
            }
        )
        baseline_table["interpretability_tradeoff"] = baseline_table["model"].map(
            {
                "Logistic Regression": "Highest transparency, lowest complexity.",
                "Gradient Boosting": "Moderate complexity, weaker local interpretability than linear models.",
                "Random Forest": "Less transparent and more variance-prone than the selected model.",
                "XGBoost": "Highest complexity here, justified only because governance performance is strongest overall.",
            }
        )

        display(
            baseline_table[
                [
                    "model",
                    "family",
                    "validation_roc_auc",
                    "validation_average_precision",
                    "validation_brier_score",
                    "overfit_gap_roc_auc",
                    "governance_value",
                    "interpretability_tradeoff",
                ]
            ]
        )

        logistic_row = baseline_table.loc[baseline_table["model"] == "Logistic Regression"]
        gradient_row = baseline_table.loc[baseline_table["model"] == "Gradient Boosting"]
        xgb_row = baseline_table.loc[baseline_table["model"] == "XGBoost"]

        if not logistic_row.empty and not gradient_row.empty and not xgb_row.empty:
            logistic_row = logistic_row.iloc[0]
            gradient_row = gradient_row.iloc[0]
            xgb_row = xgb_row.iloc[0]
            print("Baseline reading")
            print()
            print(
                f"- Logistic Regression remains the easiest model to explain, "
                f"but its validation ROC AUC ({logistic_row['validation_roc_auc']:.4f}) "
                f"and Brier score ({logistic_row['validation_brier_score']:.4f}) are clearly weaker than the selected workflow."
            )
            print(
                f"- Gradient Boosting is the closest simpler tree baseline, "
                f"but XGBoost still edges it on validation ROC AUC "
                f"({xgb_row['validation_roc_auc']:.4f} vs {gradient_row['validation_roc_auc']:.4f}) "
                f"while staying in a better governance range than Random Forest on overfit."
            )
            print(
                '- The thesis argument is therefore not "complexity for its own sake." '
                "It is that the selected model gives the strongest validated probabilities "
                "for the governed triage layer, while the notebook still preserves "
                "interpretability through SHAP and fallback explanations."
            )

# %% [notebook cell 14]
if decision_df.empty or not normal_workflow:
    print("Calibration analysis needs both the decision CSV and saved metadata.")
else:
    probability_metrics = normal_workflow["test_summary"]["probability_metrics"]
    calibration_summary = normal_workflow["calibration_summary"]
    calibration_rows = []
    for mode_name in ["uncalibrated", "calibrated"]:
        mode_metrics = calibration_summary.get(mode_name)
        if mode_metrics:
            calibration_rows.append(
                {
                    "mode": mode_name,
                    "selected_for_deployment": calibration_summary.get("selected_mode") == mode_name,
                    "validation_roc_auc": round(mode_metrics["roc_auc"], 4),
                    "validation_average_precision": round(mode_metrics["average_precision"], 4),
                    "validation_brier_score": round(mode_metrics["brier_score"], 4),
                }
            )

    if calibration_rows:
        display(pd.DataFrame(calibration_rows))

    actuals = decision_df["actual_target"].to_numpy()
    probabilities = decision_df["predicted_probability_awarded"].to_numpy()
    observed_share, mean_predicted = calibration_curve(
        actuals,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect calibration")
    ax.plot(mean_predicted, observed_share, marker="o", linewidth=2, color="#2e8b57", label="Observed")
    ax.set_title("Calibration curve on the saved holdout")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed award rate")
    ax.legend()
    plt.tight_layout()
    plt.show()

    calibration_bin_count = min(10, max(3, decision_df["predicted_probability_awarded"].nunique()))
    reliability_frame = decision_df.copy()
    reliability_frame["probability_bin"] = pd.qcut(
        reliability_frame["predicted_probability_awarded"],
        q=calibration_bin_count,
        duplicates="drop",
    )
    reliability_table = (
        reliability_frame.groupby("probability_bin", observed=False)
        .agg(
            cases=("actual_target", "size"),
            mean_predicted_probability=("predicted_probability_awarded", "mean"),
            observed_award_rate=("actual_target", "mean"),
        )
        .reset_index()
    )
    display(reliability_table)

    display(
        Markdown(
            f'''
            ### Calibration reading

            - The saved holdout Brier score is **{probability_metrics["brier_score"]:.4f}**. Lower is better: it means predicted probabilities sit closer to realized award rates.
            - The selected calibration mode is **{calibration_summary["selected_mode"]}** because: {calibration_summary["selection_reason"]}
            - This matters operationally because FAID is not only ranking applicants. It is using probabilities to decide which cases can be routed automatically and which should stay in committee review.
            '''
        )
    )

# %% [notebook cell 16]
temporal_split = resolve_temporal_validation_split(labeled_dataframe)

if bundle is None or preprocessor is None:
    print("Temporal validation needs the saved model bundle and preprocessing pipeline.")
elif not temporal_split["temporal_split_feasible"]:
    print(temporal_split["note"])
    display(pd.DataFrame(build_future_deployment_validation_plan()))
else:
    display(pd.DataFrame(temporal_split["cycle_summary"]))

    temporal_feature_frame = feature_frame.copy()
    selected_policy_name = (
        bundle.get("selected_threshold_policy")
        if bundle is not None and bundle.get("selected_threshold_policy")
        else "faid_need_sensitive"
    )

    if temporal_split["validation_cycles"]:
        temporal_train_index = temporal_split["train_index"]
        temporal_valid_index = temporal_split["validation_index"]
    else:
        earlier_features = temporal_feature_frame.loc[temporal_split["train_index"]].copy()
        earlier_target = binary_target.loc[earlier_features.index].copy()
        (
            temporal_train_features,
            temporal_valid_features,
            _,
            temporal_train_target,
            temporal_valid_target,
            _,
        ) = split_dataset(earlier_features, earlier_target)
        temporal_train_index = temporal_train_features.index.tolist()
        temporal_valid_index = temporal_valid_features.index.tolist()

    temporal_test_index = temporal_split["test_index"]

    temporal_train_raw = temporal_feature_frame.loc[temporal_train_index].copy()
    temporal_valid_raw = temporal_feature_frame.loc[temporal_valid_index].copy()
    temporal_test_raw = temporal_feature_frame.loc[temporal_test_index].copy()

    temporal_train_target = binary_target.loc[temporal_train_raw.index].copy()
    temporal_valid_target = binary_target.loc[temporal_valid_raw.index].copy()
    temporal_test_target = binary_target.loc[temporal_test_raw.index].copy()

    temporal_train_engineered = build_domain_features(temporal_train_raw)
    temporal_valid_engineered = build_domain_features(temporal_valid_raw)
    temporal_test_engineered = build_domain_features(temporal_test_raw)

    model_input_columns = []
    for _, _, columns in preprocessor.transformers_:
        if isinstance(columns, str):
            continue
        model_input_columns.extend(list(columns))
    model_input_columns = list(dict.fromkeys(model_input_columns))

    for engineered_frame in [
        temporal_train_engineered,
        temporal_valid_engineered,
        temporal_test_engineered,
    ]:
        for column in model_input_columns:
            if column not in engineered_frame.columns:
                engineered_frame[column] = np.nan

    temporal_train_model = temporal_train_engineered.reindex(columns=model_input_columns)
    temporal_valid_model = temporal_valid_engineered.reindex(columns=model_input_columns)
    temporal_test_model = temporal_test_engineered.reindex(columns=model_input_columns)

    temporal_prototype = bundle.get("explainability_estimator") or bundle.get("estimator")
    temporal_base_model = fit_model_prototype(
        temporal_prototype,
        temporal_train_model,
        temporal_train_target,
    )
    if bundle.get("calibration_mode_selected") == "calibrated":
        temporal_deployed_model = fit_calibrated_model(
            temporal_prototype,
            temporal_train_model,
            temporal_train_target,
        )
    else:
        temporal_deployed_model = temporal_base_model

    temporal_valid_probabilities = get_positive_class_probabilities(
        temporal_deployed_model,
        temporal_valid_model,
    )
    temporal_policy_grid, temporal_validation_threshold_results = tune_thresholds(
        temporal_valid_target,
        temporal_valid_probabilities,
        min_precision_floor=MIN_PRECISION_FLOOR,
    )
    temporal_policy_thresholds = {
        name: metrics.threshold
        for name, metrics in temporal_validation_threshold_results.items()
    }
    temporal_test_probabilities = get_positive_class_probabilities(
        temporal_deployed_model,
        temporal_test_model,
    )
    temporal_test_threshold_results = evaluate_threshold_policies(
        temporal_test_target,
        temporal_test_probabilities,
        temporal_policy_thresholds,
    )
    temporal_test_summary = evaluate_holdout_summary(
        dataset_label="temporal_test",
        target_true=temporal_test_target,
        probabilities=temporal_test_probabilities,
        threshold_results=temporal_test_threshold_results,
        selected_policy=selected_policy_name,
    )
    temporal_policy_sensitivity = pd.DataFrame(
        build_priority_weight_sensitivity(
            selection_target=temporal_valid_target,
            selection_probabilities=temporal_valid_probabilities,
            evaluation_target=temporal_test_target,
            evaluation_probabilities=temporal_test_probabilities,
        )
    )

    temporal_summary = pd.DataFrame(
        [
            {"metric": "temporal source", "value": temporal_split["source_column"]},
            {"metric": "train cycles", "value": ", ".join(map(str, temporal_split["train_cycles"]))},
            {"metric": "validation cycles", "value": ", ".join(map(str, temporal_split["validation_cycles"])) or "random split from earlier cycles"},
            {"metric": "test cycles", "value": ", ".join(map(str, temporal_split["test_cycles"]))},
            {"metric": "temporal test ROC AUC", "value": round(temporal_test_summary["probability_metrics"]["roc_auc"], 4)},
            {"metric": "temporal test average precision", "value": round(temporal_test_summary["probability_metrics"]["average_precision"], 4)},
            {"metric": "selected temporal threshold", "value": round(float(temporal_test_summary["selected_threshold"]), 3)},
            {"metric": "temporal precision", "value": round(temporal_test_summary["selected_policy_metrics"]["precision"], 4)},
            {"metric": "temporal recall", "value": round(temporal_test_summary["selected_policy_metrics"]["recall"], 4)},
        ]
    )
    display(temporal_summary)
    display(
        temporal_policy_sensitivity[
            [
                "missed_need_to_review_ratio",
                "selected_threshold",
                "precision",
                "recall",
                "missed_deserving_students",
                "extra_review_burden",
                "students_shortlisted",
            ]
        ]
    )
    display(
        Markdown(
            f'''
            ### Temporal reading

            A real earlier-cycle/later-cycle test is feasible here. The model is trained on earlier application cycles, validated on the next cycle, and then tested on the latest cycle. That makes the temporal section stronger than a purely random split because it asks whether the policy still holds after cycle shift.
            '''
        )
    )

# %% [notebook cell 18]
selected_threshold = (
    float(bundle.get("selected_threshold"))
    if bundle is not None and bundle.get("selected_threshold") is not None
    else float(decision_df["selected_threshold"].iloc[0])
)
selected_policy_name = (
    bundle.get("selected_threshold_policy")
    if bundle is not None and bundle.get("selected_threshold_policy")
    else "faid_need_sensitive"
)

if not normal_workflow:
    print("Metadata is missing, so decision-policy tables are limited.")
else:
    threshold_results = normal_workflow["test_summary"]["threshold_results"]
    threshold_rows = []
    for policy_name, metrics in threshold_results.items():
        confusion = extract_confusion_counts(metrics.get("confusion_matrix", {}))
        threshold_rows.append(
            {
                "policy": policy_name,
                "selected": "yes" if policy_name == selected_policy_name else "",
                "threshold": round(metrics["threshold"], 3),
                "precision": round(metrics["precision"], 3),
                "recall": round(metrics["recall"], 3),
                "f1": round(metrics["f1"], 3),
                "balanced_accuracy": round(metrics["balanced_accuracy"], 3),
                "missed_deserving_students": int(confusion["fn"]),
                "extra_review_burden": int(confusion["fp"]),
            }
        )
    display(pd.DataFrame(threshold_rows).sort_values(["selected", "recall", "precision"], ascending=[False, False, False]))
    display(
        Markdown(
            f'''
            **Policy rationale:** {normal_workflow.get("threshold_policy_rationale", "The selected policy prioritizes missed-need protection while keeping shortlist quality operationally usable.")}
            '''
        )
    )

if not decision_df.empty:
    decision_summary = pd.DataFrame(
        [
            {"metric": "selected threshold", "value": round(selected_threshold, 3)},
            {"metric": "approve threshold", "value": round(float(decision_df["approve_threshold"].iloc[0]), 3)},
            {"metric": "reject threshold", "value": round(float(decision_df["reject_threshold"].iloc[0]), 3)},
            {"metric": "approve recommendations", "value": int((decision_df["recommended_action"] == "approve").sum())},
            {"metric": "reject recommendations", "value": int((decision_df["recommended_action"] == "reject").sum())},
            {"metric": "review recommendations", "value": int((decision_df["recommended_action"] == "review").sum())},
        ]
    )
    display(decision_summary)

# %% [notebook cell 20]
if normal_workflow.get("policy_weight_sensitivity"):
    policy_weight_df = pd.DataFrame(normal_workflow["policy_weight_sensitivity"])
elif not decision_df.empty:
    policy_weight_df = pd.DataFrame(
        build_priority_weight_sensitivity(
            selection_target=decision_df["actual_target"],
            selection_probabilities=decision_df["predicted_probability_awarded"],
            evaluation_target=decision_df["actual_target"],
            evaluation_probabilities=decision_df["predicted_probability_awarded"],
        )
    )
else:
    policy_weight_df = pd.DataFrame()

if policy_weight_df.empty:
    print("Policy-weight sensitivity could not be computed.")
else:
    policy_weight_assumptions = pd.DataFrame(
        [
            {
                "policy component": "true positive benefit",
                "relative weight": 1,
                "meaning": "Correctly shortlisting a deserving student is the baseline unit of value.",
            },
            {
                "policy component": "false positive review cost",
                "relative weight": 1,
                "meaning": "One extra borderline committee review is the baseline operational cost.",
            },
            {
                "policy component": "false negative missed-need cost",
                "relative weight": "1 to 5 tested; 3 selected",
                "meaning": "Sensitivity analysis tests how much the decision policy should prioritize not missing deserving students.",
            },
        ]
    )
    display(policy_weight_assumptions)

    policy_weight_view = policy_weight_df[
        [
            "missed_need_to_review_ratio",
            "selected_threshold",
            "precision",
            "recall",
            "missed_deserving_students",
            "extra_review_burden",
            "students_shortlisted",
        ]
    ].copy()
    display(policy_weight_view)

    ratio_1 = policy_weight_df.loc[
        policy_weight_df["missed_need_to_review_ratio"] == "1:1"
    ].iloc[0]
    ratio_2 = policy_weight_df.loc[
        policy_weight_df["missed_need_to_review_ratio"] == "2:1"
    ].iloc[0]
    ratio_3 = policy_weight_df.loc[
        policy_weight_df["missed_need_to_review_ratio"] == "3:1"
    ].iloc[0]
    ratio_5 = policy_weight_df.loc[
        policy_weight_df["missed_need_to_review_ratio"] == "5:1"
    ].iloc[0]

    if int(ratio_5["missed_deserving_students"]) == int(ratio_3["missed_deserving_students"]) and int(ratio_5["extra_review_burden"]) == int(ratio_3["extra_review_burden"]):
        upper_ratio_read = (
            f"Compared with **5:1**, the **3:1** policy delivers the same holdout recall and review burden on this run. "
            "That means moving beyond `3:1` does not buy extra protection here."
        )
    else:
        upper_ratio_read = (
            f"Compared with **5:1**, the **3:1** policy avoids some extra review burden "
            f"(**{int(ratio_5['extra_review_burden'])}** down to **{int(ratio_3['extra_review_burden'])}**) "
            "while still preserving very high recall."
        )

    display(
        Markdown(
            f'''
            ### Why `3:1` is the thesis policy

            - Compared with **1:1**, the **3:1** policy cuts missed deserving students from **{int(ratio_1["missed_deserving_students"])}** to **{int(ratio_3["missed_deserving_students"])}**, while increasing extra review burden from **{int(ratio_1["extra_review_burden"])}** to **{int(ratio_3["extra_review_burden"])}**.
            - Compared with **2:1**, the **3:1** policy stays in the same high-recall regime on this run, so it preserves the stronger governance preference without forcing a more aggressive threshold.
            - {upper_ratio_read}
            - Numerically, `3:1` is therefore the smallest clearly need-sensitive ratio that reaches the maximum-recall regime under the active precision floor.
            '''
        )
    )

# %% [notebook cell 22]
if decision_df.empty:
    print("Review-band sensitivity needs the decision CSV.")
else:
    selected_threshold = float(decision_df["selected_threshold"].iloc[0])
    fairness_flags = decision_df["fairness_risk_flag"].fillna(False).astype(bool).to_numpy()
    probabilities = decision_df["predicted_probability_awarded"].to_numpy()
    actuals = decision_df["actual_target"].to_numpy()

    review_band_rows = []
    for review_margin in [0.02, 0.05, 0.10, 0.15]:
        approve_threshold = max(0.80, min(selected_threshold + review_margin, 0.95))
        reject_threshold = min(0.20, max(selected_threshold - review_margin, 0.05))
        borderline = np.abs(probabilities - selected_threshold) <= review_margin

        actions = np.where(
            fairness_flags,
            "review",
            np.where(
                borderline,
                "review",
                np.where(
                    probabilities >= approve_threshold,
                    "approve",
                    np.where(probabilities <= reject_threshold, "reject", "review"),
                ),
            ),
        )

        approve_mask = actions == "approve"
        reject_mask = actions == "reject"
        review_mask = actions == "review"

        review_band_rows.append(
            {
                "review_margin": review_margin,
                "approve_threshold": round(float(approve_threshold), 3),
                "reject_threshold": round(float(reject_threshold), 3),
                "approve_cases": int(approve_mask.sum()),
                "reject_cases": int(reject_mask.sum()),
                "review_cases": int(review_mask.sum()),
                "automation_share": round(float((~review_mask).mean()), 4),
                "harmful_auto_approves": int((approve_mask & (actuals == 0)).sum()),
                "harmful_auto_rejects": int((reject_mask & (actuals == 1)).sum()),
            }
        )

    review_band_df = pd.DataFrame(review_band_rows)
    display(review_band_df)

    current_margin_row = review_band_df.loc[review_band_df["review_margin"] == 0.10].iloc[0]
    narrow_margin_row = review_band_df.loc[review_band_df["review_margin"] == 0.02].iloc[0]

    display(
        Markdown(
            f'''
            ### Review-band reading

            - At the selected threshold (**{selected_threshold:.3f}**), a **0.10** review margin creates a governed **approve-or-review** workflow on this run: it preserves **{int(current_margin_row["approve_cases"])}** automatic approvals while sending all low-confidence and low-probability cases to review.
            - Tightening the review margin to **0.02** would create **{int(narrow_margin_row["reject_cases"])}** automatic rejections. That increases automation, but it weakens the thesis claim that FAID should be especially cautious about ruling out deserving students.
            - The chosen confidence boundaries are therefore aligned with the thesis goal: automate only the strongest approvals and keep the ambiguous or potentially harmful cases with the committee.
            '''
        )
    )

# %% [notebook cell 24]
def to_positive_class_explanation(explanation):
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float)

    if values.ndim == 3:
        values = values[:, :, -1]
    if base_values.ndim > 1:
        base_values = base_values[:, -1]

    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=np.asarray(explanation.data, dtype=float) if explanation.data is not None else None,
        feature_names=list(explanation.feature_names),
    )


def build_shap_explanation(fitted_estimator, transformed_background, transformed_examples, readable_feature_names):
    if not SHAP_AVAILABLE:
        raise RuntimeError(SHAP_IMPORT_ERROR or "SHAP is unavailable in this environment.")
    explainer = shap.Explainer(
        fitted_estimator,
        transformed_background,
        feature_names=readable_feature_names,
    )
    explanation = explainer(transformed_examples)
    return to_positive_class_explanation(explanation)


def top_shap_drivers(explanation, row_position, top_n=5):
    row_values = np.asarray(explanation.values[row_position], dtype=float)
    ranked = np.argsort(np.abs(row_values))[::-1]
    top_indices = [index for index in ranked if abs(row_values[index]) > 1e-9][:top_n]
    rows = []
    for index in top_indices:
        direction = "pushes toward award" if row_values[index] > 0 else "pushes toward denial"
        rows.append(
            {
                "feature": explanation.feature_names[index],
                "shap_value": round(float(row_values[index]), 4),
                "direction": direction,
            }
        )
    return pd.DataFrame(rows)


def build_global_feature_importance_fallback(fitted_estimator, readable_feature_names):
    importance_values = None
    if hasattr(fitted_estimator, "feature_importances_"):
        importance_values = np.asarray(fitted_estimator.feature_importances_, dtype=float)
    elif hasattr(fitted_estimator, "coef_"):
        importance_values = np.abs(np.asarray(fitted_estimator.coef_, dtype=float)).reshape(-1)

    if importance_values is None or importance_values.size == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    usable_length = min(len(readable_feature_names), int(importance_values.shape[0]))
    return (
        pd.DataFrame(
            {
                "feature": readable_feature_names[:usable_length],
                "importance": importance_values[:usable_length],
            }
        )
        .sort_values("importance", ascending=False, ignore_index=True)
    )


def build_directional_probability_association_fallback(
    transformed_matrix,
    predicted_probabilities,
    readable_feature_names,
):
    if (
        transformed_matrix.size == 0
        or len(readable_feature_names) == 0
        or len(predicted_probabilities) == 0
    ):
        return pd.DataFrame(columns=["feature", "association_with_award_probability"])

    usable_length = min(transformed_matrix.shape[1], len(readable_feature_names))
    feature_matrix = np.asarray(transformed_matrix[:, :usable_length], dtype=float)
    probability_vector = np.asarray(predicted_probabilities, dtype=float)

    centered_features = feature_matrix - feature_matrix.mean(axis=0, keepdims=True)
    centered_probabilities = probability_vector - probability_vector.mean()
    numerators = centered_features.T @ centered_probabilities
    denominators = np.sqrt(
        (centered_features**2).sum(axis=0) * (centered_probabilities**2).sum()
    )
    associations = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=float),
        where=denominators > 0,
    )
    return pd.DataFrame(
        {
            "feature": readable_feature_names[:usable_length],
            "association_with_award_probability": associations[:usable_length],
        }
    )


def render_feature_importance_fallback(importance_frame, heading):
    if importance_frame.empty:
        print("Feature-importance fallback is unavailable for this model.")
        return

    display(importance_frame.head(15))
    plot_frame = importance_frame.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(plot_frame["feature"], plot_frame["importance"], color="#4c78a8")
    plt.title(heading)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

# %% [notebook cell 25]
if pipeline is None:
    print("The saved model bundle is missing, so SHAP cannot be displayed.")
else:
    fallback_importance = build_global_feature_importance_fallback(estimator, feature_names)
    transformed_test = np.asarray(preprocessor.transform(test_model_features), dtype=float)
    probability_association_fallback = build_directional_probability_association_fallback(
        transformed_test,
        decision_df["predicted_probability_awarded"].to_numpy() if not decision_df.empty else np.array([]),
        feature_names,
    )
    background_size = min(120, len(test_model_features))
    explain_size = min(180, len(test_model_features))

    background_positions = np.linspace(0, len(test_model_features) - 1, background_size, dtype=int)
    explain_positions = np.linspace(0, len(test_model_features) - 1, explain_size, dtype=int)

    transformed_background = transformed_test[background_positions]
    transformed_explain = transformed_test[explain_positions]

    if not SHAP_AVAILABLE:
        display(
            Markdown(
                f'''
                ### Interpretability fallback

                SHAP could not be imported in this environment (`{SHAP_IMPORT_ERROR}`). The notebook therefore falls back to model feature importance so the global decision drivers remain visible.
                '''
            )
        )
        render_feature_importance_fallback(
            fallback_importance,
            heading="Fallback feature importance: overall decision drivers",
        )
        if not probability_association_fallback.empty:
            display(
                probability_association_fallback.sort_values(
                    "association_with_award_probability",
                    ascending=False,
                ).head(10)
            )
            display(
                probability_association_fallback.sort_values(
                    "association_with_award_probability",
                    ascending=True,
                ).head(10)
            )
    else:
        try:
            shap_global_explanation = build_shap_explanation(
                estimator,
                transformed_background,
                transformed_explain,
                feature_names,
            )

            global_shap = (
                pd.DataFrame(
                    {
                        "feature": feature_names,
                        "mean_abs_shap": np.abs(np.asarray(shap_global_explanation.values, dtype=float)).mean(axis=0),
                    }
                )
                .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
            )
            display(global_shap.head(15))

            shap_values_matrix = np.asarray(shap_global_explanation.values, dtype=float)
            positive_driver_table = (
                pd.DataFrame(
                    {
                        "feature": feature_names,
                        "mean_positive_shap": np.clip(shap_values_matrix, 0, None).mean(axis=0),
                    }
                )
                .sort_values("mean_positive_shap", ascending=False, ignore_index=True)
            )
            rejection_driver_table = (
                pd.DataFrame(
                    {
                        "feature": feature_names,
                        "mean_negative_shap": np.clip(-shap_values_matrix, 0, None).mean(axis=0),
                    }
                )
                .sort_values("mean_negative_shap", ascending=False, ignore_index=True)
            )
            display(positive_driver_table.head(10))
            display(rejection_driver_table.head(10))

            plt.figure(figsize=(10, 6))
            shap.plots.bar(shap_global_explanation, max_display=15, show=False)
            plt.title("Global SHAP: features that move eligibility predictions most")
            plt.tight_layout()
            plt.show()

            top_driver_lines = []
            for _, row in global_shap.head(5).iterrows():
                top_driver_lines.append(
                    f"- **{row['feature']}** has an average absolute SHAP value of **{row['mean_abs_shap']:.4f}**, so it is one of the strongest overall decision drivers."
                )

            display(
                Markdown(
                    "### Global interpretation\n\n"
                    + "\n".join(top_driver_lines)
                    + "\n\n"
                    + "The positive-direction table shows the features that most often push cases **toward approval**, while the negative-direction table shows the features that most often push cases **toward rejection or review**."
                )
            )
        except Exception as exc:
            display(
                Markdown(
                    f'''
                    ### Interpretability fallback

                    SHAP failed at runtime (`{type(exc).__name__}: {exc}`). The notebook therefore falls back to model feature importance so interpretability remains available.
                    '''
                )
            )
            render_feature_importance_fallback(
                fallback_importance,
                heading="Fallback feature importance: overall decision drivers",
            )
            if not probability_association_fallback.empty:
                display(
                    Markdown(
                        '''
                        The fallback below is directional only in an approximate sense: it shows which transformed features are most associated with higher or lower predicted award probability on the saved holdout.
                        '''
                    )
                )
                display(
                    probability_association_fallback.sort_values(
                        "association_with_award_probability",
                        ascending=False,
                    ).head(10)
                )
                display(
                    probability_association_fallback.sort_values(
                        "association_with_award_probability",
                        ascending=True,
                    ).head(10)
                )

# %% [notebook cell 26]
if pipeline is None or decision_df.empty:
    print("Case-level SHAP examples require both the model bundle and the decision CSV.")
else:
    example_frames = []
    for action in ["approve", "review", "reject"]:
        subset = decision_df[decision_df["recommended_action"] == action].copy()
        if subset.empty:
            continue
        if action == "review":
            subset["threshold_distance"] = (
                subset["predicted_probability_awarded"] - selected_threshold
            ).abs()
            subset = subset.sort_values(
                ["fairness_risk_flag", "threshold_distance", "confidence_score"],
                ascending=[False, True, False],
            )
        else:
            subset = subset.sort_values("confidence_score", ascending=False)
        example_frames.append(subset.head(1))

    example_cases = pd.concat(example_frames, axis=0).copy()
    example_positions = [int(position) for position in example_cases.index]
    example_cases = example_cases.reset_index(names="decision_row_position")

    shap_case_explanation = None
    shap_case_error = None
    transformed_test = np.asarray(preprocessor.transform(test_model_features), dtype=float)
    if SHAP_AVAILABLE:
        try:
            transformed_background = transformed_test[
                np.linspace(0, len(test_model_features) - 1, min(120, len(test_model_features)), dtype=int)
            ]
            transformed_examples = transformed_test[example_positions]
            shap_case_explanation = build_shap_explanation(
                estimator,
                transformed_background,
                transformed_examples,
                feature_names,
            )
        except Exception as exc:
            shap_case_error = f"{type(exc).__name__}: {exc}"
    else:
        shap_case_error = SHAP_IMPORT_ERROR

    driver_tables = []
    if shap_case_error:
        display(
            Markdown(
                f'''
                ### Case-level interpretability fallback

                SHAP case explanations are unavailable (`{shap_case_error}`). The notebook therefore uses the stored case explanation text so each example still has a defensible narrative.
                '''
            )
        )
    for row_position, example in example_cases.iterrows():
        display(
            Markdown(
                f'''
                #### Case {row_position + 1}: `{example['recommended_action']}`

                Probability of award: **{example['predicted_probability_awarded']:.3f}**  
                Confidence score: **{example['confidence_score']:.3f}**  
                Action reason: {example['action_reason']}
                '''
            )
        )
        if shap_case_explanation is not None:
            shap.plots.waterfall(shap_case_explanation[row_position], max_display=10, show=False)
            plt.tight_layout()
            plt.show()

            driver_table = top_shap_drivers(shap_case_explanation, row_position, top_n=5)
            driver_table.insert(0, "case", f"case_{row_position + 1}_{example['recommended_action']}")
            driver_tables.append(driver_table)
        else:
            display(Markdown(f"Interpretability fallback: {example['case_explanation']}"))

    if driver_tables:
        display(pd.concat(driver_tables, ignore_index=True))

    display(
        example_cases[
            [
                "decision_row_position",
                "parsed_school",
                "parsed_level",
                "parsed_nationality",
                "predicted_probability_awarded",
                "recommended_action",
                "action_reason",
            ]
        ]
    )

# %% [notebook cell 28]
fairness_rule_table = pd.DataFrame(
    [
        {
            "rule": f"if subgroup false-negative gap > {FAIRNESS_GAP_THRESHOLD:.2f} and sample size >= {FAIRNESS_MIN_GROUP_SIZE}",
            "action": "lower threshold and force manual review",
            "thesis_value": "Protects against missing deserving students in disadvantaged subgroups.",
        },
        {
            "rule": f"if subgroup false-positive gap > {FAIRNESS_GAP_THRESHOLD:.2f} and sample size >= {FAIRNESS_MIN_GROUP_SIZE}",
            "action": "force manual review",
            "thesis_value": "Stops automatic approvals/rejections when subgroup error patterns diverge materially.",
        },
        {
            "rule": f"if sample size < {FAIRNESS_MIN_GROUP_SIZE}",
            "action": "monitor only",
            "thesis_value": "Avoids overreacting to unstable subgroup estimates.",
        },
    ]
)
display(fairness_rule_table)

if not normal_workflow:
    print("Metadata is missing, so the fairness section cannot show saved subgroup metrics.")
else:
    fairness_summary = normal_workflow.get("test_fairness_summary", {})
    fairness_actions = pd.DataFrame(fairness_summary.get("actions", []))
    fairness_metrics = pd.DataFrame(fairness_summary.get("group_metrics", []))

    if not fairness_metrics.empty:
        cols = [
            column
            for column in [
                "group_column",
                "group_value",
                "sample_size",
                "approval_rate",
                "false_positive_rate",
                "false_negative_rate",
                "avg_predicted_probability",
            ]
            if column in fairness_metrics.columns
        ]
        display(fairness_metrics[cols].head(20))

    if not fairness_actions.empty:
        action_cols = [
            column
            for column in [
                "group_column",
                "group_value",
                "sample_size",
                "false_negative_gap",
                "false_positive_gap",
                "recommended_action",
                "action_reason",
            ]
            if column in fairness_actions.columns
        ]
        display(fairness_actions[action_cols].head(20))

        actionable = fairness_actions[
            fairness_actions["recommended_action"].isin(
                ["force_manual_review", "lower_threshold_and_force_review"]
            )
        ]
        display(
            Markdown(
                f'''
                ### Fairness interpretation

                The saved run produced **{len(actionable)}** actionable subgroup intervention rows. In other words, fairness is part of the operational governance logic: when subgroup gaps cross the thesis thresholds, the workflow stops trusting automatic routing and forces human review.
                '''
            )
        )
    else:
        print("No subgroup crossed the intervention threshold in the saved run.")

# %% [notebook cell 30]
if decision_df.empty:
    print("Fairness intervention impact needs the decision CSV.")
else:
    selected_threshold = float(decision_df["selected_threshold"].iloc[0])
    approve_threshold = float(decision_df["approve_threshold"].iloc[0])
    reject_threshold = float(decision_df["reject_threshold"].iloc[0])
    fairness_flags = decision_df["fairness_risk_flag"].fillna(False).astype(bool).to_numpy()
    probabilities = decision_df["predicted_probability_awarded"].to_numpy()
    actuals = decision_df["actual_target"].to_numpy()
    review_margin = (
        normal_workflow.get("decision_summary", {}).get("review_margin", 0.10)
        if normal_workflow
        else 0.10
    )
    borderline = np.abs(probabilities - selected_threshold) <= review_margin

    without_override_actions = np.where(
        borderline,
        "review",
        np.where(
            probabilities >= approve_threshold,
            "approve",
            np.where(probabilities <= reject_threshold, "reject", "review"),
        ),
    )
    with_override_actions = decision_df["recommended_action"].to_numpy()

    comparison_rows = []
    for scenario_name, actions in [
        ("without_fairness_override", without_override_actions),
        ("with_fairness_override", with_override_actions),
    ]:
        actions = pd.Series(actions)
        approve_mask = actions.eq("approve")
        reject_mask = actions.eq("reject")
        review_mask = actions.eq("review")
        harmful_auto_errors = int(
            (approve_mask & (actuals == 0)).sum()
            + (reject_mask & (actuals == 1)).sum()
        )
        comparison_rows.append(
            {
                "scenario": scenario_name,
                "approve_cases": int(approve_mask.sum()),
                "reject_cases": int(reject_mask.sum()),
                "review_cases": int(review_mask.sum()),
                "automation_share": round(float((~review_mask).mean()), 4),
                "fairness_risk_cases_auto_routed": int(((~review_mask) & fairness_flags).sum()),
                "harmful_auto_errors": harmful_auto_errors,
            }
        )

    fairness_override_comparison = pd.DataFrame(comparison_rows)
    display(fairness_override_comparison)

    without_row = fairness_override_comparison.loc[
        fairness_override_comparison["scenario"] == "without_fairness_override"
    ].iloc[0]
    with_row = fairness_override_comparison.loc[
        fairness_override_comparison["scenario"] == "with_fairness_override"
    ].iloc[0]

    display(
        Markdown(
            f'''
            ### Fairness override reading

            - Without fairness overrides, the workflow would auto-route **{int(without_row["fairness_risk_cases_auto_routed"])}** flagged subgroup cases.
            - With fairness overrides active, that drops to **{int(with_row["fairness_risk_cases_auto_routed"])}**. The cost is higher review load (**{int(with_row["review_cases"])}** vs **{int(without_row["review_cases"])}**), but the gain is that subgroup-risk cases are routed through review instead of automation.
            - Harmful automatic errors also fall from **{int(without_row["harmful_auto_errors"])}** to **{int(with_row["harmful_auto_errors"])}** in this saved comparison.
            - Operationally, this is the governance tradeoff: accept some extra committee work in exchange for forcing review when subgroup error gaps make automation less trustworthy.
            '''
        )
    )

# %% [notebook cell 32]
if decision_df.empty:
    print("The impact section needs the decision CSV.")
else:
    impact_summary = (
        normal_workflow.get("business_impact_summary", {})
        if normal_workflow
        else summarize_decision_impact(decision_df, selected_threshold)
    )

    threshold_grid = sorted(
        {
            round(value, 2)
            for value in np.clip(
                np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, selected_threshold]),
                0.05,
                0.95,
            )
        }
    )

    sensitivity_rows = []
    probabilities = decision_df["predicted_probability_awarded"].to_numpy()
    actuals = decision_df["actual_target"].to_numpy()

    for threshold in threshold_grid:
        predictions = (probabilities >= threshold).astype(int)
        tp = int(((predictions == 1) & (actuals == 1)).sum())
        fp = int(((predictions == 1) & (actuals == 0)).sum())
        fn = int(((predictions == 0) & (actuals == 1)).sum())

        sensitivity_rows.append(
            {
                "threshold": threshold,
                "shortlisted": int(predictions.sum()),
                "missed_deserving_students": fn,
                "extra_borderline_reviews": fp,
                "precision": precision_score(actuals, predictions, zero_division=0),
                "recall": recall_score(actuals, predictions, zero_division=0),
            }
        )

    sensitivity_df = pd.DataFrame(sensitivity_rows).sort_values("threshold").reset_index(drop=True)
    display(sensitivity_df)

    selected_row = sensitivity_df.iloc[
        (sensitivity_df["threshold"] - selected_threshold).abs().argmin()
    ]
    reference_threshold = 0.50
    reference_row = sensitivity_df.iloc[
        (sensitivity_df["threshold"] - reference_threshold).abs().argmin()
    ]

    boundary_scenarios = pd.DataFrame(evaluate_decision_boundary_scenarios(decision_df))
    display(boundary_scenarios)

    impact_highlights = pd.DataFrame(
        [
            {
                "impact": "↓ missed deserving students (vs threshold 0.50)",
                "baseline": int(reference_row["missed_deserving_students"]),
                "with_governed_policy": int(selected_row["missed_deserving_students"]),
                "improvement": int(reference_row["missed_deserving_students"] - selected_row["missed_deserving_students"]),
            },
            {
                "impact": "↓ manual reviews",
                "baseline": int(impact_summary["manual_review_baseline_cases"]),
                "with_governed_policy": int(impact_summary["review_cases_after_model"]),
                "improvement": int(impact_summary["manual_review_baseline_cases"] - impact_summary["review_cases_after_model"]),
            },
            {
                "impact": "↑ safe automation",
                "baseline": 0,
                "with_governed_policy": int(impact_summary["automated_cases"]),
                "improvement": int(impact_summary["automated_cases"]),
            },
        ]
    )
    display(impact_highlights)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        sensitivity_df["threshold"],
        sensitivity_df["missed_deserving_students"],
        marker="o",
        linewidth=2,
        color="#b22222",
        label="Missed deserving students",
    )
    axes[0].axvline(selected_threshold, linestyle="--", color="black", linewidth=1, label="Selected threshold")
    axes[0].set_title("Threshold sensitivity: missed deserving students")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    scenario_plot = boundary_scenarios.set_index("scenario")[
        ["approve_cases", "reject_cases", "review_cases"]
    ]
    scenario_plot.plot(kind="bar", stacked=True, ax=axes[1], color=["#2e8b57", "#b22222", "#d4a72c"])
    axes[1].set_title("Safe automation vs review under different boundaries")
    axes[1].set_xlabel("Scenario")
    axes[1].set_ylabel("Cases")
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()

    display(
        Markdown(
            f'''
            ### Impact interpretation

            - **↓ missed deserving students:** at threshold **{reference_row['threshold']:.2f}**, the model would miss **{int(reference_row['missed_deserving_students'])}** deserving students; at the selected threshold **{selected_row['threshold']:.2f}**, it misses **{int(selected_row['missed_deserving_students'])}**.
            - **↓ manual reviews:** an all-manual process would review **{impact_summary['manual_review_baseline_cases']}** cases, while the governed workflow keeps only **{impact_summary['review_cases_after_model']}** in review.
            - **↑ safe automation:** the workflow auto-routes **{impact_summary['automated_cases']} of {impact_summary['total_cases']}** cases and avoids **{impact_summary['harmful_auto_errors_avoided_vs_threshold_only']}** harmful automatic errors versus a threshold-only rule.
            '''
        )
    )

# %% [notebook cell 34]
if decision_df.empty:
    print("Failure-mode analysis needs the decision CSV.")
else:
    review_margin = (
        bundle.get("decision_policy", {}).get("decision_summary", {}).get("review_margin", 0.10)
        if bundle is not None
        else 0.10
    )
    threshold_distance = (
        decision_df["predicted_probability_awarded"] - selected_threshold
    ).abs()
    borderline_band = min(float(review_margin), 0.05)
    borderline_cases = int((threshold_distance <= borderline_band).sum())

    missing_share = test_model_features.isna().mean(axis=1)
    high_missing_quantile = 0.90
    high_missing_threshold = float(missing_share.quantile(high_missing_quantile))
    high_missing_cases = int((missing_share >= high_missing_threshold).sum())

    rare_profile_threshold = 5
    rare_profile_column = next(
        (
            column
            for column in ["parsed_school", "parsed_nationality", "parsed_level"]
            if column in audit_frame.columns and column in audit_test.columns
        ),
        None,
    )
    if rare_profile_column is not None:
        profile_counts = audit_frame[rare_profile_column].fillna("missing").value_counts(dropna=False)
        rare_profile_series = (
            audit_test[rare_profile_column]
            .fillna("missing")
            .map(profile_counts)
            .fillna(0)
        )
        rare_profile_cases = int(
            rare_profile_series.lt(rare_profile_threshold).sum()
        )
        rare_profile_label = rare_profile_column.replace("_", " ")
    else:
        rare_profile_cases = 0
        rare_profile_label = "profile group"
        rare_profile_series = pd.Series(0, index=decision_df.index)

    fairness_risk_cases = int(decision_df["fairness_risk_flag"].fillna(False).astype(bool).sum())
    main_policy_metrics = (
        normal_workflow.get("test_summary", {}).get("selected_policy_metrics", {})
        if normal_workflow
        else {}
    )
    temporal_row = None
    temporal_failure_text = "Temporal validation was not available, so the future-deployment validation plan remains the main safeguard against cohort drift."
    if "temporal_test_summary" in globals():
        temporal_row = {
            "failure_mode": "Temporal drift / later-cycle cohort shift",
            "trigger_count": len(temporal_test_target) if "temporal_test_target" in globals() else np.nan,
            "why_it_breaks": "A model that works on random holdout data can still weaken on a later admissions cycle when applicant mix or committee behavior shifts.",
            "governance_response": "Re-test thresholds on future cycles before expanding automation and keep prospective monitoring active.",
        }
        temporal_failure_text = (
            f"Temporal drift is visible in the harder later-cycle test: recall moves from "
            f"**{main_policy_metrics.get('recall', float('nan')):.3f}** on the main holdout "
            f"to **{temporal_test_summary['selected_policy_metrics']['recall']:.3f}** on the temporal test. "
            "That is why future-cycle monitoring belongs inside the governance story."
        )

    failure_mode_rows = [
        {
            "failure_mode": "Borderline probabilities near the decision threshold",
            "trigger_count": borderline_cases,
            "why_it_breaks": "Small score shifts can flip the recommended action, so these cases are the least stable under threshold changes.",
            "governance_response": "Keep them in manual review rather than forcing automation.",
        },
        {
            "failure_mode": "High missing-data profiles",
            "trigger_count": high_missing_cases,
            "why_it_breaks": "When many engineered inputs are missing relative to the rest of the holdout, the model has less case-specific evidence and relies more heavily on general patterns.",
            "governance_response": "Request more documentation or prioritize committee review.",
        },
        {
            "failure_mode": f"Rare {rare_profile_label} profiles",
            "trigger_count": rare_profile_cases,
            "why_it_breaks": "Very uncommon profiles may be underrepresented in training and therefore less stable at deployment.",
            "governance_response": "Use review-first handling for rare profiles until more evidence accumulates.",
        },
        {
            "failure_mode": "Subgroup fairness-risk cases",
            "trigger_count": fairness_risk_cases,
            "why_it_breaks": "Automation is less trustworthy when subgroup error gaps trigger the governance thresholds.",
            "governance_response": "Force review or lower the effective threshold according to the fairness action rules.",
        },
    ]
    if temporal_row is not None:
        failure_mode_rows.append(temporal_row)

    failure_modes = pd.DataFrame(failure_mode_rows)
    display(failure_modes)

    failure_example_frame = decision_df.copy()
    failure_example_frame["threshold_distance"] = threshold_distance
    failure_example_frame["missing_share"] = missing_share.to_numpy()
    failure_example_frame["rare_profile_flag"] = rare_profile_series.to_numpy() < rare_profile_threshold
    failure_example_frame["failure_bucket"] = "other"
    failure_example_frame.loc[
        failure_example_frame["threshold_distance"] <= borderline_band,
        "failure_bucket",
    ] = "borderline"
    failure_example_frame.loc[
        failure_example_frame["missing_share"] >= high_missing_threshold,
        "failure_bucket",
    ] = "high_missing"
    failure_example_frame.loc[
        failure_example_frame["rare_profile_flag"],
        "failure_bucket",
    ] = "rare_profile"
    failure_example_frame.loc[
        failure_example_frame["fairness_risk_flag"].fillna(False).astype(bool),
        "failure_bucket",
    ] = "fairness_risk"

    failure_examples = pd.concat(
        [
            failure_example_frame[
                failure_example_frame["threshold_distance"] <= borderline_band
            ].sort_values("threshold_distance", ascending=True).head(3),
            failure_example_frame[
                failure_example_frame["missing_share"] >= high_missing_threshold
            ].sort_values("missing_share", ascending=False).head(3),
            failure_example_frame[
                failure_example_frame["rare_profile_flag"]
            ].head(3),
            failure_example_frame[
                failure_example_frame["fairness_risk_flag"].fillna(False).astype(bool)
            ].head(3),
        ],
        axis=0,
    ).drop_duplicates().reset_index(drop=True)

    if not failure_examples.empty:
        display(
            failure_examples[
                [
                    "parsed_school",
                    "parsed_level",
                    "predicted_probability_awarded",
                    "recommended_action",
                    "failure_bucket",
                    "threshold_distance",
                    "missing_share",
                    "fairness_risk_flag",
                ]
            ]
        )

    display(
        Markdown(
            f'''
            ### Failure-mode interpretation

            - **Borderline cases:** **{borderline_cases}** cases fall very close to the selected threshold, so they should stay inside human review.
            - **Missing-data cases:** **{high_missing_cases}** cases fall into the top-decile missingness band (missing-share threshold **{high_missing_threshold:.3f}**), which means the model is working with thinner evidence than usual.
            - **Rare profiles:** **{rare_profile_cases}** cases come from rare {rare_profile_label} groups, so historical learning is less reliable for them.
            - **Fairness-risk overrides:** **{fairness_risk_cases}** cases are already being pulled out of automation because the governance safeguards fired.
            - **Temporal drift signal:** {temporal_failure_text}
            '''
        )
    )

# %% [notebook cell 36]
temporal_assessment = assess_temporal_split_feasibility(labeled_dataframe)

limitation_rows = [
    {
        "limitation": "Temporal validation is still limited by observed cycles",
        "why_it_matters": "Even with an earlier-cycle/later-cycle test, only a few historical cycles are available and cycle drift can still continue after the saved data ends.",
        "current_status": temporal_assessment["note"],
    },
    {
        "limitation": "Historical decisions as target",
        "why_it_matters": "The model learns past committee behavior, so any historical inconsistency or bias can be reflected unless fairness monitoring stays active.",
        "current_status": "Addressed partly through target filtering, fairness auditing, and review overrides.",
    },
    {
        "limitation": "SHAP is explanatory, not causal",
        "why_it_matters": "SHAP shows which features moved the prediction, but not whether those features should normatively drive aid policy.",
        "current_status": "Used here for transparency, not for causal claims.",
    },
    {
        "limitation": "Operational impact is simulated on holdout data",
        "why_it_matters": "True committee savings and intervention effects should still be checked in live workflow pilots.",
        "current_status": "Useful for policy comparison, but not a replacement for live monitoring.",
    },
]

display(pd.DataFrame(limitation_rows))

# %% [notebook cell 38]
if decision_df.empty:
    print("The conclusion section needs the decision CSV.")
else:
    impact_summary = (
        normal_workflow.get("business_impact_summary", {})
        if normal_workflow
        else summarize_decision_impact(decision_df, selected_threshold)
    )
    selected_model_name = bundle.get("selected_model_name") if bundle is not None else "saved model"
    conclusion_policy_read = ""
    if "policy_weight_df" in globals() and not policy_weight_df.empty:
        conclusion_ratio_1 = policy_weight_df.loc[
            policy_weight_df["missed_need_to_review_ratio"] == "1:1"
        ].iloc[0]
        conclusion_ratio_3 = policy_weight_df.loc[
            policy_weight_df["missed_need_to_review_ratio"] == "3:1"
        ].iloc[0]
        conclusion_policy_read = (
            f"The 3:1 policy reduces missed deserving students from "
            f"**{int(conclusion_ratio_1['missed_deserving_students'])}** under 1:1 "
            f"to **{int(conclusion_ratio_3['missed_deserving_students'])}** while keeping the model in the maximum-recall regime."
        )
    else:
        conclusion_policy_read = (
            f"The selected policy avoids **{impact_summary['harmful_auto_errors_avoided_vs_threshold_only']}** "
            "harmful automatic errors versus a threshold-only rule."
        )

    display(
        Markdown(
            f'''
            ### Final thesis conclusion

            **What we built.** A governed FAID eligibility workflow that starts from cleaned operational data, removes post-decision leakage, selects a defensible model, and translates probabilities into `approve`, `reject`, and `review` actions.

            **Why it matters.** This thesis does not merely predict aid eligibility; it operationalizes a governed triage framework balancing predictive performance, fairness safeguards, and decision efficiency.

            **What improves.** With **{selected_model_name}**, the governed decision layer reduces manual review from **{impact_summary['manual_review_baseline_cases']}** cases to **{impact_summary['review_cases_after_model']}**, auto-routes **{impact_summary['automated_cases']}** cases, and avoids **{impact_summary['harmful_auto_errors_avoided_vs_threshold_only']}** harmful automatic errors versus threshold-only automation. {conclusion_policy_read}

            **What comes next.** The next step is prospective validation on a true future admissions cycle, with ongoing fairness monitoring, threshold recalibration, and periodic review of whether the model remains aligned with committee priorities.
            '''
        )
    )

# %% [notebook cell 42]
# import subprocess
# subprocess.run(
#     [
#         "python3",
#         "faid_models/eligibility/scripts/generate_full_thesis_artifacts.py",
#     ],
#     check=True,
# )
