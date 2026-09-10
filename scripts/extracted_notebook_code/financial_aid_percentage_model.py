# Extracted notebook code for portfolio review.
# Generated from a sanitized notebook copy; outputs and execution state are intentionally excluded.


# %% [notebook cell 4]
from pathlib import Path
from html import escape
import json
import os
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display as _ipython_display

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
            (candidate / "faid_models/percentage/modeling/financial_aid_percentage_model.py").exists()
            and (candidate / "cleaned data/faid_cleaned.csv").exists()
        ):
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not locate the financial-aid-portfolio-project project root. "
        "Set FAID_PROJECT_ROOT or open the notebook from inside the project."
    )


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from faid_models.percentage.modeling import financial_aid_percentage_model as percentage_model

plt.style.use("seaborn-v0_8-whitegrid")

SOURCE_PATH = PROJECT_ROOT / "faid_models/percentage/modeling/financial_aid_percentage_model.py"
__file__ = str(SOURCE_PATH)
DATA_PATH = percentage_model.DATA_PATH.expanduser().resolve()
ARTIFACT_DIR = percentage_model.ARTIFACT_DIR.expanduser().resolve()
ARTIFACT_PATHS = percentage_model.build_artifact_paths(ARTIFACT_DIR)

warnings.filterwarnings(
    "ignore",
    message="Skipping features without any observed values: \['parsed_spouse_citizenship'\].*",
    category=UserWarning,
)


def relative_display_path(path: Path) -> str:
    resolved_path = path.expanduser().resolve()
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved_path)


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def safe_read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


metadata = safe_read_json(ARTIFACT_PATHS["metadata"])
report_text = (
    ARTIFACT_PATHS["report"].read_text(encoding="utf-8")
    if ARTIFACT_PATHS["report"].exists()
    else ""
)
business_summary_text = (
    ARTIFACT_PATHS["business_summary"].read_text(encoding="utf-8")
    if ARTIFACT_PATHS["business_summary"].exists()
    else ""
)
temporal_summary = safe_read_json(ARTIFACT_PATHS["temporal_summary"])
calibration_summary = safe_read_json(ARTIFACT_PATHS["calibration_summary"])

validation_comparison_df = safe_read_csv(ARTIFACT_PATHS["candidate_comparison_validation"])
test_comparison_df = safe_read_csv(ARTIFACT_PATHS["candidate_comparison_test"])
validation_calibration_bins_df = safe_read_csv(ARTIFACT_PATHS["calibration_bins_validation"])
test_calibration_bins_df = safe_read_csv(ARTIFACT_PATHS["calibration_bins_test"])
decision_df = safe_read_csv(ARTIFACT_PATHS["decision_recommendations_test"])
test_predictions_df = safe_read_csv(ARTIFACT_PATHS["test_predictions"])
fairness_actions_df = safe_read_csv(ARTIFACT_PATHS["fairness_actions_validation"])
segment_metrics_df = safe_read_csv(ARTIFACT_PATHS["segment_metrics"])
stage1_shap_df = safe_read_csv(ARTIFACT_PATHS["shap_stage1_global"])
stage2_shap_df = safe_read_csv(ARTIFACT_PATHS["shap_stage2_global"])

dataset_summary = metadata.get("dataset", {})
final_test_metrics = metadata.get("final_test_metrics", {})
decision_summary = metadata.get("decision_summary", {})
error_thresholds = metadata.get("error_thresholds", {})
threshold_rationale = metadata.get("threshold_rationale", {})
fairness_validation_summary = metadata.get("fairness_validation_summary", {})
fairness_test_summary = metadata.get("fairness_test_summary", {})
fairness_guardrails = metadata.get("fairness_guardrails", {})
calibration_validation_summary = metadata.get("calibration_validation_summary", {})
calibration_test_summary = metadata.get("calibration_test_summary", {})
feature_policy = metadata.get("feature_policy", {})
problem_framing = metadata.get("problem_framing", {})


def select_probability_bins(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "probability_mode" not in frame.columns:
        return frame
    subset = frame.loc[frame["probability_mode"].astype(str) == str(mode)].copy()
    return subset if not subset.empty else frame.copy()


availability = pd.DataFrame(
    [
        {"item": "project_root", "value": "."},
        {"item": "source_script", "value": relative_display_path(SOURCE_PATH)},
        {"item": "data_path", "value": relative_display_path(DATA_PATH)},
        {"item": "artifact_dir", "value": relative_display_path(ARTIFACT_DIR)},
        {"item": "metadata_loaded", "value": bool(metadata)},
        {"item": "report_loaded", "value": bool(report_text)},
        {"item": "business_summary_loaded", "value": bool(business_summary_text)},
        {"item": "decision_csv_loaded", "value": not decision_df.empty},
        {"item": "calibration_bins_loaded", "value": not validation_calibration_bins_df.empty},
    ]
)
display(availability)

# %% [notebook cell 6]
before_after_df = pd.DataFrame(
    [
        {
            "stage": "Raw decision process",
            "quality": "Manual and potentially inconsistent",
            "why": "Human judgment can capture nuance, but it is hard to audit, hard to scale, and vulnerable to inconsistency across cycles and reviewers.",
        },
        {
            "stage": "Model only",
            "quality": "Predictive but risky",
            "why": "A model can be accurate on average but still be unsafe if probabilities are misread, subgroup gaps are ignored, or uncertainty is hidden.",
        },
        {
            "stage": "Final system",
            "quality": "Controlled, fairer, and partially automatable",
            "why": "The final workflow adds leakage controls, calibration checks, dynamic fairness guardrails, uncertainty, manual-review routing, and explicit automation limits.",
        },
    ]
)
display(before_after_df)

executive_rows = pd.DataFrame(
    [
        {"headline": "Selected model", "value": problem_framing.get("selected_model_name", "n/a")},
        {"headline": "Main holdout MAE", "value": round(float(final_test_metrics.get("mae", float("nan"))), 3)},
        {"headline": "Main holdout R2", "value": round(float(final_test_metrics.get("r2", float("nan"))), 3)},
        {"headline": "Automation rate", "value": f"{float(decision_summary.get('automation_rate', 0.0)):.1%}"},
        {"headline": "Harmful auto-zero errors", "value": int(decision_summary.get("harmful_auto_zero_errors", 0))},
        {"headline": "Harmful auto-award errors", "value": int(decision_summary.get("harmful_auto_award_errors", 0))},
        {"headline": "Selected probability mode", "value": calibration_validation_summary.get("selected_mode", "n/a")},
        {"headline": "Temporal split", "value": str(temporal_summary.get("split", {}).get("train_cycles", [])) + " -> " + str(temporal_summary.get("split", {}).get("validation_cycles", [])) + " -> " + str(temporal_summary.get("split", {}).get("test_cycles", [])) if temporal_summary.get("available") else "not available"},
    ]
)
display(executive_rows)

# %% [notebook cell 8]
insight_rows = []

if DATA_PATH.exists():
    cycle_frame = pd.read_csv(
        DATA_PATH,
        usecols=[
            percentage_model.APPLICATION_TERM_COLUMN,
            percentage_model.TARGET_COLUMN,
        ],
    )
    cycle_frame[percentage_model.APPLICATION_TERM_COLUMN] = pd.to_numeric(
        cycle_frame[percentage_model.APPLICATION_TERM_COLUMN], errors='coerce'
    )
    cycle_frame[percentage_model.TARGET_COLUMN] = pd.to_numeric(
        cycle_frame[percentage_model.TARGET_COLUMN], errors='coerce'
    )
    cycle_frame['cycle_year'] = (
        cycle_frame[percentage_model.APPLICATION_TERM_COLUMN] // 100
    ).astype('Int64')
    cycle_summary_df = (
        cycle_frame.dropna(subset=['cycle_year', percentage_model.TARGET_COLUMN])
        .groupby('cycle_year', dropna=False)[percentage_model.TARGET_COLUMN]
        .agg(['count', 'mean', 'median'])
        .reset_index()
    )
    if len(cycle_summary_df) >= 2:
        first_cycle = cycle_summary_df.iloc[0]
        last_cycle = cycle_summary_df.iloc[-1]
        insight_rows.append(
            {
                'finding': 'Cycle drift was real, not hypothetical',
                'evidence': (
                    f"Average aid percentage moved from {first_cycle['mean']:.2f} in cycle {int(first_cycle['cycle_year'])} "
                    f"to {last_cycle['mean']:.2f} in cycle {int(last_cycle['cycle_year'])}."
                ),
                'why_it_matters': 'This proves that a random split alone would understate real deployment risk. Temporal validation was necessary.',
            }
        )

fairness_focus = fairness_actions_df.loc[
    (fairness_actions_df.get('sample_size', pd.Series(dtype=float)).fillna(0) >= percentage_model.FAIRNESS_MIN_GROUP_SIZE)
    & (fairness_actions_df.get('recommended_action', pd.Series(dtype=object)).astype(str) == 'force_manual_review')
].copy()
if not fairness_focus.empty:
    fairness_focus = fairness_focus.sort_values(
        by=['sample_size', 'mae_gap'],
        ascending=[False, False],
    )
    row = fairness_focus.iloc[0]
    insight_rows.append(
        {
            'finding': 'The most important fairness override was counterintuitive',
            'evidence': (
                f"The strongest manual-review trigger on validation was {row['group_column']}={row['group_value']} "
                f"(sample {int(row['sample_size'])}, MAE gap {float(row['mae_gap']):.2f}, bias gap {float(row['bias_gap']):.2f})."
            ),
            'why_it_matters': "The system discovered that risk was not just about 'low income gets missed'; it also needed protection against subgroup-specific overprediction patterns.",
        }
    )

if not stage2_shap_df.empty:
    top_stage2 = stage2_shap_df.head(3)['feature'].tolist()
    insight_rows.append(
        {
            'finding': 'Aid percentage was not driven by income alone',
            'evidence': 'Top positive-amount SHAP drivers included: ' + ', '.join(top_stage2) + '.',
            'why_it_matters': 'This suggests the portfolio encoded a richer decision logic mixing documentation, merit context, and application-track effects rather than a single linear income rule.',
        }
    )

insights_df = pd.DataFrame(insight_rows)
display(insights_df)

if not insights_df.empty:
    display(
        Markdown(
            '### Memorable takeaway\n'
            'The strongest lesson from the model is that **drift, subgroup-specific error, and non-linear decision structure were all real**. '
            'That is exactly why the final contribution had to be a governed decision system rather than a bare regressor.'
        )
    )

# %% [notebook cell 10]
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

selected_validation_mode = calibration_validation_summary.get("selected_mode", "uncalibrated")
selected_test_mode = calibration_test_summary.get("selected_mode", selected_validation_mode)
val_bins = select_probability_bins(validation_calibration_bins_df, selected_validation_mode)
test_bins = select_probability_bins(test_calibration_bins_df, selected_test_mode)

axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1, label="Perfect calibration")
if not val_bins.empty:
    axes[0, 0].plot(
        val_bins["mean_predicted_probability"],
        val_bins["empirical_nonzero_rate"],
        marker="o",
        label=f"Validation ({selected_validation_mode})",
    )
if not test_bins.empty:
    axes[0, 0].plot(
        test_bins["mean_predicted_probability"],
        test_bins["empirical_nonzero_rate"],
        marker="s",
        label=f"Test ({selected_test_mode})",
    )
axes[0, 0].set_title("Calibration Plot")
axes[0, 0].set_xlabel("Predicted probability")
axes[0, 0].set_ylabel("Observed nonzero-aid rate")
axes[0, 0].legend()

if not test_predictions_df.empty and "absolute_error" in test_predictions_df.columns:
    axes[0, 1].hist(test_predictions_df["absolute_error"].dropna(), bins=20, color="#4C78A8", alpha=0.85)
axes[0, 1].set_title("Error Distribution")
axes[0, 1].set_xlabel("Absolute error (percentage points)")
axes[0, 1].set_ylabel("Cases")

fairness_plot_df = fairness_actions_df.copy()
if not fairness_plot_df.empty:
    fairness_plot_df = fairness_plot_df.loc[
        fairness_plot_df["sample_size"].fillna(0).astype(int) >= percentage_model.FAIRNESS_MIN_GROUP_SIZE
    ].copy()
    fairness_plot_df["label"] = (
        fairness_plot_df["group_column"].astype(str)
        + "="
        + fairness_plot_df["group_value"].astype(str)
    )
    fairness_plot_df["abs_mae_gap"] = fairness_plot_df["mae_gap"].abs()
    fairness_plot_df = fairness_plot_df.sort_values("abs_mae_gap", ascending=False).head(8)
    color_map = {
        "force_manual_review": "#E45756",
        "tighten_automation_thresholds": "#F58518",
        "monitor_only": "#72B7B2",
        "monitor_only_small_sample": "#B279A2",
    }
    axes[1, 0].barh(
        fairness_plot_df["label"],
        fairness_plot_df["mae_gap"],
        color=[
            color_map.get(action, "#72B7B2")
            for action in fairness_plot_df["recommended_action"].astype(str)
        ],
    )
axes[1, 0].axvline(0, color="black", linewidth=1)
axes[1, 0].set_title("Fairness Gap Chart")
axes[1, 0].set_xlabel("MAE gap vs overall")

if not decision_df.empty and "recommended_action" in decision_df.columns:
    action_counts = decision_df["recommended_action"].value_counts()
    axes[1, 1].pie(
        action_counts.values,
        labels=action_counts.index.tolist(),
        autopct="%1.1f%%",
        startangle=90,
    )
axes[1, 1].set_title("Automation vs Review")

plt.tight_layout()
plt.show()

# %% [notebook cell 12]
problem_table = pd.DataFrame(
    [
        {"challenge": "Zero-inflated target", "why_it_matters": "Many students receive 0%, so a plain regressor can blur two different questions: any aid vs amount of aid."},
        {"challenge": "Leakage risk", "why_it_matters": "Post-decision and target-adjacent columns can make a model look excellent while being invalid in deployment."},
        {"challenge": "Operational use", "why_it_matters": "A committee needs a safe route recommendation, not just a raw numeric estimate."},
        {"challenge": "Subgroup risk", "why_it_matters": "Average accuracy can hide uneven subgroup behavior."},
        {"challenge": "Cycle drift", "why_it_matters": "Performance must survive movement from earlier to later application terms."},
    ]
)
display(problem_table)

# %% [notebook cell 14]
#!/usr/bin/env python3

from __future__ import annotations

import json
import inspect
import math
import os
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex-mpl")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores for the following reason:",
    category=UserWarning,
)

MISSING_DEPENDENCIES: list[str] = []

try:
    import joblib
except ImportError:
    MISSING_DEPENDENCIES.append("joblib")

try:
    import numpy as np
except ImportError:
    MISSING_DEPENDENCIES.append("numpy")

try:
    import pandas as pd
except ImportError:
    MISSING_DEPENDENCIES.append("pandas")

try:
    import sklearn  # noqa: F401
except ImportError:
    MISSING_DEPENDENCIES.append("scikit-learn")

if MISSING_DEPENDENCIES:
    missing_list = ", ".join(sorted(MISSING_DEPENDENCIES))
    raise ImportError(
        "Missing required Python packages: "
        f"{missing_list}. Install them with:\n"
        "/opt/homebrew/bin/python3 -m pip install "
        "joblib numpy pandas scikit-learn matplotlib shap"
    )

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    roc_auc_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_config_path(*, env_var: str, default_path: Path, base_dir: Path) -> Path:
    configured = os.environ.get(env_var)
    if not configured:
        return default_path

    candidate = Path(configured).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


DATA_PATH = _resolve_config_path(
    env_var="FAID_PERCENTAGE_DATA_PATH",
    default_path=PROJECT_ROOT / "cleaned data/faid_cleaned.csv",
    base_dir=PROJECT_ROOT,
)

ARTIFACT_DIR = _resolve_config_path(
    env_var="FAID_PERCENTAGE_ARTIFACT_DIR",
    default_path=PROJECT_ROOT / "artifacts/yasmina_percentage",
    base_dir=PROJECT_ROOT,
)

MODEL_ARTIFACT_TYPE = "yasmina_percentage_bundle"
MODEL_ARTIFACT_VERSION = 3
CANONICAL_MODULE_NAME = "faid_models.percentage.modeling.financial_aid_percentage_model"

TARGET_COLUMN = "parsed_need_pct"
RAW_SUBMISSION_DATE_COLUMN = "parsed_submission_date"
APPLICATION_TERM_COLUMN = "parsed_application_term"

RANDOM_STATE = 42
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
TOP_FEATURES_TO_SHOW = 20
TOP_LOCAL_SHAP_FEATURES = 4
MODEL_N_JOBS = 1
CATEGORY_MIN_FREQUENCY = 10
SEGMENT_MIN_GROUP_SIZE = 25
FAIRNESS_MIN_GROUP_SIZE = 30
BOOTSTRAP_ITERATIONS = 250
SHAP_BACKGROUND_SIZE = 80
SHAP_SAMPLE_SIZE = 120
CALIBRATION_BIN_COUNT = 10
TEMPORAL_VALIDATION_MIN_CYCLE_ROWS = 250
AUTO_AWARD_TOLERANCE_STEPS = 2.0
SEVERE_AWARD_ERROR_STEPS = 4.0
ZERO_UPPER_BOUND_STEPS = 2.0
FAIRNESS_MAE_GAP_MIN_STEPS = 0.5
FAIRNESS_BIAS_GAP_MIN_STEPS = 1.0
FAIRNESS_RATE_GAP_MIN = 0.10
FAIRNESS_TIGHTENED_AUTO_ZERO_MULTIPLIER = 0.50
FAIRNESS_TIGHTENED_AUTO_AWARD_DELTA = 0.10
FAIRNESS_TIGHTENED_INTERVAL_STEP_MULTIPLIER = 1.0
FAIRNESS_TIGHTENED_CONFIDENCE_DELTA = 0.10

DECISION_AUTO_ZERO_PROB_OPTIONS = (0.05, 0.10, 0.15, 0.20, 0.25)
DECISION_AUTO_AWARD_PROB_OPTIONS = (0.75, 0.80, 0.85, 0.90)
DECISION_MAX_INTERVAL_WIDTH_OPTIONS = (8.0, 10.0, 12.0, 15.0, 18.0)
DECISION_MIN_CONFIDENCE_OPTIONS = (0.50, 0.60, 0.70)

IDENTIFIER_COLUMNS = {
    "raw_source_row_number",
    "raw_source_sheet_name",
}

FAIRNESS_AUDIT_COLUMNS = (
    "parsed_level",
    "inferred_application_track",
    "parsed_nationality",
    "parsed_applicant_citizenship",
    "segment_school_group",
    "segment_household_income_bucket",
)

EXPLICIT_POST_DECISION_COLUMNS = {
    "raw_decision",
    "parsed_decision",
    "raw_bin_status",
    "parsed_bin_status",
    "raw_over_and_above_decision",
    "parsed_over_and_above_decision",
    "parsed_over_and_above_amount_awarded",
    "parsed_over_and_above_percentage_awarded",
}

POST_DECISION_KEYWORDS = (
    "decision",
    "over_and_above",
    "bin_status",
    "amount_awarded",
    "percentage_awarded",
)

TARGET_ADJACENT_COLUMNS = {
    TARGET_COLUMN,
    "raw_need",
    "parsed_need_comment",
    "raw_need_comment",
    "inferred_remaining_cap_50_rule_pct",
}

RAW_OPERATIONAL_PREFIX = "raw_"
TEXT_LIKE_TOKENS_TO_DROP = (
    "_comment",
    "_detail",
    "_clean",
    "_name",
    "position",
)
TIMESTAMP_COLUMNS = {RAW_SUBMISSION_DATE_COLUMN}

RULE_BASELINE_COLUMNS = (
    "derived_parent_gross_income_total",
    "derived_household_income_total",
    "parsed_properties_total_estimated_value",
    "parsed_properties_count",
    "parsed_siblings_at_institution_count",
    "parsed_dependents_count",
    "parsed_loans_total_amount",
    "parsed_financial_assistants_total_est_annual_amount",
    "parsed_merit_pct",
    "parsed_merit_hist_pct",
    "qa_issue_count",
    "qa_quality_score",
)

OPTIONAL_DEPENDENCY_STATUS: dict[str, dict[str, Any]] = {}


def build_artifact_paths(artifact_dir: Path) -> dict[str, Path]:
    return {
        "model": artifact_dir / "yasmina_percentage_pipeline.joblib",
        "metadata": artifact_dir / "yasmina_percentage_metadata.json",
        "report": artifact_dir / "yasmina_percentage_report.txt",
        "validation_predictions": artifact_dir / "yasmina_percentage_validation_predictions.csv",
        "test_predictions": artifact_dir / "yasmina_percentage_test_predictions.csv",
        "candidate_comparison_validation": artifact_dir
        / "yasmina_percentage_validation_candidate_comparison.csv",
        "candidate_comparison_test": artifact_dir
        / "yasmina_percentage_test_candidate_comparison.csv",
        "segment_metrics": artifact_dir / "yasmina_percentage_segment_metrics.csv",
        "fairness_metrics_validation": artifact_dir
        / "yasmina_percentage_validation_fairness_metrics.csv",
        "fairness_metrics_test": artifact_dir / "yasmina_percentage_test_fairness_metrics.csv",
        "fairness_actions_validation": artifact_dir
        / "yasmina_percentage_validation_fairness_actions.csv",
        "decision_recommendations_test": artifact_dir
        / "yasmina_percentage_test_decision_recommendations.csv",
        "decision_policy_search": artifact_dir / "yasmina_percentage_decision_policy_search.csv",
        "worst_errors_test": artifact_dir / "yasmina_percentage_test_worst_errors.csv",
        "governance_table": artifact_dir / "yasmina_percentage_governance_table.csv",
        "uncertainty_summary": artifact_dir / "yasmina_percentage_uncertainty_summary.json",
        "shap_stage1_global": artifact_dir / "yasmina_percentage_shap_stage1_global.csv",
        "shap_stage2_global": artifact_dir / "yasmina_percentage_shap_stage2_global.csv",
        "calibration_bins_validation": artifact_dir
        / "yasmina_percentage_validation_calibration_bins.csv",
        "calibration_bins_test": artifact_dir / "yasmina_percentage_test_calibration_bins.csv",
        "calibration_summary": artifact_dir / "yasmina_percentage_calibration_summary.json",
        "temporal_validation_candidate_comparison": artifact_dir
        / "yasmina_percentage_temporal_validation_candidate_comparison.csv",
        "temporal_test_candidate_comparison": artifact_dir
        / "yasmina_percentage_temporal_test_candidate_comparison.csv",
        "temporal_summary": artifact_dir / "yasmina_percentage_temporal_summary.json",
        "business_summary": artifact_dir / "yasmina_percentage_business_summary.txt",
        "decision_examples": artifact_dir / "yasmina_percentage_decision_examples.json",
    }

# %% [notebook cell 16]
governance_rows = pd.DataFrame(
    [
        {"metric": "dataset rows", "value": dataset_summary.get("rows")},
        {"metric": "dataset columns", "value": dataset_summary.get("columns")},
        {"metric": "train rows", "value": dataset_summary.get("train_rows")},
        {"metric": "validation rows", "value": dataset_summary.get("validation_rows")},
        {"metric": "test rows", "value": dataset_summary.get("test_rows")},
        {"metric": "retained feature columns", "value": len(feature_policy.get("model_feature_columns", []))},
        {"metric": "post-decision exclusions", "value": len(feature_policy.get("post_decision_columns", []))},
        {"metric": "raw operational exclusions", "value": len(feature_policy.get("raw_operational_columns", []))},
        {"metric": "text-like exclusions", "value": len(feature_policy.get("text_like_columns", []))},
    ]
)
display(governance_rows)

# %% [notebook cell 17]
@dataclass(frozen=True)
class FeaturePolicy:
    target_columns: list[str]
    post_decision_columns: list[str]
    identifier_columns: list[str]
    raw_operational_columns: list[str]
    timestamp_columns: list[str]
    text_like_columns: list[str]
    all_missing_columns: list[str]
    constant_columns: list[str]
    monitored_audit_columns: list[str]
    excluded_columns: list[str]
    model_feature_columns: list[str]
    numeric_feature_columns: list[str]
    categorical_feature_columns: list[str]


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    median_abs_error: float
    mean_signed_error: float
    mean_overprediction: float
    mean_underprediction: float
    overprediction_rate: float
    underprediction_rate: float
    p90_abs_error: float
    max_abs_error: float
    within_5_pct_points: float
    within_10_pct_points: float
    binary_nonzero_accuracy: float
    binary_nonzero_precision: float
    binary_nonzero_recall: float
    binary_nonzero_f1: float


def supports_parameter(callable_obj: Any, parameter_name: str) -> bool:
    try:
        return parameter_name in inspect.signature(callable_obj).parameters
    except Exception:
        return False


def infer_supported_percentage_step(target: pd.Series | np.ndarray) -> float:
    support = np.asarray(
        sorted(pd.Series(target).dropna().astype(float).unique()),
        dtype=float,
    )
    if len(support) < 2:
        return 5.0

    diffs = np.diff(support)
    positive_diffs = diffs[diffs > 1e-9]
    if len(positive_diffs) == 0:
        return 5.0
    return float(max(1.0, round(float(np.quantile(positive_diffs, 0.50)), 2)))


def derive_error_thresholds(target: pd.Series | np.ndarray) -> dict[str, Any]:
    support_step = infer_supported_percentage_step(target)
    auto_award_tolerance_pct = float(np.clip(support_step * AUTO_AWARD_TOLERANCE_STEPS, 5.0, 25.0))
    severe_award_error_pct = float(
        np.clip(
            support_step * SEVERE_AWARD_ERROR_STEPS,
            auto_award_tolerance_pct + support_step,
            40.0,
        )
    )
    zero_upper_bound_cap = float(np.clip(support_step * ZERO_UPPER_BOUND_STEPS, 5.0, 15.0))

    return {
        "support_step_pct": support_step,
        "auto_award_tolerance_pct": auto_award_tolerance_pct,
        "severe_award_error_pct": severe_award_error_pct,
        "zero_upper_bound_cap": zero_upper_bound_cap,
        "rationale": (
            "Historical award percentages move on an approximately "
            f"{support_step:.1f}-point grid, so a {auto_award_tolerance_pct:.1f}-point miss "
            "equals about two historical award steps and a "
            f"{severe_award_error_pct:.1f}-point miss equals about four steps."
        ),
    }


def clean_feature_name(raw_name: str) -> str:
    cleaned = raw_name.replace("num__", "").replace("cat__", "")
    cleaned = cleaned.replace("preprocessor__", "")
    cleaned = cleaned.replace("imputer__", "")
    cleaned = cleaned.replace("encoder__", "")
    return cleaned


def get_transformed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [clean_feature_name(name) for name in preprocessor.get_feature_names_out()]


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_serializable(inner) for key, inner in value.items()}
    if isinstance(value, set):
        return [to_serializable(item) for item in sorted(value, key=str)]
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def snap_to_supported_percentages(
    predictions: np.ndarray | pd.Series,
    supported_percentages: list[float],
) -> np.ndarray:
    clipped_predictions = np.clip(np.asarray(predictions, dtype=float), 0.0, 100.0)
    support = np.asarray(sorted(supported_percentages), dtype=float)
    if support.size == 0:
        return clipped_predictions
    distances = np.abs(clipped_predictions[:, None] - support[None, :])
    return support[np.argmin(distances, axis=1)]


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()

    if RAW_SUBMISSION_DATE_COLUMN in enriched.columns:
        submission_dates = pd.to_datetime(
            enriched[RAW_SUBMISSION_DATE_COLUMN],
            errors="coerce",
        )
        enriched["derived_submission_year"] = submission_dates.dt.year
        enriched["derived_submission_month"] = submission_dates.dt.month
        enriched["derived_submission_quarter"] = submission_dates.dt.quarter

    if APPLICATION_TERM_COLUMN in enriched.columns:
        application_term = enriched[APPLICATION_TERM_COLUMN].astype("Int64").astype(str)
        enriched["derived_application_year"] = pd.to_numeric(
            application_term.str[:4],
            errors="coerce",
        )
        enriched["derived_application_term_suffix"] = pd.to_numeric(
            application_term.str[-2:],
            errors="coerce",
        )

    income_groups = {
        "derived_parent_gross_income_total": (
            "parsed_father_gross_income",
            "parsed_mother_gross_income",
        ),
        "derived_parent_net_income_total": (
            "parsed_father_net_income",
            "parsed_mother_net_income",
        ),
        "derived_household_income_total": (
            "parsed_father_gross_income",
            "parsed_mother_gross_income",
            "parsed_applicant_annual_income",
            "parsed_spouse_annual_income",
        ),
    }
    for output_column, input_columns in income_groups.items():
        available = [column for column in input_columns if column in enriched.columns]
        if not available:
            continue
        enriched[output_column] = enriched[available].fillna(0.0).sum(axis=1)

    if {
        "parsed_siblings_at_institution_count",
        "parsed_dependents_count",
    }.intersection(enriched.columns):
        siblings = enriched.get("parsed_siblings_at_institution_count", 0)
        dependents = enriched.get("parsed_dependents_count", 0)
        enriched["derived_family_dependency_load"] = (
            pd.Series(siblings, index=enriched.index).fillna(0.0)
            + pd.Series(dependents, index=enriched.index).fillna(0.0)
        )

    return enriched


def load_training_dataframe(data_path: Path = DATA_PATH) -> pd.DataFrame:
    dataframe = pd.read_csv(data_path, low_memory=False)
    if TARGET_COLUMN not in dataframe.columns:
        raise KeyError(
            f"Expected target column '{TARGET_COLUMN}' in {data_path}, "
            f"but the dataset did not contain it."
        )
    return add_derived_features(dataframe)


def resolve_feature_policy(dataframe: pd.DataFrame) -> FeaturePolicy:
    target_columns = [column for column in TARGET_ADJACENT_COLUMNS if column in dataframe.columns]

    post_decision_columns = sorted(
        column
        for column in dataframe.columns
        if column != TARGET_COLUMN
        and (
            column in EXPLICIT_POST_DECISION_COLUMNS
            or any(keyword in column.lower() for keyword in POST_DECISION_KEYWORDS)
        )
    )
    identifier_columns = sorted(column for column in IDENTIFIER_COLUMNS if column in dataframe.columns)
    raw_operational_columns = sorted(
        column
        for column in dataframe.columns
        if column.startswith(RAW_OPERATIONAL_PREFIX) and column not in identifier_columns
    )
    timestamp_columns = sorted(column for column in TIMESTAMP_COLUMNS if column in dataframe.columns)
    text_like_columns = sorted(
        column
        for column in dataframe.columns
        if any(token in column.lower() for token in TEXT_LIKE_TOKENS_TO_DROP)
    )
    all_missing_columns = sorted(
        column for column in dataframe.columns if dataframe[column].dropna().empty
    )
    constant_columns = sorted(
        column
        for column in dataframe.columns
        if column not in all_missing_columns and int(dataframe[column].nunique(dropna=True)) <= 1
    )
    monitored_audit_columns = sorted(
        column for column in FAIRNESS_AUDIT_COLUMNS if column in dataframe.columns
    )

    excluded_columns = sorted(
        set(
            target_columns
            + post_decision_columns
            + identifier_columns
            + raw_operational_columns
            + timestamp_columns
            + text_like_columns
            + all_missing_columns
            + constant_columns
        )
    )

    model_feature_columns = [
        column for column in dataframe.columns if column not in excluded_columns and column != TARGET_COLUMN
    ]
    numeric_feature_columns = [
        column for column in model_feature_columns if pd.api.types.is_numeric_dtype(dataframe[column])
    ]
    categorical_feature_columns = [
        column for column in model_feature_columns if column not in numeric_feature_columns
    ]

    return FeaturePolicy(
        target_columns=sorted(target_columns),
        post_decision_columns=post_decision_columns,
        identifier_columns=identifier_columns,
        raw_operational_columns=raw_operational_columns,
        timestamp_columns=timestamp_columns,
        text_like_columns=text_like_columns,
        all_missing_columns=all_missing_columns,
        constant_columns=constant_columns,
        monitored_audit_columns=monitored_audit_columns,
        excluded_columns=excluded_columns,
        model_feature_columns=model_feature_columns,
        numeric_feature_columns=numeric_feature_columns,
        categorical_feature_columns=categorical_feature_columns,
    )


def prepare_feature_frame(dataframe: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    enriched = add_derived_features(dataframe)
    aligned = pd.DataFrame(index=enriched.index)
    for column in feature_columns:
        aligned[column] = enriched[column] if column in enriched.columns else np.nan
    return aligned


def build_sparse_preprocessor(policy: FeaturePolicy) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                policy.numeric_feature_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="[missing]"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=CATEGORY_MIN_FREQUENCY,
                            ),
                        ),
                    ]
                ),
                policy.categorical_feature_columns,
            ),
        ]
    )

# %% [notebook cell 19]
if not validation_comparison_df.empty:
    display(
        validation_comparison_df[
            [
                'model_name',
                'model_family',
                'mae',
                'rmse',
                'r2',
                'binary_nonzero_f1',
            ]
        ].round(4)
    )

display(
    Markdown(
        '### Why this model won\n'
        + str(problem_framing.get('selection_rationale', 'No saved selection rationale was found.'))
    )
)

# %% [notebook cell 20]
class SupportedValueRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimator: Any, snap_to_supported: bool = True):
        self.estimator = estimator
        self.snap_to_supported = snap_to_supported

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SupportedValueRegressor":
        self.supported_percentages_ = sorted(float(value) for value in pd.Series(y).dropna().unique())
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = np.clip(np.asarray(self.estimator_.predict(X), dtype=float), 0.0, 100.0)
        if self.snap_to_supported:
            predictions = snap_to_supported_percentages(predictions, self.supported_percentages_)
        return predictions


class RuleTreeBaseline(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        numeric_columns: tuple[str, ...] = RULE_BASELINE_COLUMNS,
        max_depth: int = 3,
        min_samples_leaf: int = 50,
        snap_to_supported: bool = True,
    ):
        self.numeric_columns = numeric_columns
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.snap_to_supported = snap_to_supported

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RuleTreeBaseline":
        self.available_columns_ = [column for column in self.numeric_columns if column in X.columns]
        if not self.available_columns_:
            raise ValueError("RuleTreeBaseline requires at least one numeric summary feature.")

        self.supported_percentages_ = sorted(float(value) for value in pd.Series(y).dropna().unique())
        self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
        transformed = self.imputer_.fit_transform(X[self.available_columns_])
        self.model_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=RANDOM_STATE,
        )
        self.model_.fit(transformed, y)
        self.rule_text_ = export_text(self.model_, feature_names=self.available_columns_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        transformed = self.imputer_.transform(X[self.available_columns_])
        predictions = np.clip(np.asarray(self.model_.predict(transformed), dtype=float), 0.0, 100.0)
        if self.snap_to_supported:
            predictions = snap_to_supported_percentages(predictions, self.supported_percentages_)
        return predictions

    def get_rule_text(self) -> str:
        return getattr(self, "rule_text_", "")


class TieredAidModel(BaseEstimator, RegressorMixin):
    def __init__(self, classifier_pipeline: Pipeline, snap_to_supported: bool = True):
        self.classifier_pipeline = classifier_pipeline
        self.snap_to_supported = snap_to_supported

    @staticmethod
    def _to_tier(target: pd.Series) -> np.ndarray:
        values = np.asarray(target, dtype=float)
        tiers = np.zeros(len(values), dtype=int)
        tiers[(values > 0.0) & (values <= 15.0)] = 1
        tiers[(values > 15.0) & (values <= 25.0)] = 2
        tiers[(values > 25.0) & (values <= 40.0)] = 3
        tiers[values > 40.0] = 4
        return tiers

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TieredAidModel":
        target_series = pd.Series(y, index=X.index)
        self.supported_percentages_ = sorted(float(value) for value in target_series.dropna().unique())
        self.classifier_pipeline_ = clone(self.classifier_pipeline)
        tiers = self._to_tier(target_series)
        self.classifier_pipeline_.fit(X, tiers)

        self.tier_medians_ = {}
        for tier in range(5):
            tier_values = target_series.loc[tiers == tier]
            if tier_values.empty:
                self.tier_medians_[tier] = float(target_series.median())
            else:
                self.tier_medians_[tier] = float(tier_values.median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predicted_tiers = np.asarray(self.classifier_pipeline_.predict(X), dtype=int)
        predictions = np.asarray(
            [self.tier_medians_.get(int(tier), 0.0) for tier in predicted_tiers],
            dtype=float,
        )
        predictions = np.clip(predictions, 0.0, 100.0)
        if self.snap_to_supported:
            predictions = snap_to_supported_percentages(predictions, self.supported_percentages_)
        return predictions


class TwoStageAidModel(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        classifier_pipeline: Pipeline,
        positive_regressor_pipeline: Pipeline,
        blend_power: float = 1.0,
        snap_to_supported: bool = True,
        calibrate_classifier: bool = True,
        calibration_method: str = "sigmoid",
        calibration_cv: int = 3,
        probability_mode: str = "calibrated",
    ):
        self.classifier_pipeline = classifier_pipeline
        self.positive_regressor_pipeline = positive_regressor_pipeline
        self.blend_power = blend_power
        self.snap_to_supported = snap_to_supported
        self.calibrate_classifier = calibrate_classifier
        self.calibration_method = calibration_method
        self.calibration_cv = calibration_cv
        self.probability_mode = probability_mode

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TwoStageAidModel":
        target_series = pd.Series(y, index=X.index).astype(float)
        positive_mask = target_series > 0.0
        if int(positive_mask.sum()) == 0:
            raise ValueError("TwoStageAidModel requires at least one positive target row.")

        self.supported_percentages_ = sorted(float(value) for value in target_series.dropna().unique())
        self.classifier_pipeline_ = clone(self.classifier_pipeline)
        self.positive_regressor_pipeline_ = clone(self.positive_regressor_pipeline)

        self.classifier_pipeline_.fit(X, positive_mask.astype(int))
        self.classifier_probability_model_ = self.classifier_pipeline_
        self.has_calibrated_probability_model_ = False
        self.selected_probability_mode_ = "uncalibrated"
        if self.calibrate_classifier:
            try:
                calibration_params: dict[str, Any] = {
                    "cv": int(self.calibration_cv),
                    "method": self.calibration_method,
                }
                base_classifier = clone(self.classifier_pipeline)
                if supports_parameter(CalibratedClassifierCV, "estimator"):
                    calibration_params["estimator"] = base_classifier
                else:
                    calibration_params["base_estimator"] = base_classifier
                calibrated_model = CalibratedClassifierCV(**calibration_params)
                calibrated_model.fit(X, positive_mask.astype(int))
                self.classifier_probability_model_ = calibrated_model
                self.has_calibrated_probability_model_ = True
                if self.probability_mode == "calibrated":
                    self.selected_probability_mode_ = "calibrated"
            except Exception as exc:
                warnings.warn(
                    f"Classifier calibration failed; falling back to uncalibrated probabilities. Reason: {exc}",
                    stacklevel=2,
                )

        self.positive_regressor_pipeline_.fit(X.loc[positive_mask], target_series.loc[positive_mask])
        return self

    def set_probability_mode(self, mode: str) -> "TwoStageAidModel":
        if mode == "calibrated" and not getattr(self, "has_calibrated_probability_model_", False):
            self.selected_probability_mode_ = "uncalibrated"
            return self
        self.selected_probability_mode_ = mode
        return self

    def _predict_nonzero_proba_with_mode(self, X: pd.DataFrame, *, mode: str) -> np.ndarray:
        model = (
            self.classifier_probability_model_
            if mode == "calibrated" and getattr(self, "has_calibrated_probability_model_", False)
            else self.classifier_pipeline_
        )
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)

    def predict_nonzero_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_nonzero_proba_with_mode(X, mode=self.selected_probability_mode_)

    def predict_nonzero_proba_uncalibrated(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_nonzero_proba_with_mode(X, mode="uncalibrated")

    def predict_nonzero_proba_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict_nonzero_proba_with_mode(X, mode="calibrated")

    def predict_positive_amount(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(
            np.asarray(self.positive_regressor_pipeline_.predict(X), dtype=float),
            0.0,
            100.0,
        )

    def predict_continuous(self, X: pd.DataFrame) -> np.ndarray:
        nonzero_probability = self.predict_nonzero_proba(X)
        positive_amount = self.predict_positive_amount(X)
        return np.clip((nonzero_probability ** self.blend_power) * positive_amount, 0.0, 100.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        predictions = self.predict_continuous(X)
        if self.snap_to_supported:
            predictions = snap_to_supported_percentages(predictions, self.supported_percentages_)
        return predictions

    def _estimate_stage2_std(self, X: pd.DataFrame) -> np.ndarray:
        preprocessor = self.positive_regressor_pipeline_.named_steps["preprocessor"]
        estimator = self.positive_regressor_pipeline_.named_steps["model"]
        transformed = preprocessor.transform(X)
        if hasattr(estimator, "estimators_"):
            tree_predictions = np.column_stack(
                [np.asarray(tree.predict(transformed), dtype=float) for tree in estimator.estimators_]
            )
            return np.std(tree_predictions, axis=1)
        return np.zeros(len(X), dtype=float)

    def predict_details(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        nonzero_probability = self.predict_nonzero_proba(X)
        positive_amount = self.predict_positive_amount(X)
        continuous_prediction = self.predict_continuous(X)
        predicted_need_pct = (
            snap_to_supported_percentages(continuous_prediction, self.supported_percentages_)
            if self.snap_to_supported
            else continuous_prediction
        )

        stage2_std = self._estimate_stage2_std(X)
        classifier_variance = np.clip(nonzero_probability * (1.0 - nonzero_probability), 0.0, None)
        classifier_amount_std = positive_amount * np.sqrt(classifier_variance)
        total_std = np.sqrt((nonzero_probability * stage2_std) ** 2 + classifier_amount_std**2)
        z_score = 1.645
        lower = np.clip(continuous_prediction - z_score * total_std, 0.0, 100.0)
        upper = np.clip(continuous_prediction + z_score * total_std, 0.0, 100.0)
        interval_width = upper - lower

        probability_confidence = 2.0 * np.abs(nonzero_probability - 0.5)
        interval_confidence = np.clip(1.0 - (interval_width / 40.0), 0.0, 1.0)
        confidence_score = np.clip(0.55 * probability_confidence + 0.45 * interval_confidence, 0.0, 1.0)

        return {
            "predicted_need_pct": np.asarray(predicted_need_pct, dtype=float),
            "predicted_continuous_need_pct": np.asarray(continuous_prediction, dtype=float),
            "predicted_nonzero_probability": np.asarray(nonzero_probability, dtype=float),
            "predicted_positive_amount": np.asarray(positive_amount, dtype=float),
            "prediction_interval_lower": np.asarray(lower, dtype=float),
            "prediction_interval_upper": np.asarray(upper, dtype=float),
            "prediction_interval_width": np.asarray(interval_width, dtype=float),
            "confidence_score": np.asarray(confidence_score, dtype=float),
        }


for _custom_class in (
    SupportedValueRegressor,
    RuleTreeBaseline,
    TieredAidModel,
    TwoStageAidModel,
):
    _custom_class.__module__ = CANONICAL_MODULE_NAME

if __name__ == "__main__":
    sys.modules[CANONICAL_MODULE_NAME] = sys.modules[__name__]

# %% [notebook cell 22]
if not test_comparison_df.empty:
    display(
        test_comparison_df[
            [
                "model_name",
                "mae",
                "rmse",
                "r2",
                "mae_improvement_vs_mean_pct",
                "mae_improvement_vs_median_pct",
                "mae_improvement_vs_rule_pct",
            ]
        ].round(4)
    )

evaluation_rows = pd.DataFrame(
    [
        {"metric": "MAE", "value": round(float(final_test_metrics.get("mae", float("nan"))), 3)},
        {"metric": "RMSE", "value": round(float(final_test_metrics.get("rmse", float("nan"))), 3)},
        {"metric": "R2", "value": round(float(final_test_metrics.get("r2", float("nan"))), 3)},
        {"metric": "P90 absolute error", "value": round(float(final_test_metrics.get("p90_abs_error", float("nan"))), 3)},
        {"metric": "Max absolute error", "value": round(float(final_test_metrics.get("max_abs_error", float("nan"))), 3)},
        {"metric": "Within 10 points", "value": f"{float(final_test_metrics.get('within_10_pct_points', 0.0)):.1%}"},
    ]
)
display(evaluation_rows)

# %% [notebook cell 23]
def build_candidate_estimators(policy: FeaturePolicy) -> dict[str, Any]:
    sparse_preprocessor = build_sparse_preprocessor(policy)
    classifier_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(sparse_preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=700,
                    min_samples_leaf=3,
                    max_features=0.7,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=MODEL_N_JOBS,
                ),
            ),
        ]
    )
    regressor_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(sparse_preprocessor)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=700,
                    min_samples_leaf=2,
                    max_features=0.7,
                    random_state=RANDOM_STATE,
                    n_jobs=MODEL_N_JOBS,
                ),
            ),
        ]
    )

    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "median_baseline": DummyRegressor(strategy="median"),
        "simple_rule_tree": RuleTreeBaseline(),
        "tiered_rf": TieredAidModel(classifier_pipeline=clone(classifier_pipeline)),
        "direct_rf_snap": SupportedValueRegressor(
            estimator=clone(regressor_pipeline),
            snap_to_supported=True,
        ),
        "two_stage_probability_x_amount": TwoStageAidModel(
            classifier_pipeline=clone(classifier_pipeline),
            positive_regressor_pipeline=clone(regressor_pipeline),
            blend_power=1.0,
            snap_to_supported=True,
        ),
    }


MODEL_FAMILY_LABELS = {
    "mean_baseline": "baseline_constant",
    "median_baseline": "baseline_constant",
    "simple_rule_tree": "baseline_rule_tree",
    "tiered_rf": "tiered_classifier",
    "direct_rf_snap": "direct_regression",
    "two_stage_probability_x_amount": "two_stage_probability_x_amount",
}


def compute_regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> RegressionMetrics:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.clip(np.asarray(predictions, dtype=float), 0.0, 100.0)
    signed_error = predicted - actual
    absolute_error = np.abs(signed_error)

    over_mask = signed_error > 1e-9
    under_mask = signed_error < -1e-9
    actual_nonzero = (actual > 0.0).astype(int)
    predicted_nonzero = (predicted >= 0.5).astype(int)

    return RegressionMetrics(
        mae=float(mean_absolute_error(actual, predicted)),
        rmse=float(math.sqrt(mean_squared_error(actual, predicted))),
        r2=float(r2_score(actual, predicted)),
        median_abs_error=float(median_absolute_error(actual, predicted)),
        mean_signed_error=float(np.mean(signed_error)),
        mean_overprediction=float(np.mean(signed_error[over_mask]) if over_mask.any() else 0.0),
        mean_underprediction=float(np.mean(signed_error[under_mask]) if under_mask.any() else 0.0),
        overprediction_rate=float(np.mean(over_mask)),
        underprediction_rate=float(np.mean(under_mask)),
        p90_abs_error=float(np.quantile(absolute_error, 0.90)),
        max_abs_error=float(np.max(absolute_error)),
        within_5_pct_points=float(np.mean(absolute_error <= 5.0)),
        within_10_pct_points=float(np.mean(absolute_error <= 10.0)),
        binary_nonzero_accuracy=float(np.mean(actual_nonzero == predicted_nonzero)),
        binary_nonzero_precision=float(
            precision_score(actual_nonzero, predicted_nonzero, zero_division=0)
        ),
        binary_nonzero_recall=float(recall_score(actual_nonzero, predicted_nonzero, zero_division=0)),
        binary_nonzero_f1=float(f1_score(actual_nonzero, predicted_nonzero, zero_division=0)),
    )


def build_prediction_frame(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    if hasattr(estimator, "predict_details"):
        details = estimator.predict_details(X)
    else:
        predicted_need_pct = np.clip(np.asarray(estimator.predict(X), dtype=float), 0.0, 100.0)
        details = {
            "predicted_need_pct": predicted_need_pct,
            "predicted_continuous_need_pct": predicted_need_pct,
            "predicted_nonzero_probability": np.clip(predicted_need_pct / 100.0, 0.0, 1.0),
            "predicted_positive_amount": predicted_need_pct,
            "prediction_interval_lower": predicted_need_pct,
            "prediction_interval_upper": predicted_need_pct,
            "prediction_interval_width": np.zeros(len(predicted_need_pct), dtype=float),
            "confidence_score": np.full(len(predicted_need_pct), np.nan, dtype=float),
        }

    frame = pd.DataFrame(index=X.index)
    for column, values in details.items():
        frame[column] = np.asarray(values, dtype=float)
    frame["actual_need_pct"] = np.asarray(y, dtype=float)
    frame["signed_error"] = frame["predicted_need_pct"] - frame["actual_need_pct"]
    frame["absolute_error"] = frame["signed_error"].abs()
    frame["is_overprediction"] = frame["signed_error"] > 0
    frame["is_underprediction"] = frame["signed_error"] < 0
    frame["actual_nonzero"] = frame["actual_need_pct"] > 0
    frame["predicted_nonzero"] = frame["predicted_need_pct"] >= 0.5
    return frame


def fit_and_score_candidates(
    candidates: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
    fitted_estimators: dict[str, Any] = {}
    prediction_frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []

    for model_name, estimator in candidates.items():
        fitted = clone(estimator)
        fitted.fit(X_train, y_train)
        predictions = build_prediction_frame(fitted, X_eval, y_eval)
        metrics = compute_regression_metrics(y_eval, predictions["predicted_need_pct"].to_numpy())

        row = {
            "model_name": model_name,
            "model_family": MODEL_FAMILY_LABELS.get(model_name, "other"),
            **asdict(metrics),
        }
        fitted_estimators[model_name] = fitted
        prediction_frames[model_name] = predictions
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(
        by=["mae", "rmse", "r2"],
        ascending=[True, True, False],
        ignore_index=True,
    )
    return comparison, fitted_estimators, prediction_frames


def select_final_model(validation_comparison: pd.DataFrame) -> tuple[str, str]:
    best_row = validation_comparison.sort_values(
        by=["mae", "rmse", "r2"],
        ascending=[True, True, False],
        ignore_index=True,
    ).iloc[0]
    two_stage_name = "two_stage_probability_x_amount"

    if two_stage_name not in set(validation_comparison["model_name"]):
        return str(best_row["model_name"]), "Validation winner selected directly."

    two_stage_row = validation_comparison.loc[
        validation_comparison["model_name"] == two_stage_name
    ].iloc[0]

    if (
        float(two_stage_row["mae"]) <= float(best_row["mae"]) * 1.03
        and float(two_stage_row["rmse"]) <= float(best_row["rmse"]) * 1.02
    ):
        return (
            two_stage_name,
            "Two-stage model selected because the target is zero-inflated and its validation "
            "error stayed within a tight tolerance of the best alternative while enabling "
            "cleaner decision logic, uncertainty, and manual-review routing.",
        )

    return (
        str(best_row["model_name"]),
        "Best validation MAE/RMSE model selected because the two-stage alternative was materially worse.",
    )


def build_candidate_improvement_table(comparison: pd.DataFrame) -> pd.DataFrame:
    frame = comparison.copy()

    mean_mae = float(frame.loc[frame["model_name"] == "mean_baseline", "mae"].iloc[0])
    median_mae = float(frame.loc[frame["model_name"] == "median_baseline", "mae"].iloc[0])
    rule_mae = float(frame.loc[frame["model_name"] == "simple_rule_tree", "mae"].iloc[0])

    frame["mae_improvement_vs_mean_pct"] = 100.0 * (mean_mae - frame["mae"]) / max(mean_mae, 1e-9)
    frame["mae_improvement_vs_median_pct"] = 100.0 * (median_mae - frame["mae"]) / max(median_mae, 1e-9)
    frame["mae_improvement_vs_rule_pct"] = 100.0 * (rule_mae - frame["mae"]) / max(rule_mae, 1e-9)
    return frame


def build_segment_frame(
    dataframe: pd.DataFrame,
    reference_index: list[int] | np.ndarray,
) -> pd.DataFrame:
    segments = pd.DataFrame(index=dataframe.index)

    if "derived_household_income_total" in dataframe.columns:
        reference_income = dataframe.loc[reference_index, "derived_household_income_total"].replace(0.0, np.nan)
        reference_income = reference_income.dropna()
        if reference_income.empty:
            q1 = q2 = q3 = 0.0
        else:
            q1, q2, q3 = reference_income.quantile([0.25, 0.50, 0.75]).tolist()

        def bucket_income(value: Any) -> str:
            if pd.isna(value) or float(value) <= 0.0:
                return "missing_or_zero"
            if float(value) <= q1:
                return "low"
            if float(value) <= q2:
                return "mid"
            if float(value) <= q3:
                return "high"
            return "very_high"

        segments["segment_household_income_bucket"] = dataframe["derived_household_income_total"].apply(bucket_income)

    if "parsed_school" in dataframe.columns:
        top_schools = (
            dataframe.loc[reference_index, "parsed_school"]
            .astype("string")
            .fillna("[missing]")
            .value_counts()
            .head(10)
            .index
        )
        segments["segment_school_group"] = (
            dataframe["parsed_school"]
            .astype("string")
            .fillna("[missing]")
            .apply(lambda value: value if value in set(top_schools) else "[other_schools]")
        )

    return segments


def build_audit_frame(dataframe: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    columns: list[str] = []
    for column in list(IDENTIFIER_COLUMNS) + list(FAIRNESS_AUDIT_COLUMNS):
        if column in dataframe.columns and column not in columns:
            columns.append(column)
        if column in segments.columns and column not in columns:
            columns.append(column)

    audit = pd.DataFrame(index=dataframe.index)
    for column in columns:
        if column in dataframe.columns:
            audit[column] = dataframe[column]
        elif column in segments.columns:
            audit[column] = segments[column]
    return audit

# %% [notebook cell 25]
risk_rows = pd.DataFrame(
    [
        {"control": "Probability mode", "value": calibration_validation_summary.get("selected_mode"), "why": calibration_validation_summary.get("selection_reason", "")},
        {"control": "Fairness threshold rationale", "value": fairness_validation_summary.get("thresholds", {}).get("rationale", ""), "why": "Dynamic thresholds use validation dispersion plus floor constraints."},
        {"control": "Auto-zero max probability", "value": threshold_rationale.get("selected_policy", {}).get("auto_zero_probability_max"), "why": "Chosen by validation search against harmful automated errors."},
        {"control": "Auto-award min probability", "value": threshold_rationale.get("selected_policy", {}).get("auto_award_probability_min"), "why": "Chosen by validation search against harmful automated errors."},
        {"control": "Award error tolerance", "value": error_thresholds.get("auto_award_tolerance_pct"), "why": error_thresholds.get("rationale", "")},
    ]
)
display(risk_rows)

if not fairness_actions_df.empty:
    display(
        fairness_actions_df[
            [
                "group_column",
                "group_value",
                "sample_size",
                "mae_gap",
                "bias_gap",
                "recommended_action",
            ]
        ].head(12)
    )

# %% [notebook cell 26]
def resolve_fairness_columns(audit_frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in FAIRNESS_AUDIT_COLUMNS:
        if column not in audit_frame.columns:
            continue
        non_null = audit_frame[column].dropna()
        if non_null.empty or int(non_null.nunique()) <= 1:
            continue
        columns.append(column)
    return columns


def _safe_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(precision_score(y_true, y_pred, zero_division=0))


def _safe_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, zero_division=0))


def compute_group_fairness_metrics(
    audit_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    *,
    group_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not group_columns:
        return pd.DataFrame(), {}

    base = audit_frame.copy()
    for column in prediction_frame.columns:
        base[column] = prediction_frame[column]

    overall_metrics = {
        "mae": float(base["absolute_error"].mean()),
        "bias": float(base["signed_error"].mean()),
        "underprediction_rate": float(base["is_underprediction"].mean()),
        "overprediction_rate": float(base["is_overprediction"].mean()),
        "within_10_pct_points": float((base["absolute_error"] <= 10.0).mean()),
    }

    rows: list[dict[str, Any]] = []
    for column in group_columns:
        grouped = (
            base.assign(_group_value=base[column].astype("string").fillna("[missing]"))
            .groupby("_group_value", dropna=False)
        )

        for group_value, group in grouped:
            actual_nonzero = np.asarray(group["actual_nonzero"], dtype=int)
            predicted_nonzero = np.asarray(group["predicted_nonzero"], dtype=int)
            rows.append(
                {
                    "group_column": column,
                    "group_value": str(group_value),
                    "sample_size": int(len(group)),
                    "actual_mean": float(group["actual_need_pct"].mean()),
                    "predicted_mean": float(group["predicted_need_pct"].mean()),
                    "mae": float(group["absolute_error"].mean()),
                    "rmse": float(math.sqrt(np.mean(np.square(group["signed_error"])))),
                    "bias": float(group["signed_error"].mean()),
                    "underprediction_rate": float(group["is_underprediction"].mean()),
                    "overprediction_rate": float(group["is_overprediction"].mean()),
                    "within_10_pct_points": float((group["absolute_error"] <= 10.0).mean()),
                    "binary_nonzero_precision": _safe_precision(actual_nonzero, predicted_nonzero),
                    "binary_nonzero_recall": _safe_recall(actual_nonzero, predicted_nonzero),
                }
            )

    return pd.DataFrame(rows), overall_metrics


def derive_fairness_thresholds(
    fairness_frame: pd.DataFrame,
    overall_metrics: dict[str, float],
    *,
    support_step_pct: float,
) -> dict[str, Any]:
    mae_floor = float(max(support_step_pct * FAIRNESS_MAE_GAP_MIN_STEPS, 1.0))
    bias_floor = float(max(support_step_pct * FAIRNESS_BIAS_GAP_MIN_STEPS, 2.0))
    rate_floor = float(FAIRNESS_RATE_GAP_MIN)

    if fairness_frame.empty:
        return {
            "mae_gap_threshold": mae_floor,
            "bias_gap_threshold": bias_floor,
            "rate_gap_threshold": rate_floor,
            "rationale": "No eligible fairness groups were available, so support-step-based minimum thresholds were used.",
            "eligible_group_count": 0,
        }

    eligible = fairness_frame.loc[
        fairness_frame["sample_size"].astype(int) >= FAIRNESS_MIN_GROUP_SIZE
    ].copy()
    if eligible.empty:
        return {
            "mae_gap_threshold": mae_floor,
            "bias_gap_threshold": bias_floor,
            "rate_gap_threshold": rate_floor,
            "rationale": "Fairness groups existed but none met the minimum sample size, so support-step-based minimum thresholds were used.",
            "eligible_group_count": 0,
        }

    eligible["mae_gap"] = np.clip(
        eligible["mae"].astype(float) - float(overall_metrics["mae"]),
        0.0,
        None,
    )
    eligible["abs_bias_gap"] = np.abs(
        eligible["bias"].astype(float) - float(overall_metrics["bias"])
    )
    eligible["max_rate_gap"] = np.maximum(
        np.abs(
            eligible["underprediction_rate"].astype(float)
            - float(overall_metrics["underprediction_rate"])
        ),
        np.abs(
            eligible["overprediction_rate"].astype(float)
            - float(overall_metrics["overprediction_rate"])
        ),
    )

    mae_gap_threshold = float(
        max(mae_floor, float(np.quantile(eligible["mae_gap"], 0.75)))
    )
    bias_gap_threshold = float(
        max(bias_floor, float(np.quantile(eligible["abs_bias_gap"], 0.75)))
    )
    rate_gap_threshold = float(
        max(rate_floor, float(np.quantile(eligible["max_rate_gap"], 0.75)))
    )

    return {
        "mae_gap_threshold": mae_gap_threshold,
        "bias_gap_threshold": bias_gap_threshold,
        "rate_gap_threshold": rate_gap_threshold,
        "eligible_group_count": int(len(eligible)),
        "mae_gap_floor": mae_floor,
        "bias_gap_floor": bias_floor,
        "rate_gap_floor": rate_floor,
        "rationale": (
            "Fairness thresholds use the larger of a support-step-based minimum and the "
            "75th percentile of validation subgroup dispersion so the policy reacts to real "
            "portfolio variability without becoming hypersensitive to noise."
        ),
    }


def derive_fairness_actions(
    fairness_frame: pd.DataFrame,
    overall_metrics: dict[str, float],
    fairness_thresholds: dict[str, Any],
) -> pd.DataFrame:
    if fairness_frame.empty:
        return fairness_frame

    actions: list[dict[str, Any]] = []
    for record in fairness_frame.to_dict(orient="records"):
        mae_gap = float(record["mae"] - overall_metrics["mae"])
        bias_gap = float(record["bias"] - overall_metrics["bias"])
        under_gap = float(record["underprediction_rate"] - overall_metrics["underprediction_rate"])
        over_gap = float(record["overprediction_rate"] - overall_metrics["overprediction_rate"])
        mae_threshold = float(fairness_thresholds["mae_gap_threshold"])
        bias_threshold = float(fairness_thresholds["bias_gap_threshold"])
        rate_threshold = float(fairness_thresholds["rate_gap_threshold"])
        moderate_threshold_multiplier = 0.60

        if int(record["sample_size"]) < FAIRNESS_MIN_GROUP_SIZE:
            action = "monitor_only_small_sample"
            reason = "Subgroup too small for intervention."
        elif (
            mae_gap >= mae_threshold
            and bias_gap <= -bias_threshold
        ):
            action = "force_manual_review"
            reason = (
                "Subgroup shows materially higher error plus systematic underprediction, "
                "so percentage recommendations should route to manual review."
            )
        elif (
            abs(bias_gap) >= bias_threshold
            or under_gap >= rate_threshold
            or over_gap >= rate_threshold
        ):
            action = "force_manual_review"
            reason = (
                "Subgroup error direction differs materially from the portfolio baseline, "
                "so automatic trust should be reduced."
            )
        elif (
            mae_gap >= mae_threshold
            or abs(bias_gap) >= bias_threshold * moderate_threshold_multiplier
            or under_gap >= rate_threshold * moderate_threshold_multiplier
            or over_gap >= rate_threshold * moderate_threshold_multiplier
        ):
            action = "tighten_automation_thresholds"
            reason = (
                "Subgroup risk is elevated but not severe enough for blanket manual review, "
                "so automation thresholds should be tightened for this subgroup."
            )
        else:
            action = "monitor_only"
            reason = "No fairness intervention threshold was crossed."

        actions.append(
            {
                **record,
                "mae_gap": float(mae_gap),
                "bias_gap": float(bias_gap),
                "underprediction_rate_gap": float(under_gap),
                "overprediction_rate_gap": float(over_gap),
                "mae_gap_threshold": mae_threshold,
                "bias_gap_threshold": bias_threshold,
                "rate_gap_threshold": rate_threshold,
                "recommended_action": action,
                "reason": reason,
            }
        )

    return pd.DataFrame(actions).sort_values(
        by=["recommended_action", "mae_gap", "sample_size"],
        ascending=[True, False, False],
        ignore_index=True,
    )


def build_fairness_guardrails(
    fairness_actions: pd.DataFrame,
    *,
    support_step_pct: float,
) -> dict[str, Any]:
    manual_review_lookup: dict[str, set[str]] = {}
    tightened_lookup: dict[str, set[str]] = {}

    if not fairness_actions.empty:
        for record in fairness_actions.to_dict(orient="records"):
            group_column = str(record["group_column"])
            group_value = str(record["group_value"])
            action = str(record["recommended_action"])
            if action == "force_manual_review":
                manual_review_lookup.setdefault(group_column, set()).add(group_value)
            elif action == "tighten_automation_thresholds":
                tightened_lookup.setdefault(group_column, set()).add(group_value)

    tightened_policy = {
        "auto_zero_probability_multiplier": float(FAIRNESS_TIGHTENED_AUTO_ZERO_MULTIPLIER),
        "auto_award_probability_delta": float(FAIRNESS_TIGHTENED_AUTO_AWARD_DELTA),
        "max_interval_width_delta": float(
            max(1.0, support_step_pct * FAIRNESS_TIGHTENED_INTERVAL_STEP_MULTIPLIER)
        ),
        "min_confidence_delta": float(FAIRNESS_TIGHTENED_CONFIDENCE_DELTA),
        "rationale": (
            "Moderately risky subgroups keep automation available, but only under stricter "
            "probability, confidence, and uncertainty thresholds."
        ),
    }

    return {
        "manual_review_lookup": manual_review_lookup,
        "tightened_lookup": tightened_lookup,
        "tightened_policy": tightened_policy,
        "manual_review_group_count": int(
            fairness_actions["recommended_action"].eq("force_manual_review").sum()
        )
        if not fairness_actions.empty
        else 0,
        "tightened_group_count": int(
            fairness_actions["recommended_action"].eq("tighten_automation_thresholds").sum()
        )
        if not fairness_actions.empty
        else 0,
    }


def summarize_fairness_assessment(
    fairness_frame: pd.DataFrame,
    fairness_actions: pd.DataFrame,
    overall_metrics: dict[str, float],
    fairness_thresholds: dict[str, Any],
    fairness_guardrails: dict[str, Any],
) -> dict[str, Any]:
    if fairness_frame.empty:
        return {
            "available": False,
            "message": "No fairness audit columns were available.",
            "overall_metrics": overall_metrics,
            "thresholds": fairness_thresholds,
            "group_metrics": [],
            "actions": [],
        }

    return {
        "available": True,
        "overall_metrics": overall_metrics,
        "thresholds": fairness_thresholds,
        "group_metrics": fairness_frame.to_dict(orient="records"),
        "actions": fairness_actions.to_dict(orient="records"),
        "flagged_group_count": int(
            fairness_actions["recommended_action"].eq("force_manual_review").sum()
        ),
        "tightened_group_count": int(
            fairness_actions["recommended_action"].eq("tighten_automation_thresholds").sum()
        ),
        "guardrails": {
            "manual_review_group_count": fairness_guardrails.get("manual_review_group_count", 0),
            "tightened_group_count": fairness_guardrails.get("tightened_group_count", 0),
            "tightened_policy": fairness_guardrails.get("tightened_policy", {}),
        },
    }


def compute_segment_metrics(
    prediction_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
    *,
    group_columns: list[str],
    min_group_size: int = SEGMENT_MIN_GROUP_SIZE,
) -> pd.DataFrame:
    combined = audit_frame.copy()
    for column in prediction_frame.columns:
        combined[column] = prediction_frame[column]

    rows: list[dict[str, Any]] = []
    for column in group_columns:
        if column not in combined.columns:
            continue
        grouped = (
            combined.assign(_group_value=combined[column].astype("string").fillna("[missing]"))
            .groupby("_group_value", dropna=False)
        )
        for group_value, group in grouped:
            if len(group) < min_group_size:
                continue
            rows.append(
                {
                    "segment_column": column,
                    "segment_value": str(group_value),
                    "sample_size": int(len(group)),
                    "actual_mean": float(group["actual_need_pct"].mean()),
                    "predicted_mean": float(group["predicted_need_pct"].mean()),
                    "mae": float(group["absolute_error"].mean()),
                    "rmse": float(math.sqrt(np.mean(np.square(group["signed_error"])))),
                    "bias": float(group["signed_error"].mean()),
                    "p90_abs_error": float(np.quantile(group["absolute_error"], 0.90)),
                    "within_10_pct_points": float((group["absolute_error"] <= 10.0).mean()),
                    "overprediction_rate": float(group["is_overprediction"].mean()),
                    "underprediction_rate": float(group["is_underprediction"].mean()),
                }
            )

    return pd.DataFrame(rows).sort_values(
        by=["mae", "sample_size"],
        ascending=[False, False],
        ignore_index=True,
    )


def build_worst_errors_table(
    prediction_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
    *,
    top_n: int = 25,
) -> pd.DataFrame:
    combined = audit_frame.copy()
    for column in prediction_frame.columns:
        combined[column] = prediction_frame[column]
    visible_columns = [
        column
        for column in (
            "raw_source_sheet_name",
            "raw_source_row_number",
            "parsed_level",
            "inferred_application_track",
            "segment_school_group",
            "segment_household_income_bucket",
            "actual_need_pct",
            "predicted_need_pct",
            "predicted_nonzero_probability",
            "prediction_interval_lower",
            "prediction_interval_upper",
            "confidence_score",
            "signed_error",
            "absolute_error",
        )
        if column in combined.columns
    ]
    return combined.sort_values("absolute_error", ascending=False).head(top_n)[visible_columns]


def apply_tightened_decision_policy(
    decision_policy: dict[str, float],
    tightened_policy: dict[str, Any],
) -> dict[str, float]:
    return {
        "auto_zero_probability_max": float(
            max(
                0.01,
                decision_policy["auto_zero_probability_max"]
                * float(tightened_policy["auto_zero_probability_multiplier"]),
            )
        ),
        "auto_award_probability_min": float(
            min(
                0.99,
                decision_policy["auto_award_probability_min"]
                + float(tightened_policy["auto_award_probability_delta"]),
            )
        ),
        "max_interval_width": float(
            max(
                5.0,
                decision_policy["max_interval_width"]
                - float(tightened_policy["max_interval_width_delta"]),
            )
        ),
        "min_confidence": float(
            min(
                0.99,
                decision_policy["min_confidence"]
                + float(tightened_policy["min_confidence_delta"]),
            )
        ),
    }

# %% [notebook cell 27]
def tune_decision_policy(
    validation_prediction_frame: pd.DataFrame,
    validation_audit_frame: pd.DataFrame,
    fairness_guardrails: dict[str, Any],
    error_thresholds: dict[str, Any],
) -> tuple[dict[str, float], pd.DataFrame]:
    search_rows: list[dict[str, Any]] = []
    best_policy: dict[str, float] | None = None
    best_score: tuple[float, float, float, float] | None = None

    for auto_zero_prob in DECISION_AUTO_ZERO_PROB_OPTIONS:
        for auto_award_prob in DECISION_AUTO_AWARD_PROB_OPTIONS:
            if auto_zero_prob >= auto_award_prob:
                continue
            for max_interval_width in DECISION_MAX_INTERVAL_WIDTH_OPTIONS:
                for min_confidence in DECISION_MIN_CONFIDENCE_OPTIONS:
                    policy = {
                        "auto_zero_probability_max": float(auto_zero_prob),
                        "auto_award_probability_min": float(auto_award_prob),
                        "max_interval_width": float(max_interval_width),
                        "min_confidence": float(min_confidence),
                    }
                    decision_frame, summary = build_decision_recommendations(
                        audit_frame=validation_audit_frame,
                        prediction_frame=validation_prediction_frame,
                        fairness_guardrails=fairness_guardrails,
                        decision_policy=policy,
                        error_thresholds=error_thresholds,
                    )
                    harmful_auto_zero = int(summary["harmful_auto_zero_errors"])
                    harmful_auto_award = int(summary["harmful_auto_award_errors"])
                    review_cases = int(summary["review_cases"])
                    severe_auto_award = int(summary["severe_auto_award_errors"])
                    score = (
                        harmful_auto_zero * 8
                        + harmful_auto_award * 5
                        + severe_auto_award * 2
                        + review_cases
                    )

                    candidate_row = {
                        **policy,
                        **summary,
                        "selection_score": float(score),
                    }
                    search_rows.append(candidate_row)

                    tie_break = (
                        float(score),
                        float(harmful_auto_zero),
                        float(harmful_auto_award),
                        float(review_cases),
                    )
                    if best_score is None or tie_break < best_score:
                        best_score = tie_break
                        best_policy = policy

    if best_policy is None:
        best_policy = {
            "auto_zero_probability_max": 0.10,
            "auto_award_probability_min": 0.85,
            "max_interval_width": 12.0,
            "min_confidence": 0.60,
        }

    search_frame = pd.DataFrame(search_rows).sort_values(
        by=[
            "selection_score",
            "harmful_auto_zero_errors",
            "harmful_auto_award_errors",
            "review_cases",
        ],
        ascending=[True, True, True, True],
        ignore_index=True,
    )
    return best_policy, search_frame


def build_decision_recommendations(
    *,
    audit_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    fairness_guardrails: dict[str, Any],
    decision_policy: dict[str, float],
    error_thresholds: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manual_review_lookup = fairness_guardrails.get("manual_review_lookup", {})
    tightened_lookup = fairness_guardrails.get("tightened_lookup", {})
    tightened_policy = fairness_guardrails.get("tightened_policy", {})

    for index, prediction_row in prediction_frame.iterrows():
        fairness_manual_review_triggers: list[str] = []
        fairness_tightened_triggers: list[str] = []
        for column, risky_values in manual_review_lookup.items():
            if column not in audit_frame.columns:
                continue
            value = audit_frame.at[index, column] if index in audit_frame.index else None
            if pd.isna(value):
                value = "[missing]"
            value = str(value)
            if value in risky_values:
                fairness_manual_review_triggers.append(f"{column}={value}")
        for column, risky_values in tightened_lookup.items():
            if column not in audit_frame.columns:
                continue
            value = audit_frame.at[index, column] if index in audit_frame.index else None
            if pd.isna(value):
                value = "[missing]"
            value = str(value)
            if value in risky_values:
                fairness_tightened_triggers.append(f"{column}={value}")

        confidence = float(prediction_row["confidence_score"])
        probability = float(prediction_row["predicted_nonzero_probability"])
        interval_width = float(prediction_row["prediction_interval_width"])
        interval_upper = float(prediction_row["prediction_interval_upper"])
        predicted_pct = float(prediction_row["predicted_need_pct"])
        actual_pct = float(prediction_row["actual_need_pct"])
        abs_error = float(prediction_row["absolute_error"])
        effective_policy = dict(decision_policy)
        fairness_guardrail_action = "none"
        fairness_guardrail_reason = ""
        tightened_thresholds_applied = False

        if fairness_manual_review_triggers:
            action = "review"
            reason = "Manual review required because fairness safeguards were triggered."
            fairness_guardrail_action = "force_manual_review"
            fairness_guardrail_reason = ", ".join(fairness_manual_review_triggers)
        else:
            if fairness_tightened_triggers:
                effective_policy = apply_tightened_decision_policy(decision_policy, tightened_policy)
                fairness_guardrail_action = "tighten_automation_thresholds"
                fairness_guardrail_reason = ", ".join(fairness_tightened_triggers)
                tightened_thresholds_applied = True

            if confidence < float(effective_policy["min_confidence"]):
                action = "review"
                reason = "Manual review required because model confidence is too low."
            elif interval_width > float(effective_policy["max_interval_width"]):
                action = "review"
                reason = "Manual review required because the prediction interval is too wide."
            elif (
                probability <= float(effective_policy["auto_zero_probability_max"])
                and interval_upper <= float(error_thresholds["zero_upper_bound_cap"])
            ):
                action = "auto_zero"
                reason = "Model is highly confident that the outcome is zero aid."
            elif (
                probability >= float(effective_policy["auto_award_probability_min"])
                and predicted_pct > 0.0
            ):
                action = "auto_award"
                reason = "Model is confident enough to recommend the predicted percentage directly."
            else:
                action = "review"
                reason = "Manual review required because the case falls between automation thresholds."

            if tightened_thresholds_applied and action != "review":
                reason = (
                    reason
                    + " Fairness mitigation tightened the automation thresholds for this subgroup."
                )

        harmful_auto_zero = action == "auto_zero" and actual_pct > 0.0
        harmful_auto_award = (
            action == "auto_award"
            and abs_error > float(error_thresholds["auto_award_tolerance_pct"])
        )
        severe_auto_award = (
            action == "auto_award"
            and abs_error > float(error_thresholds["severe_award_error_pct"])
        )

        if action == "review" and fairness_guardrail_action == "tighten_automation_thresholds":
            reason = (
                "Manual review required because tightened fairness thresholds still were not met."
            )

        record = audit_frame.loc[index].to_dict() if index in audit_frame.index else {}
        record.update(
            {
                "actual_need_pct": actual_pct,
                "predicted_need_pct": predicted_pct,
                "predicted_continuous_need_pct": float(prediction_row["predicted_continuous_need_pct"]),
                "predicted_nonzero_probability": probability,
                "prediction_interval_lower": float(prediction_row["prediction_interval_lower"]),
                "prediction_interval_upper": interval_upper,
                "prediction_interval_width": interval_width,
                "confidence_score": confidence,
                "absolute_error": abs_error,
                "signed_error": float(prediction_row["signed_error"]),
                "recommended_action": action,
                "recommended_percentage": 0.0 if action == "auto_zero" else predicted_pct,
                "action_reason": reason,
                "fairness_risk_flag": bool(
                    fairness_manual_review_triggers or fairness_tightened_triggers
                ),
                "fairness_guardrail_action": fairness_guardrail_action,
                "fairness_guardrail_reason": fairness_guardrail_reason,
                "fairness_tightened_thresholds_applied": bool(tightened_thresholds_applied),
                "effective_auto_zero_probability_max": float(
                    effective_policy["auto_zero_probability_max"]
                ),
                "effective_auto_award_probability_min": float(
                    effective_policy["auto_award_probability_min"]
                ),
                "effective_max_interval_width": float(effective_policy["max_interval_width"]),
                "effective_min_confidence": float(effective_policy["min_confidence"]),
                "harmful_auto_zero_error": bool(harmful_auto_zero),
                "harmful_auto_award_error": bool(harmful_auto_award),
                "severe_auto_award_error": bool(severe_auto_award),
            }
        )
        rows.append(record)

    decision_frame = pd.DataFrame(rows)
    summary = {
        "auto_zero_cases": int(decision_frame["recommended_action"].eq("auto_zero").sum()),
        "auto_award_cases": int(decision_frame["recommended_action"].eq("auto_award").sum()),
        "review_cases": int(decision_frame["recommended_action"].eq("review").sum()),
        "automation_rate": float(
            1.0 - decision_frame["recommended_action"].eq("review").mean()
        ),
        "harmful_auto_zero_errors": int(decision_frame["harmful_auto_zero_error"].sum()),
        "harmful_auto_award_errors": int(decision_frame["harmful_auto_award_error"].sum()),
        "severe_auto_award_errors": int(decision_frame["severe_auto_award_error"].sum()),
        "fairness_review_cases": int(
            decision_frame["fairness_guardrail_action"].eq("force_manual_review").sum()
        ),
        "fairness_tightened_cases": int(
            decision_frame["fairness_tightened_thresholds_applied"].sum()
        ),
    }
    return decision_frame, summary


def build_probability_calibration_bins(
    actual_nonzero: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = CALIBRATION_BIN_COUNT,
) -> pd.DataFrame:
    actual = np.asarray(actual_nonzero, dtype=int)
    predicted = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.digitize(predicted, edges[1:-1], right=False)

    rows: list[dict[str, Any]] = []
    for bucket_index in range(bins):
        mask = bucket == bucket_index
        if not np.any(mask):
            continue
        lower = float(edges[bucket_index])
        upper = float(edges[bucket_index + 1])
        mean_predicted_probability = float(np.mean(predicted[mask]))
        empirical_nonzero_rate = float(np.mean(actual[mask]))
        calibration_gap = float(mean_predicted_probability - empirical_nonzero_rate)
        rows.append(
            {
                "bin_index": int(bucket_index),
                "bin_lower": lower,
                "bin_upper": upper,
                "bin_label": f"[{lower:.1f}, {upper:.1f}]",
                "sample_size": int(np.sum(mask)),
                "mean_predicted_probability": mean_predicted_probability,
                "empirical_nonzero_rate": empirical_nonzero_rate,
                "calibration_gap": calibration_gap,
                "abs_calibration_gap": float(abs(calibration_gap)),
            }
        )

    return pd.DataFrame(rows)


def summarize_reference_probability(
    calibration_bins: pd.DataFrame,
    *,
    reference_probability: float = 0.80,
) -> dict[str, Any]:
    if calibration_bins.empty:
        return {
            "available": False,
            "reference_probability": reference_probability,
        }

    nearest_index = (
        calibration_bins["mean_predicted_probability"].sub(reference_probability).abs().idxmin()
    )
    record = calibration_bins.loc[nearest_index].to_dict()
    return {
        "available": True,
        "reference_probability": float(reference_probability),
        "bin_label": str(record["bin_label"]),
        "sample_size": int(record["sample_size"]),
        "mean_predicted_probability": float(record["mean_predicted_probability"]),
        "empirical_nonzero_rate": float(record["empirical_nonzero_rate"]),
        "calibration_gap": float(record["calibration_gap"]),
    }


def compute_probability_calibration_summary(
    actual_nonzero: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    actual = np.asarray(actual_nonzero, dtype=int)
    predicted = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    calibration_bins = build_probability_calibration_bins(actual, predicted)
    total_rows = int(len(actual))
    if calibration_bins.empty or total_rows == 0:
        return {
            "available": False,
            "sample_size": total_rows,
        }, calibration_bins

    weighted_abs_gap = (
        calibration_bins["sample_size"].astype(float)
        * calibration_bins["abs_calibration_gap"].astype(float)
    ).sum() / float(total_rows)

    summary = {
        "available": True,
        "sample_size": total_rows,
        "brier_score": float(brier_score_loss(actual, predicted)),
        "roc_auc": float(roc_auc_score(actual, predicted)) if len(np.unique(actual)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(actual, predicted)),
        "log_loss": float(log_loss(actual, predicted, labels=[0, 1])),
        "expected_calibration_error": float(weighted_abs_gap),
        "max_calibration_gap": float(calibration_bins["abs_calibration_gap"].max()),
        "reference_probability_check": summarize_reference_probability(
            calibration_bins,
            reference_probability=0.80,
        ),
    }
    return summary, calibration_bins


def compare_stage1_probability_calibration(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if not isinstance(estimator, TwoStageAidModel):
        return {
            "available": False,
            "message": "Probability calibration is only available for the two-stage model.",
        }, pd.DataFrame()

    actual_nonzero = (pd.Series(y, index=X.index).astype(float) > 0.0).astype(int)
    uncalibrated_summary, uncalibrated_bins = compute_probability_calibration_summary(
        actual_nonzero,
        estimator.predict_nonzero_proba_uncalibrated(X),
    )
    uncalibrated_bins = uncalibrated_bins.assign(probability_mode="uncalibrated")

    combined_bins = uncalibrated_bins.copy()
    summary: dict[str, Any] = {
        "available": True,
        "selected_mode": getattr(estimator, "selected_probability_mode_", "uncalibrated"),
        "selection_reason": (
            "Calibrated probabilities are used for automation because auto-zero and auto-award "
            "thresholds depend on absolute risk, not just ranking quality."
        ),
        "uncalibrated": uncalibrated_summary,
        "calibrated": None,
        "brier_improvement": None,
        "ece_improvement": None,
    }

    if getattr(estimator, "has_calibrated_probability_model_", False):
        calibrated_summary, calibrated_bins = compute_probability_calibration_summary(
            actual_nonzero,
            estimator.predict_nonzero_proba_calibrated(X),
        )
        calibrated_bins = calibrated_bins.assign(probability_mode="calibrated")
        combined_bins = pd.concat([combined_bins, calibrated_bins], ignore_index=True)
        summary["calibrated"] = calibrated_summary
        summary["brier_improvement"] = float(
            uncalibrated_summary["brier_score"] - calibrated_summary["brier_score"]
        )
        summary["ece_improvement"] = float(
            uncalibrated_summary["expected_calibration_error"]
            - calibrated_summary["expected_calibration_error"]
        )
    else:
        summary["selection_reason"] = (
            "Calibration wrapper was unavailable, so uncalibrated probabilities were retained."
        )

    selected_mode = summary["selected_mode"]
    if selected_mode == "calibrated" and summary["calibrated"] is not None:
        summary["selected_metrics"] = summary["calibrated"]
    else:
        summary["selected_metrics"] = summary["uncalibrated"]
    return summary, combined_bins


def choose_probability_mode(calibration_summary: dict[str, Any]) -> tuple[str, str]:
    calibrated = calibration_summary.get("calibrated")
    if not calibrated:
        return (
            "uncalibrated",
            "Calibration wrapper was unavailable, so the model kept the uncalibrated probabilities.",
        )

    uncalibrated = calibration_summary["uncalibrated"]
    brier_improvement = float(
        uncalibrated["brier_score"] - calibrated["brier_score"]
    )
    ece_improvement = float(
        uncalibrated["expected_calibration_error"]
        - calibrated["expected_calibration_error"]
    )
    roc_auc_drop = float(
        (uncalibrated.get("roc_auc") or 0.0) - (calibrated.get("roc_auc") or 0.0)
    )
    average_precision_drop = float(
        uncalibrated["average_precision"] - calibrated["average_precision"]
    )

    if (
        brier_improvement > 0.0
        and ece_improvement >= -0.005
        and roc_auc_drop <= 0.01
        and average_precision_drop <= 0.01
    ):
        return (
            "calibrated",
            "Sigmoid calibration improved probability quality on validation without materially hurting ranking quality.",
        )

    return (
        "uncalibrated",
        "Validation calibration checks showed that the raw probabilities were already stronger, so the final model keeps the uncalibrated stage-1 probabilities.",
    )


def apply_probability_mode_to_summary(
    calibration_summary: dict[str, Any],
    *,
    selected_mode: str,
    selection_reason: str,
) -> dict[str, Any]:
    updated = dict(calibration_summary)
    updated["selected_mode"] = selected_mode
    updated["selection_reason"] = selection_reason
    if selected_mode == "calibrated" and updated.get("calibrated") is not None:
        updated["selected_metrics"] = updated["calibrated"]
    else:
        updated["selected_metrics"] = updated["uncalibrated"]
    return updated


def replace_model_metrics_in_comparison(
    comparison: pd.DataFrame,
    *,
    model_name: str,
    metrics: RegressionMetrics,
) -> pd.DataFrame:
    updated = comparison.copy()
    mask = updated["model_name"] == model_name
    if not bool(mask.any()):
        return updated
    metric_values = asdict(metrics)
    for column_name, value in metric_values.items():
        if column_name in updated.columns:
            updated.loc[mask, column_name] = value
    return updated.sort_values(
        by=["mae", "rmse", "r2"],
        ascending=[True, True, False],
        ignore_index=True,
    )

# %% [notebook cell 29]
if not stage1_shap_df.empty:
    display(stage1_shap_df.head(10))
if not stage2_shap_df.empty:
    display(stage2_shap_df.head(10))

if not segment_metrics_df.empty:
    display(segment_metrics_df.head(10))

# %% [notebook cell 30]
def bootstrap_regression_intervals(
    prediction_frame: pd.DataFrame,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_STATE)
    target = prediction_frame["actual_need_pct"].to_numpy(dtype=float)
    predictions = prediction_frame["predicted_need_pct"].to_numpy(dtype=float)

    metric_values: dict[str, list[float]] = {
        "mae": [],
        "rmse": [],
        "r2": [],
        "mean_signed_error": [],
        "within_10_pct_points": [],
        "binary_nonzero_f1": [],
    }

    for _ in range(iterations):
        indices = rng.integers(0, len(target), len(target))
        sampled_target = target[indices]
        sampled_predictions = predictions[indices]
        sampled_metrics = compute_regression_metrics(sampled_target, sampled_predictions)
        metric_values["mae"].append(sampled_metrics.mae)
        metric_values["rmse"].append(sampled_metrics.rmse)
        metric_values["r2"].append(sampled_metrics.r2)
        metric_values["mean_signed_error"].append(sampled_metrics.mean_signed_error)
        metric_values["within_10_pct_points"].append(sampled_metrics.within_10_pct_points)
        metric_values["binary_nonzero_f1"].append(sampled_metrics.binary_nonzero_f1)

    summary: dict[str, Any] = {
        "iterations_requested": int(iterations),
        "iterations_used": int(iterations),
    }
    for metric_name, values in metric_values.items():
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_95_lower": float(np.quantile(values, 0.025)),
            "ci_95_upper": float(np.quantile(values, 0.975)),
        }
    summary["note"] = (
        "These intervals are bootstrap approximations around the held-out test metrics. "
        "Per-case prediction ranges come from the two-stage ensemble uncertainty approximation."
    )
    return summary


def build_governance_table(policy: FeaturePolicy) -> pd.DataFrame:
    rows = [
        {
            "column_type": "target",
            "used_in_model": "no",
            "examples": ", ".join(policy.target_columns[:5]),
            "why": "Outcome labels define the regression target and cannot be predictors.",
        },
        {
            "column_type": "post_decision",
            "used_in_model": "no",
            "examples": ", ".join(policy.post_decision_columns[:5]),
            "why": "These fields are only known after committee processing and would leak the answer.",
        },
        {
            "column_type": "identifier",
            "used_in_model": "no",
            "examples": ", ".join(policy.identifier_columns[:5]),
            "why": "Identifiers encourage memorization rather than generalizable need estimation.",
        },
        {
            "column_type": "raw_operational",
            "used_in_model": "no",
            "examples": ", ".join(policy.raw_operational_columns[:5]),
            "why": "Raw operational exports are noisy, redundant, and often not deployment-safe.",
        },
        {
            "column_type": "timestamp",
            "used_in_model": "no",
            "examples": ", ".join(policy.timestamp_columns[:5]),
            "why": "Submission timestamps can encode workflow timing instead of financial need.",
        },
        {
            "column_type": "free_text_like",
            "used_in_model": "no",
            "examples": ", ".join(policy.text_like_columns[:5]),
            "why": "Free-text-like columns were excluded to keep the model efficient and reproducible.",
        },
        {
            "column_type": "fairness_monitored",
            "used_in_model": "monitored",
            "examples": ", ".join(policy.monitored_audit_columns[:5]),
            "why": "These columns are tracked for subgroup performance and bias checks.",
        },
        {
            "column_type": "model_input",
            "used_in_model": "yes",
            "examples": "pre-decision financial, household, merit, and asset features",
            "why": "These are available before committee outcome and encode need-relevant context.",
        },
    ]
    return pd.DataFrame(rows)


def _densify(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=float)
    return np.asarray(matrix, dtype=float)


def _normalize_shap_values(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 3 and array.shape[-1] >= 2:
        return array[:, :, 1]
    if array.ndim == 3:
        return array.mean(axis=-1)
    return array


def build_pipeline_global_fallback(pipeline: Pipeline, *, top_n: int = TOP_FEATURES_TO_SHOW) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    feature_names = get_transformed_feature_names(preprocessor)
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(
            [{"feature": "fallback_unavailable", "importance": 0.0, "source": "fallback"}]
        )
    ranked = np.argsort(np.asarray(importances, dtype=float))[::-1][:top_n]
    return pd.DataFrame(
        {
            "feature": [feature_names[index] for index in ranked],
            "importance": [float(importances[index]) for index in ranked],
            "source": ["feature_importance_fallback"] * len(ranked),
        }
    )


def compute_pipeline_shap_outputs(
    pipeline: Pipeline,
    background_features: pd.DataFrame,
    sample_features: pd.DataFrame,
    example_features: pd.DataFrame,
    *,
    stage_label: str,
) -> tuple[pd.DataFrame, dict[Any, str]]:
    try:
        import shap  # type: ignore
    except Exception as exc:
        OPTIONAL_DEPENDENCY_STATUS[stage_label] = {
            "available": False,
            "generated": False,
            "message": f"SHAP unavailable: {exc}",
        }
        fallback = build_pipeline_global_fallback(pipeline)
        return fallback, {}

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        estimator = pipeline.named_steps["model"]
        feature_names = get_transformed_feature_names(preprocessor)

        background_sample = background_features.sample(
            n=min(SHAP_BACKGROUND_SIZE, len(background_features)),
            random_state=RANDOM_STATE,
        )
        sample_pool = sample_features.sample(
            n=min(SHAP_SAMPLE_SIZE, len(sample_features)),
            random_state=RANDOM_STATE,
        )

        background_matrix = _densify(preprocessor.transform(background_sample))
        sample_matrix = _densify(preprocessor.transform(sample_pool))
        explainer = shap.Explainer(estimator, background_matrix, feature_names=feature_names)
        sample_explanation = explainer(sample_matrix)
        sample_values = _normalize_shap_values(sample_explanation.values)
        mean_abs = np.mean(np.abs(sample_values), axis=0)
        ranking = np.argsort(mean_abs)[::-1][:TOP_FEATURES_TO_SHOW]
        global_frame = pd.DataFrame(
            {
                "feature": [feature_names[index] for index in ranking],
                "mean_abs_shap": [float(mean_abs[index]) for index in ranking],
                "source": ["shap"] * len(ranking),
            }
        )

        local_lookup: dict[Any, str] = {}
        if not example_features.empty:
            example_matrix = _densify(preprocessor.transform(example_features))
            example_explanation = explainer(example_matrix)
            example_values = _normalize_shap_values(example_explanation.values)

            for row_position, row_index in enumerate(example_features.index):
                values = np.asarray(example_values[row_position], dtype=float)
                ranked = np.argsort(np.abs(values))[::-1]
                top_indices = [index for index in ranked if abs(values[index]) > 1e-9][
                    :TOP_LOCAL_SHAP_FEATURES
                ]
                if not top_indices:
                    local_lookup[row_index] = "No dominant SHAP driver identified."
                    continue
                parts = [
                    f"{feature_names[index]} ({values[index]:+.3f})" for index in top_indices
                ]
                local_lookup[row_index] = "; ".join(parts)

        OPTIONAL_DEPENDENCY_STATUS[stage_label] = {
            "available": True,
            "generated": True,
            "message": "SHAP explanations generated successfully.",
        }
        return global_frame, local_lookup
    except Exception as exc:
        OPTIONAL_DEPENDENCY_STATUS[stage_label] = {
            "available": True,
            "generated": False,
            "message": f"SHAP failed, falling back to feature importances: {exc}",
        }
        fallback = build_pipeline_global_fallback(pipeline)
        return fallback, {}


def build_local_explanations(
    model: Any,
    background_features: pd.DataFrame,
    sample_features: pd.DataFrame,
    example_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[Any, str]]:
    if not isinstance(model, TwoStageAidModel):
        return pd.DataFrame(), pd.DataFrame(), {}

    stage1_global, stage1_local = compute_pipeline_shap_outputs(
        model.classifier_pipeline_,
        background_features,
        sample_features,
        example_features,
        stage_label="stage1_nonzero_probability",
    )
    stage2_global, stage2_local = compute_pipeline_shap_outputs(
        model.positive_regressor_pipeline_,
        background_features,
        sample_features,
        example_features,
        stage_label="stage2_positive_amount",
    )

    combined_local: dict[Any, str] = {}
    for row_index in example_features.index:
        stage1_text = stage1_local.get(row_index, "stage1 explanation unavailable")
        stage2_text = stage2_local.get(row_index, "stage2 explanation unavailable")
        combined_local[row_index] = (
            f"Any-aid stage: {stage1_text} | Positive-amount stage: {stage2_text}"
        )
    return stage1_global, stage2_global, combined_local


def build_decision_examples(
    decision_frame: pd.DataFrame,
    local_explanations: dict[Any, str],
) -> list[dict[str, Any]]:
    if decision_frame.empty:
        return []

    examples: list[pd.Series] = []

    auto_award_examples = (
        decision_frame[decision_frame["recommended_action"] == "auto_award"]
        .sort_values("confidence_score", ascending=False)
        .head(2)
    )
    auto_zero_examples = (
        decision_frame[decision_frame["recommended_action"] == "auto_zero"]
        .sort_values("confidence_score", ascending=False)
        .head(2)
    )
    review_examples = (
        decision_frame[decision_frame["recommended_action"] == "review"]
        .sort_values(["fairness_risk_flag", "prediction_interval_width"], ascending=[False, False])
        .head(2)
    )
    worst_error_examples = decision_frame.sort_values("absolute_error", ascending=False).head(2)

    for frame in [auto_award_examples, auto_zero_examples, review_examples, worst_error_examples]:
        examples.extend(list(frame.itertuples(index=True)))

    seen: set[Any] = set()
    output: list[dict[str, Any]] = []
    for row in examples:
        if row.Index in seen:
            continue
        seen.add(row.Index)
        payload = {
            "row_index": int(row.Index) if isinstance(row.Index, (int, np.integer)) else row.Index,
            "recommended_action": row.recommended_action,
            "actual_need_pct": float(row.actual_need_pct),
            "predicted_need_pct": float(row.predicted_need_pct),
            "predicted_nonzero_probability": float(row.predicted_nonzero_probability),
            "prediction_interval_lower": float(row.prediction_interval_lower),
            "prediction_interval_upper": float(row.prediction_interval_upper),
            "confidence_score": float(row.confidence_score),
            "absolute_error": float(row.absolute_error),
            "fairness_risk_flag": bool(row.fairness_risk_flag),
            "action_reason": row.action_reason,
            "local_explanation": local_explanations.get(row.Index, "Explanation unavailable."),
        }
        if hasattr(row, "raw_source_sheet_name"):
            payload["raw_source_sheet_name"] = row.raw_source_sheet_name
        if hasattr(row, "raw_source_row_number"):
            payload["raw_source_row_number"] = to_serializable(row.raw_source_row_number)
        if hasattr(row, "parsed_level"):
            payload["parsed_level"] = row.parsed_level
        if hasattr(row, "inferred_application_track"):
            payload["inferred_application_track"] = row.inferred_application_track
        output.append(payload)
    return output

# %% [notebook cell 32]
if temporal_summary.get("available"):
    temporal_metrics_df = pd.DataFrame(
        [
            {"metric": "train cycles", "value": temporal_summary["split"].get("train_cycles", [])},
            {"metric": "validation cycles", "value": temporal_summary["split"].get("validation_cycles", [])},
            {"metric": "test cycles", "value": temporal_summary["split"].get("test_cycles", [])},
            {"metric": "temporal MAE", "value": round(float(temporal_summary["final_test_metrics"].get("mae", float("nan"))), 3)},
            {"metric": "temporal RMSE", "value": round(float(temporal_summary["final_test_metrics"].get("rmse", float("nan"))), 3)},
            {"metric": "temporal R2", "value": round(float(temporal_summary["final_test_metrics"].get("r2", float("nan"))), 3)},
            {"metric": "temporal harmful auto-zero", "value": int(temporal_summary["decision_summary"].get("harmful_auto_zero_errors", 0))},
            {"metric": "temporal harmful auto-award", "value": int(temporal_summary["decision_summary"].get("harmful_auto_award_errors", 0))},
        ]
    )
    display(temporal_metrics_df)

    cycle_summary_records = temporal_summary.get("split", {}).get("cycle_summary", [])
    if cycle_summary_records:
        display(pd.DataFrame(cycle_summary_records))
else:
    print(temporal_summary.get("message", "Temporal validation summary unavailable."))

# %% [notebook cell 33]
def build_future_deployment_validation_plan() -> list[dict[str, str]]:
    return [
        {
            "step": "Freeze the selected model, fairness guardrails, and decision thresholds before the next cycle opens.",
            "purpose": "Prevent silent policy drift during deployment.",
        },
        {
            "step": "Score the next full admissions cycle without retraining and archive the raw predictions.",
            "purpose": "Create an honest future-cycle evaluation set.",
        },
        {
            "step": "Compare later committee outcomes against the frozen predictions and decision routes.",
            "purpose": "Measure real-world error, calibration, and review burden.",
        },
    ]


def resolve_temporal_validation_split(
    dataframe: pd.DataFrame,
    *,
    minimum_cycle_rows: int = TEMPORAL_VALIDATION_MIN_CYCLE_ROWS,
) -> dict[str, Any]:
    if APPLICATION_TERM_COLUMN not in dataframe.columns:
        return {
            "temporal_split_feasible": False,
            "source_column": None,
            "cycle_unit": None,
            "train_cycles": [],
            "validation_cycles": [],
            "test_cycles": [],
            "train_index": [],
            "validation_index": [],
            "test_index": [],
            "cycle_summary": [],
            "note": "Application-term history is unavailable, so a defendable temporal holdout cannot be built.",
        }

    raw_terms = pd.to_numeric(dataframe[APPLICATION_TERM_COLUMN], errors="coerce")
    cycle_year = (raw_terms // 100).astype("Int64")
    cycle_frame = pd.DataFrame({"cycle_year": cycle_year}, index=dataframe.index).dropna()
    cycle_summary = (
        cycle_frame.groupby("cycle_year", dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("cycle_year")
        .reset_index(drop=True)
    )
    eligible_cycles = [
        int(value)
        for value in cycle_summary.loc[
            cycle_summary["rows"] >= int(minimum_cycle_rows),
            "cycle_year",
        ].tolist()
    ]

    if len(eligible_cycles) < 3:
        return {
            "temporal_split_feasible": False,
            "source_column": APPLICATION_TERM_COLUMN,
            "cycle_unit": "application_cycle_year",
            "train_cycles": eligible_cycles[:-1],
            "validation_cycles": [],
            "test_cycles": eligible_cycles[-1:],
            "train_index": [],
            "validation_index": [],
            "test_index": [],
            "cycle_summary": cycle_summary.to_dict(orient="records"),
            "note": (
                "A strict train/validation/test temporal split requires at least three cycle years "
                f"with {minimum_cycle_rows}+ rows each."
            ),
        }

    train_cycles = eligible_cycles[:-2]
    validation_cycles = [eligible_cycles[-2]]
    test_cycles = [eligible_cycles[-1]]
    train_index = cycle_year[cycle_year.isin(train_cycles)].index.tolist()
    validation_index = cycle_year[cycle_year.isin(validation_cycles)].index.tolist()
    test_index = cycle_year[cycle_year.isin(test_cycles)].index.tolist()

    return {
        "temporal_split_feasible": True,
        "source_column": APPLICATION_TERM_COLUMN,
        "cycle_unit": "application_cycle_year",
        "train_cycles": train_cycles,
        "validation_cycles": validation_cycles,
        "test_cycles": test_cycles,
        "train_index": train_index,
        "validation_index": validation_index,
        "test_index": test_index,
        "cycle_summary": cycle_summary.to_dict(orient="records"),
        "note": (
            "Earlier cycle years train the model, the next cycle tunes thresholds and guardrails, "
            "and the latest cycle acts as a future-style holdout."
        ),
    }


def assess_temporal_split_feasibility(dataframe: pd.DataFrame) -> dict[str, Any]:
    temporal_split = resolve_temporal_validation_split(dataframe)
    temporal_split["future_deployment_validation_plan"] = build_future_deployment_validation_plan()
    return temporal_split


def summarize_decision_threshold_rationale(
    decision_policy: dict[str, float],
    decision_policy_search: pd.DataFrame,
    error_thresholds: dict[str, Any],
    fairness_thresholds: dict[str, Any],
) -> dict[str, Any]:
    selected_mask = (
        np.isclose(
            decision_policy_search["auto_zero_probability_max"].astype(float),
            float(decision_policy["auto_zero_probability_max"]),
        )
        & np.isclose(
            decision_policy_search["auto_award_probability_min"].astype(float),
            float(decision_policy["auto_award_probability_min"]),
        )
        & np.isclose(
            decision_policy_search["max_interval_width"].astype(float),
            float(decision_policy["max_interval_width"]),
        )
        & np.isclose(
            decision_policy_search["min_confidence"].astype(float),
            float(decision_policy["min_confidence"]),
        )
    )
    selected_rank = None
    selected_validation_summary: dict[str, Any] = {}
    if not decision_policy_search.empty and bool(np.any(selected_mask)):
        selected_rank = int(np.flatnonzero(np.asarray(selected_mask))[0] + 1)
        selected_validation_summary = decision_policy_search.loc[selected_mask].iloc[0].to_dict()

    return {
        "selection_objective": (
            "Validation policy search minimizes harmful auto-zero errors first, then harmful "
            "auto-award errors, then severe auto-award errors, and only then review burden "
            "(8*harmful_auto_zero + 5*harmful_auto_award + 2*severe_auto_award + review_cases)."
        ),
        "selected_policy_rank": selected_rank,
        "selected_policy": decision_policy,
        "selected_validation_summary": selected_validation_summary,
        "error_thresholds": error_thresholds,
        "fairness_thresholds": fairness_thresholds,
        "rationale": {
            "probability_thresholds": (
                "Auto-zero and auto-award probability cutoffs come from validation grid search "
                "over calibrated probabilities rather than fixed business guesses."
            ),
            "error_limits": error_thresholds["rationale"],
            "fairness_gaps": fairness_thresholds["rationale"],
        },
    }


def build_business_summary(
    *,
    decision_policy: dict[str, float],
    decision_summary: dict[str, Any],
    error_thresholds: dict[str, Any],
    fairness_thresholds: dict[str, Any],
    fairness_guardrails: dict[str, Any],
    threshold_rationale: dict[str, Any],
    calibration_summary_validation: dict[str, Any],
    temporal_summary: dict[str, Any],
) -> str:
    selected_calibration = calibration_summary_validation.get("selected_metrics", {})
    selected_probability_label = (
        "calibrated P(any aid)"
        if calibration_summary_validation.get("selected_mode") == "calibrated"
        else "validated stage-1 P(any aid)"
    )
    reference_check = selected_calibration.get("reference_probability_check", {})
    reference_line = "Unavailable."
    if reference_check.get("available"):
        reference_line = (
            f"A bin near 0.80 predicted probability averaged {reference_check['mean_predicted_probability']:.2f} "
            f"predicted vs {reference_check['empirical_nonzero_rate']:.2f} observed nonzero aid."
        )

    temporal_line = "Temporal validation was not feasible from the current term history."
    if temporal_summary.get("available"):
        temporal_metrics = temporal_summary["final_test_metrics"]
        temporal_line = (
            f"Temporal holdout trained on {temporal_summary['split']['train_cycles']}, "
            f"tuned on {temporal_summary['split']['validation_cycles']}, and tested on "
            f"{temporal_summary['split']['test_cycles']} with MAE {temporal_metrics['mae']:.2f} "
            f"and R2 {temporal_metrics['r2']:.3f}."
        )

    lines = [
        "Executive Decision Logic",
        "=" * 24,
        "",
        "When we auto-decide:",
        f"- Auto-zero only when {selected_probability_label} <= {decision_policy['auto_zero_probability_max']:.2f}, "
        f"the upper prediction bound stays <= {error_thresholds['zero_upper_bound_cap']:.1f}, "
        f"confidence >= {decision_policy['min_confidence']:.2f}, and interval width <= {decision_policy['max_interval_width']:.1f}.",
        f"- Auto-award only when {selected_probability_label} >= {decision_policy['auto_award_probability_min']:.2f}, "
        f"predicted aid is positive, confidence >= {decision_policy['min_confidence']:.2f}, "
        f"and interval width <= {decision_policy['max_interval_width']:.1f}.",
        "",
        "When we review:",
        "- Any case in a fairness manual-review subgroup goes to committee review.",
        "- Any low-confidence, high-uncertainty, or middle-probability case goes to review.",
        f"- Moderately risky subgroups keep automation only under tighter thresholds "
        f"({fairness_guardrails['tightened_policy'].get('rationale', 'tightened rules applied')}).",
        "",
        "Errors that matter most:",
        "- Harmful auto-zero: the model says no aid but the student actually received nonzero aid.",
        f"- Harmful auto-award: an automated award misses by more than {error_thresholds['auto_award_tolerance_pct']:.1f} percentage points.",
        f"- Severe auto-award: an automated award misses by more than {error_thresholds['severe_award_error_pct']:.1f} points.",
        "",
        "Why these thresholds:",
        f"- Error limits are tied to the historical {error_thresholds['support_step_pct']:.1f}-point award grid.",
        f"- Fairness gaps trigger at MAE +{fairness_thresholds['mae_gap_threshold']:.2f}, "
        f"bias {fairness_thresholds['bias_gap_threshold']:.2f}, or rate-gap {fairness_thresholds['rate_gap_threshold']:.2f}, "
        "using validation dispersion rather than arbitrary fixed cutoffs.",
        f"- Probability thresholds were chosen by validation search; selected policy rank: {threshold_rationale.get('selected_policy_rank')}.",
        f"- Calibration check: {reference_line}",
        f"- {temporal_line}",
        "",
        "Current held-out result:",
        f"- Auto-zero {decision_summary['auto_zero_cases']}, auto-award {decision_summary['auto_award_cases']}, review {decision_summary['review_cases']}.",
        f"- Harmful auto-zero {decision_summary['harmful_auto_zero_errors']}, harmful auto-award {decision_summary['harmful_auto_award_errors']}, severe auto-award {decision_summary['severe_auto_award_errors']}.",
    ]
    return "\n".join(lines)


def run_temporal_validation_experiment(
    *,
    dataframe_with_segments: pd.DataFrame,
    policy: FeaturePolicy,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    audit_full: pd.DataFrame,
    selected_model_name: str,
) -> dict[str, Any]:
    temporal_split = assess_temporal_split_feasibility(dataframe_with_segments)
    if not temporal_split["temporal_split_feasible"]:
        return {
            "available": False,
            "split": temporal_split,
            "message": temporal_split["note"],
            "future_deployment_validation_plan": temporal_split["future_deployment_validation_plan"],
        }

    train_idx = temporal_split["train_index"]
    validation_idx = temporal_split["validation_index"]
    test_idx = temporal_split["test_index"]

    temporal_candidates = build_candidate_estimators(policy)
    comparison_names = list(
        dict.fromkeys(["mean_baseline", "median_baseline", "simple_rule_tree", selected_model_name])
    )
    temporal_candidates = {name: temporal_candidates[name] for name in comparison_names}

    validation_comparison, validation_fitted, validation_predictions = fit_and_score_candidates(
        temporal_candidates,
        X_full.loc[train_idx],
        y_full.loc[train_idx],
        X_full.loc[validation_idx],
        y_full.loc[validation_idx],
    )

    temporal_error_thresholds = derive_error_thresholds(
        y_full.loc[np.concatenate([train_idx, validation_idx])]
    )
    selected_validation_model = validation_fitted[selected_model_name]
    validation_prediction_frame = validation_predictions[selected_model_name]
    validation_calibration_summary, _ = compare_stage1_probability_calibration(
        selected_validation_model,
        X_full.loc[validation_idx],
        y_full.loc[validation_idx],
    )
    temporal_probability_mode, temporal_probability_reason = choose_probability_mode(
        validation_calibration_summary
    )
    if isinstance(selected_validation_model, TwoStageAidModel):
        selected_validation_model.set_probability_mode(temporal_probability_mode)
        validation_prediction_frame = build_prediction_frame(
            selected_validation_model,
            X_full.loc[validation_idx],
            y_full.loc[validation_idx],
        )
        validation_calibration_summary, _ = compare_stage1_probability_calibration(
            selected_validation_model,
            X_full.loc[validation_idx],
            y_full.loc[validation_idx],
        )
        validation_calibration_summary = apply_probability_mode_to_summary(
            validation_calibration_summary,
            selected_mode=temporal_probability_mode,
            selection_reason=temporal_probability_reason,
        )
        validation_metrics = compute_regression_metrics(
            y_full.loc[validation_idx],
            validation_prediction_frame["predicted_need_pct"],
        )
        validation_comparison = replace_model_metrics_in_comparison(
            validation_comparison,
            model_name=selected_model_name,
            metrics=validation_metrics,
        )
    audit_validation = audit_full.loc[validation_idx]
    fairness_columns = resolve_fairness_columns(audit_validation)
    validation_fairness_frame, validation_fairness_overall = compute_group_fairness_metrics(
        audit_validation,
        validation_prediction_frame,
        group_columns=fairness_columns,
    )
    temporal_fairness_thresholds = derive_fairness_thresholds(
        validation_fairness_frame,
        validation_fairness_overall,
        support_step_pct=float(temporal_error_thresholds["support_step_pct"]),
    )
    validation_fairness_actions = derive_fairness_actions(
        validation_fairness_frame,
        validation_fairness_overall,
        temporal_fairness_thresholds,
    )
    temporal_fairness_guardrails = build_fairness_guardrails(
        validation_fairness_actions,
        support_step_pct=float(temporal_error_thresholds["support_step_pct"]),
    )
    temporal_decision_policy, temporal_policy_search = tune_decision_policy(
        validation_prediction_frame,
        audit_validation,
        temporal_fairness_guardrails,
        temporal_error_thresholds,
    )

    test_comparison, test_fitted, test_predictions = fit_and_score_candidates(
        temporal_candidates,
        X_full.loc[np.concatenate([train_idx, validation_idx])],
        y_full.loc[np.concatenate([train_idx, validation_idx])],
        X_full.loc[test_idx],
        y_full.loc[test_idx],
    )
    selected_temporal_model = test_fitted[selected_model_name]
    test_prediction_frame = test_predictions[selected_model_name]
    if isinstance(selected_temporal_model, TwoStageAidModel):
        selected_temporal_model.set_probability_mode(temporal_probability_mode)
        test_prediction_frame = build_prediction_frame(
            selected_temporal_model,
            X_full.loc[test_idx],
            y_full.loc[test_idx],
        )
        test_metrics_for_comparison = compute_regression_metrics(
            y_full.loc[test_idx],
            test_prediction_frame["predicted_need_pct"],
        )
        test_comparison = replace_model_metrics_in_comparison(
            test_comparison,
            model_name=selected_model_name,
            metrics=test_metrics_for_comparison,
        )
    test_comparison = build_candidate_improvement_table(test_comparison)
    final_test_metrics = compute_regression_metrics(
        y_full.loc[test_idx],
        test_prediction_frame["predicted_need_pct"],
    )
    test_calibration_summary, _ = compare_stage1_probability_calibration(
        selected_temporal_model,
        X_full.loc[test_idx],
        y_full.loc[test_idx],
    )
    test_calibration_summary = apply_probability_mode_to_summary(
        test_calibration_summary,
        selected_mode=temporal_probability_mode,
        selection_reason=temporal_probability_reason,
    )
    temporal_decision_frame, temporal_decision_summary = build_decision_recommendations(
        audit_frame=audit_full.loc[test_idx],
        prediction_frame=test_prediction_frame,
        fairness_guardrails=temporal_fairness_guardrails,
        decision_policy=temporal_decision_policy,
        error_thresholds=temporal_error_thresholds,
    )

    return {
        "available": True,
        "split": temporal_split,
        "selected_model_name": selected_model_name,
        "validation_candidate_comparison": validation_comparison.to_dict(orient="records"),
        "test_candidate_comparison": test_comparison.to_dict(orient="records"),
        "final_test_metrics": asdict(final_test_metrics),
        "validation_calibration_summary": validation_calibration_summary,
        "test_calibration_summary": test_calibration_summary,
        "decision_policy": temporal_decision_policy,
        "decision_summary": temporal_decision_summary,
        "decision_policy_search_top_rows": temporal_policy_search.head(10).to_dict(orient="records"),
        "note": (
            "Temporal validation reuses the chosen model architecture and policy-tuning workflow "
            "to simulate deployment into a later admissions cycle."
        ),
        "temporal_test_decisions_preview": temporal_decision_frame[
            [
                "recommended_action",
                "predicted_need_pct",
                "predicted_nonzero_probability",
                "absolute_error",
            ]
        ]
        .head(10)
        .to_dict(orient="records"),
    }

# %% [notebook cell 35]
solution_rows = pd.DataFrame(
    [
        {"component": "Prediction model", "role": "Estimates likely aid percentage"},
        {"component": "Calibration check", "role": "Verifies whether probabilities are trustworthy enough for threshold logic"},
        {"component": "Fairness guardrails", "role": "Escalates risky subgroups into review or tighter automation"},
        {"component": "Uncertainty layer", "role": "Prevents narrow trust in wide-interval cases"},
        {"component": "Decision policy", "role": "Converts predictions into auto-zero, auto-award, or review"},
        {"component": "Temporal validation", "role": "Checks whether the workflow survives movement to later cycles"},
    ]
)
display(solution_rows)

# %% [notebook cell 36]
def build_report(
    *,
    data_path: Path,
    policy: FeaturePolicy,
    selected_model_name: str,
    selection_rationale: str,
    validation_comparison: pd.DataFrame,
    test_comparison: pd.DataFrame,
    final_test_metrics: RegressionMetrics,
    bootstrap_summary: dict[str, Any],
    error_thresholds: dict[str, Any],
    decision_policy: dict[str, float],
    threshold_rationale: dict[str, Any],
    decision_summary: dict[str, Any],
    fairness_summary_validation: dict[str, Any],
    fairness_summary_test: dict[str, Any],
    calibration_summary_validation: dict[str, Any],
    calibration_summary_test: dict[str, Any],
    business_summary_text: str,
    temporal_summary: dict[str, Any],
    stage1_shap_global: pd.DataFrame,
    stage2_shap_global: pd.DataFrame,
    rule_text: str,
) -> str:
    framing_line = (
        "- Final recommendation uses a binary any-aid stage plus a positive-percentage stage because the target is zero-inflated and the structure supports cleaner review logic and uncertainty handling."
        if selected_model_name == "two_stage_probability_x_amount"
        else "- Final recommendation keeps the best non-two-stage model because it materially outperformed the binary-plus-amount alternatives on validation."
    )
    validation_reference_check = calibration_summary_validation.get("selected_metrics", {}).get(
        "reference_probability_check",
        {},
    )
    test_reference_check = calibration_summary_test.get("selected_metrics", {}).get(
        "reference_probability_check",
        {},
    )
    validation_reference_line = "Unavailable."
    if validation_reference_check.get("available"):
        validation_reference_line = (
            f"{validation_reference_check['mean_predicted_probability']:.3f} predicted vs "
            f"{validation_reference_check['empirical_nonzero_rate']:.3f} observed."
        )
    test_reference_line = "Unavailable."
    if test_reference_check.get("available"):
        test_reference_line = (
            f"{test_reference_check['mean_predicted_probability']:.3f} predicted vs "
            f"{test_reference_check['empirical_nonzero_rate']:.3f} observed."
        )

    temporal_section = ["Temporal validation:", f"- {temporal_summary.get('message', temporal_summary.get('note', 'No temporal summary available.'))}"]
    if temporal_summary.get("available"):
        temporal_metrics = temporal_summary["final_test_metrics"]
        temporal_section = [
            "Temporal validation:",
            f"- Train cycles: {temporal_summary['split']['train_cycles']}",
            f"- Validation cycles: {temporal_summary['split']['validation_cycles']}",
            f"- Test cycles: {temporal_summary['split']['test_cycles']}",
            f"- Temporal MAE: {temporal_metrics['mae']:.3f}",
            f"- Temporal RMSE: {temporal_metrics['rmse']:.3f}",
            f"- Temporal R2: {temporal_metrics['r2']:.3f}",
            f"- Temporal harmful auto-zero: {temporal_summary['decision_summary']['harmful_auto_zero_errors']}",
            f"- Temporal harmful auto-award: {temporal_summary['decision_summary']['harmful_auto_award_errors']}",
        ]

    lines = [
        "Financial Aid Percentage Model",
        "=" * 40,
        "",
        f"Data path: {data_path}",
        f"Selected model: {selected_model_name}",
        f"Selection rationale: {selection_rationale}",
        "",
        business_summary_text,
        "",
        "Problem framing:",
        "- Compared constant baselines, a transparent rule baseline, tiered prediction, direct regression, and a two-stage model.",
        framing_line,
        "",
        "Leakage policy:",
        f"- Post-decision columns excluded: {len(policy.post_decision_columns)}",
        f"- Raw operational columns excluded: {len(policy.raw_operational_columns)}",
        f"- Text-like columns excluded: {len(policy.text_like_columns)}",
        f"- Final model features retained: {len(policy.model_feature_columns)}",
        "",
        "Validation candidate comparison (sorted by MAE):",
        validation_comparison[
            [
                "model_name",
                "model_family",
                "mae",
                "rmse",
                "r2",
                "binary_nonzero_f1",
            ]
        ]
        .round(4)
        .to_string(index=False),
        "",
        "Test candidate comparison (sorted by MAE):",
        test_comparison[
            [
                "model_name",
                "mae",
                "rmse",
                "r2",
                "mae_improvement_vs_mean_pct",
                "mae_improvement_vs_median_pct",
                "mae_improvement_vs_rule_pct",
            ]
        ]
        .round(4)
        .to_string(index=False),
        "",
        "Final test metrics:",
        f"- MAE: {final_test_metrics.mae:.3f}",
        f"- RMSE: {final_test_metrics.rmse:.3f}",
        f"- R2: {final_test_metrics.r2:.3f}",
        f"- Mean signed error: {final_test_metrics.mean_signed_error:.3f}",
        f"- Overprediction rate: {final_test_metrics.overprediction_rate:.3%}",
        f"- Underprediction rate: {final_test_metrics.underprediction_rate:.3%}",
        f"- P90 absolute error: {final_test_metrics.p90_abs_error:.3f}",
        f"- Max absolute error: {final_test_metrics.max_abs_error:.3f}",
        f"- Within 10 pct points: {final_test_metrics.within_10_pct_points:.3%}",
        "",
        "Metric uncertainty (95% bootstrap intervals):",
        f"- MAE: {bootstrap_summary['mae']['ci_95_lower']:.3f} to {bootstrap_summary['mae']['ci_95_upper']:.3f}",
        f"- RMSE: {bootstrap_summary['rmse']['ci_95_lower']:.3f} to {bootstrap_summary['rmse']['ci_95_upper']:.3f}",
        f"- R2: {bootstrap_summary['r2']['ci_95_lower']:.3f} to {bootstrap_summary['r2']['ci_95_upper']:.3f}",
        "",
        "Threshold rationale:",
        f"- Validation decision-policy rank: {threshold_rationale.get('selected_policy_rank')}",
        f"- Error policy: {error_thresholds['rationale']}",
        f"- Fairness policy: {fairness_summary_validation.get('thresholds', {}).get('rationale', 'Unavailable.')}",
        f"- Fairness thresholds: MAE gap {fairness_summary_validation.get('thresholds', {}).get('mae_gap_threshold', float('nan')):.3f}, "
        f"bias gap {fairness_summary_validation.get('thresholds', {}).get('bias_gap_threshold', float('nan')):.3f}, "
        f"rate gap {fairness_summary_validation.get('thresholds', {}).get('rate_gap_threshold', float('nan')):.3f}",
        "",
        "Decision policy:",
        f"- Auto-zero probability max: {decision_policy['auto_zero_probability_max']:.2f}",
        f"- Auto-award probability min: {decision_policy['auto_award_probability_min']:.2f}",
        f"- Max interval width: {decision_policy['max_interval_width']:.1f}",
        f"- Min confidence: {decision_policy['min_confidence']:.2f}",
        f"- Zero upper-bound cap: {error_thresholds['zero_upper_bound_cap']:.1f}",
        f"- Harmful auto-award threshold: {error_thresholds['auto_award_tolerance_pct']:.1f}",
        f"- Severe auto-award threshold: {error_thresholds['severe_award_error_pct']:.1f}",
        f"- Auto-zero cases on test: {decision_summary['auto_zero_cases']}",
        f"- Auto-award cases on test: {decision_summary['auto_award_cases']}",
        f"- Review cases on test: {decision_summary['review_cases']}",
        f"- Harmful auto-zero errors on test: {decision_summary['harmful_auto_zero_errors']}",
        f"- Harmful auto-award errors on test: {decision_summary['harmful_auto_award_errors']}",
        f"- Fairness tightened-threshold cases on test: {decision_summary.get('fairness_tightened_cases', 0)}",
        "",
        "Fairness monitoring:",
        f"- Validation flagged groups: {fairness_summary_validation.get('flagged_group_count', 0)}",
        f"- Validation tightened-threshold groups: {fairness_summary_validation.get('tightened_group_count', 0)}",
        f"- Test flagged groups: {fairness_summary_test.get('flagged_group_count', 0)}",
        f"- Test tightened-threshold groups: {fairness_summary_test.get('tightened_group_count', 0)}",
        "",
        "Probability calibration:",
        f"- Selected stage-1 mode: {calibration_summary_validation.get('selected_mode', 'n/a')}",
        f"- Validation Brier: {calibration_summary_validation.get('selected_metrics', {}).get('brier_score', float('nan')):.4f}",
        f"- Validation ECE: {calibration_summary_validation.get('selected_metrics', {}).get('expected_calibration_error', float('nan')):.4f}",
        f"- Validation 0.80 check: {validation_reference_line}",
        f"- Test Brier: {calibration_summary_test.get('selected_metrics', {}).get('brier_score', float('nan')):.4f}",
        f"- Test ECE: {calibration_summary_test.get('selected_metrics', {}).get('expected_calibration_error', float('nan')):.4f}",
        f"- Test 0.80 check: {test_reference_line}",
        "",
        *temporal_section,
        "",
        "Global explainability:",
        "Stage 1 (any aid) top SHAP drivers:",
        stage1_shap_global.head(10).round(4).to_string(index=False) if not stage1_shap_global.empty else "No stage-1 SHAP output.",
        "",
        "Stage 2 (positive amount) top SHAP drivers:",
        stage2_shap_global.head(10).round(4).to_string(index=False) if not stage2_shap_global.empty else "No stage-2 SHAP output.",
        "",
        "Simple rule baseline:",
        rule_text.strip() or "Rule baseline unavailable.",
    ]
    return "\n".join(lines)

# %% [notebook cell 37]
def train_percentage_model(
    *,
    data_path: Path = DATA_PATH,
    artifact_dir: Path = ARTIFACT_DIR,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_artifact_paths(artifact_dir)

    dataframe = load_training_dataframe(data_path)
    index_array = dataframe.index.to_numpy()
    nonzero_target = (dataframe[TARGET_COLUMN].astype(float) > 0.0).astype(int)
    train_idx, temp_idx = train_test_split(
        index_array,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=nonzero_target,
    )
    relative_test_size = TEST_SIZE / (VALIDATION_SIZE + TEST_SIZE)
    validation_idx, test_idx = train_test_split(
        temp_idx,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=nonzero_target.loc[temp_idx],
    )

    train_valid_idx = np.concatenate([train_idx, validation_idx])
    segments = build_segment_frame(dataframe, train_valid_idx)
    dataframe_with_segments = pd.concat([dataframe, segments], axis=1)
    policy = resolve_feature_policy(dataframe_with_segments)
    governance_table = build_governance_table(policy)

    X_full = dataframe_with_segments[policy.model_feature_columns].copy()
    y_full = dataframe_with_segments[TARGET_COLUMN].astype(float)
    audit_full = build_audit_frame(dataframe_with_segments, segments)

    X_train = X_full.loc[train_idx]
    y_train = y_full.loc[train_idx]
    X_validation = X_full.loc[validation_idx]
    y_validation = y_full.loc[validation_idx]
    X_test = X_full.loc[test_idx]
    y_test = y_full.loc[test_idx]

    audit_validation = audit_full.loc[validation_idx]
    audit_test = audit_full.loc[test_idx]

    validation_candidates = build_candidate_estimators(policy)
    validation_comparison, validation_fitted, validation_predictions = fit_and_score_candidates(
        validation_candidates,
        X_train,
        y_train,
        X_validation,
        y_validation,
    )
    selected_model_name, selection_rationale = select_final_model(validation_comparison)
    error_thresholds = derive_error_thresholds(y_full.loc[train_valid_idx])
    selected_validation_model = validation_fitted[selected_model_name]
    validation_prediction_frame = validation_predictions[selected_model_name]
    validation_calibration_summary, validation_calibration_bins = compare_stage1_probability_calibration(
        selected_validation_model,
        X_validation,
        y_validation,
    )
    selected_probability_mode, probability_mode_reason = choose_probability_mode(
        validation_calibration_summary
    )
    if isinstance(selected_validation_model, TwoStageAidModel):
        selected_validation_model.set_probability_mode(selected_probability_mode)
        validation_prediction_frame = build_prediction_frame(
            selected_validation_model,
            X_validation,
            y_validation,
        )
        validation_calibration_summary, validation_calibration_bins = compare_stage1_probability_calibration(
            selected_validation_model,
            X_validation,
            y_validation,
        )
        validation_calibration_summary = apply_probability_mode_to_summary(
            validation_calibration_summary,
            selected_mode=selected_probability_mode,
            selection_reason=probability_mode_reason,
        )
        validation_metrics = compute_regression_metrics(
            y_validation,
            validation_prediction_frame["predicted_need_pct"],
        )
        validation_comparison = replace_model_metrics_in_comparison(
            validation_comparison,
            model_name=selected_model_name,
            metrics=validation_metrics,
        )

    fairness_columns = resolve_fairness_columns(audit_validation)
    validation_fairness_frame, validation_fairness_overall = compute_group_fairness_metrics(
        audit_validation,
        validation_prediction_frame,
        group_columns=fairness_columns,
    )
    fairness_thresholds = derive_fairness_thresholds(
        validation_fairness_frame,
        validation_fairness_overall,
        support_step_pct=float(error_thresholds["support_step_pct"]),
    )
    validation_fairness_actions = derive_fairness_actions(
        validation_fairness_frame,
        validation_fairness_overall,
        fairness_thresholds,
    )
    fairness_guardrails = build_fairness_guardrails(
        validation_fairness_actions,
        support_step_pct=float(error_thresholds["support_step_pct"]),
    )
    fairness_summary_validation = summarize_fairness_assessment(
        validation_fairness_frame,
        validation_fairness_actions,
        validation_fairness_overall,
        fairness_thresholds,
        fairness_guardrails,
    )

    decision_policy, decision_policy_search = tune_decision_policy(
        validation_prediction_frame,
        audit_validation,
        fairness_guardrails,
        error_thresholds,
    )

    test_candidates = build_candidate_estimators(policy)
    test_comparison, test_fitted, test_predictions = fit_and_score_candidates(
        test_candidates,
        X_full.loc[train_valid_idx],
        y_full.loc[train_valid_idx],
        X_test,
        y_test,
    )

    selected_deployment_model = test_fitted[selected_model_name]
    test_prediction_frame = test_predictions[selected_model_name]
    if isinstance(selected_deployment_model, TwoStageAidModel):
        selected_deployment_model.set_probability_mode(selected_probability_mode)
        test_prediction_frame = build_prediction_frame(
            selected_deployment_model,
            X_test,
            y_test,
        )
        test_metrics_for_comparison = compute_regression_metrics(
            y_test,
            test_prediction_frame["predicted_need_pct"],
        )
        test_comparison = replace_model_metrics_in_comparison(
            test_comparison,
            model_name=selected_model_name,
            metrics=test_metrics_for_comparison,
        )
    test_comparison = build_candidate_improvement_table(test_comparison)
    test_calibration_summary, test_calibration_bins = compare_stage1_probability_calibration(
        selected_deployment_model,
        X_test,
        y_test,
    )
    test_calibration_summary = apply_probability_mode_to_summary(
        test_calibration_summary,
        selected_mode=selected_probability_mode,
        selection_reason=probability_mode_reason,
    )
    final_test_metrics = compute_regression_metrics(y_test, test_prediction_frame["predicted_need_pct"])

    segment_columns = [
        column
        for column in (
            "segment_household_income_bucket",
            "parsed_level",
            "inferred_application_track",
            "segment_school_group",
            "parsed_nationality",
        )
        if column in audit_test.columns
    ]
    segment_metrics = compute_segment_metrics(
        test_prediction_frame,
        audit_test,
        group_columns=segment_columns,
    )

    test_fairness_frame, test_fairness_overall = compute_group_fairness_metrics(
        audit_test,
        test_prediction_frame,
        group_columns=fairness_columns,
    )
    test_fairness_actions = derive_fairness_actions(
        test_fairness_frame,
        test_fairness_overall,
        fairness_thresholds,
    )
    fairness_summary_test = summarize_fairness_assessment(
        test_fairness_frame,
        test_fairness_actions,
        test_fairness_overall,
        fairness_thresholds,
        fairness_guardrails,
    )

    decision_recommendations_test, decision_summary = build_decision_recommendations(
        audit_frame=audit_test,
        prediction_frame=test_prediction_frame,
        fairness_guardrails=fairness_guardrails,
        decision_policy=decision_policy,
        error_thresholds=error_thresholds,
    )

    bootstrap_summary = bootstrap_regression_intervals(test_prediction_frame)
    worst_errors = build_worst_errors_table(test_prediction_frame, audit_test)
    threshold_rationale = summarize_decision_threshold_rationale(
        decision_policy,
        decision_policy_search,
        error_thresholds,
        fairness_thresholds,
    )
    temporal_summary = run_temporal_validation_experiment(
        dataframe_with_segments=dataframe_with_segments,
        policy=policy,
        X_full=X_full,
        y_full=y_full,
        audit_full=audit_full,
        selected_model_name=selected_model_name,
    )
    business_summary_text = build_business_summary(
        decision_policy=decision_policy,
        decision_summary=decision_summary,
        error_thresholds=error_thresholds,
        fairness_thresholds=fairness_thresholds,
        fairness_guardrails=fairness_guardrails,
        threshold_rationale=threshold_rationale,
        calibration_summary_validation=validation_calibration_summary,
        temporal_summary=temporal_summary,
    )

    background_features = X_full.loc[train_valid_idx]
    sample_features = X_test
    example_indices = decision_recommendations_test.sort_values(
        by=["absolute_error", "confidence_score"],
        ascending=[False, False],
    ).head(8).index
    example_features = X_test.loc[X_test.index.intersection(example_indices)]
    stage1_shap_global, stage2_shap_global, local_explanations = build_local_explanations(
        selected_deployment_model,
        background_features,
        sample_features,
        example_features,
    )
    decision_examples = build_decision_examples(decision_recommendations_test, local_explanations)

    rule_baseline = test_fitted.get("simple_rule_tree")
    rule_text = rule_baseline.get_rule_text() if isinstance(rule_baseline, RuleTreeBaseline) else ""

    validation_with_audit = pd.concat([audit_validation, validation_prediction_frame], axis=1)
    test_with_audit = pd.concat([audit_test, test_prediction_frame], axis=1)

    governance_table.to_csv(output_paths["governance_table"], index=False)
    validation_comparison.to_csv(output_paths["candidate_comparison_validation"], index=False)
    test_comparison.to_csv(output_paths["candidate_comparison_test"], index=False)
    validation_with_audit.to_csv(output_paths["validation_predictions"], index=False)
    test_with_audit.to_csv(output_paths["test_predictions"], index=False)
    segment_metrics.to_csv(output_paths["segment_metrics"], index=False)
    validation_fairness_frame.to_csv(output_paths["fairness_metrics_validation"], index=False)
    test_fairness_frame.to_csv(output_paths["fairness_metrics_test"], index=False)
    validation_fairness_actions.to_csv(output_paths["fairness_actions_validation"], index=False)
    decision_recommendations_test.to_csv(output_paths["decision_recommendations_test"], index=False)
    decision_policy_search.to_csv(output_paths["decision_policy_search"], index=False)
    worst_errors.to_csv(output_paths["worst_errors_test"], index=False)
    stage1_shap_global.to_csv(output_paths["shap_stage1_global"], index=False)
    stage2_shap_global.to_csv(output_paths["shap_stage2_global"], index=False)
    validation_calibration_bins.to_csv(output_paths["calibration_bins_validation"], index=False)
    test_calibration_bins.to_csv(output_paths["calibration_bins_test"], index=False)
    pd.DataFrame(
        temporal_summary.get("validation_candidate_comparison", [])
    ).to_csv(output_paths["temporal_validation_candidate_comparison"], index=False)
    pd.DataFrame(
        temporal_summary.get("test_candidate_comparison", [])
    ).to_csv(output_paths["temporal_test_candidate_comparison"], index=False)
    output_paths["decision_examples"].write_text(
        json.dumps(to_serializable(decision_examples), indent=2),
        encoding="utf-8",
    )
    output_paths["uncertainty_summary"].write_text(
        json.dumps(to_serializable(bootstrap_summary), indent=2),
        encoding="utf-8",
    )
    output_paths["calibration_summary"].write_text(
        json.dumps(
            to_serializable(
                {
                    "validation": validation_calibration_summary,
                    "test": test_calibration_summary,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    output_paths["temporal_summary"].write_text(
        json.dumps(to_serializable(temporal_summary), indent=2),
        encoding="utf-8",
    )
    output_paths["business_summary"].write_text(business_summary_text, encoding="utf-8")

    report_text = build_report(
        data_path=data_path,
        policy=policy,
        selected_model_name=selected_model_name,
        selection_rationale=selection_rationale,
        validation_comparison=validation_comparison,
        test_comparison=test_comparison,
        final_test_metrics=final_test_metrics,
        bootstrap_summary=bootstrap_summary,
        error_thresholds=error_thresholds,
        decision_policy=decision_policy,
        threshold_rationale=threshold_rationale,
        decision_summary=decision_summary,
        fairness_summary_validation=fairness_summary_validation,
        fairness_summary_test=fairness_summary_test,
        calibration_summary_validation=validation_calibration_summary,
        calibration_summary_test=test_calibration_summary,
        business_summary_text=business_summary_text,
        temporal_summary=temporal_summary,
        stage1_shap_global=stage1_shap_global,
        stage2_shap_global=stage2_shap_global,
        rule_text=rule_text,
    )
    output_paths["report"].write_text(report_text, encoding="utf-8")

    metadata = {
        "artifact_type": MODEL_ARTIFACT_TYPE,
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "dataset": {
            "rows": int(len(dataframe_with_segments)),
            "columns": int(dataframe_with_segments.shape[1]),
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(validation_idx)),
            "test_rows": int(len(test_idx)),
            "target_distribution": {
                str(int(value)): int(count)
                for value, count in y_full.value_counts().sort_index().items()
            },
        },
        "problem_framing": {
            "candidates_considered": validation_comparison["model_name"].tolist(),
            "selected_model_name": selected_model_name,
            "selection_rationale": selection_rationale,
            "target_note": (
                "The target contains a large mass at 0%, so model selection explicitly compared "
                "direct regression, tiers, and a binary-plus-amount decomposition."
            ),
        },
        "feature_policy": asdict(policy),
        "validation_candidate_comparison": validation_comparison.to_dict(orient="records"),
        "test_candidate_comparison": test_comparison.to_dict(orient="records"),
        "final_test_metrics": asdict(final_test_metrics),
        "bootstrap_summary": bootstrap_summary,
        "error_thresholds": error_thresholds,
        "threshold_rationale": threshold_rationale,
        "calibration_validation_summary": validation_calibration_summary,
        "calibration_test_summary": test_calibration_summary,
        "fairness_validation_summary": fairness_summary_validation,
        "fairness_test_summary": fairness_summary_test,
        "fairness_guardrails": fairness_guardrails,
        "decision_policy": decision_policy,
        "decision_summary": decision_summary,
        "temporal_validation": temporal_summary,
        "business_summary": business_summary_text,
        "optional_dependency_status": OPTIONAL_DEPENDENCY_STATUS,
        "decision_examples": decision_examples,
    }

    model_bundle = {
        "artifact_type": MODEL_ARTIFACT_TYPE,
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "target_column": TARGET_COLUMN,
        "selected_model_name": selected_model_name,
        "feature_columns": policy.model_feature_columns,
        "supported_percentages": sorted(
            float(value)
            for value in y_full.loc[train_valid_idx].dropna().unique()
        ),
        "model": selected_deployment_model,
        "metadata": metadata,
    }

    joblib.dump(model_bundle, output_paths["model"])
    output_paths["metadata"].write_text(
        json.dumps(to_serializable(metadata), indent=2),
        encoding="utf-8",
    )

    return {
        "model_bundle": model_bundle,
        "metadata": metadata,
        "report_text": report_text,
        "paths": {label: str(path) for label, path in output_paths.items()},
    }

# %% [notebook cell 39]
def load_model_bundle(path: Path = build_artifact_paths(ARTIFACT_DIR)["model"]) -> dict[str, Any]:
    sys.modules.setdefault(CANONICAL_MODULE_NAME, sys.modules[__name__])
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        for _custom_class in (
            SupportedValueRegressor,
            RuleTreeBaseline,
            TieredAidModel,
            TwoStageAidModel,
        ):
            setattr(main_module, _custom_class.__name__, _custom_class)
    return joblib.load(path)


def predict_dataframe(
    model_bundle: dict[str, Any],
    dataframe: pd.DataFrame,
    *,
    snap_to_supported: bool = True,
) -> pd.Series:
    feature_frame = prepare_feature_frame(dataframe, model_bundle["feature_columns"])
    model = model_bundle["model"]

    if hasattr(model, "predict_details"):
        details = model.predict_details(feature_frame)
        predictions = np.asarray(details["predicted_need_pct"], dtype=float)
        if not snap_to_supported:
            predictions = np.asarray(details["predicted_continuous_need_pct"], dtype=float)
    else:
        predictions = np.clip(np.asarray(model.predict(feature_frame), dtype=float), 0.0, 100.0)
        if snap_to_supported:
            predictions = snap_to_supported_percentages(
                predictions,
                model_bundle["supported_percentages"],
            )

    return pd.Series(predictions, index=dataframe.index, name="predicted_need_pct")


def predict_with_details(
    model_bundle: dict[str, Any],
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    feature_frame = prepare_feature_frame(dataframe, model_bundle["feature_columns"])
    model = model_bundle["model"]

    if hasattr(model, "predict_details"):
        details = model.predict_details(feature_frame)
        return pd.DataFrame(details, index=dataframe.index)

    predictions = np.clip(np.asarray(model.predict(feature_frame), dtype=float), 0.0, 100.0)
    return pd.DataFrame(
        {
            "predicted_need_pct": predictions,
            "predicted_continuous_need_pct": predictions,
            "predicted_nonzero_probability": np.clip(predictions / 100.0, 0.0, 1.0),
            "predicted_positive_amount": predictions,
            "prediction_interval_lower": predictions,
            "prediction_interval_upper": predictions,
            "prediction_interval_width": np.zeros(len(predictions), dtype=float),
            "confidence_score": np.full(len(predictions), np.nan, dtype=float),
        },
        index=dataframe.index,
    )

# %% [notebook cell 41]
saved_example_columns = [
    "raw_source_row_number",
    "parsed_level",
    "parsed_nationality",
    "segment_household_income_bucket",
    "predicted_need_pct",
    "prediction_interval_lower",
    "prediction_interval_upper",
    "confidence_score",
    "recommended_action",
    "recommended_percentage",
]

if decision_df.empty:
    print("Saved decision output is unavailable, so no scored holdout examples can be displayed.")
else:
    print("Example 1: Saved scored cases from the holdout set")
    print("Focus on 'Predicted aid %' as the model's estimated financial-aid percentage.")
    print("")
    available_saved_columns = [
        column for column in saved_example_columns if column in decision_df.columns
    ]
    saved_examples = decision_df.loc[:, available_saved_columns].head(10).copy()
    saved_examples = saved_examples.rename(
        columns={
            "raw_source_row_number": "Source row",
            "parsed_level": "Level",
            "parsed_nationality": "Nationality",
            "segment_household_income_bucket": "Income bucket",
            "predicted_need_pct": "Predicted aid %",
            "prediction_interval_lower": "Lower bound",
            "prediction_interval_upper": "Upper bound",
            "confidence_score": "Confidence",
            "recommended_action": "Recommended action",
            "recommended_percentage": "Recommended %",
        }
    )
    display(saved_examples)

# %% [notebook cell 42]
bundle = load_model_bundle()
sample_input = pd.read_csv(DATA_PATH).head(5).copy()
live_prediction_details = predict_with_details(bundle, sample_input)

identity_columns = [
    "raw_source_row_number",
    "parsed_level",
    "parsed_nationality",
    "parsed_applicant_citizenship",
]
available_identity_columns = [
    column for column in identity_columns if column in sample_input.columns
]
prediction_columns = [
    "predicted_need_pct",
    "predicted_nonzero_probability",
    "prediction_interval_lower",
    "prediction_interval_upper",
    "confidence_score",
]
available_prediction_columns = [
    column for column in prediction_columns if column in live_prediction_details.columns
]

print("Example 2: Live scoring on a few rows from the dataset")
print("This is the clearest view of what the model itself predicts for new students.")
print("")

live_examples = pd.concat(
    [
        sample_input.loc[:, available_identity_columns].reset_index(drop=True),
        live_prediction_details.loc[:, available_prediction_columns].reset_index(drop=True),
    ],
    axis=1,
)
live_examples = live_examples.rename(
    columns={
        "raw_source_row_number": "Source row",
        "parsed_level": "Level",
        "parsed_nationality": "Nationality",
        "parsed_applicant_citizenship": "Citizenship",
        "predicted_need_pct": "Predicted aid %",
        "predicted_nonzero_probability": "P(any aid)",
        "prediction_interval_lower": "Lower bound",
        "prediction_interval_upper": "Upper bound",
        "confidence_score": "Confidence",
    }
)
display(live_examples)
