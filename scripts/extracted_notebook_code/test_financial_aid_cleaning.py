# Extracted notebook code for portfolio review.
# Generated from a sanitized notebook copy; outputs and execution state are intentionally excluded.


# %% [notebook cell 3]
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path.cwd()  # portfolio version: run from repository root
OUTPUT_DIR = PROJECT_ROOT / "cleaned data"
CODE_DIR = PROJECT_ROOT / "financial_aid_datacleaning"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import ast
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import patch
from xml.sax.saxutils import escape
from zipfile import ZipFile


if "__file__" in globals():
    ROOT = Path(__file__).resolve().parents[1]
elif "CODE_DIR" in globals():
    ROOT = Path(CODE_DIR).resolve()
elif "PROJECT_ROOT" in globals():
    ROOT = Path(PROJECT_ROOT).resolve() / "financial_aid_datacleaning"
else:
    candidate = Path.cwd() / "financial_aid_datacleaning"
    ROOT = candidate if candidate.exists() else Path.cwd()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import clean_faid_data as mod


def excel_column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def build_inline_string_cell(ref: str, value: object) -> str:
    text = escape("" if value is None else str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def write_test_workbook(
    path: Path,
    *,
    headers: list[str],
    rows: list[dict[str, object]],
    sheet_name: str = "Sheet1",
) -> None:
    row_xml: list[str] = []
    if headers:
        header_cells = [
            build_inline_string_cell(f"{excel_column_name(col_idx)}1", header)
            for col_idx, header in enumerate(headers, start=1)
        ]
        row_xml.append(f'<row r="1">{"".join(header_cells)}</row>')

        for row_idx, row in enumerate(rows, start=2):
            cells = [
                build_inline_string_cell(f"{excel_column_name(col_idx)}{row_idx}", row.get(header, ""))
                for col_idx, header in enumerate(headers, start=1)
            ]
            row_xml.append(f'<row r="{row_idx}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(row_xml)}</sheetData>"
        "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="{escape(sheet_name)}" sheetId="1" r:id="rId1"/></sheets>'
        "</workbook>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    with ZipFile(path, "w") as workbook_zip:
        workbook_zip.writestr("[Content_Types].xml", content_types_xml)
        workbook_zip.writestr("_rels/.rels", root_rels_xml)
        workbook_zip.writestr("xl/workbook.xml", workbook_xml)
        workbook_zip.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        workbook_zip.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_base_row(**overrides: object) -> dict[str, object]:
    row = {column: "" for column in mod.REQUIRED_SOURCE_COLUMNS}
    row.update(
        {
            "Application Type": "Regular Graduate Application",
            "LEVEL": "Graduate",
            "Application Term": "2025-2026",
            "Nationality": "Lebanese",
            "Applicant Demographic": "Citizenship: Lebanese\nMarital Status: Single\nPlan to Reside: Campus",
            "Father Work Status": (
                "Status: Employed\n"
                "Employed In: Private Sector\n"
                "Institution or Employer's Name: ABC Corp\n"
                "Annual Income: 5000"
            ),
            "Father Income Document": "5000",
            "Father Job Info": "employed",
            "Mother Work Status": (
                "Status: Employed\n"
                "Employed In: Private Sector\n"
                "Institution or Employer's Name: XYZ Corp\n"
                "Annual Income: 4000"
            ),
            "Mother Income Document": "4000",
            "Mother Job Info": "employed",
            "Travel Records Verification": "low",
            "Certificate of Ownership Verification": "correct",
            "Decision": "awarded",
            "Over and Above Decision": "No O A",
            "Submission_Date": "2025-01-15",
            "Need": "20",
            "Merit": "10",
            "Consent to share information": "Yes",
            "School": "Faculty of Arts and Sciences",
        }
    )
    row.update(overrides)
    return row


def build_pipeline_fixture_rows() -> list[dict[str, object]]:
    return [
        make_base_row(),
        make_base_row(
            **{
                "Financial Assistants": "Uncle Every 4 Months 300",
            }
        ),
        make_base_row(
            **{
                "Loans": (
                    "Reason: housing\n"
                    "Total Amount: 100,000,000 L.L.\n"
                    "Monthly Payment: 970,054 L.L. + $40\n"
                    "Remaining Balance: 7 years\n"
                    "Remaining Years: 7 years"
                ),
            }
        ),
        make_base_row(
            **{
                "Applicant Demographic": "Citizenship: Lebanese\nMarital Status: Single",
                "Applicant's Spouse Demographic": "Citizenship: Lebanese",
                "Applicant's Spouse Work Status": "Status: Employed\nAnnual Income: 1000",
                "Decision": "",
                "Bin": "Awarded",
            }
        ),
    ]


def collect_issue_codes_from_row(
    overrides: dict[str, object],
    *,
    school_mode: str | None = None,
    impute_missing_school_from_mode: bool = False,
) -> list[str]:
    issues = mod.IssueTracker()
    mod.build_cleaned_row(
        make_base_row(**overrides),
        source_row_number=999,
        source_sheet_name="Sheet1",
        school_mode=school_mode,
        impute_missing_school_from_mode=impute_missing_school_from_mode,
        issues=issues,
    )
    return [str(row["issue_code"]) for row in issues.as_rows()]


def collect_monetary_issue_codes(
    *,
    cleaned_row: dict[str, object],
    raw_row: dict[str, str],
    rules: tuple[mod.MonetaryQualityRule, ...] | None = None,
) -> list[str]:
    issues = mod.IssueTracker()
    if rules is None:
        mod.add_monetary_quality_issues(
            cleaned_row=cleaned_row,
            raw_row=raw_row,
            source_row_number=501,
            issues=issues,
        )
    else:
        with patch.object(mod, "MONETARY_QUALITY_RULES", rules):
            mod.add_monetary_quality_issues(
                cleaned_row=cleaned_row,
                raw_row=raw_row,
                source_row_number=501,
                issues=issues,
            )
    return [str(row["issue_code"]) for row in issues.as_rows()]


class WorkbookCliTestCase(unittest.TestCase):
    def run_cli(self, workbook_path: Path, output_dir: Path, *extra_args: str) -> int:
        argv = [
            "clean_faid_data.py",
            "--input",
            str(workbook_path),
            "--output-dir",
            str(output_dir),
            *extra_args,
        ]
        with patch.object(sys, "argv", argv):
            return mod.main()


ISSUE_FIXTURE_BUILDERS: dict[str, Callable[[], list[str]]] = {
    "aid_percentage_total_exceeds_100": lambda: collect_issue_codes_from_row(
        {
            "Need": "70",
            "Merit": "25",
            "Over and Above - Percentage Awarded": "10",
            "Over and Above Decision": "Yes",
        }
    ),
    "applicant_annual_income_missing_after_parse": lambda: collect_issue_codes_from_row(
        {
            "Applicant Work Status": "Status: Employed\nAnnual Income: TBD",
        }
    ),
    "application_type_inferred_from_level": lambda: collect_issue_codes_from_row({"Application Type": ""}),
    "application_type_level_mismatch": lambda: collect_issue_codes_from_row(
        {
            "LEVEL": "Graduate",
            "Application Type": "Early Merit",
        }
    ),
    "decision_missing_with_bin_status": lambda: collect_issue_codes_from_row({"Decision": "", "Bin": "Awarded"}),
    "income_present_with_non_employed_status": lambda: collect_issue_codes_from_row(
        {
            "Father Work Status": "Status: Retired\nAnnual Income: 1000",
        }
    ),
    "large_monetary_value_review": lambda: collect_monetary_issue_codes(
        cleaned_row={"loans_total_amount": 100_000_000.0},
        raw_row={"Loans": "100,000,000 L.L."},
    ),
    "large_monetary_value_without_currency_marker": lambda: collect_monetary_issue_codes(
        cleaned_row={"custom_money": 1500.0},
        raw_row={"Custom Money": "1500"},
        rules=(
            mod.MonetaryQualityRule(
                "custom_money",
                "Custom Money",
                high_threshold=1000.0,
                likely_lbp_threshold=10_000.0,
            ),
        ),
    ),
    "likely_lbp_without_currency_marker": lambda: collect_monetary_issue_codes(
        cleaned_row={"applicant_annual_income": 819_730_500.0},
        raw_row={"Applicant Work Status": "Annual Income 819,730,500"},
    ),
    "loan_balance_exceeds_total_amount": lambda: collect_issue_codes_from_row(
        {
            "Loans": (
                "Reason: housing\n"
                "Total Amount: 1000\n"
                "Remaining Balance: 1200\n"
                "Remaining Years: 2"
            ),
        }
    ),
    "loan_remaining_balance_non_monetary_text": lambda: collect_issue_codes_from_row(
        {
            "Loans": (
                "Reason: housing\n"
                "Total Amount: 100000\n"
                "Remaining Balance: 7 years\n"
                "Remaining Years: 7 years"
            ),
        }
    ),
    "merit_plus_over_and_above_exceeds_50_rule": lambda: collect_issue_codes_from_row(
        {
            "Merit": "40",
            "Over and Above - Percentage Awarded": "15",
            "Over and Above Decision": "Yes",
        }
    ),
    "mixed_currency_signal": lambda: collect_issue_codes_from_row(
        {
            "Loans": "Reason: housing\nTotal Amount: 970,054 l l + usd 40",
        }
    ),
    "negative_monetary_value": lambda: collect_issue_codes_from_row(
        {
            "Applicant Work Status": "Status: Employed\nAnnual Income: -100",
        }
    ),
    "over_and_above_flag_inconsistent": lambda: collect_issue_codes_from_row(
        {
            "Decision": "No O A",
            "Over and Above - Amount Awarded": "10",
        }
    ),
    "parent_income_needs_review": lambda: collect_issue_codes_from_row(
        {
            "Father Income Document": "unknown amount",
        }
    ),
    "parent_income_unparsed": lambda: collect_issue_codes_from_row(
        {
            "Father Income Document": "unknown amount",
        }
    ),
    "school_imputed_from_mode": lambda: collect_issue_codes_from_row(
        {
            "School": "",
        },
        school_mode="Faculty of Arts and Sciences",
        impute_missing_school_from_mode=True,
    ),
    "school_missing": lambda: collect_issue_codes_from_row({"School": ""}),
    "school_missing_undergraduate": lambda: collect_issue_codes_from_row(
        {
            "LEVEL": "Undergraduate",
            "School": "",
        }
    ),
    "spouse_data_present_while_single": lambda: collect_issue_codes_from_row(
        {
            "Applicant Demographic": "Citizenship: Lebanese\nMarital Status: Single",
            "Applicant's Spouse Demographic": "Citizenship: Lebanese",
            "Applicant's Spouse Work Status": "Status: Employed\nAnnual Income: 1000",
        }
    ),
}

# %% [notebook cell 5]
class ParseAmountTests(unittest.TestCase):
    def test_parse_plus_minus_amount_as_positive(self) -> None:
        raw_value = "family Yearly +/-6,000"
        self.assertEqual(mod.parse_amount(raw_value, strategy=mod.choose_amount_strategy(raw_value)), 6000.0)

    def test_parse_range_with_annotation_keeps_midpoint(self) -> None:
        raw_value = "$250000-300000 (+House)"
        self.assertEqual(mod.parse_amount(raw_value, strategy=mod.choose_amount_strategy(raw_value)), 275000.0)

    def test_parse_digit_amount_with_scale_word(self) -> None:
        raw_value = "37 Millions LL"
        self.assertEqual(mod.parse_amount(raw_value, strategy=mod.choose_amount_strategy(raw_value)), 37000000.0)

    def test_choose_amount_strategy_only_sums_numeric_additions(self) -> None:
        self.assertEqual(mod.choose_amount_strategy("300 + 500"), "sum")
        self.assertEqual(mod.choose_amount_strategy("970,054 L.L. + $40"), "first")

    def test_detect_currency_signal(self) -> None:
        self.assertEqual(mod.detect_currency_signal("1000 USD"), "usd")
        self.assertEqual(mod.detect_currency_signal("1,000,000 L.L."), "lbp")
        self.assertEqual(mod.detect_currency_signal("970,054 L.L. + $40"), "mixed")
        self.assertIsNone(mod.detect_currency_signal("all documents received"))

    def test_infer_currency_guess_flags_large_unmarked_values_as_likely_lbp(self) -> None:
        self.assertEqual(
            mod.infer_currency_guess("Annual Income 819,730,500", 819730500.0, likely_lbp_threshold=1_000_000),
            "likely_lbp_unmarked",
        )
        self.assertEqual(
            mod.infer_currency_guess("Annual Income 8,000", 8000.0, likely_lbp_threshold=1_000_000),
            "unknown",
        )


class WorkStatusTests(unittest.TestCase):
    def test_parse_work_status_handles_annual_income_without_colon(self) -> None:
        raw_value = (
            "Employed In: Private Sector\n"
            "Position: Junior MIS Officer\n"
            "Institution or Employer's Name: Bank of Beirut\n"
            "Annual Income 819,730,500\n"
            "Educational Benefits: 0\n"
            "Accomodation: 0\n"
            "Commission: 0\n"
            "Bonuses: 0\n"
            "Other Benefits: 0"
        )
        parsed = mod.parse_work_status(raw_value)
        self.assertEqual(parsed["status"], "Employed")
        self.assertEqual(parsed["annual_income"], 819730500.0)
        self.assertEqual(parsed["employer_name"], "Bank of Beirut")


class FinancialAssistantTests(unittest.TestCase):
    def test_parse_financial_assistants_ignores_cadence_number(self) -> None:
        parsed = mod.parse_financial_assistants("Uncle Every 4 Months 300")
        self.assertEqual(parsed["count"], 1)
        self.assertEqual(parsed["total_est_annual_amount"], 900.0)

    def test_parse_financial_assistants_handles_every_few_months_range(self) -> None:
        parsed = mod.parse_financial_assistants("Brother Every few months 300-500$")
        self.assertEqual(parsed["total_est_annual_amount"], 1600.0)

    def test_extract_financial_assistant_records_keeps_provenance(self) -> None:
        records = mod.extract_financial_assistant_records("Uncle Every 4 Months 300")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["assistant_raw_line"], "Uncle Every 4 Months 300")
        self.assertEqual(records[0]["assistant_estimated_annual_amount"], 900.0)

    def test_extract_financial_assistant_records_marks_ambiguous_amounts_for_review(self) -> None:
        records = mod.extract_financial_assistant_records("Uncle 300 every 4 months 400")
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]["assistant_amount_per_occurrence"])
        self.assertEqual(records[0]["assistant_review_flag"], 1)
        self.assertIn("ambiguous_amount_location", records[0]["assistant_review_reasons"])


class LoanParsingTests(unittest.TestCase):
    def test_parse_loans_rejects_duration_in_remaining_balance(self) -> None:
        parsed = mod.parse_loans(
            "Reason: housing\n"
            "Total Amount: 100,000,000 L.L.\n"
            "Monthly Payment: 970,054 L.L. + $40\n"
            "Remaining Balance: 7 years\n"
            "Remaining Years: 7 years"
        )
        self.assertEqual(parsed["total_amount"], 100000000.0)
        self.assertIsNone(parsed["total_remaining_balance"])
        self.assertEqual(parsed["max_remaining_years"], 7.0)

    def test_extract_loan_records_parses_block_fields(self) -> None:
        records = mod.extract_loan_records(
            "Reason: buy a house\n"
            "Source: Banque De L'habitat\n"
            "Mortgaged Asset: house\n"
            "Start Date: 1998-08-10\n"
            "End Date: 2018-08-10\n"
            "Total Amount: 37 Millions LL\n"
            "Monthly Payment: 450 LL\n"
            "Remaining Balance: 0\n"
            "Remaining Years: 0"
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["loan_reason"], "buy a house")
        self.assertEqual(records[0]["loan_total_amount"], 37000000.0)

    def test_detect_loan_record_issues_flags_balance_over_total(self) -> None:
        issues = mod.detect_loan_record_issues(
            [
                {
                    "loan_total_amount": 1000.0,
                    "loan_remaining_balance": 1200.0,
                }
            ]
        )
        self.assertEqual(issues, ["loan_balance_exceeds_total_amount"])

    def test_extract_loan_records_marks_missing_anchor_and_duplicate_labels(self) -> None:
        records = mod.extract_loan_records(
            "Total Amount: 1000\n"
            "Total Amount: 2000\n"
            "Monthly Payment: 50\n"
            "This line has no label"
        )
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0]["loan_total_amount"])
        self.assertIn("missing_anchor", records[0]["loan_review_reasons"])
        self.assertIn("duplicate_total_amount", records[0]["loan_review_reasons"])
        self.assertIn("unlabeled_lines", records[0]["loan_review_reasons"])

# %% [notebook cell 8]
class StructuredRecordTests(unittest.TestCase):
    def test_extract_property_records_parses_estimated_value(self) -> None:
        records = mod.extract_property_records(
            "Type: Land\n"
            "Location: Beqaa\n"
            "Area: About 2000\n"
            "Estimated Present Value: 37 Millions LL\n"
            "Mortgaged: No"
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["property_type"], "Land")
        self.assertEqual(records[0]["property_estimated_present_value"], 37000000.0)

    def test_extract_car_records_parses_models(self) -> None:
        records = mod.extract_car_records(
            "Type: SUV\n"
            "Model: Toyota Rav4\n"
            "Mortgaged: Yes"
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["car_model"], "Toyota Rav4")
        self.assertEqual(records[0]["car_is_mortgaged_flag"], 1)

    def test_extract_property_records_marks_unknown_and_unlabeled_lines(self) -> None:
        records = mod.extract_property_records(
            "Type: Land\n"
            "Location: Beqaa\n"
            "Estimated Present Value: 37 Millions LL\n"
            "Mystery Label: something\n"
            "This line has no label"
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["property_review_flag"], 1)
        self.assertIn("unknown_labels", records[0]["property_review_reasons"])
        self.assertIn("unlabeled_lines", records[0]["property_review_reasons"])


class QualityIssueTests(unittest.TestCase):
    def test_large_amount_without_currency_marker_is_flagged(self) -> None:
        cleaned_row = {"applicant_annual_income": 819730500.0}
        raw_row = {"Applicant Work Status": "Annual Income 819,730,500"}
        issues = mod.IssueTracker()

        mod.add_monetary_quality_issues(
            cleaned_row=cleaned_row,
            raw_row=raw_row,
            source_row_number=210,
            issues=issues,
        )

        issue_codes = [row["issue_code"] for row in issues.as_rows()]
        self.assertIn("likely_lbp_without_currency_marker", issue_codes)

    def test_build_review_rows_aggregates_issue_context(self) -> None:
        issues = mod.IssueTracker()
        issues.add(10, "School", "school_missing", "", "School is blank")
        issues.add(10, "Application Type", "application_type_inferred_from_level", "", "Derived from level")

        cleaned_rows = [
            {
                "source_row_number": 10,
                "qa_issue_count": 2,
                "qa_requires_review": 1,
                "decision": "awarded",
            },
            {
                "source_row_number": 11,
                "qa_issue_count": 0,
                "qa_requires_review": 0,
                "decision": "denied",
            },
        ]

        review_rows = mod.build_review_rows(cleaned_rows, issues)
        self.assertEqual(len(review_rows), 1)
        self.assertEqual(review_rows[0]["source_row_number"], 10)
        self.assertIn("school_missing", review_rows[0]["qa_issue_codes"])
        self.assertIn("Application Type", review_rows[0]["qa_issue_fields"])

    def test_looks_like_duration_only(self) -> None:
        self.assertTrue(mod.looks_like_duration_only("7 years"))
        self.assertFalse(mod.looks_like_duration_only("970,054 L.L. + $40"))

    def test_summarize_issue_severity(self) -> None:
        severity, quality_score, risk_categories = mod.summarize_issue_severity(
            ["school_missing", "decision_missing_with_bin_status", "loan_balance_exceeds_total_amount"]
        )
        self.assertEqual(severity, "high")
        self.assertEqual(quality_score, 70)
        self.assertEqual(risk_categories, ["modeling", "business_rule"])

    def test_build_cleaned_row_adds_cross_field_consistency_issues(self) -> None:
        issues = mod.IssueTracker()
        mod.build_cleaned_row(
            {
                "LEVEL": "Undergraduate",
                "Application Type": "",
                "Applicant Demographic": "Marital Status: Single",
                "Applicant's Spouse Demographic": "Citizenship: Lebanese",
                "Applicant's Spouse Work Status": "Status: Employed\nAnnual Income: 1000",
                "Decision": "",
                "Bin": "Awarded",
                "School": "",
            },
            source_row_number=25,
            source_sheet_name="Sheet1",
            school_mode=None,
            impute_missing_school_from_mode=False,
            issues=issues,
        )
        mod.build_cleaned_row(
            {
                "LEVEL": "Undergraduate",
                "Decision": "No O A",
                "Over and Above - Amount Awarded": "10",
            },
            source_row_number=26,
            source_sheet_name="Sheet1",
            school_mode=None,
            impute_missing_school_from_mode=False,
            issues=issues,
        )
        issue_codes_by_row: dict[int, list[str]] = {}
        for issue_row in issues.as_rows():
            issue_codes_by_row.setdefault(int(issue_row["source_row_number"]), []).append(issue_row["issue_code"])

        self.assertIn("decision_missing_with_bin_status", issue_codes_by_row[25])
        self.assertIn("school_missing_undergraduate", issue_codes_by_row[25])
        self.assertIn("spouse_data_present_while_single", issue_codes_by_row[25])
        self.assertIn("over_and_above_flag_inconsistent", issue_codes_by_row[26])

    def test_build_cleaned_row_keeps_school_null_and_uses_ops_fill_field(self) -> None:
        issues = mod.IssueTracker()
        cleaned_row = mod.build_cleaned_row(
            {
                "LEVEL": "Undergraduate",
                "School": "",
            },
            source_row_number=30,
            source_sheet_name="Sheet1",
            school_mode="Faculty of Arts and Sciences",
            impute_missing_school_from_mode=True,
            issues=issues,
        )
        self.assertIsNone(cleaned_row["school"])
        self.assertEqual(cleaned_row["school_filled_for_ops_only"], "Faculty of Arts and Sciences")
        self.assertEqual(cleaned_row["school_ops_fill_method"], "global_mode")


def extract_emitted_issue_codes() -> set[str]:
    source = inspect.getsource(mod)
    tree = ast.parse(source)
    emitted_codes: set[str] = set()

    class IssueCodeVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "add":
                if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant) and isinstance(node.args[2].value, str):
                    emitted_codes.add(node.args[2].value)
            self.generic_visit(node)

    IssueCodeVisitor().visit(tree)
    return emitted_codes


class IssueCoverageTests(unittest.TestCase):
    def test_issue_metadata_covers_all_emitted_issue_codes(self) -> None:
        emitted_codes = extract_emitted_issue_codes()
        self.assertTrue(emitted_codes.issubset(set(mod.ISSUE_METADATA)))
        self.assertEqual(set(mod.ISSUE_METADATA), set(ISSUE_FIXTURE_BUILDERS))


class ExportTests(unittest.TestCase):
    def test_build_export_rows_prefixes_column_groups(self) -> None:
        export_rows = mod.build_export_rows(
            [
                {
                    "source_row_number": 1,
                    "application_type_clean": "Regular Graduate Application",
                    "father_income_document": 1000.0,
                    "application_type_confidence": 0.7,
                    "school_raw": "Institution",
                    "qa_issue_count": 1,
                }
            ]
        )
        self.assertEqual(export_rows[0]["raw_source_row_number"], 1)
        self.assertEqual(export_rows[0]["inferred_application_type_clean"], "Regular Graduate Application")
        self.assertEqual(export_rows[0]["parsed_father_income_document"], 1000.0)
        self.assertEqual(export_rows[0]["qa_application_type_confidence"], 0.7)
        self.assertEqual(export_rows[0]["raw_school"], "Institution")

    def test_build_model_safe_rows_nulls_low_confidence_fields(self) -> None:
        model_safe_rows = mod.build_model_safe_rows(
            [
                {
                    "source_row_number": 1,
                    "source_sheet_name": "Sheet1",
                    "father_income_document": 1000.0,
                    "father_income_document_confidence": 0.6,
                    "application_term": "2025-2026",
                    "qa_issue_count": 1,
                    "qa_requires_review": 1,
                    "qa_max_severity": "medium",
                    "qa_quality_score": 80,
                    "qa_risk_categories": "parsing",
                }
            ]
        )
        self.assertEqual(model_safe_rows[0]["raw_source_row_number"], 1)
        self.assertIsNone(model_safe_rows[0]["parsed_father_income_document"])
        self.assertEqual(model_safe_rows[0]["parsed_application_term"], "2025-2026")
        self.assertEqual(model_safe_rows[0]["qa_issue_count"], 1)

# %% [notebook cell 10]
class SchemaValidationTests(WorkbookCliTestCase):
    def test_validate_source_columns_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workbook_path = Path(tmp_dir) / "missing_required.xlsx"
            write_test_workbook(
                workbook_path,
                headers=["LEVEL", "Application Type"],
                rows=[],
            )

            workbook = mod.read_xlsx_rows(workbook_path)
            with self.assertRaisesRegex(ValueError, "Input workbook is missing required columns"):
                mod.validate_source_columns(workbook.headers)

    def test_validate_source_columns_allows_extra_irrelevant_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workbook_path = Path(tmp_dir) / "extra_columns.xlsx"
            headers = sorted(mod.REQUIRED_SOURCE_COLUMNS) + ["Random Notes"]
            write_test_workbook(
                workbook_path,
                headers=headers,
                rows=[{**make_base_row(), "Random Notes": "ignore this"}],
            )

            workbook = mod.read_xlsx_rows(workbook_path)
            self.assertIn("Random Notes", workbook.headers)
            mod.validate_source_columns(workbook.headers)

    def test_read_xlsx_rows_handles_empty_sheet_and_cli_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            workbook_path = tmp_path / "empty.xlsx"
            output_dir = tmp_path / "out"
            write_test_workbook(workbook_path, headers=[], rows=[])

            workbook = mod.read_xlsx_rows(workbook_path)
            self.assertEqual(workbook.headers, [])
            self.assertEqual(workbook.rows, [])

            exit_code = self.run_cli(workbook_path, output_dir)
            self.assertEqual(exit_code, 1)


class CliOutputTests(WorkbookCliTestCase):
    def test_build_output_paths_includes_all_expected_artifacts(self) -> None:
        output_dir = Path("/tmp/fake-clean-output")
        paths = mod.build_output_paths(output_dir, "mini")
        self.assertEqual(paths.cleaned_csv.name, "mini.csv")
        self.assertEqual(paths.model_safe_csv.name, "mini_model_safe.csv")
        self.assertEqual(paths.review_csv.name, "mini_review_required.csv")
        self.assertEqual(paths.data_dictionary_json.name, "mini_data_dictionary.json")
        self.assertEqual(paths.profile_json.name, "mini_profile.json")
        self.assertEqual(paths.summary_json.name, "mini_summary.json")

    def test_cli_generates_dictionary_profile_summary_and_honors_extra_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            workbook_path = tmp_path / "pipeline.xlsx"
            output_dir = tmp_path / "out"
            headers = sorted(mod.REQUIRED_SOURCE_COLUMNS) + ["Random Notes"]
            rows = [
                {**row, "Random Notes": f"note-{idx}"}
                for idx, row in enumerate(build_pipeline_fixture_rows(), start=1)
            ]
            write_test_workbook(workbook_path, headers=headers, rows=rows)

            exit_code = self.run_cli(workbook_path, output_dir, "--output-stem", "mini")
            self.assertEqual(exit_code, 0)

            output_paths = mod.build_output_paths(output_dir, "mini")
            cleaned_rows = read_csv_rows(output_paths.cleaned_csv)
            summary = load_json(output_paths.summary_json)
            profile = load_json(output_paths.profile_json)
            dictionary = load_json(output_paths.data_dictionary_json)

            self.assertTrue(output_paths.cleaned_csv.exists())
            self.assertTrue(output_paths.model_safe_csv.exists())
            self.assertTrue(output_paths.review_csv.exists())
            self.assertTrue(output_paths.data_dictionary_json.exists())
            self.assertTrue(output_paths.profile_json.exists())
            self.assertTrue(output_paths.summary_json.exists())
            self.assertNotIn("raw_random_notes", cleaned_rows[0])
            self.assertEqual(summary["source_column_count"], len(headers))
            self.assertEqual(summary["output_files"]["model_safe_csv"], str(output_paths.model_safe_csv.resolve()))
            self.assertIn("model_safe", profile["tables"])
            self.assertEqual(dictionary["confidence_scoring"]["kind"], "rule_based_operational_confidence")
            self.assertIn("not statistically calibrated", dictionary["confidence_scoring"]["note"])

            dictionary_fields = {
                column["export_field_name"]
                for column in dictionary["tables"]["cleaned"]["columns"]
            }
            self.assertIn("raw_application_type", dictionary_fields)
            self.assertIn("parsed_loans_total_amount", dictionary_fields)
            self.assertIn("inferred_application_type_clean", dictionary_fields)
            self.assertIn("qa_loans_total_amount_confidence", dictionary_fields)


class PipelineEndToEndTests(WorkbookCliTestCase):
    def test_pipeline_end_to_end_outputs_expected_rows_and_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            workbook_path = tmp_path / "tiny_pipeline.xlsx"
            output_dir = tmp_path / "out"
            write_test_workbook(
                workbook_path,
                headers=sorted(mod.REQUIRED_SOURCE_COLUMNS),
                rows=build_pipeline_fixture_rows(),
            )

            exit_code = self.run_cli(workbook_path, output_dir, "--output-stem", "tiny")
            self.assertEqual(exit_code, 0)

            output_paths = mod.build_output_paths(output_dir, "tiny")
            cleaned_rows = read_csv_rows(output_paths.cleaned_csv)
            review_rows = read_csv_rows(output_paths.review_csv)
            model_safe_rows = read_csv_rows(output_paths.model_safe_csv)
            summary = load_json(output_paths.summary_json)

            self.assertEqual(len(cleaned_rows), 4)
            self.assertEqual(len(review_rows), 2)
            self.assertEqual(len(model_safe_rows), 4)

            cleaned_by_row = {row["raw_source_row_number"]: row for row in cleaned_rows}
            review_by_row = {row["raw_source_row_number"]: row for row in review_rows}
            model_safe_by_row = {row["raw_source_row_number"]: row for row in model_safe_rows}

            self.assertEqual(
                cleaned_by_row["3"]["parsed_financial_assistants_total_est_annual_amount"],
                "900.0",
            )
            self.assertEqual(cleaned_by_row["4"]["parsed_loans_total_amount"], "100000000.0")
            self.assertEqual(cleaned_by_row["4"]["qa_loans_total_amount_confidence"], "0.5")
            self.assertEqual(model_safe_by_row["4"]["parsed_loans_total_amount"], "")
            self.assertIn("spouse_data_present_while_single", review_by_row["5"]["qa_issue_codes"])
            self.assertIn("decision_missing_with_bin_status", review_by_row["5"]["qa_issue_codes"])
            self.assertIn("raw_application_type", cleaned_rows[0])
            self.assertIn("parsed_loans_total_amount", cleaned_rows[0])
            self.assertIn("inferred_application_type_clean", cleaned_rows[0])
            self.assertIn("qa_loans_total_amount_confidence", cleaned_rows[0])
            self.assertNotIn("inferred_application_type_clean", model_safe_rows[0])
            self.assertEqual(summary["review_output_row_count"], 2)
            self.assertEqual(summary["issue_counts"]["mixed_currency_signal"], 1)
            self.assertEqual(summary["issue_counts"]["loan_remaining_balance_non_monetary_text"], 1)
            self.assertEqual(summary["issue_counts"]["spouse_data_present_while_single"], 1)

# %% [notebook cell 12]
class ProfileTests(unittest.TestCase):
    def test_build_table_profile_infers_numeric_stats(self) -> None:
        profile = mod.build_table_profile(
            "sample",
            [
                {"amount": 1, "status": "ok"},
                {"amount": 2, "status": None},
                {"amount": None, "status": "ok"},
            ],
        )

        amount_profile = next(column for column in profile["columns"] if column["name"] == "amount")
        status_profile = next(column for column in profile["columns"] if column["name"] == "status")

        self.assertEqual(profile["row_count"], 3)
        self.assertEqual(amount_profile["inferred_type"], "int")
        self.assertEqual(amount_profile["mean"], 1.5)
        self.assertEqual(status_profile["inferred_type"], "string")
        self.assertEqual(status_profile["top_values"], {"ok": 2})


def make_issue_fixture_test(issue_code: str, builder: Callable[[], list[str]]) -> Callable[[unittest.TestCase], None]:
    def test_method(self: unittest.TestCase) -> None:
        self.assertIn(issue_code, builder())

    return test_method


for _issue_code, _builder in ISSUE_FIXTURE_BUILDERS.items():
    setattr(
        IssueCoverageTests,
        f"test_issue_fixture_{_issue_code}",
        make_issue_fixture_test(_issue_code, _builder),
    )


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        unittest.main(argv=["test_clean_faid_data.py"], exit=False)
    else:
        unittest.main()

# %% [notebook cell 14]
import unittest

suite = unittest.defaultTestLoader.discover(
    str(CODE_DIR / "tests"),
    pattern="test_*.py",
)

print("Discovered test cases:", suite.countTestCases())

test_risk_map = [
    {
        "test_family": "ParseAmountTests",
        "real_world_failure_prevented": "Scaled money like '37 Millions LL' is misread by orders of magnitude.",
    },
    {
        "test_family": "FinancialAssistantTests / LoanParsingTests",
        "real_world_failure_prevented": "Messy multiline support and debt narratives are converted into wrong annual support or remaining balance.",
    },
    {
        "test_family": "IssueCoverageTests",
        "real_world_failure_prevented": "An issue code exists in theory but is never emitted or never documented.",
    },
    {
        "test_family": "SchemaValidationTests",
        "real_world_failure_prevented": "Workbook drift breaks the pipeline silently when columns disappear or sheets are empty.",
    },
    {
        "test_family": "PipelineEndToEndTests",
        "real_world_failure_prevented": "The full cleaning/export workflow fails even though helper functions still pass in isolation.",
    },
]

for row in test_risk_map:
    print(f"- {row['test_family']}: {row['real_world_failure_prevented']}")
