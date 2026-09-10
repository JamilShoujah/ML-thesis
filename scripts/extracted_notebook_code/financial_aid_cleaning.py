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

import pandas as pd
from IPython.display import display

summary = json.loads((OUTPUT_DIR / "faid_cleaned_summary.json").read_text())
data_dictionary = json.loads((OUTPUT_DIR / "faid_cleaned_data_dictionary.json").read_text())
dictionary_frame = pd.DataFrame(data_dictionary["tables"]["cleaned"]["columns"])

display(
    pd.DataFrame(
        {
            "metric": [
                "cleaned rows",
                "review rows",
                "model-safe rows",
                "issues logged",
                "columns in cleaned export",
            ],
            "value": [
                summary["cleaned_row_count"],
                summary["review_output_row_count"],
                summary["model_safe_row_count"],
                summary["issue_count"],
                summary["cleaned_column_count"],
            ],
        }
    )
)

display(
    dictionary_frame.groupby("group", dropna=False)
    .agg(
        feature_count=("export_field_name", "count"),
        model_safe_fields=("model_safe_included", "sum"),
    )
    .reset_index()
)

# %% [notebook cell 5]
import json
import pandas as pd
from IPython.display import Markdown, display

if "summary" not in globals():
    summary = json.loads((OUTPUT_DIR / "faid_cleaned_summary.json").read_text())

cleaned = pd.read_csv(OUTPUT_DIR / "faid_cleaned.csv")
review_required = pd.read_csv(OUTPUT_DIR / "faid_cleaned_review_required.csv")

award_like = {"awarded", "awarded_affidavit_of_promise", "usaid"}
award_rate = cleaned["parsed_decision"].isin(award_like).mean()
deny_rate = (cleaned["parsed_decision"] == "denied").mean()
review_share = cleaned["qa_requires_review"].fillna(0).astype(int).mean()
school_missing_share = (
    cleaned["qa_school_was_missing"].fillna(0).astype(float).mean()
    if "qa_school_was_missing" in cleaned.columns
    else float("nan")
)
undergraduate_share = (cleaned["parsed_level"] == "Undergraduate").mean()
issue_counts = summary.get("issue_counts", {})
top_issue = max(issue_counts.items(), key=lambda item: item[1]) if issue_counts else ("none", 0)

display(
    Markdown(
        f"""
### Live Dataset Insight

This dataset shows that **{award_rate:.1%}** of historically labeled cases are award-like and **{deny_rate:.1%}** are denied, while **{review_share:.1%}** of cleaned rows still require manual review. Undergraduate applications account for **{undergraduate_share:.1%}** of the portfolio, and **{school_missing_share:.1%}** of rows carry missing-school risk.

The most common logged issue is **`{top_issue[0]}`** with **{top_issue[1]}** occurrences. That means the dominant operational challenge is not simply file parsing; it is repeated uncertainty in a small set of important policy-relevant fields.
"""
    )
)

review_by_severity = (
    review_required.groupby("qa_max_severity", dropna=False)
    .size()
    .sort_values(ascending=False)
    .rename("rows")
    .reset_index()
)
display(review_by_severity)

# %% [notebook cell 7]
#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from zipfile import ZipFile

from faid_cleaning import scoring as scoring_utils


LOGGER = logging.getLogger("clean_faid_data")
RUNNING_IN_NOTEBOOK = "ipykernel" in sys.modules

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_ID_ATTR = f"{{{REL_NS}}}id"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT = PROJECT_ROOT / "data/raw/financial_aid_export.xlsx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "cleaned data"
DEFAULT_OUTPUT_STEM = "faid_cleaned"

DROP_SOURCE_COLUMNS = {
    "FAID_Term_rolled",
    "FAID_Record_owner",
    "Regular_FAID_Then_USAID_ADM",
    "AID Type",
}

REQUIRED_SOURCE_COLUMNS = {
    "Application Type",
    "LEVEL",
    "Application Term",
    "Nationality",
    "Applicant Demographic",
    "Father Work Status",
    "Father Income Document",
    "Father Job Info",
    "Mother Work Status",
    "Mother Income Document",
    "Mother Job Info",
    "Siblings At Institution",
    "Siblings Outside Institution",
    "Source of Income",
    "Dependents",
    "Investments",
    "Financial Assistants",
    "Special Family Circumstances",
    "Loans",
    "Properties",
    "Cars",
    "Travel Records Verification",
    "Certificate of Ownership Verification",
    "Applicant Work Status",
    "Applicant's Spouse Demographic",
    "Applicant's Spouse Work Status",
    "Decision",
    "Over and Above Decision",
    "Over and Above - Amount Awarded",
    "Over and Above - Percentage Awarded",
    "Submission_Date",
    "Need",
    "Need Comment",
    "Merit",
    "Merit Hist",
    "Merit Comment",
    "Consent to share information",
    "Bin",
    "FAID Missing Documents",
    "School",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fourty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
}

KNOWN_KEY_LABELS = (
    "Institution or Employer's Name",
    "Source of Income",
    "Plan to Reside",
    "Marital Status",
    "Educational Benefits",
    "Retirement Salary",
    "Sole Owner/Partner",
    "Unemployement Date",
    "Unemployment Date",
    "Other Benefits",
    "Annual Income",
    "Gross Income",
    "Net Income",
    "Ever Worked",
    "Citizenship",
    "Work Status",
    "Accommodation",
    "Accomodation",
    "Employed In",
    "Position",
    "Indemnity",
    "Bonuses",
    "Commission",
    "Institution",
    "Company",
    "Status",
    "From",
    "Name",
)

KEY_ALIASES = {
    "institution_or_employer_s_name": "institution_or_employer_name",
    "institution_or_employer_name": "institution_or_employer_name",
    "accomodation": "accommodation",
    "unemployement_date": "unemployment_date",
}

ISSUE_FIELDNAMES = ["source_row_number", "field_name", "issue_code", "raw_value", "detail"]
NUMBER_SCALE_MULTIPLIERS = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "thousands": 1_000.0,
    "mn": 1_000_000.0,
    "mm": 1_000_000.0,
    "million": 1_000_000.0,
    "millions": 1_000_000.0,
    "bn": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
    "billions": 1_000_000_000.0,
}
NUMBER_SCALE_PATTERN = r"k|thousand|thousands|mn|mm|million|millions|bn|billion|billions"

KEY_VALUE_LINE_PATTERN = re.compile(
    rf"^(?P<key>{'|'.join(re.escape(label) for label in sorted(KNOWN_KEY_LABELS, key=len, reverse=True))})\s*:?\s*(?P<value>.*)$",
    flags=re.IGNORECASE,
)
APPROXIMATE_AMOUNT_PATTERN = re.compile(r"(?i)(?:\+/-|±|plus\s*/\s*minus)\s*(?=\d)")
APPROXIMATE_WORD_PATTERN = re.compile(r"(?i)\b(?:about|around|approx(?:\.|imately)?)\b")
ADDITIVE_NUMBER_PATTERN = re.compile(r"\d[\d,]*(?:\.\d+)?\s*\+\s*(?:\$?\s*)?\d")
SCALED_NUMBER_PATTERN = re.compile(
    rf"(?P<number>-?\d[\d,]*(?:\.\d+)?)(?:\s*(?P<scale>{NUMBER_SCALE_PATTERN}))?\b",
    flags=re.IGNORECASE,
)
SCALED_RANGE_PATTERN = re.compile(
    rf"(?P<left>-?\d[\d,]*(?:\.\d+)?(?:\s*(?:{NUMBER_SCALE_PATTERN}))?)\s*[-/]\s*"
    rf"(?P<right>-?\d[\d,]*(?:\.\d+)?(?:\s*(?:{NUMBER_SCALE_PATTERN}))?)",
    flags=re.IGNORECASE,
)
MONEY_SCALE_WORD_PATTERN = re.compile(rf"\b(?:{NUMBER_SCALE_PATTERN})\b", flags=re.IGNORECASE)
DURATION_VALUE_PATTERN = re.compile(r"\b(?:years?|months?|weeks?|days?|quarters?|semesters?)\b", flags=re.IGNORECASE)
CADENCE_SEGMENT_PATTERN = re.compile(
    r"\b(?:every\s+\d+\s+months?|every\s+few\s+months?|weekly|daily|monthly|quarterly|quarter|semester|"
    r"seasonally|yearly|annual(?:ly)?|per\s+month)\b",
    flags=re.IGNORECASE,
)


@dataclass(slots=True)
class WorkbookRows:
    sheet_name: str
    headers: list[str]
    rows: list[dict[str, str]]
    source_row_numbers: list[int]


@dataclass(slots=True)
class OutputPaths:
    cleaned_csv: Path
    model_safe_csv: Path
    review_csv: Path
    financial_assistants_csv: Path
    loans_csv: Path
    properties_csv: Path
    cars_csv: Path
    issues_csv: Path
    data_dictionary_json: Path
    profile_json: Path
    summary_json: Path


@dataclass(slots=True)
class CleaningIssue:
    source_row_number: int
    field_name: str
    issue_code: str
    raw_value: str | None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class MonetaryQualityRule:
    cleaned_field_name: str
    raw_field_name: str
    high_threshold: float | None = None
    likely_lbp_threshold: float | None = None
    negative_not_allowed: bool = True


@dataclass(slots=True)
class StructuredTables:
    financial_assistants: list[dict[str, object]]
    loans: list[dict[str, object]]
    properties: list[dict[str, object]]
    cars: list[dict[str, object]]


@dataclass(slots=True)
class SectionParseResult:
    parsed: dict[str, str]
    duplicate_labels: list[str]
    unknown_labels: list[str]
    unlabeled_lines: list[str]


ISSUE_METADATA = scoring_utils.ISSUE_METADATA
CONFIDENCE_POLICY = scoring_utils.CONFIDENCE_POLICY

LOAN_FIELD_ALIASES = {
    "reason": {"reason"},
    "source": {"source"},
    "mortgaged_asset": {"mortgaged asset"},
    "start_date": {"start date"},
    "end_date": {"end date"},
    "total_amount": {"total amount"},
    "monthly_payment": {"monthly payment"},
    "remaining_balance": {"remaining balance"},
    "remaining_years": {"remaining years"},
}

PROPERTY_FIELD_ALIASES = {
    "type": {"type"},
    "location": {"location"},
    "area": {"area"},
    "estimated_present_value": {"estimated present value"},
    "mortgaged": {"mortgaged"},
    "rented": {"rented"},
    "planted_income": {"planted income"},
}

CAR_FIELD_ALIASES = {
    "type": {"type"},
    "model": {"model"},
    "mortgaged": {"mortgaged"},
}


class IssueTracker:
    def __init__(self) -> None:
        self._issues: list[CleaningIssue] = []

    def add(
        self,
        source_row_number: int,
        field_name: str,
        issue_code: str,
        raw_value: object | None,
        detail: object | None = None,
    ) -> None:
        self._issues.append(
            CleaningIssue(
                source_row_number=source_row_number,
                field_name=field_name,
                issue_code=issue_code,
                raw_value=normalize_text(raw_value) or None,
                detail=normalize_text(detail) or None,
            )
        )

    def as_rows(self) -> list[dict[str, object]]:
        return [asdict(issue) for issue in self._issues]

    def counts(self) -> Counter[str]:
        return Counter(issue.issue_code for issue in self._issues)

    def issue_counts_by_row(self) -> Counter[int]:
        return Counter(issue.source_row_number for issue in self._issues)

    def __len__(self) -> int:
        return len(self._issues)


MONETARY_QUALITY_RULES = (
    MonetaryQualityRule("father_income_document", "Father Income Document", high_threshold=1_000_000),
    MonetaryQualityRule("mother_income_document", "Mother Income Document", high_threshold=1_000_000),
    MonetaryQualityRule("father_annual_income_from_work_status", "Father Work Status", high_threshold=1_000_000),
    MonetaryQualityRule("mother_annual_income_from_work_status", "Mother Work Status", high_threshold=1_000_000),
    MonetaryQualityRule("applicant_annual_income", "Applicant Work Status", high_threshold=1_000_000),
    MonetaryQualityRule("spouse_annual_income", "Applicant's Spouse Work Status", high_threshold=1_000_000),
    MonetaryQualityRule("financial_assistants_total_est_annual_amount", "Financial Assistants", high_threshold=1_000_000),
    MonetaryQualityRule("loans_total_amount", "Loans", high_threshold=100_000_000),
    MonetaryQualityRule("loans_total_monthly_payment", "Loans", high_threshold=5_000_000),
    MonetaryQualityRule("loans_total_remaining_balance", "Loans", high_threshold=100_000_000),
    MonetaryQualityRule("properties_total_estimated_value", "Properties", high_threshold=100_000_000),
)


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\t", " ")
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def snake_case(value: str) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def canonical_key(value: str) -> str:
    key = snake_case(value)
    return KEY_ALIASES.get(key, key)


def to_int_flag(value: bool) -> int:
    return 1 if value else 0


def column_index(ref: str) -> int:
    match = re.match(r"([A-Z]+)", ref)
    if not match:
        return -1
    idx = 0
    for ch in match.group(1):
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1

# %% [notebook cell 9]
def read_xlsx_rows(path: Path, *, sheet_name: str | None = None) -> WorkbookRows:
    if not path.exists():
        raise FileNotFoundError(f"Input workbook does not exist: {path}")

    with ZipFile(path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", NS):
                text = "".join(t.text or "" for t in si.iterfind(".//a:t", NS))
                shared_strings.append(text)

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}

        sheets_node = workbook.find("a:sheets", NS)
        if sheets_node is None or not list(sheets_node):
            raise ValueError(f"No worksheets were found in {path}")

        target_sheet = None
        if sheet_name:
            for sheet in sheets_node:
                if normalize_text(sheet.attrib.get("name")) == normalize_text(sheet_name):
                    target_sheet = sheet
                    break
            if target_sheet is None:
                available = ", ".join(sheet.attrib.get("name", "<unknown>") for sheet in sheets_node)
                raise ValueError(f"Worksheet '{sheet_name}' was not found. Available sheets: {available}")
        else:
            target_sheet = sheets_node[0]

        selected_sheet_name = target_sheet.attrib.get("name", "Sheet1")
        rel_id = target_sheet.attrib[REL_ID_ATTR]
        worksheet = ET.fromstring(zf.read("xl/" + rel_map[rel_id]))

        def cell_value(cell: ET.Element) -> str:
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                is_el = cell.find("a:is", NS)
                if is_el is None:
                    return ""
                return "".join(t.text or "" for t in is_el.iterfind(".//a:t", NS))
            value = cell.find("a:v", NS)
            if value is None:
                return ""
            text = value.text or ""
            if cell_type == "s":
                try:
                    return shared_strings[int(text)]
                except (IndexError, ValueError):
                    return text
            return text

        sheet_data = worksheet.find("a:sheetData", NS)
        if sheet_data is None:
            return WorkbookRows(sheet_name=selected_sheet_name, headers=[], rows=[], source_row_numbers=[])

        xml_rows = sheet_data.findall("a:row", NS)
        if not xml_rows:
            return WorkbookRows(sheet_name=selected_sheet_name, headers=[], rows=[], source_row_numbers=[])

        header_map: dict[int, str] = {}
        for cell in xml_rows[0].findall("a:c", NS):
            idx = column_index(cell.attrib.get("r", ""))
            if idx >= 0:
                header_map[idx] = normalize_text(cell_value(cell))

        ordered_indices = sorted(header_map)
        ordered_headers = [header_map[idx] for idx in ordered_indices]

        out_rows: list[dict[str, str]] = []
        source_row_numbers: list[int] = []
        for default_row_number, row in enumerate(xml_rows[1:], start=2):
            sparse: dict[int, str] = {}
            for cell in row.findall("a:c", NS):
                idx = column_index(cell.attrib.get("r", ""))
                if idx >= 0:
                    sparse[idx] = normalize_text(cell_value(cell))
            out_rows.append({header_map[idx]: sparse.get(idx, "") for idx in ordered_indices})
            source_row_numbers.append(int(row.attrib.get("r", default_row_number)))

    return WorkbookRows(
        sheet_name=selected_sheet_name,
        headers=ordered_headers,
        rows=out_rows,
        source_row_numbers=source_row_numbers,
    )


def validate_source_columns(headers: list[str]) -> None:
    header_set = set(headers)
    missing = sorted(REQUIRED_SOURCE_COLUMNS - header_set)
    if missing:
        raise ValueError(f"Input workbook is missing required columns: {', '.join(missing)}")

    ignored = sorted(header_set - REQUIRED_SOURCE_COLUMNS - DROP_SOURCE_COLUMNS)
    if ignored:
        LOGGER.info("Found %d additional source columns that will be carried only if explicitly parsed.", len(ignored))


def parse_date(value: str) -> str | None:
    cleaned = normalize_text(value)
    if not cleaned:
        return None

    for fmt in (
        "%m/%d/%Y",
        "%m/%d/%y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
    ):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def parse_scaled_number_fragment(text: str) -> float | None:
    cleaned = normalize_numeric_text(text)
    if not cleaned:
        return None

    match = SCALED_NUMBER_PATTERN.search(cleaned)
    if not match:
        return word_number_to_float(cleaned)

    number = float(match.group("number").replace(",", ""))
    scale = match.group("scale")
    if scale:
        number *= NUMBER_SCALE_MULTIPLIERS[scale.lower()]
    return number


def extract_amount_candidates(text: str) -> list[tuple[float, int, int]]:
    cleaned = normalize_numeric_text(text)
    candidates: list[tuple[float, int, int]] = []
    for match in SCALED_NUMBER_PATTERN.finditer(cleaned):
        number = float(match.group("number").replace(",", ""))
        scale = match.group("scale")
        if scale:
            number *= NUMBER_SCALE_MULTIPLIERS[scale.lower()]
        candidates.append((number, match.start(), match.end()))
    return candidates


def extract_numeric_tokens(text: str) -> list[float]:
    return [value for value, _, _ in extract_amount_candidates(text)]


def max_numeric_token(text: str) -> float | None:
    numbers = extract_numeric_tokens(text)
    if not numbers:
        return None
    return max(numbers)


def normalize_numeric_text(text: str) -> str:
    cleaned = normalize_text(text)
    if not cleaned:
        return ""

    cleaned = APPROXIMATE_AMOUNT_PATTERN.sub("", cleaned)
    cleaned = APPROXIMATE_WORD_PATTERN.sub("", cleaned)
    cleaned = cleaned.replace("~", " ")
    return normalize_text(cleaned)


def detect_currency_signal(text: str) -> str | None:
    cleaned = normalize_text(text)
    if not cleaned:
        return None

    lowered = cleaned.lower()
    has_usd = "$" in cleaned or bool(re.search(r"\b(?:usd|us dollars?|dollars?)\b", lowered))
    has_lbp = bool(re.search(r"(?:\blbp\b|\blira\b|(?<![a-z])l\.?\s*l\.?(?![a-z]))", lowered))

    if has_usd and has_lbp:
        return "mixed"
    if has_usd:
        return "usd"
    if has_lbp:
        return "lbp"
    return None


def infer_currency_guess(
    text: str,
    numeric_value: float | None,
    *,
    likely_lbp_threshold: float | None = None,
) -> str | None:
    signal = detect_currency_signal(text)
    if signal is not None:
        return signal
    if numeric_value is None:
        return None
    if likely_lbp_threshold is not None and abs(numeric_value) >= likely_lbp_threshold:
        return "likely_lbp_unmarked"
    return "unknown"


def build_currency_profile(
    text: str,
    numeric_value: float | None,
    *,
    likely_lbp_threshold: float | None = None,
) -> dict[str, str | None]:
    return {
        "signal": detect_currency_signal(text),
        "guess": infer_currency_guess(text, numeric_value, likely_lbp_threshold=likely_lbp_threshold),
    }


def has_mixed_currency_markers(text: str) -> bool:
    return detect_currency_signal(text) == "mixed"


def word_number_to_float(text: str) -> float | None:
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return None

    matched = False
    total = 0
    current = 0
    for token in tokens:
        if token not in NUMBER_WORDS:
            continue
        matched = True
        value = NUMBER_WORDS[token]
        if token == "hundred":
            current = max(1, current) * 100
        elif token in {"thousand", "million"}:
            total += max(1, current) * value
            current = 0
        else:
            current += value

    if not matched:
        return None
    return float(total + current)


def choose_amount_strategy(text: str) -> Literal["first", "sum", "max"]:
    cleaned = normalize_numeric_text(text)
    if ADDITIVE_NUMBER_PATTERN.search(cleaned) and not has_mixed_currency_markers(cleaned):
        return "sum"
    return "first"


def parse_amount(text: str, *, strategy: Literal["first", "sum", "max"] = "first") -> float | None:
    cleaned = normalize_numeric_text(text)
    if not cleaned:
        return None

    range_match = SCALED_RANGE_PATTERN.search(cleaned)
    if range_match and strategy != "sum":
        left = parse_scaled_number_fragment(range_match.group("left"))
        right = parse_scaled_number_fragment(range_match.group("right"))
        if left is not None and right is not None:
            if strategy == "max":
                return max(left, right)
            return (left + right) / 2

    numbers = extract_numeric_tokens(cleaned)
    if numbers:
        if strategy == "sum":
            return sum(numbers)
        if strategy == "max":
            return max(numbers)
        return numbers[0]

    return word_number_to_float(cleaned)


def extract_labeled_values(text: str, label: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    pattern = re.compile(rf"(?im)^{re.escape(label)}\s*:?\s*(.+)$")
    return [normalize_text(match.group(1)) for match in pattern.finditer(cleaned)]


def looks_like_duration_only(value: str) -> bool:
    cleaned = normalize_text(value)
    if not cleaned:
        return False
    if not DURATION_VALUE_PATTERN.search(cleaned):
        return False
    if detect_currency_signal(cleaned):
        return False
    if "$" in cleaned:
        return False
    if MONEY_SCALE_WORD_PATTERN.search(cleaned):
        return False
    return True


def parse_income_document(value: str) -> tuple[float | None, str, int]:
    text = normalize_text(value)
    if not text:
        return None, "missing", 0

    lowered = text.lower()
    if "deceased" in lowered:
        return None, "deceased", 0
    if "no contact" in lowered:
        return None, "no_contact", 0
    if "missing" in lowered:
        return None, "missing", 0
    if "divorc" in lowered:
        return None, "divorced", 0

    amount = parse_amount(text, strategy=choose_amount_strategy(text))
    review_flag = 0
    if amount is not None and amount < 0:
        amount = None
        review_flag = 1
    if has_mixed_currency_markers(text):
        review_flag = 1

    status = "parsed_with_nssf_note" if "nssf" in lowered else "parsed"
    if amount is None:
        status = "unparsed_text"
        review_flag = 1

    return amount, status, review_flag


def parse_key_value_lines(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in normalize_text(text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = KEY_VALUE_LINE_PATTERN.match(line)
        if not match:
            continue
        key = canonical_key(match.group("key"))
        value = normalize_text(match.group("value"))
        if key and value and key not in parsed:
            parsed[key] = value
    return parsed


def parse_loose_key_value_lines(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw_line in normalize_text(text).split("\n"):
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key_clean = snake_case(key)
        value_clean = normalize_text(value)
        if key_clean and key_clean not in parsed:
            parsed[key_clean] = value_clean
    return parsed


def inspect_section_key_value_lines(text: str, field_aliases: dict[str, set[str]]) -> SectionParseResult:
    alias_lookup = {
        snake_case(alias): canonical_name
        for canonical_name, aliases in field_aliases.items()
        for alias in aliases
    }
    parsed: dict[str, str] = {}
    duplicate_labels: list[str] = []
    unknown_labels: list[str] = []
    unlabeled_lines: list[str] = []
    for raw_line in normalize_text(text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            unlabeled_lines.append(line)
            continue
        key, value = line.split(":", 1)
        canonical_name = alias_lookup.get(snake_case(key))
        value_clean = normalize_text(value)
        if canonical_name is None:
            unknown_labels.append(normalize_text(key))
            continue
        if not value_clean:
            continue
        if canonical_name in parsed:
            duplicate_labels.append(canonical_name)
            continue
        parsed[canonical_name] = value_clean
    return SectionParseResult(
        parsed=parsed,
        duplicate_labels=duplicate_labels,
        unknown_labels=unknown_labels,
        unlabeled_lines=unlabeled_lines,
    )


def parse_section_key_value_lines(text: str, field_aliases: dict[str, set[str]]) -> dict[str, str]:
    return inspect_section_key_value_lines(text, field_aliases).parsed


def split_repeating_blocks(text: str, start_pattern: str) -> list[str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    pattern = re.compile(start_pattern, flags=re.IGNORECASE | re.MULTILINE)
    matches = list(pattern.finditer(cleaned))
    if not matches:
        return [cleaned]

    block_starts = [match.start() for match in matches] + [len(cleaned)]
    blocks: list[str] = []
    for idx in range(len(matches)):
        block = cleaned[block_starts[idx] : block_starts[idx + 1]].strip()
        if block:
            blocks.append(block)
    return blocks

# %% [notebook cell 11]
def parse_demographic(text: str) -> dict[str, str | None]:
    data = parse_key_value_lines(text)
    return {
        "citizenship": data.get("citizenship"),
        "marital_status": data.get("marital_status"),
        "plan_to_reside": data.get("plan_to_reside"),
    }


def parse_work_status(text: str) -> dict[str, object]:
    cleaned = normalize_text(text)
    if not cleaned:
        return {
            "status": None,
            "employment_sector": None,
            "position": None,
            "employer_name": None,
            "annual_income": None,
            "benefits_total": None,
            "self_employed_name": None,
            "gross_income": None,
            "net_income": None,
            "ever_worked": None,
            "unemployment_date": None,
            "second_occupation_flag": 0,
        }

    data = parse_key_value_lines(cleaned)

    benefits_total = 0.0
    has_benefits = False
    for field_name in (
        "educational_benefits",
        "accommodation",
        "commission",
        "bonuses",
        "other_benefits",
        "indemnity",
        "retirement_salary",
    ):
        raw_value = data.get(field_name, "")
        amount = parse_amount(raw_value, strategy=choose_amount_strategy(raw_value))
        if amount is not None:
            benefits_total += amount
            has_benefits = True

    status = data.get("status")
    lowered = cleaned.lower()
    if not status:
        if "self-employed" in lowered or "self employed" in lowered:
            status = "Self-Employed"
        elif any(
            data.get(field_name)
            for field_name in ("employed_in", "position", "institution_or_employer_name", "company", "annual_income")
        ):
            status = "Employed"
        elif "unemployed" in lowered:
            status = "Unemployed"
        elif "previous employment" in lowered:
            status = "Previous Employment"

    annual_income_text = data.get("annual_income", "")
    return {
        "status": status,
        "employment_sector": data.get("employed_in") or data.get("from"),
        "position": data.get("position"),
        "employer_name": data.get("institution_or_employer_name") or data.get("institution") or data.get("company"),
        "annual_income": parse_amount(annual_income_text, strategy=choose_amount_strategy(annual_income_text)),
        "benefits_total": benefits_total if has_benefits else None,
        "self_employed_name": data.get("name"),
        "gross_income": parse_amount(data.get("gross_income", "")),
        "net_income": parse_amount(data.get("net_income", "")),
        "ever_worked": data.get("ever_worked"),
        "unemployment_date": parse_date(data.get("unemployment_date", "")),
        "second_occupation_flag": to_int_flag("second occupation" in lowered),
    }


def categorize_job_info(text: str) -> tuple[str | None, str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return None, "missing"

    lowered = cleaned.lower()
    if "deceased" in lowered:
        return cleaned, "deceased"
    if "no contact" in lowered:
        return cleaned, "no_contact"
    if "nssf" in lowered:
        return cleaned, "nssf"
    if "retir" in lowered:
        return cleaned, "retired"
    if "tax booklet" in lowered:
        return cleaned, "tax_booklet"
    if "resident" in lowered or "residency" in lowered or "ikama" in lowered:
        return cleaned, "residency_document"
    if "housewife" in lowered:
        return cleaned, "housewife"
    if "army" in lowered:
        return cleaned, "military"
    return cleaned, "other"


def count_pattern(text: str, pattern: str) -> int:
    return len(re.findall(pattern, normalize_text(text), flags=re.IGNORECASE | re.MULTILINE))


def sum_labeled_amounts(text: str, label: str, *, reject_duration_values: bool = False) -> float | None:
    total = 0.0
    found = False
    for raw_value in extract_labeled_values(text, label):
        if reject_duration_values and looks_like_duration_only(raw_value):
            continue
        amount = parse_amount(raw_value, strategy=choose_amount_strategy(raw_value))
        if amount is not None:
            total += amount
            found = True
    return total if found else None


def parse_siblings_not_aub(text: str) -> dict[str, object]:
    cleaned = normalize_text(text)
    return {
        "count": count_pattern(cleaned, r"Sibling\s+\d+\s*:"),
        "total_tuition": sum_labeled_amounts(cleaned, "Annual Tuition"),
        "total_financial_assistance": sum_labeled_amounts(cleaned, "Financial Assistance from school/university"),
    }


def parse_dependents(text: str) -> int:
    cleaned = normalize_text(text)
    if not cleaned:
        return 0

    structured_count = count_pattern(cleaned, r"Dependent\s+\d+\s*:")
    if structured_count:
        return structured_count
    return len([line for line in cleaned.split("\n") if line.strip()])


def parse_investments(text: str) -> dict[str, object]:
    cleaned = normalize_text(text)
    return {
        "count": count_pattern(cleaned, r"Investment\s+\d+\s*:"),
        "total_annual_profit": sum_labeled_amounts(cleaned, "Annual Profit"),
    }


def estimate_frequency_multiplier(text: str) -> float:
    lowered = normalize_text(text).lower()

    every_month_match = re.search(r"every\s+(\d+)\s+months?", lowered)
    if every_month_match:
        months = max(1, int(every_month_match.group(1)))
        return 12 / months

    if "weekly" in lowered:
        return 52
    if "daily" in lowered:
        return 365
    if "monthly" in lowered or "per month" in lowered:
        return 12
    if "quarter" in lowered:
        return 4
    if "semester" in lowered:
        return 2
    if "seasonal" in lowered:
        return 4
    if "yearly" in lowered or "annual" in lowered:
        return 1
    if "every few months" in lowered:
        return 4
    return 1


def select_assistant_amount_text(text: str) -> str:
    cleaned = normalize_numeric_text(text)
    if not cleaned:
        return ""

    matches = list(CADENCE_SEGMENT_PATTERN.finditer(cleaned))
    if not matches:
        return cleaned

    match = matches[-1]
    after = cleaned[match.end() :].strip(" :-,")
    before = cleaned[: match.start()].strip(" :-,")

    if extract_amount_candidates(after):
        return after
    if extract_amount_candidates(before):
        return before
    return cleaned


def resolve_assistant_amount_text(text: str) -> tuple[str, list[str]]:
    cleaned = normalize_numeric_text(text)
    if not cleaned:
        return "", []

    review_reasons: list[str] = []
    matches = list(CADENCE_SEGMENT_PATTERN.finditer(cleaned))
    if matches:
        match = matches[-1]
        after = cleaned[match.end() :].strip(" :-,")
        before = cleaned[: match.start()].strip(" :-,")
        after_candidates = extract_amount_candidates(after)
        before_candidates = extract_amount_candidates(before)
        if after_candidates and before_candidates:
            review_reasons.append("ambiguous_amount_location")
            return "", review_reasons
        if after_candidates:
            return after, review_reasons
        if before_candidates:
            return before, review_reasons

    amount_text = select_assistant_amount_text(cleaned)
    numeric_candidates = extract_amount_candidates(amount_text)
    if (
        len(numeric_candidates) > 1
        and choose_amount_strategy(amount_text) == "first"
        and not SCALED_RANGE_PATTERN.search(amount_text)
        and not ADDITIVE_NUMBER_PATTERN.search(amount_text)
    ):
        review_reasons.append("multiple_amount_candidates")
        return "", review_reasons
    return amount_text, review_reasons


def extract_assistant_subject_text(text: str) -> str | None:
    cleaned = normalize_text(text)
    if not cleaned:
        return None

    amount_text = select_assistant_amount_text(cleaned)
    subject = cleaned
    if amount_text:
        subject = cleaned.replace(amount_text, " ", 1)
    subject = CADENCE_SEGMENT_PATTERN.sub(" ", subject)
    subject = re.sub(r"\d[\d,]*(?:\.\d+)?", " ", subject)
    subject = normalize_text(subject.strip(" -,:;")) or None
    return subject


def summarize_review_reasons(reasons: list[str]) -> str | None:
    if not reasons:
        return None
    ordered_unique = list(dict.fromkeys(reasons))
    return "; ".join(ordered_unique)


def append_section_parse_review_reasons(
    review_reasons: list[str],
    section_result: SectionParseResult,
    *,
    anchor_present: bool,
    critical_fields: set[str] | None = None,
) -> None:
    if not anchor_present:
        review_reasons.append("missing_anchor")
    if section_result.unlabeled_lines:
        review_reasons.append("unlabeled_lines")
    if section_result.unknown_labels:
        review_reasons.append("unknown_labels")
    if section_result.duplicate_labels:
        review_reasons.append("duplicate_labels")
    if critical_fields:
        duplicated_critical = sorted(set(section_result.duplicate_labels) & critical_fields)
        for field_name in duplicated_critical:
            review_reasons.append(f"duplicate_{field_name}")

# %% [notebook cell 13]
def extract_financial_assistant_records(text: str) -> list[dict[str, object]]:
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    lines = [line.strip(" |") for line in cleaned.split("\n") if line.strip(" |")]
    records: list[dict[str, object]] = []
    for record_index, line in enumerate(lines, start=1):
        amount_text, parsing_reasons = resolve_assistant_amount_text(line)
        subject = extract_assistant_subject_text(line)
        amount_per_occurrence = parse_amount(amount_text, strategy=choose_amount_strategy(amount_text))
        if amount_per_occurrence is None:
            amount_per_occurrence = max_numeric_token(amount_text)

        frequency_multiplier = estimate_frequency_multiplier(line)
        annualized_amount = amount_per_occurrence * frequency_multiplier if amount_per_occurrence is not None else None
        currency = build_currency_profile(
            line,
            annualized_amount if annualized_amount is not None else amount_per_occurrence,
            likely_lbp_threshold=1_000_000,
        )

        review_reasons: list[str] = []
        review_reasons.extend(parsing_reasons)
        if amount_per_occurrence is None:
            review_reasons.append("amount_unparsed")
        if currency["signal"] == "mixed":
            review_reasons.append("mixed_currency")

        records.append(
            {
                "record_index": record_index,
                "assistant_raw_line": line,
                "assistant_subject": subject,
                "assistant_amount_text": amount_text or None,
                "assistant_amount_per_occurrence": amount_per_occurrence,
                "assistant_frequency_multiplier": frequency_multiplier,
                "assistant_estimated_annual_amount": annualized_amount,
                "assistant_currency_signal": currency["signal"],
                "assistant_currency_guess": currency["guess"],
                "assistant_parse_method": "line_amount_then_cadence",
                "assistant_review_flag": to_int_flag(bool(review_reasons)),
                "assistant_review_reasons": summarize_review_reasons(review_reasons),
            }
        )
    return records


def parse_financial_assistants(text: str) -> dict[str, object]:
    records = extract_financial_assistant_records(text)
    annualized_amounts = [
        float(record["assistant_estimated_annual_amount"])
        for record in records
        if record["assistant_estimated_annual_amount"] is not None
    ]
    return {"count": len(records), "total_est_annual_amount": sum(annualized_amounts) if annualized_amounts else None}


def extract_loan_records(text: str) -> list[dict[str, object]]:
    cleaned = normalize_text(text)
    blocks = split_repeating_blocks(cleaned, r"^Reason\s*:")
    anchor_present = bool(re.search(r"(?im)^Reason\s*:", cleaned))
    records: list[dict[str, object]] = []
    for record_index, block in enumerate(blocks, start=1):
        section_result = inspect_section_key_value_lines(block, LOAN_FIELD_ALIASES)
        data = section_result.parsed

        total_amount_raw = data.get("total_amount")
        monthly_payment_raw = data.get("monthly_payment")
        remaining_balance_raw = data.get("remaining_balance")
        remaining_years_raw = data.get("remaining_years")

        if "total_amount" in section_result.duplicate_labels:
            total_amount_raw = None
        if "monthly_payment" in section_result.duplicate_labels:
            monthly_payment_raw = None
        if "remaining_balance" in section_result.duplicate_labels:
            remaining_balance_raw = None
        if "remaining_years" in section_result.duplicate_labels:
            remaining_years_raw = None

        total_amount = (
            parse_amount(total_amount_raw, strategy=choose_amount_strategy(total_amount_raw))
            if total_amount_raw
            else None
        )
        monthly_payment = (
            parse_amount(monthly_payment_raw, strategy=choose_amount_strategy(monthly_payment_raw))
            if monthly_payment_raw and not looks_like_duration_only(monthly_payment_raw)
            else None
        )
        remaining_balance_is_duration = looks_like_duration_only(remaining_balance_raw or "")
        remaining_balance = (
            None
            if remaining_balance_is_duration or not remaining_balance_raw
            else parse_amount(remaining_balance_raw, strategy=choose_amount_strategy(remaining_balance_raw))
        )
        remaining_years = max_numeric_token(remaining_years_raw or "")

        total_amount_currency = build_currency_profile(
            total_amount_raw or block,
            total_amount,
            likely_lbp_threshold=100_000_000,
        )

        review_reasons: list[str] = []
        append_section_parse_review_reasons(
            review_reasons,
            section_result,
            anchor_present=anchor_present and block.lstrip().lower().startswith("reason"),
            critical_fields={"total_amount", "monthly_payment", "remaining_balance", "remaining_years"},
        )
        if detect_currency_signal(block) == "mixed":
            review_reasons.append("mixed_currency")
        if total_amount_raw and total_amount is None:
            review_reasons.append("total_amount_unparsed")
        if monthly_payment_raw and monthly_payment is None and not looks_like_duration_only(monthly_payment_raw):
            review_reasons.append("monthly_payment_unparsed")
        if remaining_balance_is_duration:
            review_reasons.append("remaining_balance_non_monetary_text")
        elif remaining_balance_raw and remaining_balance is None:
            review_reasons.append("remaining_balance_unparsed")
        if remaining_years_raw and remaining_years is None:
            review_reasons.append("remaining_years_unparsed")

        records.append(
            {
                "record_index": record_index,
                "loan_reason": data.get("reason"),
                "loan_source": data.get("source"),
                "loan_mortgaged_asset": data.get("mortgaged_asset"),
                "loan_start_date": parse_date(data.get("start_date", "")),
                "loan_end_date": parse_date(data.get("end_date", "")),
                "loan_total_amount_raw": total_amount_raw,
                "loan_total_amount": total_amount,
                "loan_total_amount_currency_signal": total_amount_currency["signal"],
                "loan_total_amount_currency_guess": total_amount_currency["guess"],
                "loan_monthly_payment_raw": monthly_payment_raw,
                "loan_monthly_payment": monthly_payment,
                "loan_remaining_balance_raw": remaining_balance_raw,
                "loan_remaining_balance": remaining_balance,
                "loan_remaining_balance_is_duration_flag": to_int_flag(remaining_balance_is_duration),
                "loan_remaining_years_raw": remaining_years_raw,
                "loan_remaining_years": remaining_years,
                "loan_parse_method": "loan_section_key_value",
                "loan_review_flag": to_int_flag(bool(review_reasons)),
                "loan_review_reasons": summarize_review_reasons(review_reasons),
                "loan_raw_block": block,
            }
        )
    return records


def parse_loans(text: str) -> dict[str, object]:
    records = extract_loan_records(text)
    total_amounts = [float(record["loan_total_amount"]) for record in records if record["loan_total_amount"] is not None]
    monthly_payments = [
        float(record["loan_monthly_payment"]) for record in records if record["loan_monthly_payment"] is not None
    ]
    remaining_balances = [
        float(record["loan_remaining_balance"]) for record in records if record["loan_remaining_balance"] is not None
    ]
    remaining_years = [float(record["loan_remaining_years"]) for record in records if record["loan_remaining_years"] is not None]
    return {
        "count": len(records),
        "total_amount": sum(total_amounts) if total_amounts else None,
        "total_monthly_payment": sum(monthly_payments) if monthly_payments else None,
        "total_remaining_balance": sum(remaining_balances) if remaining_balances else None,
        "max_remaining_years": max(remaining_years) if remaining_years else None,
    }


def detect_loan_record_issues(records: list[dict[str, object]]) -> list[str]:
    issues: list[str] = []
    for record in records:
        total_amount = record.get("loan_total_amount")
        remaining_balance = record.get("loan_remaining_balance")
        if total_amount is not None and remaining_balance is not None and float(remaining_balance) > float(total_amount):
            issues.append("loan_balance_exceeds_total_amount")
    return issues


def extract_property_records(text: str) -> list[dict[str, object]]:
    cleaned = normalize_text(text)
    blocks = split_repeating_blocks(cleaned, r"^Type\s*:")
    anchor_present = bool(re.search(r"(?im)^Type\s*:", cleaned))
    records: list[dict[str, object]] = []
    for record_index, block in enumerate(blocks, start=1):
        section_result = inspect_section_key_value_lines(block, PROPERTY_FIELD_ALIASES)
        data = section_result.parsed

        area_raw = data.get("area")
        estimated_value_raw = data.get("estimated_present_value")
        rented_income_raw = data.get("rented")
        planted_income_raw = data.get("planted_income")

        if "estimated_present_value" in section_result.duplicate_labels:
            estimated_value_raw = None
        if "area" in section_result.duplicate_labels:
            area_raw = None
        if "rented" in section_result.duplicate_labels:
            rented_income_raw = None
        if "planted_income" in section_result.duplicate_labels:
            planted_income_raw = None

        area = parse_amount(area_raw, strategy=choose_amount_strategy(area_raw)) if area_raw else None
        estimated_value = (
            parse_amount(estimated_value_raw, strategy=choose_amount_strategy(estimated_value_raw))
            if estimated_value_raw
            else None
        )
        rented_income = (
            parse_amount(rented_income_raw, strategy=choose_amount_strategy(rented_income_raw))
            if rented_income_raw and not looks_like_duration_only(rented_income_raw)
            else None
        )
        planted_income = (
            parse_amount(planted_income_raw, strategy=choose_amount_strategy(planted_income_raw))
            if planted_income_raw and not looks_like_duration_only(planted_income_raw)
            else None
        )

        mortgaged_value = clean_yes_no(data.get("mortgaged", "")) or (normalize_text(data.get("mortgaged", "")) or None)
        estimated_value_currency = build_currency_profile(
            estimated_value_raw or block,
            estimated_value,
            likely_lbp_threshold=100_000_000,
        )

        review_reasons: list[str] = []
        append_section_parse_review_reasons(
            review_reasons,
            section_result,
            anchor_present=anchor_present and block.lstrip().lower().startswith("type"),
            critical_fields={"area", "estimated_present_value", "rented", "planted_income"},
        )
        if detect_currency_signal(block) == "mixed":
            review_reasons.append("mixed_currency")
        if estimated_value_raw and estimated_value is None:
            review_reasons.append("estimated_value_unparsed")

        records.append(
            {
                "record_index": record_index,
                "property_type": data.get("type"),
                "property_location": data.get("location"),
                "property_lot_section_block_number": data.get("lot_section_block_number"),
                "property_owned_shares": data.get("owned_shares"),
                "property_inherited": data.get("inherited"),
                "property_owned": data.get("owned"),
                "property_area_raw": area_raw,
                "property_area": area,
                "property_estimated_present_value_raw": estimated_value_raw,
                "property_estimated_present_value": estimated_value,
                "property_estimated_value_currency_signal": estimated_value_currency["signal"],
                "property_estimated_value_currency_guess": estimated_value_currency["guess"],
                "property_mortgaged": mortgaged_value,
                "property_is_mortgaged_flag": to_int_flag(mortgaged_value == "yes"),
                "property_rented_income_raw": rented_income_raw,
                "property_rented_income": rented_income,
                "property_planted_income_raw": planted_income_raw,
                "property_planted_income": planted_income,
                "property_parse_method": "property_section_key_value",
                "property_review_flag": to_int_flag(bool(review_reasons)),
                "property_review_reasons": summarize_review_reasons(review_reasons),
                "property_raw_block": block,
            }
        )
    return records


def parse_properties(text: str) -> dict[str, object]:
    records = extract_property_records(text)
    property_types = sorted(
        {
            normalize_text(record["property_type"]).lower()
            for record in records
            if normalize_text(record["property_type"])
        }
    )
    total_areas = [float(record["property_area"]) for record in records if record["property_area"] is not None]
    estimated_values = [
        float(record["property_estimated_present_value"])
        for record in records
        if record["property_estimated_present_value"] is not None
    ]
    rented_incomes = [
        float(record["property_rented_income"]) for record in records if record["property_rented_income"] is not None
    ]
    planted_incomes = [
        float(record["property_planted_income"]) for record in records if record["property_planted_income"] is not None
    ]
    return {
        "count": len(records),
        "property_types": "; ".join(property_types) if property_types else None,
        "total_area": sum(total_areas) if total_areas else None,
        "total_estimated_value": sum(estimated_values) if estimated_values else None,
        "mortgaged_count": sum(int(record["property_is_mortgaged_flag"]) for record in records),
        "rented_income_total": sum(rented_incomes) if rented_incomes else None,
        "planted_income_total": sum(planted_incomes) if planted_incomes else None,
    }


def extract_car_records(text: str) -> list[dict[str, object]]:
    cleaned = normalize_text(text)
    blocks = split_repeating_blocks(cleaned, r"^Type\s*:")
    anchor_present = bool(re.search(r"(?im)^Type\s*:", cleaned))
    records: list[dict[str, object]] = []
    for record_index, block in enumerate(blocks, start=1):
        section_result = inspect_section_key_value_lines(block, CAR_FIELD_ALIASES)
        data = section_result.parsed
        mortgaged_value = clean_yes_no(data.get("mortgaged", "")) or (normalize_text(data.get("mortgaged", "")) or None)
        review_reasons: list[str] = []
        append_section_parse_review_reasons(
            review_reasons,
            section_result,
            anchor_present=anchor_present and block.lstrip().lower().startswith("type"),
            critical_fields={"type", "model", "mortgaged"},
        )
        records.append(
            {
                "record_index": record_index,
                "car_type": data.get("type"),
                "car_model": data.get("model"),
                "car_mortgaged": mortgaged_value,
                "car_is_mortgaged_flag": to_int_flag(mortgaged_value == "yes"),
                "car_parse_method": "car_section_key_value",
                "car_review_flag": to_int_flag(bool(review_reasons)),
                "car_review_reasons": summarize_review_reasons(review_reasons),
                "car_raw_block": block,
            }
        )
    return records


def parse_cars(text: str) -> dict[str, object]:
    records = extract_car_records(text)
    models = [normalize_text(record["car_model"]) for record in records if normalize_text(record["car_model"])]
    return {
        "count": len(records),
        "models": "; ".join(models) if models else None,
        "mortgaged_count": sum(int(record["car_is_mortgaged_flag"]) for record in records),
    }


def parse_travel_records(text: str) -> tuple[str | None, str, int]:
    cleaned = normalize_text(text)
    if not cleaned:
        return None, "missing", 1

    lowered = cleaned.lower()
    severity = [
        ("very_high", ["very high"]),
        ("high", ["high"]),
        ("average", ["average", "moderate"]),
        ("low", ["low", "minimal"]),
        ("none", ["0", "zero", "no travel", "no travel history"]),
    ]

    category = "other"
    for label, keywords in severity:
        if any(keyword in lowered for keyword in keywords):
            category = label
            break

    if "receipt" in lowered and category == "other":
        category = "receipt_only"

    missing_flag = to_int_flag("missing" in lowered)
    return cleaned, category, missing_flag


def parse_ownership_verification(text: str) -> tuple[str | None, str]:
    cleaned = normalize_text(text)
    if not cleaned:
        return None, "missing"

    lowered = cleaned.lower()
    if lowered == "missing":
        return cleaned, "missing"
    if "fcsr" in lowered:
        return cleaned, "fcsr"
    if re.search(r"\b0\b", lowered) or "zero" in lowered:
        return cleaned, "zero_ownership"
    if "correct" in lowered:
        return cleaned, "verified"
    return cleaned, "reported_ownership"


def parse_source_of_income(text: str) -> dict[str, object]:
    cleaned = normalize_text(text)
    if not cleaned:
        return {
            "type": "not_applicable",
            "work_status": None,
            "has_record": 0,
        }

    data = parse_key_value_lines(cleaned)
    source_type = normalize_text(data.get("source_of_income", "")).lower() or "not_applicable"
    work_status = normalize_text(data.get("work_status", "")).lower() or None
    return {
        "type": source_type,
        "work_status": work_status,
        "has_record": 1,
    }


def clean_special_family_circumstances(text: str) -> tuple[str, str | None]:
    cleaned = normalize_text(text)
    if not cleaned:
        return "none", None

    first_line, *rest = cleaned.split("\n", 1)
    category = ", ".join(part.strip().lower() for part in first_line.split(",") if part.strip())
    return category or "other", rest[0] if rest else None


def employment_status_flag(status: object) -> int:
    cleaned = normalize_text(status)
    lowered = cleaned.lower()
    employed_markers = ("employed", "self-employed", "self employed")
    unemployed_markers = ("unemployed", "retired", "previous employment")
    if any(marker in lowered for marker in unemployed_markers):
        return 0
    return to_int_flag(any(marker in lowered for marker in employed_markers))


def applicant_employment_flag(work_status: dict[str, object]) -> int:
    if employment_status_flag(work_status.get("status")):
        return 1
    has_employment_fields = any(
        work_status.get(field_name)
        for field_name in ("employment_sector", "position", "employer_name", "annual_income")
    )
    return to_int_flag(bool(has_employment_fields))

# %% [notebook cell 15]
def clean_application_type(level: str, application_type: str, merit_pct: float | None) -> dict[str, object]:
    level_clean = normalize_text(level)
    raw = normalize_text(application_type)
    inferred_flag = to_int_flag(not raw)
    inconsistent_flag = 0

    if not raw and level_clean == "Graduate":
        cleaned = "Regular Graduate Application"
    elif not raw and level_clean == "Professional":
        cleaned = "Regular Professional Application"
    elif not raw and level_clean == "Undergraduate":
        cleaned = "Unknown Undergraduate Application"
    elif not raw:
        cleaned = "Unknown Application"
    else:
        cleaned = raw

    lowered = cleaned.lower()
    if level_clean in {"Graduate", "Professional"} and "early merit" in lowered:
        inconsistent_flag = 1

    if "early merit" in lowered:
        track = "early_merit"
    elif "transfer" in lowered:
        track = "transfer"
    elif "graduate" in lowered:
        track = "graduate_regular"
    elif "professional" in lowered:
        track = "professional_regular"
    elif any(keyword in lowered for keyword in ("usp", "rss", "scholarship program")):
        track = "special_program"
    elif "freshman" in lowered:
        track = "freshman_regular"
    elif "sophomore" in lowered:
        track = "sophomore_regular"
    else:
        track = "other"

    merit = merit_pct or 0.0
    remaining_cap = max(50.0 - merit, 0.0)
    return {
        "application_type_clean": cleaned,
        "application_track": track,
        "application_type_inferred_flag": inferred_flag,
        "application_has_early_merit": to_int_flag("early merit" in lowered),
        "application_type_inconsistent_flag": inconsistent_flag,
        "remaining_cap_50_rule_pct": remaining_cap,
    }


def clean_decision(value: str) -> str | None:
    cleaned = normalize_text(value)
    return snake_case(cleaned) if cleaned else None


def clean_yes_no(value: str) -> str | None:
    cleaned = normalize_text(value).lower()
    if cleaned in {"yes", "y", "true", "1"}:
        return "yes"
    if cleaned in {"no", "n", "false", "0"}:
        return "no"
    return None if not cleaned else cleaned


def count_missing_documents(value: str) -> int:
    cleaned = normalize_text(value)
    if not cleaned:
        return 0
    return len([line for line in cleaned.split("\n") if line.strip()])


def add_monetary_quality_issues(
    *,
    cleaned_row: dict[str, object],
    raw_row: dict[str, str],
    source_row_number: int,
    issues: IssueTracker,
) -> None:
    for rule in MONETARY_QUALITY_RULES:
        raw_value = raw_row.get(rule.raw_field_name, "")
        value = cleaned_row.get(rule.cleaned_field_name)
        if value in ("", None):
            continue

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        currency_signal = detect_currency_signal(raw_value) or "none"
        currency_guess = infer_currency_guess(
            raw_value,
            numeric_value,
            likely_lbp_threshold=rule.likely_lbp_threshold or rule.high_threshold,
        ) or "unknown"
        detail = (
            f"{rule.cleaned_field_name}={numeric_value}; "
            f"currency_signal={currency_signal}; "
            f"currency_guess={currency_guess}"
        )

        if rule.negative_not_allowed and numeric_value < 0:
            issues.add(
                source_row_number,
                rule.raw_field_name,
                "negative_monetary_value",
                raw_value,
                detail,
            )

        if rule.high_threshold is not None and abs(numeric_value) >= rule.high_threshold:
            if currency_signal == "none" and currency_guess == "likely_lbp_unmarked":
                issue_code = "likely_lbp_without_currency_marker"
            elif currency_signal == "none":
                issue_code = "large_monetary_value_without_currency_marker"
            else:
                issue_code = "large_monetary_value_review"
            issues.add(
                source_row_number,
                rule.raw_field_name,
                issue_code,
                raw_value,
                detail,
            )


def add_issue_batch(
    issues: IssueTracker,
    *,
    source_row_number: int,
    field_name: str,
    issue_codes: list[str],
    raw_value: object | None,
    detail: object | None = None,
) -> None:
    for issue_code in issue_codes:
        issues.add(
            source_row_number,
            field_name,
            issue_code,
            raw_value,
            detail,
        )


def issue_severity_score(issue_code: str) -> int:
    return scoring_utils.issue_severity_score(issue_code)


def issue_risk_category(issue_code: str) -> str:
    return scoring_utils.issue_risk_category(issue_code)


def summarize_issue_severity(issue_codes: list[str]) -> tuple[str, int, list[str]]:
    return scoring_utils.summarize_issue_severity(issue_codes)


def field_profile_type(values: list[object]) -> str:
    non_null_values = [value for value in values if value not in ("", None)]
    if not non_null_values:
        return "null"
    if all(isinstance(value, bool) for value in non_null_values):
        return "bool"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in non_null_values):
        return "int"
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in non_null_values):
        return "float"
    if all(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) for value in non_null_values):
        return "date"
    return "string"


def build_column_profile(rows: list[dict[str, object]], field_name: str) -> dict[str, object]:
    values = [row.get(field_name) for row in rows]
    non_null_values = [value for value in values if value not in ("", None)]
    inferred_type = field_profile_type(values)
    distinct_values = len({json.dumps(value, sort_keys=True, default=str) for value in non_null_values})

    profile: dict[str, object] = {
        "name": field_name,
        "inferred_type": inferred_type,
        "non_null_count": len(non_null_values),
        "null_count": len(values) - len(non_null_values),
        "distinct_count": distinct_values,
    }

    if inferred_type in {"int", "float"}:
        numeric_values = [float(value) for value in non_null_values]
        profile["min"] = min(numeric_values)
        profile["max"] = max(numeric_values)
        profile["mean"] = round(sum(numeric_values) / len(numeric_values), 4)
    else:
        value_counter = Counter(normalize_text(value) for value in non_null_values if normalize_text(value))
        profile["top_values"] = dict(value_counter.most_common(5))

    examples: list[object] = []
    for value in non_null_values:
        if value not in examples:
            examples.append(value)
        if len(examples) == 3:
            break
    profile["sample_values"] = examples
    return profile


def build_table_profile(table_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    fieldnames = list(rows[0].keys()) if rows else []
    return {
        "table_name": table_name,
        "row_count": len(rows),
        "column_count": len(fieldnames),
        "columns": [build_column_profile(rows, field_name) for field_name in fieldnames],
    }


def build_profile_payload(
    *,
    input_path: Path,
    workbook: WorkbookRows,
    cleaned_rows: list[dict[str, object]],
    review_rows: list[dict[str, object]],
    model_safe_rows: list[dict[str, object]],
    structured_tables: StructuredTables,
) -> dict[str, object]:
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_file": str(input_path.resolve()),
        "source_sheet_name": workbook.sheet_name,
        "tables": {
            "cleaned": build_table_profile("cleaned", cleaned_rows),
            "review_required": build_table_profile("review_required", review_rows),
            "model_safe": build_table_profile("model_safe", model_safe_rows),
            "financial_assistants": build_table_profile("financial_assistants", structured_tables.financial_assistants),
            "loans": build_table_profile("loans", structured_tables.loans),
            "properties": build_table_profile("properties", structured_tables.properties),
            "cars": build_table_profile("cars", structured_tables.cars),
        },
    }


def clamp_confidence(score: float) -> float:
    return scoring_utils.clamp_confidence(score)


def monetary_field_confidence(
    raw_text: str,
    numeric_value: float | None,
    *,
    parser_status: str | None = None,
    review_flag: bool = False,
    currency_signal: str | None = None,
    currency_guess: str | None = None,
    deterministic_bonus: float = 0.0,
    hard_fail: bool = False,
) -> float:
    return scoring_utils.monetary_field_confidence(
        raw_text,
        numeric_value,
        parser_status=parser_status,
        review_flag=review_flag,
        currency_signal=currency_signal,
        currency_guess=currency_guess,
        deterministic_bonus=deterministic_bonus,
        hard_fail=hard_fail,
    )


def categorical_field_confidence(
    raw_text: str,
    parsed_value: object | None,
    *,
    inferred: bool = False,
    missing_penalty: float = 1.0,
) -> float:
    return scoring_utils.categorical_field_confidence(
        raw_text,
        parsed_value,
        inferred=inferred,
        missing_penalty=missing_penalty,
    )


def status_implies_no_income(status: object | None) -> bool:
    lowered = normalize_text(status).lower()
    return lowered in {"unemployed", "retired"} or "previous employment" in lowered


def build_cleaned_row(
    row: dict[str, str],
    *,
    source_row_number: int,
    source_sheet_name: str,
    school_mode: str | None,
    impute_missing_school_from_mode: bool,
    issues: IssueTracker,
) -> dict[str, object]:
    level = normalize_text(row.get("LEVEL", ""))
    raw_application_type = normalize_text(row.get("Application Type", "")) or None
    raw_father_income_document = normalize_text(row.get("Father Income Document", "")) or None
    raw_mother_income_document = normalize_text(row.get("Mother Income Document", "")) or None
    raw_father_work_status = normalize_text(row.get("Father Work Status", "")) or None
    raw_mother_work_status = normalize_text(row.get("Mother Work Status", "")) or None
    raw_applicant_work_status = normalize_text(row.get("Applicant Work Status", "")) or None
    raw_spouse_work_status = normalize_text(row.get("Applicant's Spouse Work Status", "")) or None
    raw_financial_assistants = normalize_text(row.get("Financial Assistants", "")) or None
    raw_loans = normalize_text(row.get("Loans", "")) or None
    raw_properties = normalize_text(row.get("Properties", "")) or None
    raw_cars = normalize_text(row.get("Cars", "")) or None
    raw_decision = normalize_text(row.get("Decision", "")) or None
    raw_bin = normalize_text(row.get("Bin", "")) or None
    raw_over_and_above_decision = normalize_text(row.get("Over and Above Decision", "")) or None

    applicant_demo = parse_demographic(row.get("Applicant Demographic", ""))
    spouse_demo = parse_demographic(row.get("Applicant's Spouse Demographic", ""))

    father_income, father_income_status, father_income_review_flag = parse_income_document(
        row.get("Father Income Document", "")
    )
    mother_income, mother_income_status, mother_income_review_flag = parse_income_document(
        row.get("Mother Income Document", "")
    )

    need_pct = parse_amount(row.get("Need", ""))
    merit_pct = parse_amount(row.get("Merit", ""))
    merit_hist_pct = parse_amount(row.get("Merit Hist", ""))
    application_type = clean_application_type(level, row.get("Application Type", ""), merit_pct)

    father_work = parse_work_status(row.get("Father Work Status", ""))
    mother_work = parse_work_status(row.get("Mother Work Status", ""))
    applicant_work = parse_work_status(row.get("Applicant Work Status", ""))
    spouse_work = parse_work_status(row.get("Applicant's Spouse Work Status", ""))

    father_job_info_clean, father_job_info_category = categorize_job_info(row.get("Father Job Info", ""))
    mother_job_info_clean, mother_job_info_category = categorize_job_info(row.get("Mother Job Info", ""))

    siblings_other = parse_siblings_not_aub(row.get("Siblings Outside Institution", ""))
    source_of_income = parse_source_of_income(row.get("Source of Income", ""))
    investments = parse_investments(row.get("Investments", ""))
    assistant_records = extract_financial_assistant_records(row.get("Financial Assistants", ""))
    assistants = parse_financial_assistants(row.get("Financial Assistants", ""))
    loan_records = extract_loan_records(row.get("Loans", ""))
    loans = parse_loans(row.get("Loans", ""))
    property_records = extract_property_records(row.get("Properties", ""))
    properties = parse_properties(row.get("Properties", ""))
    cars = parse_cars(row.get("Cars", ""))
    travel_clean, travel_category, travel_missing = parse_travel_records(row.get("Travel Records Verification", ""))
    ownership_clean, ownership_status = parse_ownership_verification(
        row.get("Certificate of Ownership Verification", "")
    )
    special_category, special_detail = clean_special_family_circumstances(
        row.get("Special Family Circumstances", "")
    )
    decision = clean_decision(row.get("Decision", ""))
    bin_status = clean_decision(row.get("Bin", ""))

    school_raw = normalize_text(row.get("School", ""))
    school_was_missing = to_int_flag(not school_raw)
    school = school_raw or None
    school_filled_for_ops_only = None
    school_ops_fill_method = None
    if not school and impute_missing_school_from_mode and school_mode:
        school_filled_for_ops_only = school_mode
        school_ops_fill_method = "global_mode"
        issues.add(
            source_row_number,
            "School",
            "school_imputed_from_mode",
            row.get("School", ""),
            f"Prepared ops-only school fill with '{school_mode}'",
        )

    if application_type["application_type_inferred_flag"]:
        issues.add(
            source_row_number,
            "Application Type",
            "application_type_inferred_from_level",
            row.get("Application Type", ""),
            f"Derived from level '{level or 'unknown'}'",
        )

    if father_income_status == "unparsed_text":
        issues.add(
            source_row_number,
            "Father Income Document",
            "parent_income_unparsed",
            row.get("Father Income Document", ""),
            "Could not confidently parse a numeric amount",
        )
    if mother_income_status == "unparsed_text":
        issues.add(
            source_row_number,
            "Mother Income Document",
            "parent_income_unparsed",
            row.get("Mother Income Document", ""),
            "Could not confidently parse a numeric amount",
        )

    if father_income_review_flag:
        issues.add(
            source_row_number,
            "Father Income Document",
            "parent_income_needs_review",
            row.get("Father Income Document", ""),
            father_income_status,
        )
    if mother_income_review_flag:
        issues.add(
            source_row_number,
            "Mother Income Document",
            "parent_income_needs_review",
            row.get("Mother Income Document", ""),
            mother_income_status,
        )

    if raw_applicant_work_status and applicant_work["annual_income"] is None and "annual income" in raw_applicant_work_status.lower():
        issues.add(
            source_row_number,
            "Applicant Work Status",
            "applicant_annual_income_missing_after_parse",
            row.get("Applicant Work Status", ""),
            "Employment text contained an annual income label but no amount was parsed",
        )

    if not decision and bin_status:
        issues.add(
            source_row_number,
            "Decision",
            "decision_missing_with_bin_status",
            row.get("Decision", ""),
            f"Decision is blank while Bin contains '{bin_status}'",
        )

    for field_name in ("Financial Assistants", "Loans", "Properties"):
        raw_text = row.get(field_name, "")
        if has_mixed_currency_markers(raw_text):
            issues.add(
                source_row_number,
                field_name,
                "mixed_currency_signal",
                raw_text,
                "Contains both USD and LBP/L.L. markers",
            )

    loan_remaining_balance_values = extract_labeled_values(row.get("Loans", ""), "Remaining Balance")
    if any(looks_like_duration_only(value) for value in loan_remaining_balance_values):
        issues.add(
            source_row_number,
            "Loans",
            "loan_remaining_balance_non_monetary_text",
            row.get("Loans", ""),
            "Remaining Balance contains duration text instead of a monetary amount",
        )

    if school_was_missing:
        issues.add(
            source_row_number,
            "School",
            "school_missing",
            row.get("School", ""),
            "School is blank in the source export",
        )
        if level == "Undergraduate":
            issues.add(
                source_row_number,
                "School",
                "school_missing_undergraduate",
                row.get("School", ""),
                "School is blank for an undergraduate application",
            )

    over_and_above_amount = parse_amount(row.get("Over and Above - Amount Awarded", ""))
    over_and_above_pct = parse_amount(row.get("Over and Above - Percentage Awarded", ""))
    over_and_above_decision = clean_decision(row.get("Over and Above Decision", ""))
    has_over_and_above = to_int_flag(
        over_and_above_amount is not None
        or over_and_above_pct is not None
        or (over_and_above_decision is not None and over_and_above_decision != "no_o_a")
    )
    if has_over_and_above and decision == "no_o_a":
        issues.add(
            source_row_number,
            "Decision",
            "over_and_above_flag_inconsistent",
            row.get("Decision", ""),
            "Over-and-above data is populated while Decision is 'no_o_a'",
        )

    if over_and_above_pct is not None and (merit_pct or 0.0) + over_and_above_pct > 50.0:
        issues.add(
            source_row_number,
            "Over and Above - Percentage Awarded",
            "merit_plus_over_and_above_exceeds_50_rule",
            row.get("Over and Above - Percentage Awarded", ""),
            f"merit_pct={merit_pct}; over_and_above_pct={over_and_above_pct}",
        )

    total_award_pct = sum(value for value in (need_pct, merit_pct, over_and_above_pct) if value is not None)
    if total_award_pct > 100.0:
        issues.add(
            source_row_number,
            "Need",
            "aid_percentage_total_exceeds_100",
            row.get("Need", ""),
            f"need_pct={need_pct}; merit_pct={merit_pct}; over_and_above_pct={over_and_above_pct}",
        )

    spouse_has_data = any(
        value
        for value in (
            spouse_demo["citizenship"],
            spouse_demo["marital_status"],
            spouse_demo["plan_to_reside"],
            spouse_work["status"],
            spouse_work["employment_sector"],
            spouse_work["position"],
            spouse_work["employer_name"],
            spouse_work["annual_income"],
        )
    )
    if applicant_demo["marital_status"] == "Single" and spouse_has_data:
        issues.add(
            source_row_number,
            "Applicant's Spouse Demographic",
            "spouse_data_present_while_single",
            row.get("Applicant's Spouse Demographic", ""),
            "Applicant marital status is Single but spouse fields contain data",
        )

    for field_name, status_value, income_value, raw_value in (
        ("Father Work Status", father_work["status"], father_work["annual_income"], raw_father_work_status),
        ("Mother Work Status", mother_work["status"], mother_work["annual_income"], raw_mother_work_status),
        ("Applicant Work Status", applicant_work["status"], applicant_work["annual_income"], raw_applicant_work_status),
        ("Applicant's Spouse Work Status", spouse_work["status"], spouse_work["annual_income"], raw_spouse_work_status),
    ):
        if income_value is not None and status_implies_no_income(status_value):
            issues.add(
                source_row_number,
                field_name,
                "income_present_with_non_employed_status",
                raw_value,
                f"status={status_value}; income={income_value}",
            )

    if application_type["application_type_inconsistent_flag"]:
        issues.add(
            source_row_number,
            "Application Type",
            "application_type_level_mismatch",
            row.get("Application Type", ""),
            f"level={level}; application_type={application_type['application_type_clean']}",
        )

    add_issue_batch(
        issues,
        source_row_number=source_row_number,
        field_name="Loans",
        issue_codes=detect_loan_record_issues(loan_records),
        raw_value=row.get("Loans", ""),
        detail="Remaining balance exceeds total amount for at least one parsed loan block",
    )

    father_income_document_currency = build_currency_profile(
        row.get("Father Income Document", ""),
        father_income,
        likely_lbp_threshold=1_000_000,
    )
    mother_income_document_currency = build_currency_profile(
        row.get("Mother Income Document", ""),
        mother_income,
        likely_lbp_threshold=1_000_000,
    )
    father_work_income_currency = build_currency_profile(
        row.get("Father Work Status", ""),
        father_work["annual_income"],
        likely_lbp_threshold=1_000_000,
    )
    mother_work_income_currency = build_currency_profile(
        row.get("Mother Work Status", ""),
        mother_work["annual_income"],
        likely_lbp_threshold=1_000_000,
    )
    applicant_income_currency = build_currency_profile(
        row.get("Applicant Work Status", ""),
        applicant_work["annual_income"],
        likely_lbp_threshold=1_000_000,
    )
    spouse_income_currency = build_currency_profile(
        row.get("Applicant's Spouse Work Status", ""),
        spouse_work["annual_income"],
        likely_lbp_threshold=1_000_000,
    )
    financial_assistants_currency = build_currency_profile(
        row.get("Financial Assistants", ""),
        assistants["total_est_annual_amount"],
        likely_lbp_threshold=1_000_000,
    )
    loans_amount_currency = build_currency_profile(
        row.get("Loans", ""),
        loans["total_amount"],
        likely_lbp_threshold=100_000_000,
    )
    properties_value_currency = build_currency_profile(
        row.get("Properties", ""),
        properties["total_estimated_value"],
        likely_lbp_threshold=100_000_000,
    )
    father_work_income_confidence = monetary_field_confidence(
        raw_father_work_status or "",
        father_work["annual_income"],
        parser_status="direct",
        review_flag=status_implies_no_income(father_work["status"]),
        currency_signal=father_work_income_currency["signal"],
        currency_guess=father_work_income_currency["guess"],
    )
    mother_work_income_confidence = monetary_field_confidence(
        raw_mother_work_status or "",
        mother_work["annual_income"],
        parser_status="direct",
        review_flag=status_implies_no_income(mother_work["status"]),
        currency_signal=mother_work_income_currency["signal"],
        currency_guess=mother_work_income_currency["guess"],
    )
    applicant_income_confidence = monetary_field_confidence(
        raw_applicant_work_status or "",
        applicant_work["annual_income"],
        parser_status="direct",
        review_flag=status_implies_no_income(applicant_work["status"]),
        currency_signal=applicant_income_currency["signal"],
        currency_guess=applicant_income_currency["guess"],
    )
    spouse_income_confidence = monetary_field_confidence(
        raw_spouse_work_status or "",
        spouse_work["annual_income"],
        parser_status="direct",
        review_flag=status_implies_no_income(spouse_work["status"]),
        currency_signal=spouse_income_currency["signal"],
        currency_guess=spouse_income_currency["guess"],
    )
    field_confidences = {
        "application_type_confidence": categorical_field_confidence(
            raw_application_type or level,
            application_type["application_type_clean"],
            inferred=bool(application_type["application_type_inferred_flag"]),
            missing_penalty=0.7,
        ),
        "school_confidence": clamp_confidence(0.4 if school_filled_for_ops_only else (1.0 if school else 0.0)),
        "father_income_document_confidence": monetary_field_confidence(
            raw_father_income_document or "",
            father_income,
            parser_status=father_income_status,
            review_flag=bool(father_income_review_flag),
            currency_signal=father_income_document_currency["signal"],
            currency_guess=father_income_document_currency["guess"],
        ),
        "mother_income_document_confidence": monetary_field_confidence(
            raw_mother_income_document or "",
            mother_income,
            parser_status=mother_income_status,
            review_flag=bool(mother_income_review_flag),
            currency_signal=mother_income_document_currency["signal"],
            currency_guess=mother_income_document_currency["guess"],
        ),
        "father_work_income_confidence": father_work_income_confidence,
        "mother_work_income_confidence": mother_work_income_confidence,
        "applicant_income_confidence": applicant_income_confidence,
        "spouse_income_confidence": spouse_income_confidence,
        "financial_assistants_amount_confidence": monetary_field_confidence(
            raw_financial_assistants or "",
            assistants["total_est_annual_amount"],
            parser_status="direct",
            review_flag=any(bool(record["assistant_review_flag"]) for record in assistant_records),
            currency_signal=financial_assistants_currency["signal"],
            currency_guess=financial_assistants_currency["guess"],
            deterministic_bonus=0.05,
        ),
        "loans_total_amount_confidence": monetary_field_confidence(
            raw_loans or "",
            loans["total_amount"],
            parser_status="direct",
            review_flag=any(bool(record["loan_review_flag"]) for record in loan_records),
            currency_signal=loans_amount_currency["signal"],
            currency_guess=loans_amount_currency["guess"],
            deterministic_bonus=0.05,
        ),
        "loans_monthly_payment_confidence": monetary_field_confidence(
            raw_loans or "",
            loans["total_monthly_payment"],
            parser_status="direct",
            review_flag=any(bool(record["loan_review_flag"]) for record in loan_records),
            currency_signal=detect_currency_signal(raw_loans or ""),
            currency_guess=infer_currency_guess(raw_loans or "", loans["total_monthly_payment"], likely_lbp_threshold=5_000_000),
        ),
        "loans_remaining_balance_confidence": monetary_field_confidence(
            raw_loans or "",
            loans["total_remaining_balance"],
            parser_status="direct",
            review_flag=any(bool(record["loan_remaining_balance_is_duration_flag"]) for record in loan_records),
            currency_signal=detect_currency_signal(raw_loans or ""),
            currency_guess=infer_currency_guess(raw_loans or "", loans["total_remaining_balance"], likely_lbp_threshold=100_000_000),
            hard_fail=any(bool(record["loan_remaining_balance_is_duration_flag"]) for record in loan_records),
        ),
        "properties_total_estimated_value_confidence": monetary_field_confidence(
            raw_properties or "",
            properties["total_estimated_value"],
            parser_status="direct",
            review_flag=any(bool(record["property_review_flag"]) for record in property_records),
            currency_signal=properties_value_currency["signal"],
            currency_guess=properties_value_currency["guess"],
            deterministic_bonus=0.05,
        ),
    }

    cleaned_row = {
        "source_sheet_name": source_sheet_name,
        "source_row_number": source_row_number,
        "application_type_raw": raw_application_type,
        "father_work_status_raw": raw_father_work_status,
        "mother_work_status_raw": raw_mother_work_status,
        "applicant_work_status_raw": raw_applicant_work_status,
        "spouse_work_status_raw": raw_spouse_work_status,
        "father_income_document_raw": raw_father_income_document,
        "mother_income_document_raw": raw_mother_income_document,
        "financial_assistants_raw": raw_financial_assistants,
        "loans_raw": raw_loans,
        "properties_raw": raw_properties,
        "cars_raw": raw_cars,
        "decision_raw": raw_decision,
        "bin_status_raw": raw_bin,
        "over_and_above_decision_raw": raw_over_and_above_decision,
        **application_type,
        "level": level or None,
        "application_term": normalize_text(row.get("Application Term", "")) or None,
        "nationality": normalize_text(row.get("Nationality", "")) or None,
        "applicant_citizenship": applicant_demo["citizenship"],
        "applicant_marital_status": applicant_demo["marital_status"],
        "applicant_plan_to_reside": applicant_demo["plan_to_reside"],
        "father_status": father_work["status"],
        "father_is_employed": employment_status_flag(father_work["status"]),
        "father_employment_sector": father_work["employment_sector"],
        "father_position": father_work["position"],
        "father_employer_name": father_work["employer_name"],
        "father_annual_income_from_work_status": father_work["annual_income"],
        "father_annual_income_currency_signal": father_work_income_currency["signal"],
        "father_annual_income_currency_guess": father_work_income_currency["guess"],
        "father_work_status_benefits_total": father_work["benefits_total"],
        "father_self_employed_name": father_work["self_employed_name"],
        "father_gross_income": father_work["gross_income"],
        "father_net_income": father_work["net_income"],
        "father_ever_worked": father_work["ever_worked"],
        "father_unemployment_date": father_work["unemployment_date"],
        "father_second_occupation_flag": father_work["second_occupation_flag"],
        "father_job_info_clean": father_job_info_clean,
        "father_job_info_category": father_job_info_category,
        "father_income_document": father_income,
        "father_income_document_status": father_income_status,
        "father_income_document_review_flag": father_income_review_flag,
        "father_income_document_currency_signal": father_income_document_currency["signal"],
        "father_income_document_currency_guess": father_income_document_currency["guess"],
        "mother_status": mother_work["status"],
        "mother_is_employed": employment_status_flag(mother_work["status"]),
        "mother_employment_sector": mother_work["employment_sector"],
        "mother_position": mother_work["position"],
        "mother_employer_name": mother_work["employer_name"],
        "mother_annual_income_from_work_status": mother_work["annual_income"],
        "mother_annual_income_currency_signal": mother_work_income_currency["signal"],
        "mother_annual_income_currency_guess": mother_work_income_currency["guess"],
        "mother_work_status_benefits_total": mother_work["benefits_total"],
        "mother_self_employed_name": mother_work["self_employed_name"],
        "mother_gross_income": mother_work["gross_income"],
        "mother_net_income": mother_work["net_income"],
        "mother_ever_worked": mother_work["ever_worked"],
        "mother_unemployment_date": mother_work["unemployment_date"],
        "mother_second_occupation_flag": mother_work["second_occupation_flag"],
        "mother_job_info_clean": mother_job_info_clean,
        "mother_job_info_category": mother_job_info_category,
        "mother_income_document": mother_income,
        "mother_income_document_status": mother_income_status,
        "mother_income_document_review_flag": mother_income_review_flag,
        "mother_income_document_currency_signal": mother_income_document_currency["signal"],
        "mother_income_document_currency_guess": mother_income_document_currency["guess"],
        "siblings_at_institution_count": count_pattern(row.get("Siblings At Institution", ""), r"Sibling\s+\d+\s*:"),
        "siblings_outside_institution_count": siblings_other["count"],
        "siblings_other_total_tuition": siblings_other["total_tuition"],
        "siblings_other_total_financial_assistance": siblings_other["total_financial_assistance"],
        "source_of_income_type": source_of_income["type"],
        "source_of_income_work_status": source_of_income["work_status"],
        "source_of_income_has_record": source_of_income["has_record"],
        "dependents_count": parse_dependents(row.get("Dependents", "")),
        "investments_count": investments["count"],
        "investments_total_annual_profit": investments["total_annual_profit"],
        "financial_assistants_count": assistants["count"],
        "financial_assistants_total_est_annual_amount": assistants["total_est_annual_amount"],
        "financial_assistants_currency_signal": financial_assistants_currency["signal"],
        "financial_assistants_currency_guess": financial_assistants_currency["guess"],
        "special_family_circumstances_category": special_category,
        "special_family_circumstances_detail": special_detail,
        "loans_count": loans["count"],
        "loans_total_amount": loans["total_amount"],
        "loans_total_amount_currency_signal": loans_amount_currency["signal"],
        "loans_total_amount_currency_guess": loans_amount_currency["guess"],
        "loans_total_monthly_payment": loans["total_monthly_payment"],
        "loans_total_remaining_balance": loans["total_remaining_balance"],
        "loans_max_remaining_years": loans["max_remaining_years"],
        "properties_count": properties["count"],
        "property_types": properties["property_types"],
        "properties_total_area": properties["total_area"],
        "properties_total_estimated_value": properties["total_estimated_value"],
        "properties_total_estimated_value_currency_signal": properties_value_currency["signal"],
        "properties_total_estimated_value_currency_guess": properties_value_currency["guess"],
        "properties_mortgaged_count": properties["mortgaged_count"],
        "properties_rented_income_total": properties["rented_income_total"],
        "properties_planted_income_total": properties["planted_income_total"],
        "cars_count": cars["count"],
        "car_models": cars["models"],
        "cars_mortgaged_count": cars["mortgaged_count"],
        "travel_records_clean": travel_clean,
        "travel_records_category": travel_category,
        "travel_records_missing_flag": travel_missing,
        "certificate_ownership_clean": ownership_clean,
        "certificate_ownership_status": ownership_status,
        "applicant_status": applicant_work["status"],
        "applicant_is_employed": applicant_employment_flag(applicant_work),
        "applicant_employment_sector": applicant_work["employment_sector"],
        "applicant_position": applicant_work["position"],
        "applicant_employer_name": applicant_work["employer_name"],
        "applicant_annual_income": applicant_work["annual_income"],
        "applicant_annual_income_currency_signal": applicant_income_currency["signal"],
        "applicant_annual_income_currency_guess": applicant_income_currency["guess"],
        "applicant_gross_income": applicant_work["gross_income"],
        "applicant_net_income": applicant_work["net_income"],
        "applicant_second_occupation_flag": applicant_work["second_occupation_flag"],
        "spouse_citizenship": spouse_demo["citizenship"],
        "spouse_marital_status": spouse_demo["marital_status"],
        "spouse_plan_to_reside": spouse_demo["plan_to_reside"],
        "spouse_status": spouse_work["status"],
        "spouse_is_employed": applicant_employment_flag(spouse_work),
        "spouse_employment_sector": spouse_work["employment_sector"],
        "spouse_position": spouse_work["position"],
        "spouse_employer_name": spouse_work["employer_name"],
        "spouse_annual_income": spouse_work["annual_income"],
        "spouse_annual_income_currency_signal": spouse_income_currency["signal"],
        "spouse_annual_income_currency_guess": spouse_income_currency["guess"],
        "decision": decision,
        "over_and_above_decision": over_and_above_decision,
        "over_and_above_amount_awarded": over_and_above_amount,
        "over_and_above_percentage_awarded": over_and_above_pct,
        "has_over_and_above": has_over_and_above,
        "submission_date": parse_date(row.get("Submission_Date", "")),
        "need_pct": need_pct,
        "need_comment": normalize_text(row.get("Need Comment", "")) or None,
        "merit_pct": merit_pct,
        "merit_hist_pct": merit_hist_pct,
        "merit_comment": normalize_text(row.get("Merit Comment", "")) or None,
        "consent_to_share_information": clean_yes_no(row.get("Consent to share information", "")),
        "bin_status": bin_status,
        "faid_missing_documents": normalize_text(row.get("FAID Missing Documents", "")) or None,
        "faid_missing_documents_count": count_missing_documents(row.get("FAID Missing Documents", "")),
        "school_raw": school_raw or None,
        "school": school,
        "school_was_missing": school_was_missing,
        "school_filled_for_ops_only": school_filled_for_ops_only,
        "school_ops_fill_method": school_ops_fill_method,
        **field_confidences,
    }
    add_monetary_quality_issues(
        cleaned_row=cleaned_row,
        raw_row=row,
        source_row_number=source_row_number,
        issues=issues,
    )
    return cleaned_row

# %% [notebook cell 17]
def build_cleaned_rows(
    workbook: WorkbookRows,
    *,
    impute_missing_school_from_mode: bool,
) -> tuple[list[dict[str, object]], IssueTracker]:
    school_counter = Counter(
        normalize_text(row.get("School", ""))
        for row in workbook.rows
        if normalize_text(row.get("School", ""))
    )
    school_mode = school_counter.most_common(1)[0][0] if school_counter else None

    issues = IssueTracker()
    cleaned_rows: list[dict[str, object]] = []
    for source_row_number, row in zip(workbook.source_row_numbers, workbook.rows):
        row_without_drops = {key: value for key, value in row.items() if key not in DROP_SOURCE_COLUMNS}
        cleaned_rows.append(
            build_cleaned_row(
                row_without_drops,
                source_row_number=source_row_number,
                source_sheet_name=workbook.sheet_name,
                school_mode=school_mode,
                impute_missing_school_from_mode=impute_missing_school_from_mode,
                issues=issues,
            )
        )

    issue_rows_by_number: dict[int, list[dict[str, object]]] = {}
    for issue_row in issues.as_rows():
        issue_rows_by_number.setdefault(int(issue_row["source_row_number"]), []).append(issue_row)

    for cleaned_row in cleaned_rows:
        row_number = int(cleaned_row["source_row_number"])
        issue_rows = issue_rows_by_number.get(row_number, [])
        issue_codes = [str(issue_row["issue_code"]) for issue_row in issue_rows]
        issue_count = len(issue_codes)
        max_severity, quality_score, risk_categories = summarize_issue_severity(issue_codes)
        cleaned_row["qa_issue_count"] = issue_count
        cleaned_row["qa_requires_review"] = to_int_flag(issue_count > 0)
        cleaned_row["qa_max_severity"] = max_severity
        cleaned_row["qa_quality_score"] = quality_score
        cleaned_row["qa_risk_categories"] = "; ".join(risk_categories) if risk_categories else None

    return cleaned_rows, issues


def drop_all_null_columns(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[str]]:
    if not rows:
        return rows, []

    dropped_columns: list[str] = []
    for field_name in rows[0]:
        has_value = any(row.get(field_name) not in ("", None) for row in rows)
        if not has_value:
            dropped_columns.append(field_name)

    if not dropped_columns:
        return rows, []

    trimmed_rows = [{key: value for key, value in row.items() if key not in dropped_columns} for row in rows]
    return trimmed_rows, dropped_columns


def validate_cleaned_rows(workbook: WorkbookRows, cleaned_rows: list[dict[str, object]]) -> None:
    if not workbook.rows:
        raise ValueError("The source workbook contains no data rows.")
    if len(cleaned_rows) != len(workbook.rows):
        raise ValueError(
            f"Row-count mismatch after cleaning: expected {len(workbook.rows)}, got {len(cleaned_rows)}"
        )
    if not cleaned_rows:
        raise ValueError("No cleaned rows were produced.")

    required_output_columns = {
        "source_sheet_name",
        "source_row_number",
        "application_type_clean",
        "level",
        "decision",
        "school",
    }
    missing_output_columns = sorted(required_output_columns - set(cleaned_rows[0]))
    if missing_output_columns:
        raise ValueError(f"Missing required cleaned columns: {', '.join(missing_output_columns)}")

    source_row_numbers = [row["source_row_number"] for row in cleaned_rows]
    if len(source_row_numbers) != len(set(source_row_numbers)):
        raise ValueError("Cleaned data contains duplicate source_row_number values.")


def build_review_rows(
    cleaned_rows: list[dict[str, object]],
    issues: IssueTracker,
) -> list[dict[str, object]]:
    issues_by_row: dict[int, list[dict[str, object]]] = {}
    for issue_row in issues.as_rows():
        issues_by_row.setdefault(int(issue_row["source_row_number"]), []).append(issue_row)

    review_rows: list[dict[str, object]] = []
    for cleaned_row in cleaned_rows:
        row_number = int(cleaned_row["source_row_number"])
        row_issues = issues_by_row.get(row_number)
        if not row_issues:
            continue

        unique_codes = list(dict.fromkeys(str(issue["issue_code"]) for issue in row_issues))
        unique_fields = list(dict.fromkeys(str(issue["field_name"]) for issue in row_issues))
        detail_fragments = [
            f"{issue['field_name']}:{issue['issue_code']}"
            + (f" ({issue['detail']})" if issue.get("detail") else "")
            for issue in row_issues[:8]
        ]

        review_row = dict(cleaned_row)
        review_row["qa_issue_codes"] = "; ".join(unique_codes)
        review_row["qa_issue_fields"] = "; ".join(unique_fields)
        review_row["qa_issue_summary"] = " | ".join(detail_fragments)
        review_rows.append(review_row)

    review_rows.sort(key=lambda row: (-int(row["qa_issue_count"]), int(row["source_row_number"])))
    return review_rows


def build_structured_tables(
    workbook: WorkbookRows,
    cleaned_rows: list[dict[str, object]],
) -> StructuredTables:
    financial_assistants: list[dict[str, object]] = []
    loans: list[dict[str, object]] = []
    properties: list[dict[str, object]] = []
    cars: list[dict[str, object]] = []

    for raw_row, cleaned_row in zip(workbook.rows, cleaned_rows):
        base_context = {
            "source_sheet_name": cleaned_row["source_sheet_name"],
            "source_row_number": cleaned_row["source_row_number"],
            "application_term": cleaned_row["application_term"],
            "level": cleaned_row["level"],
            "decision": cleaned_row["decision"],
            "school": cleaned_row["school"],
            "parent_qa_requires_review": cleaned_row["qa_requires_review"],
            "parent_qa_issue_count": cleaned_row["qa_issue_count"],
        }

        for record in extract_financial_assistant_records(raw_row.get("Financial Assistants", "")):
            financial_assistants.append({**base_context, **record})
        for record in extract_loan_records(raw_row.get("Loans", "")):
            loans.append({**base_context, **record})
        for record in extract_property_records(raw_row.get("Properties", "")):
            properties.append({**base_context, **record})
        for record in extract_car_records(raw_row.get("Cars", "")):
            cars.append({**base_context, **record})

    return StructuredTables(
        financial_assistants=financial_assistants,
        loans=loans,
        properties=properties,
        cars=cars,
    )


def output_field_group(field_name: str) -> str:
    if field_name in {"source_sheet_name", "source_row_number"} or field_name.endswith("_raw"):
        return "raw"
    if field_name.startswith("qa_") or field_name.endswith("_confidence"):
        return "qa"
    if field_name in {"school_was_missing", "father_income_document_status", "mother_income_document_status"}:
        return "qa"
    if field_name.endswith("_review_flag") or field_name.endswith("_review_reasons"):
        return "qa"
    if field_name.endswith("_currency_guess"):
        return "inferred"
    if field_name in {
        "application_type_clean",
        "application_track",
        "application_type_inferred_flag",
        "application_has_early_merit",
        "application_type_inconsistent_flag",
        "remaining_cap_50_rule_pct",
        "has_over_and_above",
        "school_filled_for_ops_only",
        "school_ops_fill_method",
    }:
        return "inferred"
    return "parsed"


def export_field_name(field_name: str) -> str:
    group = output_field_group(field_name)
    if field_name.endswith("_raw"):
        return f"raw_{field_name[:-4]}"
    if group == "qa" and field_name.startswith("qa_"):
        return field_name
    if group == "raw":
        return f"raw_{field_name}"
    return f"{group}_{field_name}"


def ordered_export_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return []

    group_order = {"raw": 0, "parsed": 1, "inferred": 2, "qa": 3}
    seen: list[str] = []
    for row in rows:
        for field_name in row:
            export_name = export_field_name(field_name)
            if export_name not in seen:
                seen.append(export_name)
    return sorted(
        seen,
        key=lambda export_name: (
            group_order.get(export_name.split("_", 1)[0], 99),
            export_name,
        ),
    )


def ordered_export_row_fieldnames(rows: list[dict[str, object]]) -> list[str]:
    if not rows:
        return []

    group_order = {"raw": 0, "parsed": 1, "inferred": 2, "qa": 3}
    seen: list[str] = []
    for row in rows:
        for field_name in row:
            if field_name not in seen:
                seen.append(field_name)
    return sorted(
        seen,
        key=lambda field_name: (
            group_order.get(field_name.split("_", 1)[0], 99),
            field_name,
        ),
    )


def build_export_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    export_rows: list[dict[str, object]] = []
    for row in rows:
        export_row: dict[str, object] = {}
        for field_name, value in row.items():
            export_row[export_field_name(field_name)] = value
        export_rows.append(export_row)
    return export_rows


MODEL_SAFE_CONFIDENCE_FIELDS = {
    "school": "school_confidence",
    "father_income_document": "father_income_document_confidence",
    "mother_income_document": "mother_income_document_confidence",
    "father_annual_income_from_work_status": "father_work_income_confidence",
    "mother_annual_income_from_work_status": "mother_work_income_confidence",
    "applicant_annual_income": "applicant_income_confidence",
    "spouse_annual_income": "spouse_income_confidence",
    "financial_assistants_total_est_annual_amount": "financial_assistants_amount_confidence",
    "loans_total_amount": "loans_total_amount_confidence",
    "loans_total_monthly_payment": "loans_monthly_payment_confidence",
    "loans_total_remaining_balance": "loans_remaining_balance_confidence",
    "properties_total_estimated_value": "properties_total_estimated_value_confidence",
}

MODEL_SAFE_EXCLUDED_FIELDS = {
    "decision",
    "bin_status",
    "need_pct",
    "need_comment",
    "merit_pct",
    "merit_hist_pct",
    "merit_comment",
    "over_and_above_decision",
    "over_and_above_amount_awarded",
    "over_and_above_percentage_awarded",
    "has_over_and_above",
    "faid_missing_documents",
    "special_family_circumstances_detail",
    "submission_date",
}

MODEL_SAFE_ALLOWED_QA_FIELDS = {
    "qa_issue_count",
    "qa_requires_review",
    "qa_max_severity",
    "qa_quality_score",
    "qa_risk_categories",
    "application_type_confidence",
    "school_confidence",
    "father_income_document_confidence",
    "mother_income_document_confidence",
    "father_work_income_confidence",
    "mother_work_income_confidence",
    "applicant_income_confidence",
    "spouse_income_confidence",
    "financial_assistants_amount_confidence",
    "loans_total_amount_confidence",
    "loans_monthly_payment_confidence",
    "loans_remaining_balance_confidence",
    "properties_total_estimated_value_confidence",
}


def build_model_safe_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    model_safe_rows: list[dict[str, object]] = []
    for row in rows:
        safe_row: dict[str, object] = {}
        for field_name, value in row.items():
            if field_name in {"source_sheet_name", "source_row_number"}:
                safe_row[export_field_name(field_name)] = value
                continue
            group = output_field_group(field_name)
            if group == "raw" or group == "inferred":
                continue
            if group == "qa":
                if field_name in MODEL_SAFE_ALLOWED_QA_FIELDS:
                    safe_row[export_field_name(field_name)] = value
                continue
            if field_name in MODEL_SAFE_EXCLUDED_FIELDS:
                continue

            safe_value = value
            confidence_field = MODEL_SAFE_CONFIDENCE_FIELDS.get(field_name)
            if confidence_field:
                confidence_value = row.get(confidence_field)
                if confidence_value is None or float(confidence_value) < 0.75:
                    safe_value = None
            if field_name == "school" and row.get("school_was_missing"):
                safe_value = None
            safe_row[export_field_name(field_name)] = safe_value

        model_safe_rows.append(safe_row)
    return model_safe_rows


def infer_export_field_description(internal_field_name: str, group: str) -> str:
    base_name = internal_field_name[:-4] if internal_field_name.endswith("_raw") else internal_field_name
    label = base_name.replace("_", " ")
    if group == "raw":
        return f"Raw preserved source text or provenance for {label}."
    if group == "parsed":
        return f"Parsed or standardized value for {label}."
    if group == "inferred":
        return f"Derived or inferred value for {label}."
    return f"QA signal or confidence metric for {label}."


def infer_export_field_derivation(internal_field_name: str, group: str) -> str:
    if group == "raw":
        return "Copied from the source workbook without semantic transformation."
    if group == "parsed":
        return "Produced by direct parsing or normalization of source workbook content."
    if group == "inferred":
        return "Derived from parsed values, heuristics, or policy logic."
    return "Generated by validation, issue scoring, or field-level confidence logic."


def build_data_dictionary(
    internal_rows: list[dict[str, object]],
    export_rows: list[dict[str, object]],
    model_safe_rows: list[dict[str, object]],
) -> dict[str, object]:
    export_profile = build_table_profile("cleaned", export_rows)
    model_safe_fields = set(model_safe_rows[0].keys()) if model_safe_rows else set()
    internal_field_by_export_name = {}
    if internal_rows:
        for internal_field_name in internal_rows[0]:
            internal_field_by_export_name[export_field_name(internal_field_name)] = internal_field_name
    dictionary_columns: list[dict[str, object]] = []
    for column in export_profile["columns"]:
        export_name = str(column["name"])
        internal_field_name = internal_field_by_export_name.get(export_name)
        if not internal_field_name:
            continue

        group = output_field_group(internal_field_name)
        dictionary_columns.append(
            {
                "export_field_name": export_name,
                "internal_field_name": internal_field_name,
                "group": group,
                "inferred_type": column["inferred_type"],
                "description": infer_export_field_description(internal_field_name, group),
                "derivation": infer_export_field_derivation(internal_field_name, group),
                "model_safe_included": export_name in model_safe_fields,
                "confidence_gate_field": export_field_name(MODEL_SAFE_CONFIDENCE_FIELDS[internal_field_name])
                if internal_field_name in MODEL_SAFE_CONFIDENCE_FIELDS
                else None,
            }
        )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "confidence_scoring": CONFIDENCE_POLICY,
        "tables": {
            "cleaned": {
                "column_count": len(dictionary_columns),
                "columns": dictionary_columns,
            }
        },
    }


def write_csv(path: Path, rows: list[dict[str, object]], *, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def value_counts(rows: list[dict[str, object]], field_name: str, *, limit: int = 10) -> dict[str, int]:
    counts = Counter(normalize_text(row.get(field_name)) for row in rows if normalize_text(row.get(field_name)))
    return dict(counts.most_common(limit))


def build_output_paths(output_dir: Path, output_stem: str) -> OutputPaths:
    return OutputPaths(
        cleaned_csv=output_dir / f"{output_stem}.csv",
        model_safe_csv=output_dir / f"{output_stem}_model_safe.csv",
        review_csv=output_dir / f"{output_stem}_review_required.csv",
        financial_assistants_csv=output_dir / f"{output_stem}_financial_assistants.csv",
        loans_csv=output_dir / f"{output_stem}_loans.csv",
        properties_csv=output_dir / f"{output_stem}_properties.csv",
        cars_csv=output_dir / f"{output_stem}_cars.csv",
        issues_csv=output_dir / f"{output_stem}_issues.csv",
        data_dictionary_json=output_dir / f"{output_stem}_data_dictionary.json",
        profile_json=output_dir / f"{output_stem}_profile.json",
        summary_json=output_dir / f"{output_stem}_summary.json",
    )


def build_summary(
    *,
    input_path: Path,
    workbook: WorkbookRows,
    cleaned_rows: list[dict[str, object]],
    cleaned_export_rows: list[dict[str, object]],
    review_rows: list[dict[str, object]],
    review_export_rows: list[dict[str, object]],
    model_safe_rows: list[dict[str, object]],
    structured_tables: StructuredTables,
    issues: IssueTracker,
    output_paths: OutputPaths,
    dropped_all_null_columns: list[str],
    impute_missing_school_from_mode: bool,
) -> dict[str, object]:
    issue_counts = issues.counts()
    issue_risk_counts = Counter(issue_risk_category(issue_code) for issue_code, count in issue_counts.items() for _ in range(count))
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_file": str(input_path.resolve()),
        "source_sheet_name": workbook.sheet_name,
        "source_row_count": len(workbook.rows),
        "source_column_count": len(workbook.headers),
        "cleaned_row_count": len(cleaned_export_rows),
        "cleaned_column_count": len(cleaned_export_rows[0]) if cleaned_export_rows else 0,
        "dropped_source_columns": sorted(DROP_SOURCE_COLUMNS & set(workbook.headers)),
        "dropped_all_null_output_columns": dropped_all_null_columns,
        "missing_school_rows": sum(int(row["school_was_missing"]) for row in cleaned_rows),
        "prepare_school_fill_for_ops_only": impute_missing_school_from_mode,
        "rows_requiring_review": sum(int(row["qa_requires_review"]) for row in cleaned_rows),
        "review_output_row_count": len(review_export_rows),
        "model_safe_row_count": len(model_safe_rows),
        "model_safe_column_count": len(model_safe_rows[0]) if model_safe_rows else 0,
        "qa_max_severity_counts": value_counts(cleaned_rows, "qa_max_severity"),
        "issue_risk_category_counts": dict(sorted(issue_risk_counts.items())),
        "structured_output_row_counts": {
            "financial_assistants": len(structured_tables.financial_assistants),
            "loans": len(structured_tables.loans),
            "properties": len(structured_tables.properties),
            "cars": len(structured_tables.cars),
        },
        "issue_count": len(issues),
        "issue_counts": dict(sorted(issue_counts.items())),
        "top_levels": value_counts(cleaned_rows, "level"),
        "top_application_tracks": value_counts(cleaned_rows, "application_track"),
        "top_decisions": value_counts(cleaned_rows, "decision"),
        "top_bin_statuses": value_counts(cleaned_rows, "bin_status"),
        "output_files": {
            "cleaned_csv": str(output_paths.cleaned_csv.resolve()),
            "model_safe_csv": str(output_paths.model_safe_csv.resolve()),
            "review_csv": str(output_paths.review_csv.resolve()),
            "financial_assistants_csv": str(output_paths.financial_assistants_csv.resolve()),
            "loans_csv": str(output_paths.loans_csv.resolve()),
            "properties_csv": str(output_paths.properties_csv.resolve()),
            "cars_csv": str(output_paths.cars_csv.resolve()),
            "issues_csv": str(output_paths.issues_csv.resolve()),
            "data_dictionary_json": str(output_paths.data_dictionary_json.resolve()),
            "profile_json": str(output_paths.profile_json.resolve()),
            "summary_json": str(output_paths.summary_json.resolve()),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the FAID export into a model-ready CSV with an audit trail."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the raw FAID Excel export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where cleaned artifacts should be written.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Base file name for the generated artifacts.",
    )
    parser.add_argument(
        "--sheet-name",
        default=None,
        help="Worksheet name to read. Defaults to the first sheet.",
    )
    parser.add_argument(
        "--impute-missing-school-from-mode",
        action="store_true",
        help="Prepare an ops-only school fill field from the batch mode without overwriting the cleaned school value.",
    )
    parser.add_argument(
        "--drop-all-null-columns",
        action="store_true",
        help="Drop only columns that are entirely null in the current cleaned output.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    try:
        workbook = read_xlsx_rows(args.input, sheet_name=args.sheet_name)
        validate_source_columns(workbook.headers)

        cleaned_rows, issues = build_cleaned_rows(
            workbook,
            impute_missing_school_from_mode=args.impute_missing_school_from_mode,
        )

        dropped_all_null_columns: list[str] = []
        if args.drop_all_null_columns:
            cleaned_rows, dropped_all_null_columns = drop_all_null_columns(cleaned_rows)

        validate_cleaned_rows(workbook, cleaned_rows)
        review_rows = build_review_rows(cleaned_rows, issues)
        structured_tables = build_structured_tables(workbook, cleaned_rows)
        cleaned_export_rows = build_export_rows(cleaned_rows)
        review_export_rows = build_export_rows(review_rows)
        model_safe_rows = build_model_safe_rows(cleaned_rows)

        output_paths = build_output_paths(args.output_dir, args.output_stem)
        write_csv(
            output_paths.cleaned_csv,
            cleaned_export_rows,
            fieldnames=ordered_export_fieldnames(cleaned_rows),
        )
        write_csv(
            output_paths.model_safe_csv,
            model_safe_rows,
            fieldnames=ordered_export_row_fieldnames(model_safe_rows),
        )
        write_csv(
            output_paths.review_csv,
            review_export_rows,
            fieldnames=ordered_export_fieldnames(review_rows),
        )
        write_csv(output_paths.financial_assistants_csv, structured_tables.financial_assistants)
        write_csv(output_paths.loans_csv, structured_tables.loans)
        write_csv(output_paths.properties_csv, structured_tables.properties)
        write_csv(output_paths.cars_csv, structured_tables.cars)
        write_csv(output_paths.issues_csv, issues.as_rows(), fieldnames=ISSUE_FIELDNAMES)
        data_dictionary = build_data_dictionary(cleaned_rows, cleaned_export_rows, model_safe_rows)
        write_json(output_paths.data_dictionary_json, data_dictionary)
        profile_payload = build_profile_payload(
            input_path=args.input,
            workbook=workbook,
            cleaned_rows=cleaned_export_rows,
            review_rows=review_export_rows,
            model_safe_rows=model_safe_rows,
            structured_tables=structured_tables,
        )
        write_json(output_paths.profile_json, profile_payload)

        summary = build_summary(
            input_path=args.input,
            workbook=workbook,
            cleaned_rows=cleaned_rows,
            cleaned_export_rows=cleaned_export_rows,
            review_rows=review_rows,
            review_export_rows=review_export_rows,
            model_safe_rows=model_safe_rows,
            structured_tables=structured_tables,
            issues=issues,
            output_paths=output_paths,
            dropped_all_null_columns=dropped_all_null_columns,
            impute_missing_school_from_mode=args.impute_missing_school_from_mode,
        )
        write_json(output_paths.summary_json, summary)
    except Exception:
        LOGGER.exception("Cleaning failed")
        return 1

    LOGGER.info("Wrote %d cleaned rows to %s", len(cleaned_rows), output_paths.cleaned_csv)
    LOGGER.info("Wrote %d model-safe rows to %s", len(model_safe_rows), output_paths.model_safe_csv)
    LOGGER.info("Wrote %d review rows to %s", len(review_rows), output_paths.review_csv)
    LOGGER.info(
        "Wrote structured child tables: %d assistants, %d loans, %d properties, %d cars",
        len(structured_tables.financial_assistants),
        len(structured_tables.loans),
        len(structured_tables.properties),
        len(structured_tables.cars),
    )
    LOGGER.info("Wrote %d data-quality issues to %s", len(issues), output_paths.issues_csv)
    LOGGER.info("Wrote data dictionary to %s", output_paths.data_dictionary_json)
    LOGGER.info("Wrote data profile to %s", output_paths.profile_json)
    LOGGER.info("Wrote run summary to %s", output_paths.summary_json)
    return 0


if __name__ == "__main__":
    if RUNNING_IN_NOTEBOOK:
        main([])
    else:
        raise SystemExit(main())

# %% [notebook cell 19]
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

# %% [notebook cell 21]
import pandas as pd
from IPython.display import display

review_required = pd.read_csv(OUTPUT_DIR / "faid_cleaned_review_required.csv")
issues = pd.read_csv(OUTPUT_DIR / "faid_cleaned_issues.csv")

top_issue_table = (
    issues.groupby("issue_code", dropna=False)
    .size()
    .sort_values(ascending=False)
    .rename("row_count")
    .reset_index()
)
display(top_issue_table.head(15))

review_snapshot = (
    review_required.groupby("qa_max_severity", dropna=False)
    .agg(
        rows=("raw_source_row_number", "count"),
        avg_issue_count=("qa_issue_count", "mean"),
        avg_quality_score=("qa_quality_score", "mean"),
    )
    .reset_index()
)
display(review_snapshot)

# %% [notebook cell 23]
import pandas as pd
from IPython.display import display

cleaned = pd.read_csv(OUTPUT_DIR / "faid_cleaned.csv")


def operational_route(row: pd.Series) -> str:
    if bool(row.get("qa_requires_review")) or row.get("qa_max_severity") in {"high", "medium"}:
        return "manual_review_required"
    if row.get("qa_quality_score", 0.0) >= 90 and row.get("qa_issue_count", 99) == 0:
        return "high_trust_case_ready_for_analysis"
    if row.get("qa_quality_score", 0.0) >= 75:
        return "usable_for_modeling_with_monitoring"
    return "usable_for_descriptive_reporting_only"


routes = cleaned.assign(decision_route=cleaned.apply(operational_route, axis=1))
display(routes["decision_route"].value_counts(dropna=False).rename_axis("decision_route").reset_index(name="rows"))
