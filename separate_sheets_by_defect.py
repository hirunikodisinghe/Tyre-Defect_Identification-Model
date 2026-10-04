from pathlib import Path
import csv
import re
from collections import defaultdict

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    r"E:\Hiruni\TireDefectProject"
)

# Change only this filename if your current file has
# a different name.
#
# This can be either a .csv file or a .xlsx file.
INPUT_FILE = (
    PROJECT_DIR
    / "manual_inspection_with_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "manual_inspection_separate_sheets.xlsx"
)


# ============================================================
# 2. POSSIBLE COLUMN NAMES
# ============================================================

DEFECT_COLUMN_CANDIDATES = [
    "TyreDefectName",
    "ActualDefect",
    "TyreDefect",
    "Defect",
]

STATUS_COLUMN_CANDIDATES = [
    "PredictionStatus",
    "Status",
]

CONFIDENCE_COLUMN_CANDIDATES = [
    "ConfidencePercent",
    "ModelConfidence",
    "Confidence",
]

IMAGE_NAME_COLUMN_CANDIDATES = [
    "ImageName",
    "FileName",
    "Image",
]

IMAGE_PATH_COLUMN_CANDIDATES = [
    "ImagePath",
    "FilePath",
    "Path",
]

MODEL_COLUMN_CANDIDATES = [
    "ModelPrediction",
    "CLIPPrediction",
    "Prediction",
]

MANUAL_STATUS_COLUMN_CANDIDATES = [
    "ManualStatus",
    "ReviewStatus",
]

CORRECT_DEFECT_COLUMN_CANDIDATES = [
    "CorrectDefectName",
    "CorrectDefect",
]

REMOVE_COLUMN_CANDIDATES = [
    "RemoveImage",
    "Remove",
]


# ============================================================
# 3. DEFECT CLASS ORDER
# ============================================================

DEFAULT_CLASS_ORDER = [
    "Abnormal_Wear",
    "Bead_Damage",
    "Bulge",
    "Cut",
    "Shoulder_Crack",
    "Sidewall_Crack",
    "Sidewall_Damage",
    "Tire_Burst",
    "Tread_Chunking",
    "Tread_Damage",
    "Tread_Separation",
    "noDefect",
]


# ============================================================
# 4. STYLE SETTINGS
# ============================================================

HEADER_FILL = "1F4E78"
HEADER_FONT = "FFFFFF"

MATCH_FILL = "C6EFCE"
MISMATCH_FILL = "FFC7CE"
ERROR_FILL = "FCE4D6"

ACTUAL_DEFECT_FILL = "E2F0D9"
MODEL_PREDICTION_FILL = "FFF2CC"
MANUAL_INPUT_FILL = "DDEBF7"

IMPORTED_FONT = "008000"
MANUAL_INPUT_FONT = "0000FF"


# ============================================================
# 5. HELPER FUNCTIONS
# ============================================================

def canonical_name(value):
    """
    Convert a column name into a simple comparable value.

    Example:
    'Actual Defect' -> 'actualdefect'
    """

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower(),
    )


def find_column(headers, candidates):
    """
    Find a column even if capitalization or spacing differs.
    """

    header_lookup = {
        canonical_name(header): header
        for header in headers
    }

    for candidate in candidates:

        result = header_lookup.get(
            canonical_name(candidate)
        )

        if result:
            return result

    return None


def safe_sheet_name(name, used_names):
    """
    Remove characters that Excel does not allow
    in worksheet names.
    """

    cleaned = re.sub(
        r'[:\\/?*\[\]]',
        "_",
        str(name).strip(),
    )

    if not cleaned:
        cleaned = "Unassigned"

    cleaned = cleaned[:31]

    candidate = cleaned
    counter = 2

    while candidate in used_names:

        suffix = f"_{counter}"

        candidate = (
            cleaned[:31 - len(suffix)]
            + suffix
        )

        counter += 1

    used_names.add(candidate)

    return candidate


def parse_number(value):
    """
    Convert a confidence value into a number
    for sorting.
    """

    try:

        return float(
            str(value)
            .replace("%", "")
            .strip()
        )

    except (TypeError, ValueError):

        return -1.0


def create_file_uri(path_value):
    """
    Convert a Windows file path into an Excel hyperlink.
    """

    if not path_value:
        return None

    try:

        file_path = Path(
            str(path_value)
        )

        if file_path.is_absolute():
            return file_path.as_uri()

    except (TypeError, ValueError):

        return None

    return None


# ============================================================
# 6. READ CSV OR XLSX INPUT
# ============================================================

def read_input_file(file_path):

    extension = file_path.suffix.lower()

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    if extension == ".csv":

        with file_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:

            reader = csv.DictReader(
                csv_file
            )

            if not reader.fieldnames:

                raise ValueError(
                    "The CSV file has no header row."
                )

            headers = [
                str(header).strip()
                for header in reader.fieldnames
            ]

            rows = []

            for source_row in reader:

                row = {
                    header: (
                        source_row.get(header, "")
                        or ""
                    )
                    for header in headers
                }

                rows.append(row)

        return headers, rows

    # --------------------------------------------------------
    # READ XLSX
    # --------------------------------------------------------

    if extension == ".xlsx":

        source_workbook = load_workbook(
            file_path,
            read_only=True,
            data_only=False,
        )

        source_sheet = (
            source_workbook.active
        )

        raw_headers = [
            cell.value
            for cell in source_sheet[1]
        ]

        headers = []
        used_headers = set()

        for index, raw_header in enumerate(
            raw_headers,
            start=1,
        ):

            if raw_header in (None, ""):
                base_header = f"Column_{index}"
            else:
                base_header = str(
                    raw_header
                ).strip()

            header = base_header
            counter = 2

            while header in used_headers:

                header = (
                    f"{base_header}_{counter}"
                )

                counter += 1

            headers.append(header)
            used_headers.add(header)

        rows = []

        for values in source_sheet.iter_rows(
            min_row=2,
            values_only=True,
        ):

            if all(
                value in (None, "")
                for value in values
            ):
                continue

            row = {}

            for index, header in enumerate(
                headers
            ):

                if (
                    index < len(values)
                    and values[index] is not None
                ):

                    row[header] = values[index]

                else:

                    row[header] = ""

            rows.append(row)

        source_workbook.close()

        return headers, rows

    raise ValueError(
        "INPUT_FILE must be a CSV or XLSX file."
    )


# ============================================================
# 7. CHECK INPUT FILE
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nInput file not found:\n"
        f"{INPUT_FILE}\n\n"
        "Change INPUT_FILE in the script so that it "
        "matches your actual CSV or Excel filename."
    )


# ============================================================
# 8. LOAD DATA
# ============================================================

print("=" * 70)
print("SEPARATE EXCEL SHEETS BY TYRE DEFECT")
print("=" * 70)

print("\nInput file:")
print(INPUT_FILE)

headers, source_rows = read_input_file(
    INPUT_FILE
)

print(
    f"\nRows loaded: {len(source_rows)}"
)


# ============================================================
# 9. DETECT IMPORTANT COLUMNS
# ============================================================

actual_defect_column = find_column(
    headers,
    DEFECT_COLUMN_CANDIDATES,
)

status_column = find_column(
    headers,
    STATUS_COLUMN_CANDIDATES,
)

confidence_column = find_column(
    headers,
    CONFIDENCE_COLUMN_CANDIDATES,
)

image_name_column = find_column(
    headers,
    IMAGE_NAME_COLUMN_CANDIDATES,
)

image_path_column = find_column(
    headers,
    IMAGE_PATH_COLUMN_CANDIDATES,
)

model_prediction_column = find_column(
    headers,
    MODEL_COLUMN_CANDIDATES,
)

manual_status_column = find_column(
    headers,
    MANUAL_STATUS_COLUMN_CANDIDATES,
)

correct_defect_column = find_column(
    headers,
    CORRECT_DEFECT_COLUMN_CANDIDATES,
)

remove_image_column = find_column(
    headers,
    REMOVE_COLUMN_CANDIDATES,
)


if actual_defect_column is None:

    raise ValueError(
        "\nCould not find the actual-defect column.\n\n"
        "Expected one of:\n"
        + "\n".join(
            DEFECT_COLUMN_CANDIDATES
        )
        + "\n\nColumns found:\n"
        + "\n".join(headers)
    )


print(
    "\nDefect column detected:",
    actual_defect_column,
)


# ============================================================
# 10. GROUP ROWS BY ACTUAL DEFECT
# ============================================================

rows_by_defect = defaultdict(list)


for row in source_rows:

    defect_name = str(
        row.get(
            actual_defect_column,
            "",
        )
    ).strip()

    if not defect_name:
        defect_name = "Unassigned"

    rows_by_defect[
        defect_name
    ].append(row)


# ============================================================
# 11. DETERMINE SHEET ORDER
# ============================================================

class_order = []


# First use class_names.txt if available
class_names_file = (
    PROJECT_DIR
    / "class_names.txt"
)


if class_names_file.exists():

    class_order = [
        line.strip()
        for line in class_names_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


# Otherwise use the default list
if not class_order:

    class_order = (
        DEFAULT_CLASS_ORDER.copy()
    )


# Add unexpected class names
for defect_name in sorted(
    rows_by_defect
):

    if defect_name not in class_order:

        class_order.append(
            defect_name
        )


# ============================================================
# 12. SORT MISMATCHES FIRST
# ============================================================

status_priority = {
    "MISMATCH": 0,
    "ERROR": 1,
    "MATCH": 2,
}


for defect_name, rows in rows_by_defect.items():

    rows.sort(
        key=lambda row: (

            status_priority.get(
                str(
                    row.get(
                        status_column,
                        "",
                    )
                ).strip().upper(),
                3,
            )
            if status_column
            else 3,

            -parse_number(
                row.get(
                    confidence_column,
                    "",
                )
            )
            if confidence_column
            else 0,

            str(
                row.get(
                    image_name_column,
                    "",
                )
            ).lower()
            if image_name_column
            else "",
        )
    )


# ============================================================
# 13. CREATE WORKBOOK
# ============================================================

workbook = Workbook()

summary_sheet = workbook.active
summary_sheet.title = "Summary"

summary_sheet.sheet_view.showGridLines = False


# ============================================================
# 14. CREATE HIDDEN DROPDOWN LIST SHEET
# ============================================================

list_sheet = workbook.create_sheet(
    "_Lists"
)


for row_number, class_name in enumerate(
    class_order,
    start=1,
):

    list_sheet.cell(
        row=row_number,
        column=1,
        value=class_name,
    )


manual_status_options = [
    "Pending",
    "Correct Label",
    "Wrong Label",
    "Model Error",
    "Unclear",
    "Bad Image",
]


for row_number, status in enumerate(
    manual_status_options,
    start=1,
):

    list_sheet.cell(
        row=row_number,
        column=2,
        value=status,
    )


remove_options = [
    "No",
    "Yes",
]


for row_number, option in enumerate(
    remove_options,
    start=1,
):

    list_sheet.cell(
        row=row_number,
        column=3,
        value=option,
    )


list_sheet.sheet_state = "hidden"


# ============================================================
# 15. COLUMN WIDTHS
# ============================================================

COLUMN_WIDTHS = {
    "ImageName": 42,
    "FileName": 42,
    "Split": 11,
    "TyreDefectName": 23,
    "ActualDefect": 23,
    "ModelPrediction": 23,
    "CLIPPrediction": 23,
    "ConfidencePercent": 19,
    "ModelConfidence": 19,
    "Confidence": 19,
    "PredictionStatus": 19,
    "Top1Prediction": 32,
    "Top2Prediction": 32,
    "Top3Prediction": 32,
    "ImagePath": 80,
    "ManualStatus": 20,
    "ReviewStatus": 20,
    "CorrectDefectName": 25,
    "RemoveImage": 15,
    "Notes": 45,
    "Note": 45,
    "ChatGPT results": 45,
    "ChatGPT Results": 45,
}


# ============================================================
# 16. CREATE ONE SHEET PER DEFECT
# ============================================================

used_sheet_names = {
    "Summary",
    "_Lists",
}

defect_sheet_names = {}


for defect_name in class_order:

    if defect_name not in rows_by_defect:
        continue

    sheet_name = safe_sheet_name(
        defect_name,
        used_sheet_names,
    )

    defect_sheet_names[
        defect_name
    ] = sheet_name

    worksheet = workbook.create_sheet(
        sheet_name
    )

    worksheet.sheet_view.showGridLines = False

    worksheet.freeze_panes = "A2"

    worksheet.row_dimensions[
        1
    ].height = 24


    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    for column_number, header in enumerate(
        headers,
        start=1,
    ):

        cell = worksheet.cell(
            row=1,
            column=column_number,
            value=header,
        )

        cell.font = Font(
            bold=True,
            color=HEADER_FONT,
        )

        cell.fill = PatternFill(
            fill_type="solid",
            fgColor=HEADER_FILL,
        )

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

        column_letter = get_column_letter(
            column_number
        )

        worksheet.column_dimensions[
            column_letter
        ].width = COLUMN_WIDTHS.get(
            header,
            20,
        )


    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    for row_data in rows_by_defect[
        defect_name
    ]:

        worksheet.append([
            row_data.get(
                header,
                "",
            )
            for header in headers
        ])


    header_positions = {
        header: index + 1
        for index, header in enumerate(
            headers
        )
    }


    actual_column_number = (
        header_positions.get(
            actual_defect_column
        )
    )

    model_column_number = (
        header_positions.get(
            model_prediction_column
        )
        if model_prediction_column
        else None
    )

    status_column_number = (
        header_positions.get(
            status_column
        )
        if status_column
        else None
    )

    confidence_column_number = (
        header_positions.get(
            confidence_column
        )
        if confidence_column
        else None
    )

    image_name_column_number = (
        header_positions.get(
            image_name_column
        )
        if image_name_column
        else None
    )

    image_path_column_number = (
        header_positions.get(
            image_path_column
        )
        if image_path_column
        else None
    )

    manual_status_column_number = (
        header_positions.get(
            manual_status_column
        )
        if manual_status_column
        else None
    )

    correct_defect_column_number = (
        header_positions.get(
            correct_defect_column
        )
        if correct_defect_column
        else None
    )

    remove_image_column_number = (
        header_positions.get(
            remove_image_column
        )
        if remove_image_column
        else None
    )


    # --------------------------------------------------------
    # FORMAT DATA ROWS
    # --------------------------------------------------------

    for row_number in range(
        2,
        worksheet.max_row + 1,
    ):

        for cell in worksheet[
            row_number
        ]:

            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )

            cell.font = Font(
                color=IMPORTED_FONT
            )


        # Actual defect
        if actual_column_number:

            worksheet.cell(
                row=row_number,
                column=actual_column_number,
            ).fill = PatternFill(
                fill_type="solid",
                fgColor=ACTUAL_DEFECT_FILL,
            )


        # Model prediction
        if model_column_number:

            worksheet.cell(
                row=row_number,
                column=model_column_number,
            ).fill = PatternFill(
                fill_type="solid",
                fgColor=MODEL_PREDICTION_FILL,
            )


        # Manual input columns
        for manual_column in [
            manual_status_column_number,
            correct_defect_column_number,
            remove_image_column_number,
        ]:

            if manual_column:

                cell = worksheet.cell(
                    row=row_number,
                    column=manual_column,
                )

                cell.fill = PatternFill(
                    fill_type="solid",
                    fgColor=MANUAL_INPUT_FILL,
                )

                cell.font = Font(
                    color=MANUAL_INPUT_FONT,
                )


        # Confidence number format
        if confidence_column_number:

            worksheet.cell(
                row=row_number,
                column=confidence_column_number,
            ).number_format = "0.00"


        # Match / mismatch status
        if status_column_number:

            status_cell = worksheet.cell(
                row=row_number,
                column=status_column_number,
            )

            status_value = str(
                status_cell.value or ""
            ).strip().upper()


            if status_value == "MATCH":

                status_cell.fill = PatternFill(
                    fill_type="solid",
                    fgColor=MATCH_FILL,
                )


            elif status_value == "MISMATCH":

                status_cell.fill = PatternFill(
                    fill_type="solid",
                    fgColor=MISMATCH_FILL,
                )


            elif status_value == "ERROR":

                status_cell.fill = PatternFill(
                    fill_type="solid",
                    fgColor=ERROR_FILL,
                )


        # Add hyperlink to ImageName
        if (
            image_name_column_number
            and image_path_column_number
        ):

            path_value = worksheet.cell(
                row=row_number,
                column=image_path_column_number,
            ).value

            file_uri = create_file_uri(
                path_value
            )

            if file_uri:

                image_cell = worksheet.cell(
                    row=row_number,
                    column=image_name_column_number,
                )

                image_cell.hyperlink = file_uri

                image_cell.style = (
                    "Hyperlink"
                )


    # --------------------------------------------------------
    # CORRECT DEFECT DROPDOWN
    # --------------------------------------------------------

    if correct_defect_column_number:

        defect_validation = DataValidation(
            type="list",
            formula1=(
                f"'_Lists'!$A$1:"
                f"$A${len(class_order)}"
            ),
            allow_blank=True,
        )

        worksheet.add_data_validation(
            defect_validation
        )

        correct_letter = get_column_letter(
            correct_defect_column_number
        )

        defect_validation.add(
            f"{correct_letter}2:"
            f"{correct_letter}"
            f"{max(worksheet.max_row, 2)}"
        )


    # --------------------------------------------------------
    # MANUAL STATUS DROPDOWN
    # --------------------------------------------------------

    if manual_status_column_number:

        status_validation = DataValidation(
            type="list",
            formula1=(
                f"'_Lists'!$B$1:"
                f"$B${len(manual_status_options)}"
            ),
            allow_blank=True,
        )

        worksheet.add_data_validation(
            status_validation
        )

        manual_letter = get_column_letter(
            manual_status_column_number
        )

        status_validation.add(
            f"{manual_letter}2:"
            f"{manual_letter}"
            f"{max(worksheet.max_row, 2)}"
        )


    # --------------------------------------------------------
    # REMOVE IMAGE DROPDOWN
    # --------------------------------------------------------

    if remove_image_column_number:

        remove_validation = DataValidation(
            type="list",
            formula1=(
                f"'_Lists'!$C$1:"
                f"$C${len(remove_options)}"
            ),
            allow_blank=True,
        )

        worksheet.add_data_validation(
            remove_validation
        )

        remove_letter = get_column_letter(
            remove_image_column_number
        )

        remove_validation.add(
            f"{remove_letter}2:"
            f"{remove_letter}"
            f"{max(worksheet.max_row, 2)}"
        )


    last_column_letter = get_column_letter(
        len(headers)
    )

    worksheet.auto_filter.ref = (
        f"A1:"
        f"{last_column_letter}"
        f"{worksheet.max_row}"
    )


# ============================================================
# 17. CREATE SUMMARY SHEET
# ============================================================

summary_headers = [
    "Defect",
    "Total Images",
    "MATCH",
    "MISMATCH",
    "ERROR",
    "Open Sheet",
]


for column_number, header in enumerate(
    summary_headers,
    start=1,
):

    cell = summary_sheet.cell(
        row=1,
        column=column_number,
        value=header,
    )

    cell.font = Font(
        bold=True,
        color=HEADER_FONT,
    )

    cell.fill = PatternFill(
        fill_type="solid",
        fgColor=HEADER_FILL,
    )

    cell.alignment = Alignment(
        horizontal="center",
        vertical="center",
    )


summary_sheet.freeze_panes = "A2"

summary_sheet.column_dimensions[
    "A"
].width = 28

summary_sheet.column_dimensions[
    "B"
].width = 15

summary_sheet.column_dimensions[
    "C"
].width = 15

summary_sheet.column_dimensions[
    "D"
].width = 15

summary_sheet.column_dimensions[
    "E"
].width = 15

summary_sheet.column_dimensions[
    "F"
].width = 18


summary_row = 2


for defect_name in class_order:

    sheet_name = defect_sheet_names.get(
        defect_name
    )

    if not sheet_name:
        continue


    rows = rows_by_defect[
        defect_name
    ]


    match_count = 0
    mismatch_count = 0
    error_count = 0


    if status_column:

        for row in rows:

            status_value = str(
                row.get(
                    status_column,
                    "",
                )
            ).strip().upper()


            if status_value == "MATCH":
                match_count += 1

            elif status_value == "MISMATCH":
                mismatch_count += 1

            elif status_value == "ERROR":
                error_count += 1


    summary_sheet.cell(
        row=summary_row,
        column=1,
        value=defect_name,
    )

    summary_sheet.cell(
        row=summary_row,
        column=2,
        value=len(rows),
    )

    summary_sheet.cell(
        row=summary_row,
        column=3,
        value=match_count,
    )

    summary_sheet.cell(
        row=summary_row,
        column=4,
        value=mismatch_count,
    )

    summary_sheet.cell(
        row=summary_row,
        column=5,
        value=error_count,
    )


    open_cell = summary_sheet.cell(
        row=summary_row,
        column=6,
        value="Open",
    )

    open_cell.hyperlink = (
        f"#'{sheet_name}'!A1"
    )

    open_cell.style = (
        "Hyperlink"
    )


    summary_row += 1


# ============================================================
# 18. SUMMARY TOTAL
# ============================================================

summary_sheet.cell(
    row=summary_row,
    column=1,
    value="TOTAL",
).font = Font(
    bold=True
)


summary_sheet.cell(
    row=summary_row,
    column=2,
    value=len(source_rows),
).font = Font(
    bold=True
)


summary_sheet.cell(
    row=summary_row,
    column=3,
    value=sum(
        1
        for row in source_rows
        if status_column
        and str(
            row.get(
                status_column,
                "",
            )
        ).strip().upper()
        == "MATCH"
    ),
).font = Font(
    bold=True
)


summary_sheet.cell(
    row=summary_row,
    column=4,
    value=sum(
        1
        for row in source_rows
        if status_column
        and str(
            row.get(
                status_column,
                "",
            )
        ).strip().upper()
        == "MISMATCH"
    ),
).font = Font(
    bold=True
)


summary_sheet.cell(
    row=summary_row,
    column=5,
    value=sum(
        1
        for row in source_rows
        if status_column
        and str(
            row.get(
                status_column,
                "",
            )
        ).strip().upper()
        == "ERROR"
    ),
).font = Font(
    bold=True
)


summary_sheet.auto_filter.ref = (
    f"A1:F{summary_row - 1}"
)


# ============================================================
# 19. SAVE WORKBOOK
# ============================================================

workbook.save(
    OUTPUT_FILE
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "SEPARATE DEFECT SHEETS CREATED"
)

print("=" * 70)


print(
    f"Input rows     : "
    f"{len(source_rows)}"
)

print(
    f"Defect sheets : "
    f"{len(defect_sheet_names)}"
)

print(
    f"\nOutput file:\n"
    f"{OUTPUT_FILE}"
)


print(
    "\nThe source file and dataset images "
    "were not changed."
)

print("\nDone.")