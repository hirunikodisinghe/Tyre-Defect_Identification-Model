from pathlib import Path
import csv


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(r"E:\Hiruni\TireDefectProject")

INPUT_CSV = (
    PROJECT_DIR
    / "manual_inspection_with_predictions.csv"
)

OUTPUT_CSV = (
    PROJECT_DIR
    / "manual_inspection_mismatches.csv"
)


# ============================================================
# 2. PRIORITY CLASSES
# ============================================================

# Weakest / most important classes first
PRIORITY_CLASSES = [
    "Sidewall_Damage",
    "Tread_Damage",
    "Tread_Chunking",
    "Abnormal_Wear",
    "Cut",
    "Sidewall_Crack",
    "Bulge",
]


# ============================================================
# 3. CHECK INPUT FILE
# ============================================================

if not INPUT_CSV.exists():

    raise FileNotFoundError(
        f"CSV file not found:\n{INPUT_CSV}"
    )


# ============================================================
# 4. READ CSV
# ============================================================

with open(
    INPUT_CSV,
    "r",
    encoding="utf-8-sig",
    newline=""
) as csv_file:

    reader = csv.DictReader(csv_file)

    fieldnames = reader.fieldnames

    if fieldnames is None:

        raise ValueError(
            "CSV does not contain column headers."
        )

    rows = list(reader)


# ============================================================
# 5. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "PredictionStatus",
    "TyreDefectName"
]


for column in required_columns:

    if column not in fieldnames:

        raise ValueError(
            f"Required column not found: {column}"
        )


# ============================================================
# 6. FILTER MISMATCHES
# ============================================================

mismatch_rows = []


for row in rows:

    prediction_status = (
        row.get(
            "PredictionStatus",
            ""
        )
        .strip()
        .upper()
    )

    tyre_defect = (
        row.get(
            "TyreDefectName",
            ""
        )
        .strip()
    )


    # Only model mismatches
    if prediction_status != "MISMATCH":
        continue


    # Only priority weak classes
    if tyre_defect not in PRIORITY_CLASSES:
        continue


    mismatch_rows.append(row)


# ============================================================
# 7. SORT BY PRIORITY CLASS
# ============================================================

priority_order = {

    class_name: index

    for index, class_name
    in enumerate(PRIORITY_CLASSES)

}


def sort_key(row):

    class_name = row.get(
        "TyreDefectName",
        ""
    ).strip()


    confidence_text = row.get(
        "ConfidencePercent",
        "0"
    ).strip()


    try:

        confidence = float(
            confidence_text
        )

    except ValueError:

        confidence = 0


    return (
        priority_order.get(
            class_name,
            999
        ),

        -confidence
    )


mismatch_rows.sort(
    key=sort_key
)


# ============================================================
# 8. SAVE FILTERED CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        mismatch_rows
    )


# ============================================================
# 9. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "MANUAL INSPECTION MISMATCH CSV CREATED"
)

print("=" * 70)


print(
    f"Total original rows : {len(rows)}"
)

print(
    f"Rows for inspection : {len(mismatch_rows)}"
)


print("\nImages requiring inspection by class:")

print("-" * 70)


for class_name in PRIORITY_CLASSES:

    count = sum(

        1

        for row in mismatch_rows

        if (
            row.get(
                "TyreDefectName",
                ""
            ).strip()
            ==
            class_name
        )
    )


    print(
        f"{class_name:25s}: {count}"
    )


print("\nPriority order:")

for index, class_name in enumerate(
    PRIORITY_CLASSES,
    start=1
):

    print(
        f"{index}. {class_name}"
    )


print("\nCSV saved to:")

print(
    OUTPUT_CSV
)


print("\nDone.")