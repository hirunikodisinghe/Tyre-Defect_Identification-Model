from pathlib import Path
import shutil
import csv

from openpyxl import load_workbook


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    r"E:\Hiruni\TireDefectProject"
)

EXCEL_FILE = (
    PROJECT_DIR
    / "manual_inspection_by_defect.xlsx"
)

DATASET_DIR = (
    PROJECT_DIR
    / "clean_dataset"
)

MISMATCH_REMOVED_DIR = (
    PROJECT_DIR
    / "model_mismatch_removed"
)

MOVE_LOG_FILE = (
    PROJECT_DIR
    / "moved_model_mismatches.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

MISMATCH_STATUS = "MISMATCH"


# ============================================================
# 3. VALIDATE PATHS
# ============================================================

if not EXCEL_FILE.exists():
    raise FileNotFoundError(
        f"Excel file not found:\n{EXCEL_FILE}"
    )

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_DIR}"
    )


MISMATCH_REMOVED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 4. LOAD EXCEL
# ============================================================

print("=" * 70)
print("MOVE MODEL MISMATCH IMAGES FROM CLEAN_DATASET")
print("=" * 70)

print("\nExcel file:")
print(EXCEL_FILE)

print("\nDataset:")
print(DATASET_DIR)

print("\nDestination:")
print(MISMATCH_REMOVED_DIR)


workbook = load_workbook(
    EXCEL_FILE,
    data_only=True
)


# ============================================================
# 5. COUNTERS
# ============================================================

total_rows_checked = 0
mismatch_rows = 0
moved_count = 0
not_found_count = 0
skipped_count = 0


move_log = []


# ============================================================
# 6. PROCESS EACH DEFECT SHEET
# ============================================================

for sheet_name in workbook.sheetnames:

    if sheet_name.lower() in [
        "summary",
        "_lists"
    ]:
        continue


    worksheet = workbook[
        sheet_name
    ]


    print("\n" + "-" * 70)
    print(f"Checking sheet: {sheet_name}")


    # ========================================================
    # 7. READ HEADERS
    # ========================================================

    headers = {}


    for column_number, cell in enumerate(
        worksheet[1],
        start=1
    ):

        if cell.value is not None:

            headers[
                str(cell.value).strip()
            ] = column_number


    # ========================================================
    # 8. FIND REQUIRED COLUMNS
    # ========================================================

    status_column = headers.get(
        "PredictionStatus"
    )

    image_name_column = headers.get(
        "ImageName"
    )

    actual_defect_column = (
        headers.get("TyreDefectName")
        or
        headers.get("ActualDefect")
    )

    model_prediction_column = headers.get(
        "ModelPrediction"
    )

    confidence_column = headers.get(
        "ConfidencePercent"
    )


    # ========================================================
    # 9. VALIDATE COLUMNS
    # ========================================================

    if not status_column:

        print(
            "PredictionStatus column not found. "
            "Skipping this sheet."
        )

        continue


    if not image_name_column:

        print(
            "ImageName column not found. "
            "Skipping this sheet."
        )

        continue


    # ========================================================
    # 10. PROCESS ROWS
    # ========================================================

    for row_number in range(
        2,
        worksheet.max_row + 1
    ):

        total_rows_checked += 1


        status_value = worksheet.cell(
            row=row_number,
            column=status_column
        ).value


        if status_value is None:
            continue


        status = str(
            status_value
        ).strip().upper()


        # ----------------------------------------------------
        # ONLY MISMATCH
        # ----------------------------------------------------

        if status != MISMATCH_STATUS:
            continue


        mismatch_rows += 1


        # ====================================================
        # 11. GET VALUES
        # ====================================================

        image_name = worksheet.cell(
            row=row_number,
            column=image_name_column
        ).value


        if not image_name:

            skipped_count += 1

            print(
                f"SKIPPED row {row_number}: "
                f"ImageName missing"
            )

            continue


        image_name = str(
            image_name
        ).strip()


        # Use column if available,
        # otherwise sheet name = defect
        actual_defect = (
            worksheet.cell(
                row=row_number,
                column=actual_defect_column
            ).value
            if actual_defect_column
            else sheet_name
        )


        actual_defect = str(
            actual_defect
        ).strip()


        model_prediction = (
            worksheet.cell(
                row=row_number,
                column=model_prediction_column
            ).value
            if model_prediction_column
            else ""
        )


        confidence = (
            worksheet.cell(
                row=row_number,
                column=confidence_column
            ).value
            if confidence_column
            else ""
        )


        # ====================================================
        # 12. FIND ORIGINAL IMAGE INSIDE clean_dataset
        # ====================================================

        class_folder = (
            DATASET_DIR
            / actual_defect
        )


        if not class_folder.exists():

            skipped_count += 1

            print(
                f"CLASS FOLDER NOT FOUND:"
            )

            print(
                class_folder
            )

            move_log.append({
                "Sheet": sheet_name,
                "Row": row_number,
                "ImageName": image_name,
                "ActualDefect": actual_defect,
                "ModelPrediction": model_prediction,
                "ConfidencePercent": confidence,
                "OriginalPath": "",
                "MovedTo": "",
                "Result": "CLASS FOLDER NOT FOUND"
            })

            continue


        # First try exact path
        source_path = (
            class_folder
            / image_name
        )


        # If exact file not found,
        # search recursively inside class folder
        if not source_path.exists():

            matches = list(
                class_folder.rglob(
                    image_name
                )
            )


            if len(matches) == 1:

                source_path = matches[0]


            elif len(matches) > 1:

                skipped_count += 1

                print(
                    f"MULTIPLE FILES FOUND:"
                )

                print(
                    image_name
                )

                for match in matches:
                    print(
                        f"  {match}"
                    )


                move_log.append({
                    "Sheet": sheet_name,
                    "Row": row_number,
                    "ImageName": image_name,
                    "ActualDefect": actual_defect,
                    "ModelPrediction": model_prediction,
                    "ConfidencePercent": confidence,
                    "OriginalPath": "",
                    "MovedTo": "",
                    "Result": "MULTIPLE FILES FOUND"
                })

                continue


            else:

                not_found_count += 1

                print(
                    f"NOT FOUND:"
                )

                print(
                    class_folder
                    / image_name
                )


                move_log.append({
                    "Sheet": sheet_name,
                    "Row": row_number,
                    "ImageName": image_name,
                    "ActualDefect": actual_defect,
                    "ModelPrediction": model_prediction,
                    "ConfidencePercent": confidence,
                    "OriginalPath": str(
                        class_folder
                        / image_name
                    ),
                    "MovedTo": "",
                    "Result": "NOT FOUND"
                })

                continue


        # ====================================================
        # 13. DESTINATION
        # ====================================================

        destination_folder = (
            MISMATCH_REMOVED_DIR
            / actual_defect
        )


        destination_folder.mkdir(
            parents=True,
            exist_ok=True
        )


        destination = (
            destination_folder
            / source_path.name
        )


        # ====================================================
        # 14. PREVENT OVERWRITE
        # ====================================================

        if destination.exists():

            counter = 1


            while destination.exists():

                destination = (
                    destination_folder
                    /
                    (
                        f"{source_path.stem}"
                        f"_mismatch_{counter}"
                        f"{source_path.suffix}"
                    )
                )

                counter += 1


        # ====================================================
        # 15. MOVE IMAGE
        # ====================================================

        try:

            shutil.move(
                str(source_path),
                str(destination)
            )


            moved_count += 1


            print(
                f"MOVED [{actual_defect}] "
                f"{image_name}"
            )

            print(
                f"   FROM: {source_path}"
            )

            print(
                f"   TO  : {destination}"
            )


            move_log.append({
                "Sheet": sheet_name,
                "Row": row_number,
                "ImageName": image_name,
                "ActualDefect": actual_defect,
                "ModelPrediction": model_prediction,
                "ConfidencePercent": confidence,
                "OriginalPath": str(source_path),
                "MovedTo": str(destination),
                "Result": "MOVED"
            })


        except Exception as error:

            skipped_count += 1


            print(
                f"ERROR moving:"
            )

            print(
                source_path
            )

            print(
                error
            )


            move_log.append({
                "Sheet": sheet_name,
                "Row": row_number,
                "ImageName": image_name,
                "ActualDefect": actual_defect,
                "ModelPrediction": model_prediction,
                "ConfidencePercent": confidence,
                "OriginalPath": str(source_path),
                "MovedTo": "",
                "Result": f"ERROR - {error}"
            })


# ============================================================
# 16. CLOSE WORKBOOK
# ============================================================

workbook.close()


# ============================================================
# 17. SAVE MOVE LOG
# ============================================================

with MOVE_LOG_FILE.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    fieldnames = [
        "Sheet",
        "Row",
        "ImageName",
        "ActualDefect",
        "ModelPrediction",
        "ConfidencePercent",
        "OriginalPath",
        "MovedTo",
        "Result",
    ]


    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )


    writer.writeheader()

    writer.writerows(
        move_log
    )


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MISMATCH MOVE COMPLETED")
print("=" * 70)


print(
    f"Rows checked           : "
    f"{total_rows_checked}"
)

print(
    f"Mismatch rows found    : "
    f"{mismatch_rows}"
)

print(
    f"Images moved           : "
    f"{moved_count}"
)

print(
    f"Images not found       : "
    f"{not_found_count}"
)

print(
    f"Images skipped/errors  : "
    f"{skipped_count}"
)


print(
    "\nMoved images are stored in:"
)

print(
    MISMATCH_REMOVED_DIR
)


print(
    "\nMove log:"
)

print(
    MOVE_LOG_FILE
)