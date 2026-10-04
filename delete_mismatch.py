import os
import shutil
from openpyxl import load_workbook

# ============================================================
# SETTINGS
# ============================================================

EXCEL_FILE = (
    r"E:\Hiruni\TireDefectProject\results"
    r"\manual_inspection_separate_sheets.xlsx"
)

CLEAN_DATASET_DIR = (
    r"E:\Hiruni\TireDefectProject\clean_dataset"
)

REMOVED_DIR = (
    r"E:\Hiruni\TireDefectProject"
    r"\removed_mismatch_from_clean_dataset"
)

# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(EXCEL_FILE):
    raise FileNotFoundError(
        f"Excel file not found:\n{EXCEL_FILE}"
    )

if not os.path.exists(CLEAN_DATASET_DIR):
    raise FileNotFoundError(
        f"Clean dataset folder not found:\n{CLEAN_DATASET_DIR}"
    )

os.makedirs(REMOVED_DIR, exist_ok=True)

# ============================================================
# LOAD EXCEL
# ============================================================

workbook = load_workbook(
    EXCEL_FILE,
    data_only=True
)

print("Excel loaded successfully.")

# ============================================================
# COUNTERS
# ============================================================

total_mismatch = 0
moved_count = 0
already_moved_count = 0
not_found_count = 0
error_count = 0

# ============================================================
# PROCESS SHEETS
# ============================================================

for sheet_name in workbook.sheetnames:

    if sheet_name in ["Summary", "_Lists"]:
        continue

    sheet = workbook[sheet_name]

    print("\n" + "=" * 60)
    print("Processing:", sheet_name)
    print("=" * 60)

    # ========================================================
    # GET HEADERS
    # ========================================================

    headers = {}

    for cell in sheet[1]:
        if cell.value is not None:
            headers[str(cell.value).strip()] = cell.column

    required_columns = [
        "ImageName",
        "TyreDefectName",
        "PredictionStatus"
    ]

    missing = [
        col for col in required_columns
        if col not in headers
    ]

    if missing:
        print(
            f"Skipping {sheet_name}. "
            f"Missing columns: {missing}"
        )
        continue

    image_name_col = headers["ImageName"]
    class_col = headers["TyreDefectName"]
    status_col = headers["PredictionStatus"]

    # ========================================================
    # PROCESS ROWS
    # ========================================================

    for row_number in range(
        2,
        sheet.max_row + 1
    ):

        image_name = sheet.cell(
            row=row_number,
            column=image_name_col
        ).value

        class_name = sheet.cell(
            row=row_number,
            column=class_col
        ).value

        status = sheet.cell(
            row=row_number,
            column=status_col
        ).value

        if (
            image_name is None
            or class_name is None
            or status is None
        ):
            continue

        image_name = str(image_name).strip()
        class_name = str(class_name).strip()
        status = str(status).strip().upper()

        # ====================================================
        # ONLY MISMATCH
        # ====================================================

        if status != "MISMATCH":
            continue

        total_mismatch += 1

        # ====================================================
        # SOURCE IN clean_dataset
        # ====================================================

        source_path = os.path.join(
            CLEAN_DATASET_DIR,
            class_name,
            image_name
        )

        # ====================================================
        # DESTINATION
        # ====================================================

        destination_class_dir = os.path.join(
            REMOVED_DIR,
            class_name
        )

        os.makedirs(
            destination_class_dir,
            exist_ok=True
        )

        destination_path = os.path.join(
            destination_class_dir,
            image_name
        )

        # ====================================================
        # MOVE FILE
        # ====================================================

        try:

            if os.path.exists(source_path):

                # Avoid overwrite
                if os.path.exists(destination_path):

                    name, ext = os.path.splitext(
                        image_name
                    )

                    counter = 1

                    while os.path.exists(destination_path):

                        new_name = (
                            f"{name}_{counter}{ext}"
                        )

                        destination_path = os.path.join(
                            destination_class_dir,
                            new_name
                        )

                        counter += 1

                # Removes from clean_dataset
                # and stores in removed folder
                shutil.move(
                    source_path,
                    destination_path
                )

                moved_count += 1

                print(
                    f"MOVED: "
                    f"{class_name}\\{image_name}"
                )

            else:

                # Check whether already moved
                original_destination = os.path.join(
                    destination_class_dir,
                    image_name
                )

                if os.path.exists(original_destination):

                    already_moved_count += 1

                    print(
                        f"ALREADY MOVED: "
                        f"{class_name}\\{image_name}"
                    )

                else:

                    not_found_count += 1

                    print(
                        f"NOT FOUND: "
                        f"{source_path}"
                    )

        except Exception as e:

            error_count += 1

            print(
                f"ERROR: "
                f"{class_name}\\{image_name}"
            )

            print("Reason:", e)

# ============================================================
# COUNT REMAINING IMAGES
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)

remaining_images = 0

for root, dirs, files in os.walk(
    CLEAN_DATASET_DIR
):
    for file in files:
        if file.lower().endswith(
            IMAGE_EXTENSIONS
        ):
            remaining_images += 1

# ============================================================
# COUNT REMOVED IMAGES
# ============================================================

removed_images = 0

for root, dirs, files in os.walk(
    REMOVED_DIR
):
    for file in files:
        if file.lower().endswith(
            IMAGE_EXTENSIONS
        ):
            removed_images += 1

# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("CLEAN DATASET MISMATCH REMOVAL COMPLETED")
print("=" * 60)

print("Mismatch records found :", total_mismatch)
print("Images moved           :", moved_count)
print("Already moved          :", already_moved_count)
print("Images not found       :", not_found_count)
print("Errors                 :", error_count)
print("Remaining clean images :", remaining_images)
print("Stored removed images  :", removed_images)

print("\nRemoved images stored in:")
print(REMOVED_DIR)

print("\nDone.")