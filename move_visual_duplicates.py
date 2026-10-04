import csv
import shutil
from pathlib import Path
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"E:\Hiruni\TireDefectProject")

DATASET_DIR = PROJECT_DIR / "clean_dataset"

REPORT_FILE = PROJECT_DIR / "visual_duplicate_report.csv"

BACKUP_DIR = PROJECT_DIR / "removed_visual_duplicates"


# ============================================================
# CHECK FILES
# ============================================================

if not REPORT_FILE.exists():
    raise FileNotFoundError(
        f"Report not found:\n{REPORT_FILE}"
    )

BACKUP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# READ REPORT
# ============================================================

groups = defaultdict(list)

with open(
    REPORT_FILE,
    "r",
    encoding="utf-8-sig"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        group_number = row["group_number"]

        groups[group_number].append(
            Path(row["original_file"])
        )


# ============================================================
# PROCESS GROUPS
# ============================================================

groups_processed = 0
images_moved = 0
missing_files = 0


print("=" * 70)
print("VISUAL DUPLICATE CLEANER")
print("=" * 70)


for group_number, images in groups.items():

    existing_images = [
        image
        for image in images
        if image.exists()
    ]

    if len(existing_images) < 2:
        continue


    groups_processed += 1


    # --------------------------------------------------------
    # KEEP FIRST IMAGE
    # --------------------------------------------------------

    keep_image = existing_images[0]

    print("\n" + "-" * 70)

    print(
        f"Group {group_number}"
    )

    print(
        f"KEEP : {keep_image}"
    )


    # --------------------------------------------------------
    # MOVE REMAINING IMAGES
    # --------------------------------------------------------

    for duplicate_path in existing_images[1:]:

        try:

            relative_path = (
                duplicate_path.relative_to(
                    DATASET_DIR
                )
            )

        except ValueError:

            print(
                f"Unexpected path, skipped: "
                f"{duplicate_path}"
            )

            continue


        destination = (
            BACKUP_DIR
            / relative_path
        )


        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        # Prevent overwriting existing files

        if destination.exists():

            counter = 1

            original_destination = destination

            while destination.exists():

                destination = (
                    original_destination.parent
                    /
                    (
                        f"{original_destination.stem}"
                        f"_dup_{counter}"
                        f"{original_destination.suffix}"
                    )
                )

                counter += 1


        shutil.move(
            str(duplicate_path),
            str(destination)
        )


        images_moved += 1


        print(
            f"MOVED: {duplicate_path}"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINISHED")
print("=" * 70)

print(
    f"Groups processed : {groups_processed}"
)

print(
    f"Images moved     : {images_moved}"
)

print(
    "\nBackup location:"
)

print(
    BACKUP_DIR
)

print(
    "\nNo images were permanently deleted."
)