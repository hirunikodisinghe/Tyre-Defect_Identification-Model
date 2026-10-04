import os
import shutil
import csv
from pathlib import Path

from PIL import Image
import imagehash


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    r"E:\Hiruni\TireDefectProject"
)

DATASET_DIR = (
    PROJECT_DIR
    / "clean_dataset"
)

REMOVED_DIR = (
    PROJECT_DIR
    / "removed_visual_duplicates"
)

REPORT_FILE = (
    PROJECT_DIR
    / "moved_visual_duplicates.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

HASH_THRESHOLD = 1

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
)


# ============================================================
# 3. CHECK PATH
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"Dataset folder not found:\n"
        f"{DATASET_DIR}"
    )


# ============================================================
# 4. SCAN IMAGES
# ============================================================

images = []

print("=" * 70)
print("VISUAL DUPLICATE REMOVER")
print("=" * 70)

print("\nDataset:")
print(DATASET_DIR)

print("\nScanning images...")


for root, dirs, files in os.walk(
    DATASET_DIR
):

    for file in files:

        if not file.lower().endswith(
            VALID_EXTENSIONS
        ):
            continue


        image_path = (
            Path(root)
            / file
        )


        try:

            with Image.open(
                image_path
            ) as img:

                img_hash = imagehash.phash(
                    img.convert("RGB")
                )


            images.append(
                (
                    image_path,
                    img_hash
                )
            )


        except Exception as error:

            print(
                f"\nCannot read:"
            )

            print(
                image_path
            )

            print(
                error
            )


print(
    f"\nImages scanned: "
    f"{len(images)}"
)


# ============================================================
# 5. FIND DUPLICATE GROUPS
# ============================================================

duplicate_groups = []

used = set()


print(
    "\nFinding duplicate groups..."
)


for i in range(
    len(images)
):

    if i in used:
        continue


    base_path = (
        images[i][0]
    )

    base_hash = (
        images[i][1]
    )


    group = [
        base_path
    ]


    for j in range(
        i + 1,
        len(images)
    ):

        if j in used:
            continue


        compare_path = (
            images[j][0]
        )

        compare_hash = (
            images[j][1]
        )


        hash_difference = (
            base_hash
            -
            compare_hash
        )


        if (
            hash_difference
            <=
            HASH_THRESHOLD
        ):

            group.append(
                compare_path
            )

            used.add(
                j
            )


    if len(group) > 1:

        duplicate_groups.append(
            group
        )


print(
    f"Duplicate groups found: "
    f"{len(duplicate_groups)}"
)


# ============================================================
# 6. CREATE REMOVED FOLDER
# ============================================================

REMOVED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 7. MOVE EXTRA DUPLICATES
# ============================================================

moved_count = 0

report_rows = []


for group_number, group in enumerate(
    duplicate_groups,
    start=1
):

    print(
        "\n" + "-" * 70
    )

    print(
        f"Group {group_number}"
    )


    # ========================================================
    # KEEP FIRST IMAGE
    # ========================================================

    keep_image = (
        group[0]
    )


    print(
        f"KEEP:"
    )

    print(
        keep_image
    )


    keep_class = (
        keep_image.parent.name
    )


    # ========================================================
    # MOVE OTHER DUPLICATES
    # ========================================================

    for duplicate_path in group[1:]:

        duplicate_path = Path(
            duplicate_path
        )


        duplicate_class = (
            duplicate_path.parent.name
        )


        # ----------------------------------------------------
        # Relative path inside clean_dataset
        # ----------------------------------------------------

        try:

            relative_path = (
                duplicate_path.relative_to(
                    DATASET_DIR
                )
            )


        except ValueError:

            print(
                "Skipping unexpected path:"
            )

            print(
                duplicate_path
            )

            continue


        # ----------------------------------------------------
        # Destination
        # ----------------------------------------------------

        destination = (
            REMOVED_DIR
            /
            relative_path
        )


        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        # ----------------------------------------------------
        # Prevent overwrite
        # ----------------------------------------------------

        if destination.exists():

            counter = 1


            while destination.exists():

                destination = (

                    destination.parent

                    /

                    (
                        f"{duplicate_path.stem}"
                        f"_dup_{counter}"
                        f"{duplicate_path.suffix}"
                    )
                )


                counter += 1


        # ----------------------------------------------------
        # MOVE FILE
        # ----------------------------------------------------

        shutil.move(
            str(
                duplicate_path
            ),
            str(
                destination
            )
        )


        moved_count += 1


        print(
            f"MOVED:"
        )

        print(
            duplicate_path
        )

        print(
            f"TO:"
        )

        print(
            destination
        )


        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        report_rows.append({

            "GroupNumber":
                group_number,

            "KeepImage":
                str(
                    keep_image
                ),

            "KeepClass":
                keep_class,

            "MovedImage":
                str(
                    duplicate_path
                ),

            "MovedClass":
                duplicate_class,

            "MovedTo":
                str(
                    destination
                ),

        })


# ============================================================
# 8. SAVE CSV REPORT
# ============================================================

with REPORT_FILE.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    fieldnames = [

        "GroupNumber",

        "KeepImage",

        "KeepClass",

        "MovedImage",

        "MovedClass",

        "MovedTo",

    ]


    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )


    writer.writeheader()


    writer.writerows(
        report_rows
    )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DUPLICATE REMOVAL COMPLETED"
)

print(
    "=" * 70
)


print(
    f"Images scanned             : "
    f"{len(images)}"
)


print(
    f"Duplicate groups           : "
    f"{len(duplicate_groups)}"
)


print(
    f"Duplicate images moved     : "
    f"{moved_count}"
)


print(
    f"\nRemoved images folder:"
)

print(
    REMOVED_DIR
)


print(
    f"\nReport:"
)

print(
    REPORT_FILE
)


print(
    "\nOriginal duplicate files were moved, "
    "not permanently deleted."
)