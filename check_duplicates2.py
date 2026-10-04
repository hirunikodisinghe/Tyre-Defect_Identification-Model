import os
import csv
import shutil
from pathlib import Path

from PIL import Image, ImageFile
import imagehash


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(r"E:\Hiruni\TireDefectProject")

DATASET_DIR = PROJECT_DIR / "clean_dataset"

REVIEW_DIR = PROJECT_DIR / "visual_duplicate_review"

CSV_REPORT = PROJECT_DIR / "visual_duplicate_report.csv"


# Lower = stricter
#
# 0 = identical perceptual hash
# 1 = extremely similar
# 2 = strong near-duplicate candidate
# 3+ = increasingly less strict
#
# Start with 1.
HASH_THRESHOLD = 1


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}


ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_DIR}"
    )


# ============================================================
# PREPARE REVIEW FOLDER
# ============================================================

# Remove OLD review folder so old results do not mix
# with the new scan.
#
# IMPORTANT:
# This deletes only visual_duplicate_review,
# NOT clean_dataset.

if REVIEW_DIR.exists():

    shutil.rmtree(REVIEW_DIR)


REVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SCAN IMAGES
# ============================================================

images = []

corrupted_images = []


print("=" * 70)
print("VISUAL DUPLICATE IMAGE CHECKER")
print("=" * 70)

print(f"\nDataset:")
print(DATASET_DIR)

print(f"\nHash threshold: {HASH_THRESHOLD}")

print("\nScanning images...")


for root, dirs, files in os.walk(DATASET_DIR):

    for file in files:

        file_path = Path(root) / file

        if file_path.suffix.lower() not in VALID_EXTENSIONS:
            continue

        try:

            with Image.open(file_path) as img:

                rgb_image = img.convert("RGB")

                img_hash = imagehash.phash(
                    rgb_image
                )


            # Get class relative to clean_dataset
            relative_path = file_path.relative_to(
                DATASET_DIR
            )

            class_name = relative_path.parts[0]


            images.append({
                "path": file_path,
                "hash": img_hash,
                "class": class_name,
            })


        except Exception as error:

            corrupted_images.append(
                str(file_path)
            )

            print(
                f"\nCannot read: {file_path}"
            )

            print(
                f"Error: {error}"
            )


print(
    f"\nImages successfully scanned: {len(images)}"
)

print(
    f"Unreadable/corrupted images: {len(corrupted_images)}"
)


# ============================================================
# FIND VISUALLY SIMILAR IMAGES
# ============================================================

print("\nComparing images...")

duplicate_groups = []

used = set()


for i in range(len(images)):

    if i in used:
        continue


    base_image = images[i]

    group = [
        {
            **base_image,
            "hash_difference": 0
        }
    ]


    for j in range(i + 1, len(images)):

        if j in used:
            continue


        comparison_image = images[j]


        hash_difference = (
            base_image["hash"]
            -
            comparison_image["hash"]
        )


        if hash_difference <= HASH_THRESHOLD:

            group.append({
                **comparison_image,
                "hash_difference":
                    hash_difference
            })

            used.add(j)


    if len(group) > 1:

        duplicate_groups.append(
            group
        )

        used.add(i)


# ============================================================
# SAFE COPY FUNCTION
# ============================================================

def safe_copy(source_path, destination_folder):

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    destination = (
        destination_folder
        /
        source_path.name
    )


    counter = 1


    while destination.exists():

        destination = (
            destination_folder
            /
            (
                f"{source_path.stem}"
                f"_{counter}"
                f"{source_path.suffix}"
            )
        )

        counter += 1


    shutil.copy2(
        source_path,
        destination
    )


    return destination


# ============================================================
# CREATE REVIEW FOLDERS + CSV DATA
# ============================================================

report_rows = []

same_class_groups = 0

cross_class_groups = 0


print("\n" + "=" * 70)
print("VISUAL DUPLICATE IMAGE REPORT")
print("=" * 70)


if not duplicate_groups:

    print(
        "\nNo visual duplicate candidates found."
    )


else:

    for group_number, group in enumerate(
        duplicate_groups,
        start=1
    ):


        # Get all class names in this group

        classes = {
            item["class"]
            for item in group
        }


        if len(classes) == 1:

            group_type = "SAME_CLASS"

            same_class_groups += 1

        else:

            group_type = "CROSS_CLASS"

            cross_class_groups += 1


        # Example:
        # Group_001_SAME_CLASS
        # Group_002_CROSS_CLASS

        group_folder = (
            REVIEW_DIR
            /
            (
                f"Group_{group_number:03d}"
                f"_{group_type}"
            )
        )


        group_folder.mkdir(
            parents=True,
            exist_ok=True
        )


        print(
            f"\nDuplicate Group {group_number}"
        )

        print(
            f"Type: {group_type}"
        )

        print("-" * 70)


        for item_number, item in enumerate(
            group,
            start=1
        ):

            image_path = item["path"]

            class_name = item["class"]

            hash_difference = item[
                "hash_difference"
            ]


            size_mb = (
                image_path.stat().st_size
                /
                (1024 * 1024)
            )


            print(
                f"Class     : {class_name}"
            )

            print(
                f"File      : {image_path}"
            )

            print(
                f"Hash diff : {hash_difference}"
            )

            print(
                f"Size      : {size_mb:.3f} MB"
            )

            print()


            # Create filename showing original class
            copy_name = (
                f"{item_number:02d}"
                f"__{class_name}"
                f"__{image_path.name}"
            )


            copy_destination = (
                group_folder
                /
                copy_name
            )


            counter = 1


            while copy_destination.exists():

                copy_destination = (
                    group_folder
                    /
                    (
                        f"{item_number:02d}"
                        f"__{class_name}"
                        f"__{image_path.stem}"
                        f"_{counter}"
                        f"{image_path.suffix}"
                    )
                )

                counter += 1


            shutil.copy2(
                image_path,
                copy_destination
            )


            report_rows.append({

                "group_number":
                    group_number,

                "group_type":
                    group_type,

                "class_name":
                    class_name,

                "original_file":
                    str(image_path),

                "file_name":
                    image_path.name,

                "hash_difference":
                    hash_difference,

                "size_mb":
                    f"{size_mb:.4f}",

                "review_copy":
                    str(copy_destination),
            })


# ============================================================
# SAVE CSV REPORT
# ============================================================

with open(
    CSV_REPORT,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:


    fieldnames = [

        "group_number",

        "group_type",

        "class_name",

        "original_file",

        "file_name",

        "hash_difference",

        "size_mb",

        "review_copy",
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
# SUMMARY
# ============================================================

duplicate_image_count = sum(
    len(group)
    for group in duplicate_groups
)


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Images scanned              : {len(images)}"
)

print(
    f"Corrupted/unreadable        : {len(corrupted_images)}"
)

print(
    f"Visual duplicate groups     : {len(duplicate_groups)}"
)

print(
    f"Images in duplicate groups  : {duplicate_image_count}"
)

print(
    f"Same-class groups           : {same_class_groups}"
)

print(
    f"Cross-class groups          : {cross_class_groups}"
)


print("\nReview folder:")

print(
    REVIEW_DIR
)


print("\nCSV report:")

print(
    CSV_REPORT
)


print("\nIMPORTANT:")

print(
    "No images were deleted or moved from clean_dataset."
)

print(
    "Candidate images were COPIED to visual_duplicate_review."
)

print(
    "Review the candidate groups manually before removing anything."
)

print("=" * 70)