from pathlib import Path
import hashlib
import csv
import shutil

from PIL import Image


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "clean_dataset"

DUPLICATE_BACKUP_DIR = PROJECT_DIR / "duplicate_backup"

REPORT_PATH = PROJECT_DIR / "duplicate_report.csv"

CORRUPT_REPORT_PATH = PROJECT_DIR / "corrupt_images.csv"


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}

# True  = move duplicate files to duplicate_backup
# False = only report them
MOVE_DUPLICATES = False


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_DIR}"
    )


print("=" * 70)
print("TIRE DATASET DUPLICATE CHECKER")
print("=" * 70)

print("\nDataset:")
print(DATASET_DIR)

print("\nMove duplicates:", MOVE_DUPLICATES)


# ============================================================
# 4. GET ALL IMAGE FILES
# ============================================================

image_files = []

for class_folder in DATASET_DIR.iterdir():

    if not class_folder.is_dir():
        continue

    for file_path in class_folder.rglob("*"):

        if (
            file_path.is_file()
            and file_path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            image_files.append(file_path)


print("\nTotal images found:", len(image_files))


# ============================================================
# 5. CHECK CORRUPTED IMAGES
# ============================================================

valid_images = []

corrupt_images = []


print("\nChecking corrupted/unreadable images...")


for index, image_path in enumerate(image_files, start=1):

    try:

        with Image.open(image_path) as img:
            img.verify()

        valid_images.append(image_path)

    except Exception as exc:

        corrupt_images.append(
            {
                "file": str(image_path),
                "class": image_path.parent.name,
                "error": str(exc)
            }
        )

    if index % 250 == 0:
        print(
            f"Checked {index}/{len(image_files)} images..."
        )


# ============================================================
# 6. SAVE CORRUPT IMAGE REPORT
# ============================================================

with open(
    CORRUPT_REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "class",
        "file",
        "error"
    ])

    for item in corrupt_images:

        writer.writerow([
            item["class"],
            item["file"],
            item["error"]
        ])


print(
    "\nCorrupted/unreadable images:",
    len(corrupt_images)
)


# ============================================================
# 7. SHA-256 HASH FUNCTION
# ============================================================

def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 8. FIND EXACT DUPLICATES
# ============================================================

print("\nChecking exact duplicate images...")


hash_map = {}

duplicates = []


for index, image_path in enumerate(valid_images, start=1):

    try:

        image_hash = calculate_sha256(
            image_path
        )

    except Exception as exc:

        print(
            f"Could not hash: {image_path}"
        )
        print(exc)
        continue


    if image_hash not in hash_map:

        hash_map[image_hash] = image_path

    else:

        original_path = hash_map[image_hash]

        duplicate_info = {
            "hash": image_hash,

            "original_class":
                original_path.parent.name,

            "original_file":
                str(original_path),

            "duplicate_class":
                image_path.parent.name,

            "duplicate_file":
                str(image_path),

            "cross_class":
                original_path.parent.name
                != image_path.parent.name
        }

        duplicates.append(
            duplicate_info
        )


    if index % 250 == 0:

        print(
            f"Hashed {index}/{len(valid_images)} images..."
        )


# ============================================================
# 9. SAVE DUPLICATE REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow([
        "hash",
        "original_class",
        "original_file",
        "duplicate_class",
        "duplicate_file",
        "cross_class"
    ])


    for item in duplicates:

        writer.writerow([
            item["hash"],
            item["original_class"],
            item["original_file"],
            item["duplicate_class"],
            item["duplicate_file"],
            item["cross_class"]
        ])


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

same_class_duplicates = [
    item
    for item in duplicates
    if not item["cross_class"]
]

cross_class_duplicates = [
    item
    for item in duplicates
    if item["cross_class"]
]


print("\n" + "=" * 70)
print("DUPLICATE CHECK SUMMARY")
print("=" * 70)


print(
    "Total images scanned       :",
    len(image_files)
)

print(
    "Valid images               :",
    len(valid_images)
)

print(
    "Corrupted images           :",
    len(corrupt_images)
)

print(
    "Exact duplicate files      :",
    len(duplicates)
)

print(
    "Same-class duplicates      :",
    len(same_class_duplicates)
)

print(
    "Cross-class duplicates     :",
    len(cross_class_duplicates)
)


# ============================================================
# 11. SHOW CROSS-CLASS DUPLICATES
# ============================================================

if cross_class_duplicates:

    print(
        "\nWARNING: Cross-class duplicate images found!"
    )

    print(
        "These require manual review because the same image"
        " appears under different defect labels."
    )


    for item in cross_class_duplicates[:20]:

        print("\n----------------------------------")

        print(
            "Original:"
        )

        print(
            item["original_file"]
        )

        print(
            "Duplicate:"
        )

        print(
            item["duplicate_file"]
        )


# ============================================================
# 12. OPTIONAL MOVE DUPLICATES
# ============================================================

if MOVE_DUPLICATES:

    print(
        "\nMoving duplicate files..."
    )


    for item in duplicates:

        duplicate_path = Path(
            item["duplicate_file"]
        )


        relative_path = duplicate_path.relative_to(
            DATASET_DIR
        )


        destination = (
            DUPLICATE_BACKUP_DIR
            / relative_path
        )


        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        # Avoid overwriting existing files
        if destination.exists():

            stem = destination.stem

            suffix = destination.suffix

            counter = 1

            while destination.exists():

                destination = (
                    destination.parent
                    / f"{stem}_duplicate_{counter}{suffix}"
                )

                counter += 1


        shutil.move(
            str(duplicate_path),
            str(destination)
        )


    print(
        "\nDuplicates moved to:"
    )

    print(
        DUPLICATE_BACKUP_DIR
    )


else:

    print(
        "\nNo files were moved."
    )

    print(
        "Review duplicate_report.csv first."
    )


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print("\nReports created:")

print(
    "Duplicate report:"
)

print(
    REPORT_PATH
)

print(
    "\nCorrupt image report:"
)

print(
    CORRUPT_REPORT_PATH
)

print("\nDone.")