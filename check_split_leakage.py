# make sure your test and validate data are truely independent from your training data

from pathlib import Path
import hashlib
from collections import defaultdict


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "train_split_dataset"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"


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
    ".tiff",
}


# ============================================================
# 3. HASH FUNCTION
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
# 4. COLLECT IMAGES FROM ONE SPLIT
# ============================================================

def collect_images(split_name, split_dir):

    results = []

    for class_folder in split_dir.iterdir():

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        for image_path in class_folder.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in IMAGE_EXTENSIONS
            ):

                try:

                    image_hash = calculate_sha256(
                        image_path
                    )

                    results.append({
                        "split": split_name,
                        "class": class_name,
                        "path": image_path,
                        "hash": image_hash,
                    })

                except Exception as exc:

                    print(
                        f"Could not read: {image_path}"
                    )

                    print(exc)

    return results


# ============================================================
# 5. CHECK FOLDERS
# ============================================================

for folder in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
]:

    if not folder.exists():

        raise FileNotFoundError(
            f"Folder not found:\n{folder}"
        )


# ============================================================
# 6. LOAD ALL IMAGES
# ============================================================

print("=" * 70)
print("TRAIN / VAL / TEST LEAKAGE CHECK")
print("=" * 70)

print("\nScanning TRAIN...")

train_images = collect_images(
    "train",
    TRAIN_DIR
)

print(
    "Train images:",
    len(train_images)
)


print("\nScanning VAL...")

val_images = collect_images(
    "val",
    VAL_DIR
)

print(
    "Val images:",
    len(val_images)
)


print("\nScanning TEST...")

test_images = collect_images(
    "test",
    TEST_DIR
)

print(
    "Test images:",
    len(test_images)
)


all_images = (
    train_images
    + val_images
    + test_images
)


# ============================================================
# 7. GROUP BY HASH
# ============================================================

hash_groups = defaultdict(list)

for item in all_images:

    hash_groups[
        item["hash"]
    ].append(item)


# ============================================================
# 8. FIND CROSS-SPLIT DUPLICATES
# ============================================================

leakage_groups = []

for image_hash, items in hash_groups.items():

    splits = {
        item["split"]
        for item in items
    }

    if len(splits) > 1:

        leakage_groups.append(
            (
                image_hash,
                items
            )
        )


# ============================================================
# 9. DISPLAY LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("LEAKAGE RESULTS")
print("=" * 70)


if not leakage_groups:

    print(
        "\nNo exact duplicate leakage found."
    )

else:

    print(
        f"\nWARNING: "
        f"{len(leakage_groups)} "
        f"cross-split duplicate groups found."
    )


    for group_number, (
        image_hash,
        items
    ) in enumerate(
        leakage_groups,
        start=1
    ):

        print(
            f"\nLeakage Group {group_number}"
        )

        print("-" * 70)

        for item in items:

            print(
                f"Split : {item['split']}"
            )

            print(
                f"Class : {item['class']}"
            )

            print(
                f"File  : {item['path']}"
            )

            print()


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    "Train images:",
    len(train_images)
)

print(
    "Val images  :",
    len(val_images)
)

print(
    "Test images :",
    len(test_images)
)

print(
    "Total images:",
    len(all_images)
)

print(
    "Cross-split duplicate groups:",
    len(leakage_groups)
)


if len(leakage_groups) == 0:

    print(
        "\nRESULT: PASS"
    )

    print(
        "No exact duplicate leakage "
        "between train, validation, and test."
    )

else:

    print(
        "\nRESULT: FAIL"
    )

    print(
        "Duplicate images exist across splits."
    )

    print(
        "Fix these before model training."
    )


print("\nDone.")