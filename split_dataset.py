from pathlib import Path
import random
import shutil


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

SOURCE_DIR = PROJECT_DIR / "clean_dataset"

OUTPUT_DIR = PROJECT_DIR / "train_split_dataset"


# ============================================================
# 2. SPLIT SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

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
# 3. CHECK SETTINGS
# ============================================================

if abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) > 1e-9:
    raise ValueError(
        "Train, validation, and test ratios must add up to 1.0."
    )


if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"clean_dataset folder not found:\n{SOURCE_DIR}"
    )


# ============================================================
# 4. IMPORTANT SAFETY CHECK
# ============================================================

if OUTPUT_DIR.exists():

    raise FileExistsError(
        f"\nThe output folder already exists:\n{OUTPUT_DIR}\n\n"
        "Rename or delete the old train_split_dataset folder "
        "before running this script."
    )


# ============================================================
# 5. CREATE OUTPUT FOLDERS
# ============================================================

for split_name in [
    "train",
    "val",
    "test",
]:

    (
        OUTPUT_DIR / split_name
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 6. RANDOM GENERATOR
# ============================================================

random.seed(
    RANDOM_SEED
)


# ============================================================
# 7. SUMMARY COUNTERS
# ============================================================

total_source = 0

total_train = 0
total_val = 0
total_test = 0


print("=" * 70)
print("TIRE DEFECT DATASET SPLITTER")
print("=" * 70)

print("\nSource dataset:")
print(SOURCE_DIR)

print("\nOutput dataset:")
print(OUTPUT_DIR)

print(
    "\nSplit ratio:"
)

print(
    f"Train      : {TRAIN_RATIO * 100:.0f}%"
)

print(
    f"Validation : {VAL_RATIO * 100:.0f}%"
)

print(
    f"Test       : {TEST_RATIO * 100:.0f}%"
)


# ============================================================
# 8. PROCESS EACH CLASS
# ============================================================

class_folders = sorted(
    [
        folder
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    ]
)


if not class_folders:

    raise ValueError(
        "No class folders were found inside clean_dataset."
    )


for class_folder in class_folders:

    class_name = class_folder.name


    # --------------------------------------------------------
    # GET IMAGES
    # --------------------------------------------------------

    images = sorted(
        [
            image
            for image in class_folder.rglob("*")
            if (
                image.is_file()
                and image.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]
    )


    total = len(images)


    if total == 0:

        print(
            f"\nWARNING: {class_name} contains 0 images."
        )

        continue


    # --------------------------------------------------------
    # RANDOM SHUFFLE
    # --------------------------------------------------------

    random.shuffle(
        images
    )


    # --------------------------------------------------------
    # CALCULATE COUNTS
    # --------------------------------------------------------

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    test_count = (
        total
        - train_count
        - val_count
    )


    # --------------------------------------------------------
    # CREATE SPLITS
    # --------------------------------------------------------

    train_images = images[
        :train_count
    ]

    val_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]


    # --------------------------------------------------------
    # CREATE CLASS FOLDERS
    # --------------------------------------------------------

    train_class_dir = (
        OUTPUT_DIR
        / "train"
        / class_name
    )

    val_class_dir = (
        OUTPUT_DIR
        / "val"
        / class_name
    )

    test_class_dir = (
        OUTPUT_DIR
        / "test"
        / class_name
    )


    train_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    val_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    test_class_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # COPY TRAIN IMAGES
    # --------------------------------------------------------

    for image_path in train_images:

        destination = (
            train_class_dir
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination
        )


    # --------------------------------------------------------
    # COPY VALIDATION IMAGES
    # --------------------------------------------------------

    for image_path in val_images:

        destination = (
            val_class_dir
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination
        )


    # --------------------------------------------------------
    # COPY TEST IMAGES
    # --------------------------------------------------------

    for image_path in test_images:

        destination = (
            test_class_dir
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination
        )


    # --------------------------------------------------------
    # UPDATE TOTALS
    # --------------------------------------------------------

    total_source += total

    total_train += len(
        train_images
    )

    total_val += len(
        val_images
    )

    total_test += len(
        test_images
    )


    # --------------------------------------------------------
    # PRINT CLASS SUMMARY
    # --------------------------------------------------------

    print(
        f"\n{class_name}"
    )

    print(
        f"Total : {total}"
    )

    print(
        f"Train : {len(train_images)}"
    )

    print(
        f"Val   : {len(val_images)}"
    )

    print(
        f"Test  : {len(test_images)}"
    )


# ============================================================
# 9. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DATASET SPLIT COMPLETED")
print("=" * 70)

print(
    f"Total source images : {total_source}"
)

print(
    f"Training images     : {total_train}"
)

print(
    f"Validation images   : {total_val}"
)

print(
    f"Test images         : {total_test}"
)

print(
    f"Total split images  : "
    f"{total_train + total_val + total_test}"
)


# ============================================================
# 10. FINAL CONSISTENCY CHECK
# ============================================================

if (
    total_source
    !=
    total_train
    + total_val
    + total_test
):

    raise RuntimeError(
        "ERROR: Split image count does not match source count."
    )


print(
    "\nAll images were split successfully."
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_DIR
)

print("\nDone.")