from pathlib import Path
from collections import defaultdict


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(r"E:\Hiruni\TireDefectProject")

SPLIT_DIR = PROJECT_DIR / "train_split_dataset"

TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
TEST_DIR = SPLIT_DIR / "test"


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# CHECK FOLDERS
# ============================================================

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

    if not folder.exists():

        raise FileNotFoundError(
            f"Folder not found:\n{folder}"
        )


# ============================================================
# FUNCTION TO COUNT IMAGES
# ============================================================

def count_images(folder):

    counts = defaultdict(int)

    for class_folder in folder.iterdir():

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        for file_path in class_folder.rglob("*"):

            if (
                file_path.is_file()
                and
                file_path.suffix.lower() in VALID_EXTENSIONS
            ):

                counts[class_name] += 1

    return dict(counts)


# ============================================================
# COUNT EACH SPLIT
# ============================================================

train_counts = count_images(TRAIN_DIR)
val_counts = count_images(VAL_DIR)
test_counts = count_images(TEST_DIR)


# ============================================================
# GET ALL CLASS NAMES
# ============================================================

all_classes = sorted(
    set(train_counts)
    | set(val_counts)
    | set(test_counts)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 85)
print("TRAIN / VALIDATION / TEST SPLIT CHECK")
print("=" * 85)

print(
    f"\n{'Class':<25}"
    f"{'Train':>10}"
    f"{'Val':>10}"
    f"{'Test':>10}"
    f"{'Total':>10}"
)


print("-" * 65)


grand_train = 0
grand_val = 0
grand_test = 0


for class_name in all_classes:

    train = train_counts.get(
        class_name,
        0
    )

    val = val_counts.get(
        class_name,
        0
    )

    test = test_counts.get(
        class_name,
        0
    )

    total = train + val + test


    grand_train += train
    grand_val += val
    grand_test += test


    print(
        f"{class_name:<25}"
        f"{train:>10}"
        f"{val:>10}"
        f"{test:>10}"
        f"{total:>10}"
    )


# ============================================================
# TOTALS
# ============================================================

grand_total = (
    grand_train
    + grand_val
    + grand_test
)


print("-" * 65)

print(
    f"{'TOTAL':<25}"
    f"{grand_train:>10}"
    f"{grand_val:>10}"
    f"{grand_test:>10}"
    f"{grand_total:>10}"
)


# ============================================================
# PERCENTAGES
# ============================================================

print("\n" + "=" * 85)
print("SPLIT PERCENTAGES")
print("=" * 85)


if grand_total > 0:

    train_percentage = (
        grand_train
        / grand_total
        * 100
    )

    val_percentage = (
        grand_val
        / grand_total
        * 100
    )

    test_percentage = (
        grand_test
        / grand_total
        * 100
    )


    print(
        f"Train      : "
        f"{grand_train} images "
        f"({train_percentage:.2f}%)"
    )

    print(
        f"Validation : "
        f"{grand_val} images "
        f"({val_percentage:.2f}%)"
    )

    print(
        f"Test       : "
        f"{grand_test} images "
        f"({test_percentage:.2f}%)"
    )

    print(
        f"Total      : "
        f"{grand_total} images"
    )


# ============================================================
# CHECK MISSING CLASSES
# ============================================================

print("\n" + "=" * 85)
print("CLASS CHECK")
print("=" * 85)


problems = []


for class_name in all_classes:

    if train_counts.get(class_name, 0) == 0:

        problems.append(
            f"{class_name} is missing from TRAIN"
        )


    if val_counts.get(class_name, 0) == 0:

        problems.append(
            f"{class_name} is missing from VALIDATION"
        )


    if test_counts.get(class_name, 0) == 0:

        problems.append(
            f"{class_name} is missing from TEST"
        )


if problems:

    print("\nWARNING:")

    for problem in problems:

        print(
            f" - {problem}"
        )

else:

    print(
        "\nPASS: Every class exists in "
        "train, validation and test."
    )


# ============================================================
# APPROXIMATE 70 / 15 / 15 CHECK
# ============================================================

print("\n" + "=" * 85)
print("EXPECTED SPLIT: APPROXIMATELY 70% / 15% / 15%")
print("=" * 85)


if grand_total > 0:

    # Small differences are expected because individual
    # class counts must be rounded to whole images.

    train_ok = (
        68 <= train_percentage <= 72
    )

    val_ok = (
        13 <= val_percentage <= 17
    )

    test_ok = (
        13 <= test_percentage <= 17
    )


    if train_ok and val_ok and test_ok:

        print(
            "\nPASS: Overall split is close "
            "to 70 / 15 / 15."
        )

    else:

        print(
            "\nWARNING: Overall split is not "
            "close to 70 / 15 / 15."
        )


print("\n" + "=" * 85)
print("CHECK COMPLETE")
print("=" * 85)