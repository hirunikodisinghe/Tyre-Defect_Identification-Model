import os
import shutil
from pathlib import Path

from PIL import Image
import imagehash


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "clean_dataset"

BACKUP_DIR = PROJECT_DIR / "removed_visual_duplicates"


# ============================================================
# 2. SETTINGS
# ============================================================

HASH_THRESHOLD = 3

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
# 3. SCAN IMAGES
# ============================================================

images = []

print("=" * 70)
print("VISUAL DUPLICATE MOVER")
print("=" * 70)

print("\nDataset:")
print(DATASET_DIR)

print("\nScanning images...")


for root, dirs, files in os.walk(DATASET_DIR):

    for file in files:

        if file.lower().endswith(VALID_EXTENSIONS):

            path = Path(root) / file

            try:

                with Image.open(path) as img:

                    img_hash = imagehash.phash(
                        img.convert("RGB")
                    )

                    images.append(
                        (
                            path,
                            img_hash
                        )
                    )

            except Exception as exc:

                print(
                    f"Cannot read: {path}"
                )

                print(exc)


print(
    f"\nImages scanned: {len(images)}"
)


# ============================================================
# 4. FIND DUPLICATE GROUPS
# ============================================================

duplicate_groups = []

used = set()

print("\nFinding visual duplicate groups...")


for i in range(len(images)):

    if i in used:
        continue

    group = [
        images[i][0]
    ]

    base_hash = images[i][1]

    for j in range(
        i + 1,
        len(images)
    ):

        if j in used:
            continue

        hash_difference = (
            base_hash -
            images[j][1]
        )

        if hash_difference <= HASH_THRESHOLD:

            group.append(
                images[j][0]
            )

            used.add(j)

    if len(group) > 1:

        duplicate_groups.append(
            group
        )


print(
    f"Duplicate groups found: "
    f"{len(duplicate_groups)}"
)


# ============================================================
# 5. MOVE EXTRA IMAGES
# ============================================================

BACKUP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

moved_count = 0


for group_number, group in enumerate(
    duplicate_groups,
    start=1
):

    # Keep the FIRST image in the group
    keep_image = group[0]

    print("\n" + "-" * 70)

    print(
        f"Group {group_number}"
    )

    print(
        f"KEEP: {keep_image}"
    )


    # Move the remaining images
    for duplicate_path in group[1:]:

        duplicate_path = Path(
            duplicate_path
        )

        try:

            relative_path = (
                duplicate_path.relative_to(
                    DATASET_DIR
                )
            )

        except ValueError:

            print(
                f"Skipping unexpected path: "
                f"{duplicate_path}"
            )

            continue


        destination = (
            BACKUP_DIR /
            relative_path
        )


        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )


        # Prevent overwrite
        if destination.exists():

            counter = 1

            while destination.exists():

                destination = (
                    destination.parent
                    / (
                        f"{duplicate_path.stem}"
                        f"_dup_{counter}"
                        f"{duplicate_path.suffix}"
                    )
                )

                counter += 1


        shutil.move(
            str(duplicate_path),
            str(destination)
        )


        moved_count += 1


        print(
            f"MOVED: {duplicate_path}"
        )


# ============================================================
# 6. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINISHED")
print("=" * 70)

print(
    f"Duplicate groups processed : "
    f"{len(duplicate_groups)}"
)

print(
    f"Duplicate images moved     : "
    f"{moved_count}"
)

print(
    "\nBackup folder:"
)

print(
    BACKUP_DIR
)

print(
    "\nNo files were permanently deleted."
)