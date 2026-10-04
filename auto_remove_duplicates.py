from pathlib import Path
import hashlib
import shutil
from collections import defaultdict


PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "clean_dataset"

BACKUP_DIR = PROJECT_DIR / "removed_exact_duplicates"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def calculate_sha256(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


print("=" * 70)
print("SAFE EXACT DUPLICATE CLEANER")
print("=" * 70)

print("\nScanning dataset...")


hash_groups = defaultdict(list)

image_count = 0


for class_folder in DATASET_DIR.iterdir():

    if not class_folder.is_dir():
        continue

    for image_path in class_folder.rglob("*"):

        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):

            image_count += 1

            try:

                image_hash = calculate_sha256(
                    image_path
                )

                hash_groups[
                    image_hash
                ].append(
                    image_path
                )

            except Exception as error:

                print(
                    f"Could not read: {image_path}"
                )

                print(error)


print(
    f"\nImages scanned: {image_count}"
)


# ============================================================
# PROCESS DUPLICATES
# ============================================================

same_class_moved = 0

cross_class_groups = 0

duplicate_groups = 0


BACKUP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for image_hash, group in hash_groups.items():

    if len(group) <= 1:
        continue

    duplicate_groups += 1


    classes = {
        path.parent.name
        for path in group
    }


    # ========================================================
    # SAME CLASS DUPLICATES
    # ========================================================

    if len(classes) == 1:

        # Keep first image
        keep_image = group[0]

        print("\n" + "-" * 70)

        print(
            f"KEEP: {keep_image}"
        )


        # Move remaining duplicates
        for duplicate_path in group[1:]:

            relative_path = duplicate_path.relative_to(
                DATASET_DIR
            )

            destination = (
                BACKUP_DIR
                / relative_path
            )

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )


            counter = 1

            original_destination = destination


            while destination.exists():

                destination = (
                    original_destination.parent
                    / (
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


            same_class_moved += 1


            print(
                f"MOVED: {duplicate_path}"
            )


    # ========================================================
    # CROSS CLASS DUPLICATES
    # ========================================================

    else:

        cross_class_groups += 1

        print("\n" + "=" * 70)

        print(
            "CROSS-CLASS DUPLICATE FOUND"
        )

        print(
            "NOT MOVED AUTOMATICALLY"
        )

        print("=" * 70)


        for path in group:

            print(
                f"{path.parent.name} -> {path}"
            )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("SUMMARY")

print("=" * 70)


print(
    f"Images scanned             : {image_count}"
)

print(
    f"Duplicate groups found     : {duplicate_groups}"
)

print(
    f"Same-class duplicates moved: {same_class_moved}"
)

print(
    f"Cross-class groups skipped : {cross_class_groups}"
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

print(
    "Cross-class duplicates were left untouched."
)

print("\nDone.")