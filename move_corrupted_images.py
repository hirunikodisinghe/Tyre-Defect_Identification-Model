from pathlib import Path
import shutil
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "clean_dataset"
BACKUP_DIR = PROJECT_DIR / "corrupted_images"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

BACKUP_DIR.mkdir(
    parents=True,
    exist_ok=True
)

moved = 0

for class_folder in DATASET_DIR.iterdir():

    if not class_folder.is_dir():
        continue

    for image_path in class_folder.rglob("*"):
    
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ):

            try:

                with Image.open(image_path) as img:
                    img.verify()

            except Exception as exc:

                destination_dir = (
                    BACKUP_DIR / class_folder.name
                )

                destination_dir.mkdir(
                    parents=True,
                    exist_ok=True
                )

                destination = (
                    destination_dir /
                    image_path.name
                )

                print(
                    f"Moving corrupted image:\n"
                    f"{image_path}"
                )

                print(
                    f"Reason: {exc}\n"
                )

                shutil.move(
                    str(image_path),
                    str(destination)
                )

                moved += 1


print("=" * 60)
print("CORRUPTED IMAGE CLEANING FINISHED")
print("=" * 60)

print(
    "Moved corrupted images:",
    moved
)

print(
    "Backup location:",
    BACKUP_DIR
)