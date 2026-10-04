from pathlib import Path
import shutil
import csv

from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(r"E:\Hiruni\TireDefectProject")

DATASET_DIR = PROJECT_DIR / "clean_dataset"
MODEL_PATH = PROJECT_DIR / "best_tire_defect_model.pth"
CLASS_NAMES_PATH = PROJECT_DIR / "class_names.txt"

REVIEW_DIR = PROJECT_DIR / "manual_review"
UNCERTAIN_DIR = PROJECT_DIR / "uncertain_review"

REPORT_PATH = PROJECT_DIR / "label_review_report.csv"

IMAGE_SIZE = 224

# If prediction is different from folder label and confidence
# is >= 85%, copy to manual_review.
CONFIDENCE_THRESHOLD = 0.85

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
}

# Allows PIL to read partially truncated files.
# However, corrupted images are still reported if loading fails.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("TIRE DATASET AUTOMATIC LABEL REVIEW")
print("=" * 70)

print(f"\nDevice  : {DEVICE}")
print(f"Dataset : {DATASET_DIR}")
print(f"Model   : {MODEL_PATH}")


# ============================================================
# CHECK FILES
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"\nClass names file not found:\n{CLASS_NAMES_PATH}"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]


num_classes = len(class_names)

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index:2d}: {class_name}")

print(f"\nNumber of classes: {num_classes}")


# ============================================================
# CHECK DATASET FOLDERS
# ============================================================

dataset_classes = sorted([
    folder.name
    for folder in DATASET_DIR.iterdir()
    if folder.is_dir()
])

print("\nDataset folders:")

for folder_name in dataset_classes:
    print(f" - {folder_name}")


unknown_folders = [
    name
    for name in dataset_classes
    if name not in class_names
]

if unknown_folders:

    print("\nWARNING:")
    print("These folders are not present in class_names.txt:")

    for folder_name in unknown_folders:
        print(f" - {folder_name}")


# ============================================================
# CREATE MODEL
# ============================================================

print("\nLoading EfficientNet-B0 model...")

model = efficientnet_b0(
    weights=None
)

input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    num_classes
)


# ============================================================
# LOAD MODEL WEIGHTS
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)


# Support both:
# 1. direct state_dict
# 2. checkpoint dictionary

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    state_dict = checkpoint["model_state_dict"]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)

model = model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ============================================================
# IMAGE TRANSFORM
#
# IMPORTANT:
# This should match the validation/test transform used
# during model training.
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# CREATE REVIEW DIRECTORIES
# ============================================================

REVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

UNCERTAIN_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COUNTERS
# ============================================================

total_images = 0

matched_images = 0

high_confidence_mismatches = 0

low_confidence_mismatches = 0

corrupted_images = 0


# ============================================================
# HELPER: SAFE COPY
# ============================================================

def safe_copy(source, destination_folder):

    destination_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = destination_folder / source.name

    # Prevent overwriting files with identical filenames
    counter = 1

    while destination.exists():

        destination = (
            destination_folder
            / f"{source.stem}_{counter}{source.suffix}"
        )

        counter += 1

    shutil.copy2(
        source,
        destination
    )

    return destination


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        image_tensor = transform(
            image
        ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_index.item()

    predicted_class = class_names[
        predicted_index
    ]

    confidence_value = confidence.item()

    return (
        predicted_class,
        confidence_value
    )


# ============================================================
# REPORT
# ============================================================

report_rows = []


# ============================================================
# SCAN DATASET
# ============================================================

print("\n" + "=" * 70)
print("STARTING DATASET REVIEW")
print("=" * 70)


for class_folder in sorted(DATASET_DIR.iterdir()):

    if not class_folder.is_dir():
        continue

    actual_class = class_folder.name

    # Skip unexpected folders
    if actual_class not in class_names:

        print(
            f"\nSkipping unknown folder: "
            f"{actual_class}"
        )

        continue


    print(
        f"\nChecking class: {actual_class}"
    )


    image_files = [

        file_path

        for file_path in class_folder.rglob("*")

        if (
            file_path.is_file()
            and
            file_path.suffix.lower()
            in VALID_EXTENSIONS
        )
    ]


    print(
        f"Images: {len(image_files)}"
    )


    for image_path in image_files:

        total_images += 1


        try:

            predicted_class, confidence = (
                predict_image(
                    image_path
                )
            )


            # =================================================
            # CASE 1:
            # Prediction matches folder label
            # =================================================

            if predicted_class == actual_class:

                status = "MATCH"

                matched_images += 1

                copied_path = ""


            # =================================================
            # CASE 2:
            # Different prediction + HIGH confidence
            # =================================================

            elif confidence >= CONFIDENCE_THRESHOLD:

                status = "HIGH_CONFIDENCE_MISMATCH"

                high_confidence_mismatches += 1


                review_folder_name = (
                    f"{actual_class}"
                    f"__predicted_"
                    f"{predicted_class}"
                )


                destination_folder = (
                    REVIEW_DIR
                    / review_folder_name
                )


                copied_path = safe_copy(
                    image_path,
                    destination_folder
                )


                print(
                    f"\nSUSPICIOUS:"
                )

                print(
                    f"Actual     : {actual_class}"
                )

                print(
                    f"Predicted  : {predicted_class}"
                )

                print(
                    f"Confidence : "
                    f"{confidence * 100:.2f}%"
                )

                print(
                    f"Image      : "
                    f"{image_path.name}"
                )


            # =================================================
            # CASE 3:
            # Different prediction + LOW confidence
            # =================================================

            else:

                status = "LOW_CONFIDENCE_MISMATCH"

                low_confidence_mismatches += 1


                uncertain_folder_name = (
                    f"{actual_class}"
                    f"__predicted_"
                    f"{predicted_class}"
                )


                destination_folder = (
                    UNCERTAIN_DIR
                    / uncertain_folder_name
                )


                copied_path = safe_copy(
                    image_path,
                    destination_folder
                )


            # =================================================
            # ADD REPORT ROW
            # =================================================

            report_rows.append({

                "image_path":
                    str(image_path),

                "actual_class":
                    actual_class,

                "predicted_class":
                    predicted_class,

                "confidence":
                    f"{confidence:.6f}",

                "confidence_percent":
                    f"{confidence * 100:.2f}",

                "status":
                    status,

                "copied_to":
                    str(copied_path)
                    if copied_path
                    else ""
            })


        except Exception as error:

            corrupted_images += 1


            print(
                f"\nERROR reading image:"
            )

            print(
                image_path
            )

            print(
                error
            )


            report_rows.append({

                "image_path":
                    str(image_path),

                "actual_class":
                    actual_class,

                "predicted_class":
                    "",

                "confidence":
                    "",

                "confidence_percent":
                    "",

                "status":
                    "ERROR",

                "copied_to":
                    ""
            })


# ============================================================
# SAVE CSV REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    fieldnames = [

        "image_path",

        "actual_class",

        "predicted_class",

        "confidence",

        "confidence_percent",

        "status",

        "copied_to"
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

print("\n")
print("=" * 70)
print("AUTOMATIC LABEL REVIEW SUMMARY")
print("=" * 70)

print(
    f"Total images checked          : "
    f"{total_images}"
)

print(
    f"Prediction matches label      : "
    f"{matched_images}"
)

print(
    f"High-confidence mismatches    : "
    f"{high_confidence_mismatches}"
)

print(
    f"Low-confidence mismatches     : "
    f"{low_confidence_mismatches}"
)

print(
    f"Images with reading errors    : "
    f"{corrupted_images}"
)


if total_images > 0:

    match_percentage = (
        matched_images
        / total_images
        * 100
    )

    print(
        f"Model/folder agreement        : "
        f"{match_percentage:.2f}%"
    )


print("\nReview folders:")

print(
    f"High-confidence review:"
    f"\n{REVIEW_DIR}"
)

print(
    f"\nUncertain review:"
    f"\n{UNCERTAIN_DIR}"
)

print(
    f"\nCSV report:"
    f"\n{REPORT_PATH}"
)


print("\nIMPORTANT:")
print(
    "Original clean_dataset images "
    "were NOT deleted or moved."
)

print(
    "Suspicious images were copied "
    "for manual review."
)

print("=" * 70)
print("DONE")
print("=" * 70)