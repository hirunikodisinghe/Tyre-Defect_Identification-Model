from pathlib import Path
import csv

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(r"E:\Hiruni\TireDefectProject")

DATASET_DIR = PROJECT_DIR / "train_split_dataset"

MODEL_PATH = PROJECT_DIR / "best_tire_defect_model.pth"

CLASS_NAMES_PATH = PROJECT_DIR / "class_names.txt"

OUTPUT_CSV = (
    PROJECT_DIR
    / "manual_inspection_with_predictions.csv"
)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224

BATCH_CLASSES = [
    "train",
    "val",
    "test"
]

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("MANUAL INSPECTION + MODEL PREDICTION CSV")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# 3. CHECK FILES
# ============================================================

if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_DIR}"
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


if not CLASS_NAMES_PATH.exists():

    raise FileNotFoundError(
        f"Class names file not found:\n"
        f"{CLASS_NAMES_PATH}"
    )


# ============================================================
# 4. LOAD CLASS NAMES
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

for index, class_name in enumerate(
    class_names
):

    print(
        f"{index}: {class_name}"
    )


# ============================================================
# 5. CREATE MODEL
# ============================================================

model = efficientnet_b0(
    weights=None
)


input_features = (
    model.classifier[1].in_features
)


model.classifier[1] = nn.Linear(
    input_features,
    num_classes
)


# ============================================================
# 6. LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict
)


model = model.to(
    DEVICE
)

model.eval()


print("\nModel loaded successfully.")


# ============================================================
# 7. IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 8. PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )

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


        confidence, predicted_index = (
            torch.max(
                probabilities,
                dim=1
            )
        )


    predicted_index = (
        predicted_index.item()
    )


    predicted_class = (
        class_names[
            predicted_index
        ]
    )


    confidence_percentage = (
        confidence.item()
        * 100
    )


    # --------------------------------------------------------
    # TOP 3
    # --------------------------------------------------------

    top_probabilities, top_indices = (
        torch.topk(
            probabilities,
            k=min(
                3,
                num_classes
            )
        )
    )


    top_predictions = []


    for rank in range(
        top_indices.shape[1]
    ):

        class_index = (
            top_indices[
                0
            ][rank].item()
        )

        probability = (
            top_probabilities[
                0
            ][rank].item()
            * 100
        )


        top_predictions.append(
            (
                class_names[
                    class_index
                ],
                probability
            )
        )


    return (
        predicted_class,
        confidence_percentage,
        top_predictions
    )


# ============================================================
# 9. SCAN SPLIT DATASET
# ============================================================

rows = []

total_images = 0

matching_predictions = 0

different_predictions = 0


for split_name in BATCH_CLASSES:

    split_folder = (
        DATASET_DIR
        / split_name
    )


    if not split_folder.exists():

        print(
            f"\nWARNING: "
            f"{split_folder} "
            f"does not exist."
        )

        continue


    print(
        f"\nProcessing {split_name.upper()}..."
    )


    for class_folder in sorted(
        split_folder.iterdir()
    ):

        if not class_folder.is_dir():
            continue


        tyre_defect_name = (
            class_folder.name
        )


        image_files = sorted([

            file_path

            for file_path
            in class_folder.rglob("*")

            if (
                file_path.is_file()
                and
                file_path.suffix.lower()
                in VALID_EXTENSIONS
            )

        ])


        for image_path in image_files:

            total_images += 1


            try:

                (
                    model_prediction,
                    confidence,
                    top_predictions
                ) = predict_image(
                    image_path
                )


                # --------------------------------------------
                # MATCH / MISMATCH
                # --------------------------------------------

                if (
                    model_prediction
                    ==
                    tyre_defect_name
                ):

                    prediction_status = (
                        "MATCH"
                    )

                    matching_predictions += 1

                else:

                    prediction_status = (
                        "MISMATCH"
                    )

                    different_predictions += 1


                # --------------------------------------------
                # TOP 3 TEXT
                # --------------------------------------------

                top1 = (
                    f"{top_predictions[0][0]} "
                    f"({top_predictions[0][1]:.2f}%)"
                )


                top2 = (
                    f"{top_predictions[1][0]} "
                    f"({top_predictions[1][1]:.2f}%)"
                )


                top3 = (
                    f"{top_predictions[2][0]} "
                    f"({top_predictions[2][1]:.2f}%)"
                )


                rows.append({

                    "ImageName":
                        image_path.name,

                    "Split":
                        split_name,

                    "TyreDefectName":
                        tyre_defect_name,

                    "ModelPrediction":
                        model_prediction,

                    "ConfidencePercent":
                        round(
                            confidence,
                            2
                        ),

                    "PredictionStatus":
                        prediction_status,

                    "Top1Prediction":
                        top1,

                    "Top2Prediction":
                        top2,

                    "Top3Prediction":
                        top3,

                    "ImagePath":
                        str(image_path),

                    "ManualStatus":
                        "",

                    "CorrectDefectName":
                        "",

                    "RemoveImage":
                        "",

                    "Notes":
                        ""
                })


            except Exception as error:

                print(
                    f"\nERROR:"
                )

                print(
                    image_path
                )

                print(
                    error
                )


                rows.append({

                    "ImageName":
                        image_path.name,

                    "Split":
                        split_name,

                    "TyreDefectName":
                        tyre_defect_name,

                    "ModelPrediction":
                        "ERROR",

                    "ConfidencePercent":
                        "",

                    "PredictionStatus":
                        "ERROR",

                    "Top1Prediction":
                        "",

                    "Top2Prediction":
                        "",

                    "Top3Prediction":
                        "",

                    "ImagePath":
                        str(image_path),

                    "ManualStatus":
                        "",

                    "CorrectDefectName":
                        "",

                    "RemoveImage":
                        "",

                    "Notes":
                        str(error)
                })


            if total_images % 100 == 0:

                print(
                    f"Processed "
                    f"{total_images} images..."
                )


# ============================================================
# 10. SAVE CSV
# ============================================================

fieldnames = [

    "ImageName",

    "Split",

    "TyreDefectName",

    "ModelPrediction",

    "ConfidencePercent",

    "PredictionStatus",

    "Top1Prediction",

    "Top2Prediction",

    "Top3Prediction",

    "ImagePath",

    "ManualStatus",

    "CorrectDefectName",

    "RemoveImage",

    "Notes"

]


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )


    writer.writeheader()

    writer.writerows(
        rows
    )


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "CSV CREATION COMPLETED"
)

print("=" * 70)


print(
    f"Total images       : "
    f"{total_images}"
)


print(
    f"Model matches      : "
    f"{matching_predictions}"
)


print(
    f"Model mismatches   : "
    f"{different_predictions}"
)


if total_images > 0:

    agreement = (
        matching_predictions
        /
        total_images
        *
        100
    )


    print(
        f"Model agreement    : "
        f"{agreement:.2f}%"
    )


print(
    f"\nCSV saved to:\n"
    f"{OUTPUT_CSV}"
)


print("\nDone.")