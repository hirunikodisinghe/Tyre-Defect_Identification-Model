from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(r"E:\Hiruni\TireDefectProject")

DATASET_DIR = PROJECT_DIR / "train_split_dataset"

MODEL_PATH = PROJECT_DIR / "best_tire_defect_model.pth"

CLASS_NAMES_PATH = PROJECT_DIR / "class_names.txt"

OUTPUT_XLSX = (
    PROJECT_DIR
    / "manual_inspection_by_defect.xlsx"
)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224

SPLITS = [
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
    ".tiff"
}


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 70)
print("TIRE DEFECT MANUAL INSPECTION EXCEL")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# 3. CHECK REQUIRED FILES
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
        f"Class names not found:\n{CLASS_NAMES_PATH}"
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
# 6. LOAD MODEL
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


    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=min(3, num_classes)
    )


    top_predictions = []


    for rank in range(
        top_indices.shape[1]
    ):

        class_index = (
            top_indices[0][rank].item()
        )

        probability = (
            top_probabilities[0][rank].item()
            * 100
        )


        top_predictions.append(
            (
                class_names[class_index],
                probability
            )
        )


    predicted_class = (
        top_predictions[0][0]
    )

    confidence = (
        top_predictions[0][1]
    )


    return (
        predicted_class,
        confidence,
        top_predictions
    )


# ============================================================
# 9. CREATE EXCEL WORKBOOK
# ============================================================

workbook = Workbook()

default_sheet = workbook.active

workbook.remove(
    default_sheet
)


HEADERS = [
    "ImageName",
    "Note",
    "ChatGPT Results",
    "Split",
    "ActualDefect",
    "ModelPrediction",
    "ModelConfidence",
    "PredictionStatus",
    "Top1Prediction",
    "Top2Prediction",
    "Top3Prediction",
    "ManualStatus",
    "CorrectDefectName",
    "ImagePath"
]


# ============================================================
# 10. CREATE SEPARATE SHEET FOR EACH DEFECT
# ============================================================

worksheets = {}


for class_name in class_names:

    sheet_name = class_name[:31]

    worksheet = workbook.create_sheet(
        title=sheet_name
    )

    worksheets[
        class_name
    ] = worksheet


    for column_number, header in enumerate(
        HEADERS,
        start=1
    ):

        cell = worksheet.cell(
            row=1,
            column=column_number,
            value=header
        )

        cell.font = Font(
            bold=True,
            color="FFFFFF"
        )

        cell.fill = PatternFill(
            fill_type="solid",
            fgColor="1F4E78"
        )

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center"
        )


    worksheet.freeze_panes = "A2"


# ============================================================
# 11. COUNTERS
# ============================================================

total_images = 0

matches = 0

mismatches = 0

errors = 0

class_counts = {
    class_name: 0
    for class_name in class_names
}

class_matches = {
    class_name: 0
    for class_name in class_names
}

class_mismatches = {
    class_name: 0
    for class_name in class_names
}

class_errors = {
    class_name: 0
    for class_name in class_names
}


# ============================================================
# 12. PROCESS TRAIN / VAL / TEST
# ============================================================

for split_name in SPLITS:

    split_folder = (
        DATASET_DIR
        / split_name
    )


    if not split_folder.exists():

        print(
            f"\nWARNING: "
            f"{split_folder} does not exist."
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


        # -----------------------------------------------
        # ACTUAL DEFECT NAME
        #
        # This comes from the folder name.
        # -----------------------------------------------

        actual_defect = (
            class_folder.name
        )


        if actual_defect not in worksheets:

            print(
                f"Unknown class skipped: "
                f"{actual_defect}"
            )

            continue


        worksheet = worksheets[
            actual_defect
        ]


        image_files = sorted([

            image_path

            for image_path
            in class_folder.rglob("*")

            if (
                image_path.is_file()
                and
                image_path.suffix.lower()
                in VALID_EXTENSIONS
            )

        ])


        for image_path in image_files:

            total_images += 1

            class_counts[
                actual_defect
            ] += 1


            try:

                (
                    model_prediction,
                    model_confidence,
                    top_predictions
                ) = predict_image(
                    image_path
                )


                # ==========================================
                # MATCH / MISMATCH
                # ==========================================

                if (
                    model_prediction
                    ==
                    actual_defect
                ):

                    prediction_status = (
                        "MATCH"
                    )

                    note = (
                        "Model prediction matches "
                        "the dataset label."
                    )

                    matches += 1

                    class_matches[
                        actual_defect
                    ] += 1

                else:

                    prediction_status = (
                        "MISMATCH"
                    )

                    note = (
                        "Dataset label and model "
                        "prediction differ. "
                        "Manual inspection required."
                    )

                    mismatches += 1

                    class_mismatches[
                        actual_defect
                    ] += 1


                # ==========================================
                # TOP 3
                # ==========================================

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


                # ==========================================
                # ADD ROW
                # ==========================================

                worksheet.append([

                    # ImageName
                    image_path.name,

                    # Note
                    note,

                    # ChatGPT Results
                    # Fill manually later if needed.
                    "",

                    # Split
                    split_name,

                    # ActualDefect
                    actual_defect,

                    # ModelPrediction
                    model_prediction,

                    # ModelConfidence
                    round(
                        model_confidence,
                        2
                    ),

                    # PredictionStatus
                    prediction_status,

                    # Top predictions
                    top1,
                    top2,
                    top3,

                    # ManualStatus
                    "",

                    # CorrectDefectName
                    "",

                    # ImagePath
                    str(image_path)

                ])


            except Exception as error:

                errors += 1

                class_errors[
                    actual_defect
                ] += 1

                worksheet.append([

                    image_path.name,

                    f"Error reading image: {error}",

                    "",

                    split_name,

                    actual_defect,

                    "ERROR",

                    "",

                    "ERROR",

                    "",

                    "",

                    "",

                    "",

                    "",

                    str(image_path)

                ])


            if total_images % 100 == 0:

                print(
                    f"Processed "
                    f"{total_images} images..."
                )


# ============================================================
# 13. FORMAT EACH DEFECT SHEET
# ============================================================

for class_name, worksheet in worksheets.items():

    worksheet.column_dimensions["A"].width = 35
    worksheet.column_dimensions["B"].width = 55
    worksheet.column_dimensions["C"].width = 45
    worksheet.column_dimensions["D"].width = 12
    worksheet.column_dimensions["E"].width = 24
    worksheet.column_dimensions["F"].width = 24
    worksheet.column_dimensions["G"].width = 18
    worksheet.column_dimensions["H"].width = 18
    worksheet.column_dimensions["I"].width = 32
    worksheet.column_dimensions["J"].width = 32
    worksheet.column_dimensions["K"].width = 32
    worksheet.column_dimensions["L"].width = 20
    worksheet.column_dimensions["M"].width = 25
    worksheet.column_dimensions["N"].width = 80


    for row in worksheet.iter_rows(
        min_row=2
    ):

        status_cell = row[7]


        # MATCH = light green
        if status_cell.value == "MATCH":

            status_cell.fill = PatternFill(
                fill_type="solid",
                fgColor="D9EAD3"
            )


        # MISMATCH = light red
        elif status_cell.value == "MISMATCH":

            status_cell.fill = PatternFill(
                fill_type="solid",
                fgColor="F4CCCC"
            )


        # ERROR = orange
        elif status_cell.value == "ERROR":

            status_cell.fill = PatternFill(
                fill_type="solid",
                fgColor="FCE5CD"
            )


        # Actual/model data = green text
        for cell in row[3:11]:

            cell.font = Font(
                color="008000"
            )


        # Manual / ChatGPT review columns = blue
        row[2].font = Font(
            color="0000FF"
        )

        row[11].font = Font(
            color="0000FF"
        )

        row[12].font = Font(
            color="0000FF"
        )


    worksheet.auto_filter.ref = (
        worksheet.dimensions
    )


# ============================================================
# 14. CREATE SUMMARY SHEET
# ============================================================

summary_sheet = workbook.create_sheet(
    title="Summary",
    index=0
)


summary_headers = [
    "Defect",
    "Image Count",
    "Matches",
    "Mismatches",
    "Errors",
    "Match %",
    "Mismatch %",
    "Error %",
    "Dataset %",
    "Model Agreement %"
]


for column_number, header in enumerate(
    summary_headers,
    start=1
):

    cell = summary_sheet.cell(
        row=1,
        column=column_number,
        value=header
    )

    cell.font = Font(
        bold=True,
        color="FFFFFF"
    )

    cell.fill = PatternFill(
        fill_type="solid",
        fgColor="1F4E78"
    )

    cell.alignment = Alignment(
        horizontal="center",
        vertical="center"
    )


summary_sheet.freeze_panes = "A2"


row_number = 2


for class_name in class_names:

    image_count = class_counts[
        class_name
    ]

    class_match_count = class_matches[
        class_name
    ]

    class_mismatch_count = class_mismatches[
        class_name
    ]

    class_error_count = class_errors[
        class_name
    ]


    match_percentage = (
        class_match_count
        / image_count
        if image_count > 0
        else 0
    )


    mismatch_percentage = (
        class_mismatch_count
        / image_count
        if image_count > 0
        else 0
    )


    error_percentage = (
        class_error_count
        / image_count
        if image_count > 0
        else 0
    )


    dataset_percentage = (
        image_count
        / total_images
        if total_images > 0
        else 0
    )


    model_agreement = match_percentage


    summary_sheet.cell(
        row=row_number,
        column=1,
        value=class_name
    )


    summary_sheet.cell(
        row=row_number,
        column=2,
        value=image_count
    )


    summary_sheet.cell(
        row=row_number,
        column=3,
        value=class_match_count
    )


    summary_sheet.cell(
        row=row_number,
        column=4,
        value=class_mismatch_count
    )


    summary_sheet.cell(
        row=row_number,
        column=5,
        value=class_error_count
    )


    summary_sheet.cell(
        row=row_number,
        column=6,
        value=match_percentage
    )


    summary_sheet.cell(
        row=row_number,
        column=7,
        value=mismatch_percentage
    )


    summary_sheet.cell(
        row=row_number,
        column=8,
        value=error_percentage
    )


    summary_sheet.cell(
        row=row_number,
        column=9,
        value=dataset_percentage
    )


    summary_sheet.cell(
        row=row_number,
        column=10,
        value=model_agreement
    )


    # --------------------------------------------------------
    # COLOR MATCH / MISMATCH / ERROR COUNTS
    # --------------------------------------------------------

    summary_sheet.cell(
        row=row_number,
        column=3
    ).fill = PatternFill(
        fill_type="solid",
        fgColor="D9EAD3"
    )


    summary_sheet.cell(
        row=row_number,
        column=4
    ).fill = PatternFill(
        fill_type="solid",
        fgColor="F4CCCC"
    )


    summary_sheet.cell(
        row=row_number,
        column=5
    ).fill = PatternFill(
        fill_type="solid",
        fgColor="FCE5CD"
    )


    # Percentage formatting
    for percentage_column in [
        6,
        7,
        8,
        9,
        10
    ]:

        summary_sheet.cell(
            row=row_number,
            column=percentage_column
        ).number_format = "0.00%"


    row_number += 1


# ============================================================
# TOTAL ROW
# ============================================================

total_row = row_number


summary_sheet.cell(
    row=total_row,
    column=1,
    value="TOTAL"
)


summary_sheet.cell(
    row=total_row,
    column=2,
    value=total_images
)


summary_sheet.cell(
    row=total_row,
    column=3,
    value=matches
)


summary_sheet.cell(
    row=total_row,
    column=4,
    value=mismatches
)


summary_sheet.cell(
    row=total_row,
    column=5,
    value=errors
)


overall_match_percentage = (
    matches / total_images
    if total_images > 0
    else 0
)


overall_mismatch_percentage = (
    mismatches / total_images
    if total_images > 0
    else 0
)


overall_error_percentage = (
    errors / total_images
    if total_images > 0
    else 0
)


summary_sheet.cell(
    row=total_row,
    column=6,
    value=overall_match_percentage
)


summary_sheet.cell(
    row=total_row,
    column=7,
    value=overall_mismatch_percentage
)


summary_sheet.cell(
    row=total_row,
    column=8,
    value=overall_error_percentage
)


summary_sheet.cell(
    row=total_row,
    column=9,
    value=1.0
)


summary_sheet.cell(
    row=total_row,
    column=10,
    value=overall_match_percentage
)


for percentage_column in [
    6,
    7,
    8,
    9,
    10
]:

    summary_sheet.cell(
        row=total_row,
        column=percentage_column
    ).number_format = "0.00%"


# TOTAL row formatting
for column_number in range(
    1,
    len(summary_headers) + 1
):

    cell = summary_sheet.cell(
        row=total_row,
        column=column_number
    )

    cell.font = Font(
        bold=True,
        color="FFFFFF"
    )

    cell.fill = PatternFill(
        fill_type="solid",
        fgColor="1F4E78"
    )


# ============================================================
# SUMMARY SHEET COLUMN WIDTHS
# ============================================================

summary_sheet.column_dimensions["A"].width = 30
summary_sheet.column_dimensions["B"].width = 15
summary_sheet.column_dimensions["C"].width = 15
summary_sheet.column_dimensions["D"].width = 15
summary_sheet.column_dimensions["E"].width = 12
summary_sheet.column_dimensions["F"].width = 15
summary_sheet.column_dimensions["G"].width = 15
summary_sheet.column_dimensions["H"].width = 12
summary_sheet.column_dimensions["I"].width = 15
summary_sheet.column_dimensions["J"].width = 20


summary_sheet.auto_filter.ref = (
    summary_sheet.dimensions
)


# ============================================================
# 15. SAVE WORKBOOK
# ============================================================

workbook.save(
    OUTPUT_XLSX
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXCEL CREATION COMPLETED")
print("=" * 70)


print(
    f"Total images       : "
    f"{total_images}"
)


print(
    f"Model matches      : "
    f"{matches}"
)


print(
    f"Model mismatches   : "
    f"{mismatches}"
)


print(
    f"Errors             : "
    f"{errors}"
)


if total_images > 0:

    agreement = (
        matches
        /
        total_images
        *
        100
    )

    mismatch_rate = (
        mismatches
        /
        total_images
        *
        100
    )

    error_rate = (
        errors
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
        f"Mismatch rate      : "
        f"{mismatch_rate:.2f}%"
    )

    print(
        f"Error rate         : "
        f"{error_rate:.2f}%"
    )


print("\n" + "=" * 70)
print("PER-CLASS SUMMARY")
print("=" * 70)


for class_name in class_names:

    count = class_counts[
        class_name
    ]

    class_match_count = class_matches[
        class_name
    ]

    class_mismatch_count = class_mismatches[
        class_name
    ]

    class_error_count = class_errors[
        class_name
    ]


    if count > 0:

        class_match_percentage = (
            class_match_count
            / count
            * 100
        )

        class_mismatch_percentage = (
            class_mismatch_count
            / count
            * 100
        )

    else:

        class_match_percentage = 0
        class_mismatch_percentage = 0


    print(
        f"{class_name:20} | "
        f"Total: {count:4} | "
        f"Match: {class_match_count:4} "
        f"({class_match_percentage:6.2f}%) | "
        f"Mismatch: {class_mismatch_count:4} "
        f"({class_mismatch_percentage:6.2f}%) | "
        f"Errors: {class_error_count:3}"
    )


print(
    f"\nExcel saved to:\n"
    f"{OUTPUT_XLSX}"
)


print("\nDone.")