from pathlib import Path
import shutil

from PIL import Image

import torch

from transformers import (
    CLIPModel,
    CLIPProcessor,
)

from openpyxl import Workbook
from openpyxl.styles import (
    Font,
    PatternFill,
)


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(
    r"E:\Hiruni\TireDefectProject"
)

DATASET_DIR = (
    PROJECT_DIR
    / "clean_dataset"
)

OUTPUT_EXCEL = (
    PROJECT_DIR
    / "clip_dataset_review.xlsx"
)

REVIEW_DIR = (
    PROJECT_DIR
    / "clip_mismatch_review"
)


# ============================================================
# 2. SETTINGS
# ============================================================

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


# ============================================================
# 3. DEFECT CLASSES
# ============================================================

CLASS_NAMES = [
    "Abnormal_Wear",
    "Bead_Damage",
    "Bulge",
    "Cut",
    "Shoulder_Crack",
    "Sidewall_Crack",
    "Sidewall_Damage",
    "Tire_Burst",
    "Tread_Chunking",
    "Tread_Damage",
    "Tread_Separation",
    "noDefect",
]


# ============================================================
# 4. TEXT PROMPTS FOR ZERO-SHOT CLIP
# ============================================================

TEXT_PROMPTS = {

    "Abnormal_Wear":
        "an industrial tire with severe abnormal or uneven tread wear",

    "Bead_Damage":
        "an industrial tire with visible damage around the tire bead",

    "Bulge":
        "an industrial tire with a visible bulge or swelling on the sidewall",

    "Cut":
        "an industrial tire with a clear cut or laceration in the rubber",

    "Shoulder_Crack":
        "an industrial tire with visible cracking around the shoulder area",

    "Sidewall_Crack":
        "an industrial tire with visible cracks on the sidewall",

    "Sidewall_Damage":
        "an industrial tire with visible physical damage on the sidewall",

    "Tire_Burst":
        "an industrial tire that has burst or ruptured",

    "Tread_Chunking":
        "an industrial tire with chunks of tread rubber broken or missing",

    "Tread_Damage":
        "an industrial tire with visible damage on the tread surface",

    "Tread_Separation":
        "an industrial tire with tread separating from the tire body",

    "noDefect":
        "a normal industrial tire with no visible defect or damage",
}


# ============================================================
# 5. START
# ============================================================

print("=" * 70)
print("PRETRAINED CLIP TIRE DATASET REVIEW")
print("=" * 70)

print("Device:", DEVICE)


if not DATASET_DIR.exists():

    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_DIR}"
    )


# ============================================================
# 6. LOAD PRETRAINED CLIP
# ============================================================

print("\nLoading pretrained CLIP model...")


MODEL_NAME = (
    "openai/clip-vit-base-patch32"
)


processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)


model = CLIPModel.from_pretrained(
    MODEL_NAME
)


model = model.to(
    DEVICE
)


model.eval()


print(
    "CLIP model loaded successfully."
)


# ============================================================
# 7. PREPARE TEXT PROMPTS
# ============================================================

prompt_texts = [
    TEXT_PROMPTS[
        class_name
    ]
    for class_name in CLASS_NAMES
]


text_inputs = processor(
    text=prompt_texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
)


text_inputs = {
    key: value.to(DEVICE)
    for key, value
    in text_inputs.items()
}


# ============================================================
# 8. CREATE TEXT FEATURES
#
# Corrected for newer transformers versions
# ============================================================

with torch.no_grad():

    text_outputs = model.text_model(

        input_ids=text_inputs[
            "input_ids"
        ],

        attention_mask=text_inputs.get(
            "attention_mask"
        ),
    )


    pooled_text = (
        text_outputs.pooler_output
    )


    text_features = (
        model.text_projection(
            pooled_text
        )
    )


# Normalize text embeddings
text_features = (
    text_features
    /
    text_features.norm(
        dim=-1,
        keepdim=True
    )
)


print(
    "Text features prepared successfully."
)


# ============================================================
# 9. CLIP PREDICTION FUNCTION
# ============================================================

def predict_clip(
    image_path
):

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    with Image.open(
        image_path
    ) as image:

        image = image.convert(
            "RGB"
        )


        image_inputs = processor(
            images=image,
            return_tensors="pt"
        )


    pixel_values = (
        image_inputs[
            "pixel_values"
        ].to(DEVICE)
    )


    # --------------------------------------------------------
    # CREATE IMAGE FEATURES
    # --------------------------------------------------------

    with torch.no_grad():

        image_outputs = model.vision_model(
            pixel_values=pixel_values
        )


        pooled_image = (
            image_outputs.pooler_output
        )


        image_features = (
            model.visual_projection(
                pooled_image
            )
        )


    # Normalize image embeddings
    image_features = (
        image_features
        /
        image_features.norm(
            dim=-1,
            keepdim=True
        )
    )


    # --------------------------------------------------------
    # IMAGE-TEXT SIMILARITY
    # --------------------------------------------------------

    similarity = (
        image_features
        @
        text_features.T
    )


    # CLIP's learned logit scaling
    logit_scale = (
        model.logit_scale.exp()
    )


    similarity = (
        similarity
        *
        logit_scale
    )


    probabilities = torch.softmax(
        similarity,
        dim=1
    )


    # --------------------------------------------------------
    # TOP 3
    # --------------------------------------------------------

    top_probabilities, top_indices = (
        torch.topk(

            probabilities,

            k=min(
                3,
                len(CLASS_NAMES)
            ),

            dim=1
        )
    )


    top_predictions = []


    for rank in range(
        top_indices.shape[1]
    ):

        index = (
            top_indices[
                0,
                rank
            ].item()
        )


        probability = (
            top_probabilities[
                0,
                rank
            ].item()
            *
            100
        )


        top_predictions.append(
            (
                CLASS_NAMES[index],
                probability
            )
        )


    return top_predictions


# ============================================================
# 10. PREPARE MISMATCH REVIEW DIRECTORY
# ============================================================

if REVIEW_DIR.exists():

    print(
        "\nRemoving previous review folder..."
    )

    shutil.rmtree(
        REVIEW_DIR
    )


REVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 11. CREATE EXCEL WORKBOOK
# ============================================================

workbook = Workbook()

worksheet = workbook.active

worksheet.title = (
    "CLIP Review"
)


HEADERS = [

    "ImageName",

    "ActualDefect",

    "CLIPPrediction",

    "ConfidencePercent",

    "PredictionStatus",

    "Top1Prediction",

    "Top2Prediction",

    "Top3Prediction",

    "ImagePath",

    "ReviewDecision",

    "CorrectDefectName",

    "Notes",

]


# ============================================================
# 12. FORMAT HEADER
# ============================================================

for column, header in enumerate(
    HEADERS,
    start=1
):

    cell = worksheet.cell(
        row=1,
        column=column,
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


worksheet.freeze_panes = (
    "A2"
)


# ============================================================
# 13. COUNTERS
# ============================================================

total_images = 0

matches = 0

mismatches = 0

errors = 0


# ============================================================
# 14. SCAN DATASET
# ============================================================

for class_folder in sorted(
    DATASET_DIR.iterdir()
):

    if not class_folder.is_dir():

        continue


    actual_defect = (
        class_folder.name
    )


    if actual_defect not in CLASS_NAMES:

        print(
            f"Skipping unknown class: "
            f"{actual_defect}"
        )

        continue


    print(
        f"\nChecking class: "
        f"{actual_defect}"
    )


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


    print(
        f"Images found: "
        f"{len(image_files)}"
    )


    # --------------------------------------------------------
    # PROCESS EACH IMAGE
    # --------------------------------------------------------

    for image_path in image_files:

        total_images += 1


        try:

            predictions = predict_clip(
                image_path
            )


            predicted_defect = (
                predictions[0][0]
            )


            confidence = (
                predictions[0][1]
            )


            # ------------------------------------------------
            # MATCH
            # ------------------------------------------------

            if (
                predicted_defect
                ==
                actual_defect
            ):

                status = (
                    "MATCH"
                )

                matches += 1


            # ------------------------------------------------
            # MISMATCH
            # ------------------------------------------------

            else:

                status = (
                    "MISMATCH"
                )

                mismatches += 1


                # ============================================
                # COPY MISMATCH TO REVIEW FOLDER
                # ============================================

                review_folder = (

                    REVIEW_DIR

                    /

                    (
                        f"{actual_defect}"
                        f"__predicted_"
                        f"{predicted_defect}"
                    )
                )


                review_folder.mkdir(
                    parents=True,
                    exist_ok=True
                )


                destination = (
                    review_folder
                    /
                    image_path.name
                )


                # Prevent overwrite
                counter = 1


                while destination.exists():

                    destination = (

                        review_folder

                        /

                        (
                            f"{image_path.stem}"
                            f"_{counter}"
                            f"{image_path.suffix}"
                        )
                    )

                    counter += 1


                shutil.copy2(
                    image_path,
                    destination
                )


            # ------------------------------------------------
            # TOP 3 TEXT
            # ------------------------------------------------

            top1 = (
                f"{predictions[0][0]} "
                f"({predictions[0][1]:.2f}%)"
            )


            top2 = (
                f"{predictions[1][0]} "
                f"({predictions[1][1]:.2f}%)"
            )


            top3 = (
                f"{predictions[2][0]} "
                f"({predictions[2][1]:.2f}%)"
            )


            # ------------------------------------------------
            # EXCEL ROW
            # ------------------------------------------------

            worksheet.append([

                image_path.name,

                actual_defect,

                predicted_defect,

                round(
                    confidence,
                    2
                ),

                status,

                top1,

                top2,

                top3,

                str(image_path),

                "",

                "",

                "",

            ])


        except Exception as error:

            errors += 1


            worksheet.append([

                image_path.name,

                actual_defect,

                "ERROR",

                "",

                "ERROR",

                "",

                "",

                "",

                str(image_path),

                "",

                "",

                str(error),

            ])


            print(
                f"\nERROR processing:"
            )

            print(
                image_path
            )

            print(
                error
            )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            total_images % 100
            ==
            0
        ):

            print(
                f"Processed "
                f"{total_images} images..."
            )


# ============================================================
# 15. COLOR STATUS CELLS
# ============================================================

for row_number in range(
    2,
    worksheet.max_row + 1
):

    status_cell = worksheet.cell(
        row=row_number,
        column=5
    )


    if (
        status_cell.value
        ==
        "MATCH"
    ):

        status_cell.fill = PatternFill(
            fill_type="solid",
            fgColor="C6EFCE"
        )


    elif (
        status_cell.value
        ==
        "MISMATCH"
    ):

        status_cell.fill = PatternFill(
            fill_type="solid",
            fgColor="FFC7CE"
        )


    elif (
        status_cell.value
        ==
        "ERROR"
    ):

        status_cell.fill = PatternFill(
            fill_type="solid",
            fgColor="FCE5CD"
        )


# ============================================================
# 16. EXCEL COLUMN WIDTHS
# ============================================================

worksheet.column_dimensions[
    "A"
].width = 42


worksheet.column_dimensions[
    "B"
].width = 24


worksheet.column_dimensions[
    "C"
].width = 24


worksheet.column_dimensions[
    "D"
].width = 20


worksheet.column_dimensions[
    "E"
].width = 18


worksheet.column_dimensions[
    "F"
].width = 34


worksheet.column_dimensions[
    "G"
].width = 34


worksheet.column_dimensions[
    "H"
].width = 34


worksheet.column_dimensions[
    "I"
].width = 80


worksheet.column_dimensions[
    "J"
].width = 22


worksheet.column_dimensions[
    "K"
].width = 26


worksheet.column_dimensions[
    "L"
].width = 50


worksheet.auto_filter.ref = (
    worksheet.dimensions
)


# ============================================================
# 17. SAVE WORKBOOK
# ============================================================

workbook.save(
    OUTPUT_EXCEL
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print(
    "CLIP REVIEW COMPLETED"
)

print("=" * 70)


print(
    f"Images checked : "
    f"{total_images}"
)


print(
    f"Matches        : "
    f"{matches}"
)


print(
    f"Mismatches     : "
    f"{mismatches}"
)


print(
    f"Errors         : "
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


    print(
        f"CLIP agreement : "
        f"{agreement:.2f}%"
    )


print(
    f"\nExcel report:\n"
    f"{OUTPUT_EXCEL}"
)


print(
    f"\nMismatch review folders:\n"
    f"{REVIEW_DIR}"
)


print("\nIMPORTANT:")

print(
    "Original images were NOT moved or deleted."
)

print(
    "Only mismatch images were COPIED "
    "to the review directory."
)

print(
    "Do not automatically relabel based only on CLIP."
)


print("\nDone.")