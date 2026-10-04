from pathlib import Path
from tkinter import Tk, filedialog, messagebox
import math

from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_DIR / "best_tire_defect_model.pth"
CLASS_NAMES_PATH = PROJECT_DIR / "class_names.txt"


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------------------
# REJECTION SETTINGS
# ------------------------------------------------------------
#
# These values help reject uncertain / unusual images.
#
# They DO NOT replace a real Tire-vs-Non-Tire model.
#
# You can tune these later using validation images.
# ------------------------------------------------------------

MIN_CONFIDENCE = 0.60

MIN_MARGIN = 0.15

MAX_NORMALIZED_ENTROPY = 0.65


print("=" * 70)
print("TIRE DEFECT IDENTIFICATION")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


if not CLASS_NAMES_PATH.exists():

    raise FileNotFoundError(
        f"Class names file not found:\n{CLASS_NAMES_PATH}"
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


for index, class_name in enumerate(class_names):

    print(
        f"{index}: {class_name}"
    )


print(
    "\nNumber of classes:",
    num_classes
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


# Supports:
#
# 1. direct state_dict
# 2. checkpoint containing model_state_dict

if (
    isinstance(checkpoint, dict)
    and
    "model_state_dict" in checkpoint
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


print(
    "\nModel loaded successfully."
)


# ============================================================
# 7. IMAGE TRANSFORM
#
# Must match validation/test preprocessing used during training.
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
# 8. ENTROPY FUNCTION
# ============================================================

def calculate_normalized_entropy(probabilities):

    probabilities = probabilities.clamp(
        min=1e-12
    )

    entropy = -torch.sum(
        probabilities
        *
        torch.log(probabilities)
    ).item()


    max_entropy = math.log(
        len(probabilities)
    )


    normalized_entropy = (
        entropy
        /
        max_entropy
    )


    return normalized_entropy


# ============================================================
# 9. MODEL PREDICTION WITH TTA
# ============================================================

def predict_with_tta(image):

    # --------------------------------------------------------
    # ORIGINAL IMAGE
    # --------------------------------------------------------

    image_tensor = transform(
        image
    ).unsqueeze(0)


    image_tensor = image_tensor.to(
        DEVICE
    )


    # --------------------------------------------------------
    # FLIPPED IMAGE
    # --------------------------------------------------------

    flipped_tensor = torch.flip(
        image_tensor,
        dims=[3]
    )


    with torch.no_grad():

        # Original prediction
        original_outputs = model(
            image_tensor
        )


        original_probabilities = torch.softmax(
            original_outputs,
            dim=1
        )


        # Flipped prediction
        flipped_outputs = model(
            flipped_tensor
        )


        flipped_probabilities = torch.softmax(
            flipped_outputs,
            dim=1
        )


        # Average predictions
        probabilities = (

            original_probabilities
            +
            flipped_probabilities

        ) / 2.0


    return probabilities


# ============================================================
# 10. SELECT IMAGE FROM FOLDER
# ============================================================

root = Tk()

root.withdraw()

root.attributes(
    "-topmost",
    True
)


image_path = filedialog.askopenfilename(

    title="Select Tire Image",

    filetypes=[

        (
            "Image Files",
            "*.jpg *.jpeg *.png *.bmp *.webp"
        ),

        (
            "All Files",
            "*.*"
        )
    ]
)


root.destroy()


# ============================================================
# 11. CHECK USER SELECTION
# ============================================================

if not image_path:

    print(
        "\nNo image selected."
    )

    raise SystemExit


print(
    "\nSelected image:"
)

print(
    image_path
)


# ============================================================
# 12. LOAD IMAGE
# ============================================================

try:

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


except Exception as error:

    print(
        "\nERROR: Unable to open image."
    )

    print(
        error
    )

    raise SystemExit


# ============================================================
# 13. RUN PREDICTION
# ============================================================

probabilities = predict_with_tta(
    image
)


# ============================================================
# 14. TOP PREDICTIONS
# ============================================================

top_probabilities, top_indices = torch.topk(

    probabilities,

    k=min(
        3,
        num_classes
    )
)


top_probabilities = top_probabilities[
    0
]

top_indices = top_indices[
    0
]


# ============================================================
# 15. GET TOP-1 AND TOP-2
# ============================================================

top1_probability = (
    top_probabilities[0].item()
)

top1_index = (
    top_indices[0].item()
)


top2_probability = (
    top_probabilities[1].item()
)


predicted_class = class_names[
    top1_index
]


confidence_percentage = (
    top1_probability
    *
    100
)


# ============================================================
# 16. CALCULATE PREDICTION MARGIN
# ============================================================

prediction_margin = (

    top1_probability
    -
    top2_probability

)


# ============================================================
# 17. CALCULATE ENTROPY
# ============================================================

normalized_entropy = calculate_normalized_entropy(

    probabilities[0]

)


# ============================================================
# 18. CHECK IF PREDICTION IS RELIABLE
# ============================================================

rejection_reasons = []


# ------------------------------------------------------------
# LOW CONFIDENCE
# ------------------------------------------------------------

if top1_probability < MIN_CONFIDENCE:

    rejection_reasons.append(
        "low prediction confidence"
    )


# ------------------------------------------------------------
# SMALL DIFFERENCE BETWEEN TOP 1 AND TOP 2
# ------------------------------------------------------------

if prediction_margin < MIN_MARGIN:

    rejection_reasons.append(
        "prediction is ambiguous"
    )


# ------------------------------------------------------------
# HIGH ENTROPY
# ------------------------------------------------------------

if (
    normalized_entropy
    >
    MAX_NORMALIZED_ENTROPY
):

    rejection_reasons.append(
        "model is uncertain"
    )


# ============================================================
# 19. DISPLAY TOP 3
# ============================================================

print("\n" + "=" * 70)

print(
    "TOP 3 MODEL PREDICTIONS"
)

print("=" * 70)


for rank in range(
    top_indices.shape[0]
):

    class_index = (
        top_indices[rank].item()
    )


    probability = (
        top_probabilities[rank].item()
        *
        100
    )


    print(

        f"{rank + 1}. "
        f"{class_names[class_index]:25s} "
        f"{probability:.2f}%"

    )


# ============================================================
# 20. DISPLAY DIAGNOSTIC VALUES
# ============================================================

print("\n" + "-" * 70)

print(
    f"Top confidence      : "
    f"{confidence_percentage:.2f}%"
)

print(
    f"Top-1 / Top-2 margin: "
    f"{prediction_margin * 100:.2f}%"
)

print(
    f"Prediction entropy  : "
    f"{normalized_entropy:.4f}"
)


# ============================================================
# 21. FINAL DECISION
# ============================================================

print("\n" + "=" * 70)

print(
    "FINAL RESULT"
)

print("=" * 70)


# ------------------------------------------------------------
# REJECT / UNCERTAIN
# ------------------------------------------------------------

if rejection_reasons:

    print(
        "\n⚠ IMAGE REJECTED / PREDICTION UNCERTAIN"
    )


    print(
        "\nThe model cannot confidently classify "
        "this image as a known tire condition."
    )


    print(
        "\nPossible reasons:"
    )


    for reason in rejection_reasons:

        print(
            f"- {reason}"
        )


    print(
        "\nPlease upload a clear tire image."
    )


    messagebox.showwarning(

        "Invalid or Uncertain Image",

        (
            "The system could not confidently classify "
            "this image.\n\n"
            "Please upload a clear tire image.\n\n"
            f"Highest prediction: "
            f"{predicted_class}\n"
            f"Confidence: "
            f"{confidence_percentage:.2f}%"
        )
    )


# ------------------------------------------------------------
# ACCEPT PREDICTION
# ------------------------------------------------------------

else:

    print(
        f"\nPredicted Defect : "
        f"{predicted_class}"
    )


    print(
        f"Confidence       : "
        f"{confidence_percentage:.2f}%"
    )


    # -----------------------------------------------
    # HUMAN-FRIENDLY NAME
    # -----------------------------------------------

    display_class = predicted_class.replace(
        "_",
        " "
    )


    # -----------------------------------------------
    # NORMAL TIRE
    # -----------------------------------------------

    if predicted_class.lower() in [
        "nodefect",
        "nodamage",
        "no_defect",
        "no_damage"
    ]:

        result_message = (

            "No tire defect was identified.\n\n"

            f"Confidence: "
            f"{confidence_percentage:.2f}%"

        )


    # -----------------------------------------------
    # DEFECT
    # -----------------------------------------------

    else:

        result_message = (

            f"Detected Defect:\n"
            f"{display_class}\n\n"

            f"Confidence: "
            f"{confidence_percentage:.2f}%"

        )


    messagebox.showinfo(

        "Tire Inspection Result",

        result_message
    )


print("\n" + "=" * 70)

print(
    "Prediction completed."
)

print("=" * 70)