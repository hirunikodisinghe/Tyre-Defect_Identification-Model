from pathlib import Path
from tkinter import Tk, filedialog
import base64
import mimetypes

from openai import OpenAI


# ============================================================
# 1. SETTINGS
# ============================================================

PROJECT_DIR = Path(r"E:\Hiruni\TireDefectProject")

ALLOWED_CLASSES = [
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
    "Unclear",
]


# ============================================================
# 2. OPENAI CLIENT
# ============================================================

# Automatically reads OPENAI_API_KEY
# from the environment variable.
client = OpenAI()


# ============================================================
# 3. SELECT IMAGE
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
            "*.jpg *.jpeg *.png *.webp"
        ),
        (
            "All Files",
            "*.*"
        )
    ]
)

root.destroy()


if not image_path:

    print("No image selected.")

    raise SystemExit


image_path = Path(image_path)


print("\nSelected image:")
print(image_path)


# ============================================================
# 4. CONVERT IMAGE TO BASE64
# ============================================================

mime_type, _ = mimetypes.guess_type(
    str(image_path)
)


if mime_type is None:

    mime_type = "image/jpeg"


with open(
    image_path,
    "rb"
) as image_file:

    encoded_image = base64.b64encode(
        image_file.read()
    ).decode("utf-8")


image_data_url = (
    f"data:{mime_type};base64,"
    f"{encoded_image}"
)


# ============================================================
# 5. PROMPT
# ============================================================

class_text = "\n".join(
    f"- {class_name}"
    for class_name in ALLOWED_CLASSES
)


prompt = f"""
You are reviewing an industrial tire inspection image.

Classify the visible tire condition using ONLY one of the
following labels:

{class_text}

Important rules:

1. Inspect the visible tire region carefully.
2. Select the single best label.
3. Use "Unclear" if the image is not clearly a tire,
   the defect is not visible enough, or the image cannot
   be classified reliably.
4. Do not invent a new defect category.
5. Do not rely on filename information.

Return exactly this format:

Defect: <class name>
Confidence: <0-100>
Reason: <short visual explanation>
"""


# ============================================================
# 6. SEND IMAGE TO OPENAI
# ============================================================

print("\nSending image for AI review...")


try:

    response = client.responses.create(

        model="gpt-5.6-sol",

        input=[
            {
                "role": "user",

                "content": [

                    {
                        "type": "input_text",
                        "text": prompt
                    },

                    {
                        "type": "input_image",
                        "image_url": image_data_url
                    }

                ]
            }
        ]
    )


except Exception as error:

    print("\nAPI ERROR:")
    print(error)

    raise SystemExit


# ============================================================
# 7. DISPLAY RESULT
# ============================================================

result = response.output_text


print("\n" + "=" * 70)

print(
    "CHATGPT / OPENAI VISION REVIEW"
)

print("=" * 70)

print(result)

print("=" * 70)