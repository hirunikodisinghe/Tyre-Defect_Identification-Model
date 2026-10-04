from pathlib import Path

from openpyxl import load_workbook
from openpyxl.styles import (
    Font,
    PatternFill,
)


PROJECT_DIR = Path(
    r"E:\Hiruni\TireDefectProject"
)


EXCEL_PATH = (
    PROJECT_DIR
    / "manual_inspection_by_defect.xlsx"
)


if not EXCEL_PATH.exists():

    raise FileNotFoundError(
        f"Excel file not found:\n{EXCEL_PATH}"
    )


workbook = load_workbook(
    EXCEL_PATH
)


updated = 0


for worksheet in workbook.worksheets:

    if worksheet.title == "Summary":

        continue


    # --------------------------------------------------------
    # Find columns using header names
    # --------------------------------------------------------

    headers = {}


    for cell in worksheet[1]:

        headers[
            str(cell.value).strip()
        ] = cell.column


    required = [

        "Note",

        "ActualDefect",

        "ModelPrediction",

        "CorrectDefectName",

        "ReviewStatus",

    ]


    if not all(
        name in headers
        for name in required
    ):

        print(
            f"Skipping sheet: "
            f"{worksheet.title}"
        )

        continue


    for row_number in range(
        2,
        worksheet.max_row + 1
    ):

        actual = worksheet.cell(

            row=row_number,

            column=headers[
                "ActualDefect"
            ]

        ).value


        prediction = worksheet.cell(

            row=row_number,

            column=headers[
                "ModelPrediction"
            ]

        ).value


        correct = worksheet.cell(

            row=row_number,

            column=headers[
                "CorrectDefectName"
            ]

        ).value


        note_cell = worksheet.cell(

            row=row_number,

            column=headers[
                "Note"
            ]

        )


        review_cell = worksheet.cell(

            row=row_number,

            column=headers[
                "ReviewStatus"
            ]

        )


        # ----------------------------------------------------
        # No manual review yet
        # ----------------------------------------------------

        if (
            correct is None
            or
            str(correct).strip() == ""
        ):

            note_cell.value = (
                "Manual inspection required."
            )

            review_cell.value = (
                "Pending"
            )

            review_cell.fill = PatternFill(
                fill_type="solid",
                fgColor="FFF2CC"
            )

            continue


        actual = str(
            actual
        ).strip()


        prediction = str(
            prediction
        ).strip()


        correct = str(
            correct
        ).strip()


        # ----------------------------------------------------
        # CASE 1
        # ALL THREE AGREE
        # ----------------------------------------------------

        if (
            correct == actual
            and
            correct == prediction
        ):

            note = (
                "Dataset label and model "
                "prediction are correct."
            )

            review_status = (
                "Correct"
            )

            fill = "C6EFCE"


        # ----------------------------------------------------
        # CASE 2
        # MODEL IS CORRECT
        # DATASET LABEL IS WRONG
        # ----------------------------------------------------

        elif (
            correct == prediction
            and
            correct != actual
        ):

            note = (
                "Model prediction is correct, "
                "but the image is incorrectly labelled."
            )

            review_status = (
                "Relabel Dataset"
            )

            fill = "F4B183"


        # ----------------------------------------------------
        # CASE 3
        # DATASET LABEL IS CORRECT
        # MODEL IS WRONG
        # ----------------------------------------------------

        elif (
            correct == actual
            and
            correct != prediction
        ):

            note = (
                "Dataset label is correct, "
                "but model prediction is incorrect."
            )

            review_status = (
                "Model Error"
            )

            fill = "FFC7CE"


        # ----------------------------------------------------
        # CASE 4
        # BOTH ACTUAL AND MODEL ARE WRONG
        # ----------------------------------------------------

        else:

            note = (
                "Both dataset label and "
                "model prediction are incorrect. "
                f"Correct defect: {correct}."
            )

            review_status = (
                "Relabel Dataset"
            )

            fill = "FFD966"


        note_cell.value = (
            note
        )


        review_cell.value = (
            review_status
        )


        review_cell.fill = PatternFill(
            fill_type="solid",
            fgColor=fill
        )


        updated += 1


workbook.save(
    EXCEL_PATH
)


print("=" * 70)

print(
    "REVIEW NOTES UPDATED"
)

print("=" * 70)

print(
    f"Reviewed rows updated: "
    f"{updated}"
)

print(
    f"\nSaved:\n{EXCEL_PATH}"
)

print("\nDone.")