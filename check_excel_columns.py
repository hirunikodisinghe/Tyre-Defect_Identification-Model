from openpyxl import load_workbook

EXCEL_FILE = r"E:\Hiruni\TireDefectProject\results\manual_inspection_separate_sheets.xlsx"

workbook = load_workbook(EXCEL_FILE, data_only=True)

for sheet_name in workbook.sheetnames:
    sheet = workbook[sheet_name]

    print("\nSHEET:", sheet_name)

    headers = []
    for cell in sheet[1]:
        headers.append(cell.value)

    print("HEADERS:", headers)

    for row in sheet.iter_rows(min_row=2, max_row=5, values_only=True):
        print(row)