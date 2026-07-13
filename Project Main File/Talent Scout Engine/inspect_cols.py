import pandas as pd, sys
xl = pd.ExcelFile(r'c:/Users/HP/OneDrive/Desktop/4h Semester Projects/Machine Learning & Data Science Project/Data Set/Data Set For DS & ML Project.xlsx')
bat_sheets = [s for s in xl.sheet_names if '_BAT' in s.upper()]

# Test a few that are being skipped
for sheet in bat_sheets[1:4]:
    raw = pd.read_excel(xl, sheet_name=sheet, header=None)
    print(f"\n=== {sheet} | Shape {raw.shape} ===")
    for i, row in raw.head(10).iterrows():
        vals = [v for v in row.values if pd.notna(v)]
        print(f"  Row {i}: {vals[:8]}")
