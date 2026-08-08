import pandas as pd

df = pd.read_excel("june.xlsx", sheet_name="july", skiprows=1)
df = df.drop(columns=["Unnamed: 12"], errors="ignore")

# Strip extra whitespace from column names (fixes 'Protein ' -> 'Protein')
df.columns = df.columns.str.strip()

# Replace blank cells with 0 in all category columns (not the Date column)
category_cols = [c for c in df.columns if c != "Date"]
df[category_cols] = df[category_cols].fillna(0)

print(df.head(10))
print("\nColumn names:", list(df.columns))
print("\nTotal spend per category:")
print(df[category_cols].sum())