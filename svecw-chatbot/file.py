import pandas as pd

# Load the Excel file
file_path = "./data/svecw2024.xlsx"
xls = pd.ExcelFile(file_path)

# Load data from the first sheet
df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

# Clean the data by selecting relevant columns and renaming them
df_cleaned = df.iloc[3:, [0, 1, 2, 3, 4]].copy()
df_cleaned.columns = ["Department", "Category", "Min Rank", "Max Rank", "No. of Students"]

df_cleaned = df_cleaned.dropna(subset=["Department", "Category"])

# Filter out numeric department and category values
df_cleaned = df_cleaned[~df_cleaned["Department"].astype(str).str.isnumeric()]
df_cleaned = df_cleaned[~df_cleaned["Category"].astype(str).str.isnumeric()]

# Convert numeric columns to proper format
df_cleaned["Min Rank"] = pd.to_numeric(df_cleaned["Min Rank"], errors='coerce')
df_cleaned["Max Rank"] = pd.to_numeric(df_cleaned["Max Rank"], errors='coerce')
df_cleaned["No. of Students"] = pd.to_numeric(df_cleaned["No. of Students"], errors='coerce')

# Group by Department and Category
df_grouped = df_cleaned.groupby(["Department", "Category"], as_index=False).agg({
    "Min Rank": "min",
    "Max Rank": "max",
    "No. of Students": "sum"
})

# Save the processed data to a new Excel file
output_file_path = "./data/grouped_data.xlsx"
df_grouped.to_excel(output_file_path, index=False)

print("File saved successfully at:", output_file_path)
