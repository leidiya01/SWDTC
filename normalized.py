import pandas as pd

# Step 1: Load the data from Excel
data = pd.read_excel(r'C:\Users\leidy\Desktop\initial-data.xlsx')

# Step 2: Rename the columns to X1 to X7
column_mapping = {
    '绿色信贷': 'X1',
    '绿色投资': 'X2',
    '绿色保险': 'X3',
    '绿色债券': 'X4',
    '绿色支持': 'X5',
    '绿色基金': 'X6',
    '绿色权益': 'X7'
}
data_renamed = data.rename(columns=column_mapping)

# Step 3: Define the positive and negative columns
positive_columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7']


# Step 4: Apply Min-Max normalization
for col in positive_columns:
    data_renamed[col] = (data_renamed[col] - data_renamed[col].min()) / (data_renamed[col].max() - data_renamed[col].min())

# Step 5: Save the normalized data to a new Excel file
output_path = 'normalized_data.xlsx'  # Change the path if needed
data_renamed.to_excel(output_path, index=False)

print(f"Normalized data saved to {output_path}")
