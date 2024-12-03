

import pandas as pd

# File paths
original_file_path = '/Users/yuhanye/Desktop/451 final/chosen_cancer_patient_data.csv'
imputed_file_path = '/Users/yuhanye/Desktop/451 final/KNN_RF_chosen_cancer_patient_data.csv'

# Load datasets
original_data = pd.read_csv(original_file_path)
imputed_data = pd.read_csv(imputed_file_path)

# Check 1: Ensure no missing values in the imputed file
remaining_na = imputed_data.isnull().sum().sum()
if remaining_na == 0:
    print("Check 1 Passed: No missing values in the KNN_RF file.")
else:
    print(f"Check 1 Failed: There are {remaining_na} missing values in the KNN_RF file.")

# Check 2: Validate that no non-NA values in the original dataset were changed
tolerance = 1e-6  # Floating-point precision tolerance
non_na_unchanged = True

# Compare quantitative variables
quantitative_columns = original_data.select_dtypes(include=['float64', 'int64']).columns
quantitative_changed = {}
for col in quantitative_columns:
    original_values = original_data[col].dropna()
    imputed_values = imputed_data[col].loc[original_values.index]
    differences = (original_values - imputed_values).abs() > tolerance
    if differences.any():
        quantitative_changed[col] = original_values[differences]
        non_na_unchanged = False

# Compare categorical variables
categorical_columns = original_data.select_dtypes(include=['object', 'category']).columns
categorical_changed = {}
for col in categorical_columns:
    original_non_na = original_data[col].dropna()
    imputed_non_na = imputed_data.loc[original_non_na.index, col]
    if not original_non_na.equals(imputed_non_na):
        categorical_changed[col] = original_non_na.compare(imputed_non_na)
        non_na_unchanged = False

# Output Results
if remaining_na == 0:
    print("Check 1 Passed: No missing values in the KNN_RF file.")
else:
    print(f"Check 1 Failed: There are {remaining_na} missing values in the KNN_RF file.")

if non_na_unchanged:
    print("Check 2 Passed: No non-NA values in the original file were changed in the KNN_RF file.")
else:
    print("Check 2 Failed: Some non-NA values in the original file were changed.")
    if quantitative_changed:
        print("Quantitative changes detected in the following columns:")
        for col, changes in quantitative_changed.items():
            print(f"  - {col}: {changes}")
    if categorical_changed:
        print("Categorical changes detected in the following columns:")
        for col, changes in categorical_changed.items():
            print(f"  - {col}:")
            print(changes)

# Summary
if remaining_na == 0 and non_na_unchanged:
    print("All checks passed: The KNN_RF file has no NA values, and no non-NA values in the original file were changed.")
else:
    print("Some checks failed: Please review the issues noted above.")
