
import pandas as pd
from sklearn.impute import KNNImputer
import os

# Set file paths
input_path = '/Users/yuhanye/Desktop/451 final/chosen_cancer_patient_data.csv'
output_directory = '/Users/yuhanye/Desktop/451 final'
output_file_name = 'KNN_chosen_cancer_patient_data.csv'
output_path = f"{output_directory}/{output_file_name}"

# Debug Step 1: Print the output path
print(f"Output path: {output_path}")

# Load the dataset
try:
    data = pd.read_csv(input_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Step 1: Identify quantitative variables (numerical columns)
quantitative_data = data.select_dtypes(include=['float64', 'int64'])

# Step 2: Apply KNN Imputer to fill missing values in quantitative variables
try:
    imputer = KNNImputer(n_neighbors=5)
    quantitative_data_imputed = pd.DataFrame(imputer.fit_transform(quantitative_data), 
                                             columns=quantitative_data.columns)
    print("KNN imputation completed successfully.")
except Exception as e:
    print(f"Error during KNN imputation: {e}")

# Replace original quantitative columns with imputed values
data[quantitative_data.columns] = quantitative_data_imputed

# Step 3: Check for any changes to column names or row indices
original_data = pd.read_csv(input_path)

# Debug Step 2: Check for column and row changes
column_names_changed = not (data.columns.equals(original_data.columns))
row_names_changed = not (data.shape[0] == original_data.shape[0])
original_quantitative_data = original_data.select_dtypes(include=['float64', 'int64'])
non_na_values_changed = not (original_quantitative_data.fillna(0).equals(quantitative_data.fillna(0)))

print("Column names changed:", column_names_changed)
print("Row indexes changed:", row_names_changed)
print("Non-NA values changed:", non_na_values_changed)

# Step 4: Save the imputed dataset
try:
    data.to_csv(output_path, index=False)
    print("File saved successfully.")
except Exception as e:
    print(f"Error saving file: {e}")

# Step 5: Debugging the output directory
try:
    files_in_directory = os.listdir(output_directory)
    print("Files in the output directory:", files_in_directory)
except Exception as e:
    print(f"Error listing files in directory: {e}")

# Step 6: Check for temporary save as a backup
try:
    temp_output_path = '/tmp/chosen_cancer_patient_data_KNN_imputed.csv'
    data.to_csv(temp_output_path, index=False)
    print(f"Temporary backup saved to: {temp_output_path}")
except Exception as e:
    print(f"Error saving to temporary path: {e}")
