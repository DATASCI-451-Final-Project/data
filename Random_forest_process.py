

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Set file paths
input_path = '/Users/yuhanye/Desktop/451 final/KNN_chosen_cancer_patient_data.csv'
output_path = '/Users/yuhanye/Desktop/451 final/KNN_RF_chosen_cancer_patient_data.csv'

# Load the dataset
data = pd.read_csv(input_path)

# Step 1: Identify categorical variables
categorical_columns = data.select_dtypes(include=['object', 'category']).columns
quantitative_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Encode all categorical predictors and the target column
encoded_data = data.copy()
label_encoders = {}

for col in categorical_columns:
    # Create a LabelEncoder for each categorical column
    label_encoders[col] = LabelEncoder()
    # Fill missing values temporarily to encode the column
    encoded_data[col] = label_encoders[col].fit_transform(data[col].fillna('Missing'))

# Step 2: Perform Random Forest Imputation for Categorical Variables
for col in categorical_columns:
    if data[col].isnull().sum() > 0:
        # Separate rows with and without missing values
        train_data = encoded_data[encoded_data[col] != label_encoders[col].transform(['Missing'])[0]]
        test_data = encoded_data[encoded_data[col] == label_encoders[col].transform(['Missing'])[0]]

        # Ensure there's enough data to train the model
        if len(train_data) > 0 and len(test_data) > 0:
            predictors = [p for p in encoded_data.columns if p != col]
            
            # Use other columns as predictors (ensuring no missing values in predictors)
            train_X = train_data[predictors].dropna()
            train_y = train_data.loc[train_X.index, col]
            
            test_X = test_data[predictors].dropna()
            
            if len(train_X) > 0 and len(test_X) > 0:
                # Train Random Forest Classifier
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(train_X, train_y)
                
                # Predict missing values
                predictions = rf.predict(test_X)
                
                # Replace the missing values in the original dataset
                data.loc[test_X.index, col] = label_encoders[col].inverse_transform(predictions.astype(int))

# Step 3: Verify No Changes in Quantitative Variables
original_data = pd.read_csv(input_path)

# Check for unchanged column names
column_names_unchanged = original_data.columns.equals(data.columns)

# Check for unchanged row indices
row_indices_unchanged = original_data.shape[0] == data.shape[0]

# Check for unchanged quantitative values
quantitative_values_unchanged = original_data[quantitative_columns].fillna(0).equals(
    data[quantitative_columns].fillna(0)
)

# Step 4: Save the Imputed Dataset
data.to_csv(output_path, index=False)

# Output Results
print("Column names unchanged:", column_names_unchanged)
print("Row indices unchanged:", row_indices_unchanged)
print("Quantitative values unchanged:", quantitative_values_unchanged)
print(f"Imputed dataset saved to: {output_path}")
