

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
dataset_path = '/Users/yuhanye/Desktop/451 final/data/KNN_RF_chosen_cancer_patient_data.csv'  # Specify the path to the dataset (CSV file) you want to use
df = pd.read_csv(dataset_path)  # Read the CSV file into a DataFrame using pandas

# Define the response variable and the predictors
response_variable = 'MCQ230A'  # Set the response variable (target variable) as 'MCQ230A', which corresponds to cancer types
predictors = df.columns[df.columns != response_variable]  # Define the predictors as all columns except the response variable

# Split the dataset into training and test sets (80:20 ratio), stratified by the response variable
# train_test_split is used to randomly split the dataset into training and test sets
# Stratification is applied to ensure that the distribution of cancer types is similar in both sets
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[response_variable], random_state=42)

# Save the split datasets to CSV files
# The training set is saved to 'cancer_data_train.csv'
train_df.to_csv('/Users/yuhanye/Desktop/451 final/data/cancer_data_train.csv', index=False)  # Save the training DataFrame to a CSV file without the index
# The test set is saved to 'cancer_data_test.csv'
test_df.to_csv('/Users/yuhanye/Desktop/451 final/data/cancer_data_test.csv', index=False)  # Save the test DataFrame to a CSV file without the index

# Print a message to indicate that the data splitting is complete
print("Data splitting completed.")
