import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Load the dataset from the Excel file
file_path = 'nothing.xlsx'
df = pd.read_excel(file_path)

# Convert 'Dates' to datetime format
df['Dates'] = pd.to_datetime(df['Dates'], format='%Y-%m-%d')

# Fill missing values only for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Set 'Dates' as the index
df.set_index('Dates', inplace=True)

# Separate the features and target variable
X = df.drop(columns=['Consumption'])
y = df['Consumption']

# Convert dates to numerical values for Random Forest
X = X.copy()
X['Dates'] = X.index.map(pd.Timestamp.toordinal)

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

with open("rf-model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("model saved successfully")


# # Evaluate the model on the test set
# y_pred = rf.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error on test set: {mse}')

# # Prepare the input date range for predictions
# start_date = pd.to_datetime('2022-01-01')
# end_date = pd.to_datetime('2022-12-31')
# date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# # Create a DataFrame for the input dates with the same features as in the training set
# input_df = pd.DataFrame(date_range, columns=['Dates'])
# input_df['Dates'] = input_df['Dates'].map(pd.Timestamp.toordinal)

# # Add mean values for the other features
# for column in X.columns:
#     if column != 'Dates':  # Skip 'Dates' since it's already handled
#         input_df[column] = X_train[column].mean()

# # Ensure input_df has all the same columns as X_train
# input_df = input_df[X_train.columns]

# # Predict consumption for the input date range
# predictions = rf.predict(input_df)

# # Create a DataFrame for the results
# results = pd.DataFrame({
#     'Dates': pd.date_range(start=start_date, end=end_date, freq='D'),
#     'Predicted_Consumption': predictions
# })

# # Save the results to a new CSV file
# output_file_path = 'predicted_consumption.csv'
# results.to_csv(output_file_path, index=False)

# print(f"Predictions saved to '{output_file_path}'")