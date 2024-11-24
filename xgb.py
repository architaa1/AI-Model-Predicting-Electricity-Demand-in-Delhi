# import pandas as pd
# from xgboost import XGBRegressor

# # Load the dataset
# file_path = 'Database.xlsx'
# df = pd.read_excel(file_path)

# # Select the relevant features
# features = ['Dates', 'Consumption', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']
# df = df[features]

# # Drop rows with missing values (optional, depending on your dataset)
# df.dropna(inplace=True)

# # Convert Dates to datetime
# df['Dates'] = pd.to_datetime(df['Dates'])

# # Split the data into features (X) and target (y)
# X = df[['Dates', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']]
# y = df['Consumption']

# # Convert Dates to numerical representation for modeling
# X['Dates'] = X['Dates'].map(pd.Timestamp.toordinal)

# # Train the XGBRegressor model on the entire dataset
# model = XGBRegressor()
# model.fit(X, y)

# # Prepare the data for the specified date range 1st January 2022 to 31st December 2022
# start_date = pd.to_datetime('2024-01-01')
# end_date = pd.to_datetime('2024-05-30')

# df_predict = df[(df['Dates'] >= start_date) & (df['Dates'] <= end_date)]
# X_predict = df_predict[['Dates', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']]

# # Convert Dates to numerical representation
# X_predict['Dates'] = X_predict['Dates'].map(pd.Timestamp.toordinal)

# # Predict consumption values for the given date range
# predictions = model.predict(X_predict)

# # Create a new DataFrame with Dates and the predicted Consumption
# output_df = pd.DataFrame({
#     'Dates': pd.to_datetime(X_predict['Dates'].map(pd.Timestamp.fromordinal)),  # Convert back to datetime
#     'Predicted Consumption': predictions
# })

# # Save the predictions to a new CSV file with the specified date range
# output_df.to_csv('predicted_consumption_2024.csv', index=False)

# print("Predictions for 2022 have been saved to 'predicted_consumption_2024.csv'")
# import pandas as pd
# from xgboost import XGBRegressor

# # Load the dataset
# file_path = 'Database.xlsx'
# df = pd.read_excel(file_path)

# # Select the relevant features
# features = ['Dates', 'Consumption', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']
# df = df[features]

# # Drop rows with missing values
# df.dropna(inplace=True)

# # Convert Dates to datetime
# df['Dates'] = pd.to_datetime(df['Dates'])

# # Extract day and month to match future dates with historical features
# df['Day'] = df['Dates'].dt.day
# df['Month'] = df['Dates'].dt.month

# # Split the data into features (X) and target (y)
# X = df[['Dates', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']]
# y = df['Consumption']

# # Convert Dates to numerical representation for modeling
# X['Dates'] = X['Dates'].map(pd.Timestamp.toordinal)

# # Train the XGBRegressor model on the historical dataset
# model = XGBRegressor()
# model.fit(X, y)

# # Define the future date range for prediction (e.g., 2024)
# start_date = pd.to_datetime('2024-01-01')
# end_date = pd.to_datetime('2024-05-30')
# date_range = pd.date_range(start=start_date, end=end_date)

# # Prepare the future dataframe with historical features
# future_df = pd.DataFrame({
#     'Dates': date_range,
#     'Day': date_range.day,
#     'Month': date_range.month
# })

# # Merge future dates with historical data to get the corresponding features
# future_df = future_df.merge(df[['Day', 'Month', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']], 
#                             on=['Day', 'Month'], how='left')

# # Convert future Dates to numerical representation
# future_df['Dates'] = future_df['Dates'].map(pd.Timestamp.toordinal)

# # Predict consumption values for the future dates
# X_predict = future_df[['Dates', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']]
# predictions = model.predict(X_predict)

# # Create a new DataFrame with Dates and the predicted Consumption
# output_df = pd.DataFrame({
#     'Dates': pd.to_datetime(future_df['Dates'].map(pd.Timestamp.fromordinal)),  # Convert back to datetime
#     'Predicted Consumption': predictions
# })

# # Save the predictions to a new CSV file with the specified date range
# output_df.to_csv('predicted_consumption_2024.csv', index=False)

# print("Predictions for 2024 have been saved to 'predicted_consumption_2024.csv'")
# import pandas as pd
# from xgboost import XGBRegressor

# # Load the dataset
# file_path = 'Database.xlsx'
# df = pd.read_excel(file_path)

# # Select the relevant features
# features = ['Dates', 'Consumption', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']
# df = df[features]

# # Drop rows with missing values
# df.dropna(inplace=True)

# # Convert Dates to datetime
# df['Dates'] = pd.to_datetime(df['Dates'])

# # Extract year, day, and month to match future dates with historical features
# df['Year'] = df['Dates'].dt.year
# df['Day'] = df['Dates'].dt.day
# df['Month'] = df['Dates'].dt.month

# # Use only the most recent year's data
# latest_year = df['Year'].max()
# df_latest = df[df['Year'] == latest_year]

# # Train the model on historical data
# X = df[['Dates', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']]
# y = df['Consumption']

# # Convert Dates to numerical representation for modeling
# X['Dates'] = X['Dates'].map(pd.Timestamp.toordinal)

# # Train the XGBRegressor model
# model = XGBRegressor()
# model.fit(X, y)

# # Prepare future dates (e.g., 2024)
# start_date = pd.to_datetime('2018-01-01')
# end_date = pd.to_datetime('2027-12-31')
# date_range = pd.date_range(start=start_date, end=end_date)

# # Prepare future DataFrame
# future_df = pd.DataFrame({
#     'Dates': date_range,
#     'Day': date_range.day,
#     'Month': date_range.month
# })

# # Merge future dates with the most recent historical features
# future_df = future_df.merge(df_latest[['Day', 'Month', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']], 
#                             on=['Day', 'Month'], how='left')

# # Convert Dates to numerical representation
# future_df['Dates'] = future_df['Dates'].map(pd.Timestamp.toordinal)

# # Predict consumption values for future dates
# X_predict = future_df[['Dates', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']]
# predictions = model.predict(X_predict)

# # Create a DataFrame with Dates and predicted Consumption
# output_df = pd.DataFrame({
#     'Dates': pd.to_datetime(future_df['Dates'].map(pd.Timestamp.fromordinal)),  # Convert back to datetime
#     'Predicted Consumption': predictions
# })

# # Save the predictions to a CSV file
# output_df.to_csv('predicted_consumption_2018-27.csv', index=False)

# print("Predictions for 2018-27 have been saved to 'predicted_consumption_2018-27.csv'")

import pandas as pd
from xgboost import XGBRegressor
import pickle

file_path = 'Database.xlsx'
df = pd.read_excel(file_path)

features = ['Dates', 'Consumption', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']
df = df[features]

df.dropna(inplace=True)

df['Dates'] = pd.to_datetime(df['Dates'])

df['Year'] = df['Dates'].dt.year
df['Day'] = df['Dates'].dt.day
df['Month'] = df['Dates'].dt.month

latest_year = df['Year'].max()
df_latest = df[df['Year'] == latest_year]

X = df[['Dates', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']]
y = df['Consumption']

X['Dates'] = X['Dates'].map(pd.Timestamp.toordinal)

model = XGBRegressor()
model.fit(X, y)

with open("xgbr-model.pkl", "wb") as f:
    pickle.dump(model, f)

print("model saved successfully")

# # Prepare future dates (e.g., 2024)
# start_date = pd.to_datetime('2018-01-01')
# end_date = pd.to_datetime('2027-12-31')
# date_range = pd.date_range(start=start_date, end=end_date)

# # Prepare future DataFrame
# future_df = pd.DataFrame({
#     'Dates': date_range,
#     'Day': date_range.day,
#     'Month': date_range.month
# })

# # Merge future dates with the most recent historical features
# future_df = future_df.merge(df_latest[['Day', 'Month', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']], 
#                             on=['Day', 'Month'], how='left')

# # Convert Dates to numerical representation
# future_df['Dates'] = future_df['Dates'].map(pd.Timestamp.toordinal)

# # Predict consumption values for future dates
# X_predict = future_df[['Dates', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']]
# predictions = model.predict(X_predict)

# # Create a DataFrame with Dates and predicted Consumption
# output_df = pd.DataFrame({
#     'Dates': pd.to_datetime(future_df['Dates'].map(pd.Timestamp.fromordinal)),  # Convert back to datetime
#     'Predicted Consumption': predictions
# })

# # Save the predictions to a CSV file
# output_df.to_csv('predicted_consumption_2018-27.csv', index=False)

# print("Predictions for 2018-27 have been saved to 'predicted_consumption_2018-27.csv'")