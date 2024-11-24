# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score
# from xgboost import XGBRegressor
# from sklearn.impute import SimpleImputer

# # Read the dataset
# file_path = 'Database.xlsx'
# data = pd.read_excel(file_path)

# # Display first few rows to understand the data
# data.head(), data.columns
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.neighbors import KNeighborsRegressor

# # Select relevant features
# features = ['Dates', 'Consumption', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']
# target = 'Consumption'

# # Convert 'Dates' to datetime and extract relevant features like year, month, and day
# data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%Y')
# data['Year'] = data['Dates'].dt.year
# data['Month'] = data['Dates'].dt.month
# data['Day'] = data['Dates'].dt.day

# # Add extracted date features to the list of features
# features.extend(['Year', 'Month', 'Day'])

# # Preparing data for training
# X = data[features]
# y = data[target]

# # Handle missing values by imputing with the mean for numerical columns
# imputer = SimpleImputer(strategy='mean')
# X_imputed = imputer.fit_transform(X)

# # Split the imputed dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Define models to train
# models = {
#     "Linear Regression": LinearRegression(),
#     "Random Forest": RandomForestRegressor(random_state=42),
#     "Gradient Boosting": GradientBoostingRegressor(random_state=42),
#     "Decision Tree": DecisionTreeRegressor(random_state=42),
#     "K-Nearest Neighbors": KNeighborsRegressor(),
#     "XGBoost Regressor": XGBRegressor(random_state=42, objective='reg:squarederror')
# }

# # Train and evaluate each model
# best_model_name = None
# best_model_score = float('inf')
# results = {}

# for model_name, model in models.items():
#     # Train the model
#     model.fit(X_train_scaled, y_train)
    
#     # Predict on the test set
#     y_pred = model.predict(X_test_scaled)
    
#     # Evaluate the model using Mean Squared Error (MSE)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
    
#     # Store the results
#     results[model_name] = {'MSE': mse, 'R2 Score': r2}
    
#     # Determine the best model based on MSE
#     if mse < best_model_score:
#         best_model_score = mse
#         best_model_name = model_name

# # Output the results
# print(results, best_model_name, best_model_score)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

file_path = 'Database.xlsx'
data = pd.read_excel(file_path)

data['Dates'] = pd.to_datetime(data['Dates'], format='%m/%d/%Y')
data['Year'] = data['Dates'].dt.year
data['Month'] = data['Dates'].dt.month
data['Day'] = data['Dates'].dt.day

features = ['tempmax', 'tempmin', 'temp', 'humidity', 'windspeed', 'Year', 'Month', 'Day']
target = 'Consumption'

X = data[features]
y = data[target]

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "XGBoost Regressor": XGBRegressor(random_state=42, objective='reg:squarederror')
}

best_model_name = None
best_model_score = float('inf')
results = {}

for model_name, model in models.items():
    
    model.fit(X_train_scaled, y_train)
    
    
    y_pred = model.predict(X_test_scaled)
    
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    
    results[model_name] = {'MSE': mse, 'R2 Score': r2}
    
    
    if mse < best_model_score:
        best_model_score = mse
        best_model_name = model_name


print("Model Evaluation Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: MSE = {metrics['MSE']:.4f}, R2 Score = {metrics['R2 Score']:.4f}")

print(f"\nBest Model: {best_model_name} with MSE = {best_model_score:.4f}")
