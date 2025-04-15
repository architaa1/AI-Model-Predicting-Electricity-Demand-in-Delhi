# AI Model for Predicting Electricity Demand in Delhi

This project uses machine learning to forecast electricity demand in Delhi based on historical data. The model helps in energy planning by providing accurate predictions of future electricity consumption.

Demo Video Lini: https://drive.google.com/file/d/1Z-Z5mBmer5xdiOA9K0zUYH7n8bFDeFrv/view?usp=sharing

## 📌 Features

- Forecasts electricity demand using advanced ML models
- Supports XGBoost Regression for high accuracy
- Includes visual evaluation (MSE and R² Score)
- Containerized with Docker for easy deployment
- User interface for interacting with the model

## 📁 Repository Structure

├── Database.xlsx # Raw dataset 

├── model.py # Model training and preprocessing 

├── xgb.py # XGBoost regression model 

├── xgbr-model.pkl # Trained model file 

├── frontend.py # Streamlit-based frontend UI 

├── Dockerfile # Docker container instructions 

├── steps to run docker.txt # Docker usage guide 

├── MSE.png # Visualization of Mean Squared Error 

├── R2 Score.png # Visualization of R² Score 

├── Both Chart.png # Combined chart

## 🚀 Getting Started

### 1. Clone the Repository

git clone https://github.com/architaa1/AI-Model-Predicting-Electricity-Demand-in-Delhi.git

cd AI-Model-Predicting-Electricity-Demand-in-Delhi

2. Install Dependencies

Make sure Python and pip are installed. Then run:

pip install -r requirements.txt

If requirements.txt is not available, install manually:

pip install pandas numpy scikit-learn xgboost streamlit

3. Run the Application

streamlit run frontend.py

This will open the UI in your browser where you can input data and get predictions.

🐳 Running with Docker

1. Build the Docker Image

docker build -t electricity-demand-predictor .

2. Run the Container

docker run -p 8501:8501 electricity-demand-predictor

Visit http://localhost:8501 in your browser to interact with the model.

📊 Model Evaluation

The model is evaluated using the following metrics:

Mean Squared Error (MSE)

R² Score (Coefficient of Determination)

Refer to MSE.png, R2 Score.png, and Both Chart.png for performance visuals.


Developed by Archita. This project aims to contribute towards smart energy management in urban cities like Delhi.







