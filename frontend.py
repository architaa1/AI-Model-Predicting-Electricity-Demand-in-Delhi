

import streamlit as st
from streamlit_lottie import st_lottie 
import json
import pandas as pd
import pickle


file_path = 'Database.xlsx'
df = pd.read_excel(file_path)


df['Dates'] = pd.to_datetime(df['Dates'])


with open("xgbr-model.pkl", "rb") as f:
    loaded_model = pickle.load(f)


with open("Animation - 1725291110869.json", "r") as file:
    url = json.load(file)

with open("Animation - 4.json", "r") as file:
    url2 = json.load(file)

with open("save energy - 1.json", "r") as file:
    start_animation = json.load(file)


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    
    .custom-font {
        font-family: 'Roboto', sans-serif;
        font-size: 18px; /* Adjust font size if needed */
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1 style='text-align: center;'>Unveiling Delhi’s Electricity Needs</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>One <span style='color: red;'>Prediction</span> at a Time</h1>", unsafe_allow_html=True)

st_lottie(url, 
    reverse=True, 
    height=450, 
    width=800, 
    speed=1, 
    loop=True, 
    quality='high', 
    key='Car'
)

st.subheader("Predicting the Pulse of Delhi’s Power Needs")

with st.container():
    st.write("-----")
    left_column, right_column = st.columns(2)
    with left_column:
        date = st.date_input("ENTER DATE: ")
        
    with right_column:
        options = ["CENTRAL", "NORTH", "SOUTH", "EAST", "WEST", "NORTH-EAST", "NORTH-WEST", "SOUTH-EAST", "SOUTH-WEST"]
        locality = st.selectbox("ENTER LOCALITY: ", options)

    
    if st.button("SUBMIT", type="primary"):
        prop = 1
        target_date = pd.to_datetime(date)
        latest_date_in_dataset = df['Dates'].max()
        
        
        if target_date > latest_date_in_dataset:
            latest_year = latest_date_in_dataset.year
            target_date_in_latest_year = target_date.replace(year=latest_year)
            
            
            feature_values = df.loc[
                (df['Dates'].dt.month == target_date_in_latest_year.month) & 
                (df['Dates'].dt.day == target_date_in_latest_year.day) & 
                (df['Dates'].dt.year == latest_year),
                ['Dates', 'temp', 'tempmax', 'tempmin', 'humidity', 'windspeed']
            ]
        else:
            
            feature_values = df.loc[df['Dates'] == target_date, ['Dates', 'temp', 'tempmax', 'tempmin', 'humidity', 'windspeed']]

        
        if feature_values.empty:
            st.write(f"No data found for {target_date.date()}")
        else:
            
            locality_prop_map = {
                "CENTRAL": 0.09, "NORTH": 0.11, "SOUTH": 0.18, "EAST": 0.10,
                "WEST": 0.16, "NORTH-EAST": 0.06, "NORTH-WEST": 0.15,
                "SOUTH-EAST": 0.05, "SOUTH-WEST": 0.10
            }
            prop = locality_prop_map.get(locality, 1)  # Default to 1 if not found

            
            feature_values['Dates'] = feature_values['Dates'].map(pd.Timestamp.toordinal)

            
            X_predict = feature_values[['Dates', 'tempmax', 'tempmin', 'temp', 'humidity', 'windspeed']]
            predictions = loaded_model.predict(X_predict)

            
            # st.write(f"Feature values on {target_date.date()}:\n{feature_values.iloc[0].to_dict()}")
            #st.subheader(f"⚡Predicted : {predictions[0] * prop:.2f} MU")
            # st.subheader(f"{st.markdown("<h1 style='text-align: center;'>Predicted <span style='color: red;'>Consumption</span></h1>", unsafe_allow_html=True)}")
            # Change color to red
            st.markdown(f"<h4>⚡Predicted <span style='color:red;'>Consumption: </span>{predictions[0] * prop:.2f}</h4>", unsafe_allow_html=True)
            st.markdown(f"<h4>🌡️Predicted <span style='color:red;'>Temperature: </span>{feature_values.iloc[0][1]}\u00B0C</h4>", unsafe_allow_html=True)
            st.markdown(f"<h4>🌡️Predicted <span style='color:red;'>Maximum Temperature: </span>{feature_values.iloc[0][2]}\u00B0C</h4>", unsafe_allow_html=True)
            st.markdown(f"<h4>🌡️Predicted <span style='color:red;'>Minimum Temperature: </span>{feature_values.iloc[0][3]}\u00B0C</h4>", unsafe_allow_html=True)
            st.markdown(f"<h4>💧Predicted <span style='color:red;'>Humidity: </span>{feature_values.iloc[0][4]} %</h4>", unsafe_allow_html=True)
            st.markdown(f"<h4>🍃Predicted <span style='color:red;'>Wind Speed: </span>{feature_values.iloc[0][5]} KM/HR</h4>", unsafe_allow_html=True)



            # st.subheader(f"🌡️Predicted Temperature: {feature_values.iloc[0][1]}\u00B0C")
            # st.subheader(f"🌡️Predicted Maximum Temperature: {feature_values.iloc[0][2]}\u00B0C")
            # st.subheader(f"🌡️Predicted Minimum Temperature: {feature_values.iloc[0][3]}\u00B0C")
            # st.subheader(f"💧Predicted Humidity: {feature_values.iloc[0][4]} %")
            # st.subheader(f"🍃Predicted Wind Speed: {feature_values.iloc[0][5]} KM/HR")
            # st.write(f"Predicted consumption: {predictions[0] * prop:.2f}")

            


st.title("Data-Driven Insights for a Sustainable Tomorrow")

st.markdown("""
    <ul style='list-style-type: disc; padding-left: 20px;'>
        <li><h3 class='custom-font' style='text-align: left;'>Solar energy contributes to reducing peak load by up to 20%, helping Discoms manage high-demand periods more efficiently.</h3></li>
        <li><h3 class='custom-font' style='text-align: left;'>With over 2 million streetlights, Delhi's illumination consumes as much electricity as a small city!</h3></li>
        <li><h3 class='custom-font' style='text-align: left;'>During peak hours, Delhi’s power consumption spikes dramatically, highlighting the need for robust grid infrastructure.</h3></li>
        <li><h3 class='custom-font' style='text-align: left;'>Delhi’s energy demand surges by over 50% during summer months, driven by widespread air conditioning use.</h3></li>    
    </ul>
""", unsafe_allow_html=True)

st_lottie(url2, reverse=True, height=400, width=900, speed=1, loop=True, quality='high', key='Car2')

st.image("R2 Score.png", caption="Accuracy", use_column_width=True)
st.image("MSE.png", caption="Error", use_column_width=True)
st.image("Both Chart.png", caption="Comparison", use_column_width=True)

st_lottie(start_animation, reverse=True, height=300, width=300, speed=1, loop=True, quality='high', key='StartAnimation')

st.markdown(
    "<h2 style='text-align: center;'>Powering Tomorrow with Less Today</h2>",
    unsafe_allow_html=True
)

print("server is running at port 8501")