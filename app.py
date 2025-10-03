import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
#   
# from sklearn.preprocessing import LabelEncoder()

try:
    model = joblib.load(open('my_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found! Make sure 'my_model.pkl' is in the same folder.")
    st.stop()


st.set_page_config(
    page_title="Insurance Charge Predictor",
    page_icon="🏥",
    layout="centered"
)



st.title("🏥 Insurance Charge Predictor")
st.write("Enter the details below to get a prediction of the insurance charges.")



col1, col2 = st.columns(2)

with col1:
    age = st.number_input('**Age**', min_value=18, max_value=100, value=25)
    sex = st.selectbox('**Sex**', ('Male', 'Female'))
    bmi = st.number_input('**BMI**', min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    children = st.slider('**Number of Children**', min_value=0, max_value=5, value=0)
    smoker = st.selectbox('**Smoker**', ('No', 'Yes'))
    region = st.selectbox('**Region**', ('Southwest', 'Southeast', 'Northwest', 'Northeast'))


# --- Data Preprocessing and Prediction ---
if st.button('**Predict Charges**', type="primary"):

    # Map categorical inputs to numerical values
    # NOTE: This mapping must match the one used during model training!
    sex_map = {'Male': 0, 'Female': 1}
    smoker_map = {'No': 0, 'Yes': 1}
    region_map = {'Southwest': 0, 'Southeast': 1, 'Northwest': 2, 'Northeast': 3}

    # Create a DataFrame from the inputs
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_map[sex]],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_map[smoker]],
        'region': [region_map[region]]
    })

    


    scaler =  joblib.load(open('my_scaler.pkl' , 'rb'))
    new_data = scaler.transform(input_data)
    # Make prediction
    prediction = model.predict(new_data)
    predicted_charge = prediction[0]

    st.success(f"**Predicted Insurance Charge: ${predicted_charge:,.2f}**")