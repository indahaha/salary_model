import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# Load model
model = load_model('best_salary_model')

st.title("Prediksi Gaji Karyawan")
st.write("Aplikasi ini memprediksi gaji berdasarkan data karyawan.")

# Form input
age = st.number_input("Usia", min_value=18, max_value=65, value=30)
gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
department = st.selectbox("Departemen", ["HR", "Finance", "Engineering", "Sales", "Marketing"])
job_title = st.text_input("Jabatan", "Staff")
experience_years = st.number_input("Pengalaman (tahun)", min_value=0, max_value=40, value=5)
education_level = st.selectbox("Pendidikan", ["High School", "Bachelor's", "Master's", "PhD"])
location = st.selectbox("Lokasi", ["New York", "Los Angeles", "Chicago", "Houston", "Miami"])

# Prediksi
if st.button("Prediksi Gaji"):
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Department': [department],
        'Job_Title': [job_title],
        'Experience_Years': [experience_years],
        'Education_Level': [education_level],
        'Location': [location]
    })

    prediction = predict_model(model, data=input_data)
    salary_pred = prediction['prediction_label'][0]

    st.success(f"Perkiraan gaji: ${salary_pred:,.2f}")
