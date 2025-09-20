import streamlit as st
import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load Model dan Scaler
model = joblib.load("C:/Users/AtikahDR/Downloads/Model Deployment 2 Materials/diabetes_model.pkl")
scaler = joblib.load("C:/Users/AtikahDR/Downloads/Model Deployment 2 Materials/scaler.pkl")

# Judul
st.title("Prediksi Risiko Diabetes dengan Machine Learning")
st.write("Isi data pasien untuk memprediksi kemungkinan diabetes. Prediksi ini **bukan diagnosis medis**.")

# Input Data Pasien
st.subheader("Input Data Pasien")

gender = st.radio("Jenis Kelamin", ['Perempuan', 'Laki-laki'], index=0)
age = st.number_input("Usia (tahun)", min_value=0, max_value=120, value=30, step=1)
urea = st.number_input("Kadar Urea dalam Darah (mg/dL)", min_value=0.0, max_value=200.0, value=30.0, step=0.1)
cr = st.number_input("Kadar Kreatinin dalam Darah (mg/dL)", min_value=0.0, max_value=20.0, value=1.0, step=0.01)
hba1c = st.number_input("Hemoglobin Terglikasi (%)", min_value=3.0, max_value=15.0, value=5.5, step=0.1)
chol = st.number_input("Kolesterol Total (mg/dL)", min_value=100.0, max_value=400.0, value=200.0, step=1.0)
hdl = st.number_input("HDL (mg/dL)", min_value=20.0, max_value=100.0, value=50.0, step=1.0)
ldl = st.number_input("LDL (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=1.0)
vldl = st.number_input("VLDL (mg/dL)", min_value=5.0, max_value=50.0, value=20.0, step=1.0)
bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
urea_cr_ratio = st.number_input("Urea/Creatinine Ratio", min_value=0.0, max_value=50.0, value=30.0, step=0.1)
bmi_hba1c = st.number_input("BMI x HbA1c", min_value=0.0, max_value=1000.0, value=150.0, step=0.1)
age_bmi = st.number_input("Usia x BMI", min_value=0.0, max_value=10000.0, value=750.0, step=1.0)

# Encode Gender jadi angka
gender_encoded = 0 if gender == "Perempuan" else 1

# Array Input
input_data = np.array([[gender_encoded, age, urea, cr, hba1c, chol, hdl, ldl, vldl, bmi, urea_cr_ratio, 
                        bmi_hba1c, age_bmi]])

# Tombol Tampilkan Data
# Membuat dua kolom dengan lebar sama
col1, col2 = st.columns(2)

# Kolom kiri: Tampilkan Data
with col1:
    if st.button("Tampilkan Data"):
        st.json({
            "Gender": gender,
            "AGE": age,
            "Urea": urea,
            "Cr": cr,
            "HbA1c": hba1c,
            "Chol": chol,
            "HDL": hdl,
            "LDL": ldl,
            "VLDL": vldl,
            "BMI": bmi,
            "Urea_Cr_Ratio": urea_cr_ratio,
            "BMI_HbA1c": bmi_hba1c,
            "AGE_BMI": age_bmi
        })

# Kolom kanan: Prediksi
with col2:
    if st.button("Prediksi"):
        # Skalakan input
        input_scaled = scaler.transform(input_data)

        # Prediksi model
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0]
        print(type(probability))
        print(probability.shape)

        # Placeholder Hasil pediksi
        st.subheader("Hasil Prediksi")
        if prediction[0] == 0:
                st.success("✅ Pasien TIDAK BERESIKO terkena Diabetes")
        else:
            if prediction[0] == 1:  
                st.success("⚠️ Pasien BERESIKO TINGGI terkena Diabetes Type 1")
            else:
                 st.success("⚠️ Pasien BERESIKO TINGGI terkena Diabetes Type 2")

        # Visualisas Probabilitas
        st.subheader("Probabilitas Prediksi")
        labels = ["Tidak Diabetes", "Diabetes Type 1", "Diabetes Type 2"]
        fig, ax = plt.subplots()
        ax.bar(labels, probability, color=["green", "yellow", "red"])
        ax.set_ylim([0,1])
        ax.set_ylabel("Probabilitas")
        st.pyplot(fig)

    