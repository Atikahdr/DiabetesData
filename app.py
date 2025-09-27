import os
import subprocess
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
  
# Load Model dan Scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Session State untuk navigasi
if "page" not in st.session_state:
    st.session_state.page = "input"

if "input_data" not in st.session_state:
    st.session_state.input_data = None

# Sidebar Button
if st.sidebar.button("📝 Input Data Pasien"):
    st.session_state.page = "input"

# Sidebar Menu dengan Dropdown
with st.sidebar.expander("📊 Machine Learning", expanded=True):
    if st.button("Result by Table Data"):
        st.session_state.page = "table"
    if st.button("Result by Line Chart"):
        st.session_state.page = "line_chart"
    if st.button("Result by Pie Chart"):
        st.session_state.page = "pie_chart"
    if st.button("Result by Bar Chart  "):
        st.session_state.page = "bar_chart"
    
# Fungsi Prediksi
def predict_diabetes(model, scaler, input_data):
    """Fungsi untuk scaling data, prediksi, dan probabilitas"""

    # Scaled
    input_scaled = scaler.transform(input_data)

    # Prediksi model
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0]

    return prediction, probability

# Input Data Pasien
if st.session_state.page == "input":
    st.title("Prediksi Risiko Diabetes dengan Machine Learning")
    st.write("Isi data pasien untuk memprediksi kemungkinan diabetes. "
             "Prediksi ini **bukan diagnosis medis**.")

    gender = st.radio("Jenis Kelamin", ['Perempuan', 'Laki-laki'], index=0)
    age = st.number_input("Usia (tahun)", min_value=25, max_value=77, value=30, step=1)
    urea = st.number_input("Kadar Urea (mg/dL)", min_value=1.1, max_value=26.40, value=5.0, step=0.1)
    cr = st.number_input("Kadar Kreatinin (mg/dL)", min_value=0.1, max_value=9.0, value=0.7, step=0.1)
    hba1c = st.number_input("Hemoglobin Terglikasi (%)", min_value=0.9, max_value=14.6, value=5.5, step=0.1)
    chol = st.number_input("Kolesterol (mg/dL)", min_value=81, max_value=367, value=150, step=1)
    tg = st.number_input("Triglycerides (mg/dL)", min_value=53, max_value=771, value=100, step=1)
    hdl = st.number_input("High Density Lipoprotein (mg/dL)", min_value=15, max_value=97, value=25, step=1)
    ldl = st.number_input("Low Density Lipoprotein (mg/dL)", min_value=12, max_value=217, value=98, step=1)
    vldl = st.number_input("Very Low Density Lipoprotein(mg/dL)", min_value=10.6, max_value=154.2, value=25.2, step=0.1)
    bmi = st.number_input("Body Mass Index (kg/m²)", min_value=19.0, max_value=43.25, value=23.0, step=0.1)
    urea_cr_ratio = st.number_input("Urea/Creatinine", min_value=1.0, max_value=39.0, value=25.0, step=0.1)
    bmi_hba1c = st.number_input("BMI x HbA1c", min_value=19.8, max_value=475.2, value=175.2, step=0.1)
    age_bmi = st.number_input("Usia x BMI", min_value=550.0, max_value=2553.0, value=750.0, step=1.0)

    gender_encoded = 0 if gender == "Perempuan" else 1

    st.session_state.input_data = np.array([[gender_encoded, age, urea, cr, hba1c, chol, 
                                             tg, hdl, ldl, vldl, bmi, urea_cr_ratio,
                                             bmi_hba1c, age_bmi]])
                                             
# Table Page
elif st.session_state.page == "table":
    st.title("📋 Tabel Data Pasien")
    if st.session_state.input_data is None:
        st.warning("⚠️ Silakan isi data pasien terlebih dahulu di menu *Input Data Pasien*.")
    else:
        st.session_state.data = {
            "Gender": ["Laki-laki" if st.session_state.input_data[0][0] == 1 else "Perempuan"],
            "Age": [round(float(st.session_state.input_data[0][1]), 2)],
            "Urea": [round(float(st.session_state.input_data[0][2]), 2)],
            "Cr": [round(float(st.session_state.input_data[0][3]), 2)],
            "HbA1c": [round(float(st.session_state.input_data[0][4]), 2)],
            "Chol": [round(float(st.session_state.input_data[0][5]), 2)],
            "TG": [round(float(st.session_state.input_data[0][6]), 2)],
            "HDL": [round(float(st.session_state.input_data[0][7]), 2)],
            "LDL": [round(float(st.session_state.input_data[0][8]), 2)],
            "VLDL": [round(float(st.session_state.input_data[0][9]), 2)],
            "BMI": [round(float(st.session_state.input_data[0][10]), 2)],
            "Urea/Cr": [round(float(st.session_state.input_data[0][11]), 2)],
            "BMI x HbA1c": [round(float(st.session_state.input_data[0][12]), 2)],
            "AGE x BMI": [round(float(st.session_state.input_data[0][13]), 2)]
        }

        data_input = pd.DataFrame(st.session_state.data)
        st.session_state.data_input = data_input
        data_input_transpose = data_input.T
        data_input_transpose.rename(columns={0: "Data Pasien"}, inplace=True)
        st.dataframe(data_input_transpose)

        # Prediksi
        prediction, prob = predict_diabetes(model, scaler, st.session_state.input_data)

        st.title("Hasil Prediksi")
        if prediction[0] == 0:
            st.success("✅ Pasien TIDAK BERISIKO terkena Diabetes")
        elif prediction[0] == 1:
            st.warning("⚠️ Pasien BERISIKO TINGGI terkena Diabetes Type 1")
        else:
            st.error("⚠️ Pasien BERISIKO TINGGI terkena Diabetes Type 2")

# Line Chart Page
elif st.session_state.page == "line_chart":
    st.title("Hasil Prediksi & Probabilitas")
    if st.session_state.input_data is None or 'data' not in st.session_state:
        st.warning("⚠️ Silakan isi data pasien terlebih dahulu di menu Input Data Pasien.")
    else:
        data_input = st.session_state.data_input

        # Menampilkan data nama, tahun, BMI
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jenis Kelamin", data_input["Gender"][0])
        with col2:
            st.metric("Umur", f"{int(data_input['Age'][0])} tahun")
        with col3:
            st.metric("BMI", f"{data_input['BMI'][0]:.1f} kg/m²")

        # Line Chart per kategori biar tidak turun drastis
        st.subheader("📈 Visualisasi Data Pasien (Line Chart)")

        # Lipid profile
        lipid_features = ["Chol", "TG", "HDL", "LDL", "VLDL"]
        lipid_values = [data_input[feat][0] for feat in lipid_features]
        df_lipid = pd.DataFrame({"Fitur": lipid_features, "Nilai": lipid_values})
        st.markdown("**🧪 Lipid Profile (Profil Lemak)**")
        st.line_chart(df_lipid.set_index("Fitur"), y="Nilai", height=250)

        # Renal function
        renal_features = ["Urea", "Cr", "Urea/Cr"]
        renal_values = [data_input[feat][0] for feat in renal_features]
        df_renal = pd.DataFrame({"Fitur": renal_features, "Nilai": renal_values})
        st.markdown("**💧 Renal Function (Fungsi Ginjal)**")
        st.line_chart(df_renal.set_index("Fitur"), y="Nilai", height=250)

        # Glycemic
        glycemic_features = ["HbA1c", "BMI x HbA1c"]
        glycemic_values = [data_input[feat][0] for feat in glycemic_features]
        df_glycemic = pd.DataFrame({"Fitur": glycemic_features, "Nilai": glycemic_values})
        st.markdown("**💉 Glycemic (Gula Darah)**")
        st.line_chart(df_glycemic.set_index("Fitur"), y="Nilai", height=250)

        # Prediksi
        prediction, prob = predict_diabetes(model, scaler, st.session_state.input_data)

        if prediction[0] == 0:
            st.success("✅ Pasien TIDAK BERISIKO terkena Diabetes")
        elif prediction[0] == 1:
            st.warning("⚠️ Pasien BERISIKO TINGGI terkena Diabetes Type 1")
        else:
            st.error("⚠️ Pasien BERISIKO TINGGI terkena Diabetes Type 2")

# Pie Chart Page
elif st.session_state.page == "pie_chart":
    st.title("Hasil Prediksi & Probabilitas")
    if st.session_state.input_data is None or 'data' not in st.session_state:
        st.warning("⚠️ Silakan isi data pasien terlebih dahulu di menu Input Data Pasien.")
    else:
        data_input = st.session_state.data_input

        # Menampilkan data nama, tahun, BMI
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jenis Kelamin", data_input["Gender"][0])
        with col2:
            st.metric("Umur", f"{int(data_input['Age'][0])} tahun")
        with col3:
            st.metric("BMI", f"{data_input['BMI'][0]:.1f} kg/m²")

        # Prediksi
        prediction, prob = predict_diabetes(model, scaler, st.session_state.input_data)

        # Pie chart probabilitas
        all_labels = ["Tidak Diabetes", "Diabetes Type 1", "Diabetes Type 2"]
        all_colors = ["forestgreen", "darkorange", "firebrick"]

        # Filter hanya yang lebih dari 1%
        threshold = 0.01
        labels = [l for l, p in zip(all_labels, prob) if p > threshold]
        colors = [c for c, p in zip(all_colors, prob) if p > threshold]
        probability = [p for p in prob if p > threshold]

        # Pesan hasil prediksi
        if prediction[0] == 0:
            st.success("✅ Pasien TIDAK BERISIKO terkena Diabetes")
        elif prediction[0] == 1:
            st.warning("⚠️ Pasien BERISIKO TINGGI terkena Diabetes Type 1")
        else:
            st.error("⚠️ Pasien BERISIKO TINGGI terkena Diabetes Type 2")

        # Fungsi untuk menampilkan persentase hanya jika > 0
        def autopct_format(pct):
            return ("%1.1f%%" % pct) if pct > 0 else ""

        # Pie Chart
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            probability, labels=labels, colors=colors,
            autopct=autopct_format, startangle=60,
            counterclock=False, radius=0.6,
            labeldistance=1.1, pctdistance=0.5,
            textprops={'fontsize': 8}
        )
        ax.axis("equal")
        plt.tight_layout()
        st.pyplot(fig)


# Bar Chart Page
elif st.session_state.page == "bar_chart":
    st.title("Hasil Prediksi & Probabilitas")
    if st.session_state.input_data is None or 'data' not in st.session_state:
        st.warning("⚠️ Silakan isi data pasien terlebih dahulu di menu Input Data Pasien.")
    else:
        data_input = st.session_state.data_input

        # Menampilkan data nama, tahun, BMI
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Jenis Kelamin", data_input["Gender"][0])
        with col2:
            st.metric("Umur", f"{int(data_input['Age'][0])} tahun")
        with col3:
            st.metric("BMI", f"{data_input['BMI'][0]:.1f} kg/m²")

        # Bar Chart Terkelompok
        # 1. "Profil Fungsi Ginjal & Risiko Diabetes"
        fitur_ginjal = ["Urea", "Cr", "HbA1c"]
        nilai_ginjal = [data_input[feat][0] for feat in fitur_ginjal]
        df_ginjal = pd.DataFrame({"Fitur": fitur_ginjal, "Nilai": nilai_ginjal})
        st.subheader("**⚖️ Profil Fungsi Ginjal & Risiko Diabetes**")
        st.bar_chart(df_ginjal.set_index("Fitur"), y="Nilai", height=250)

        # 2. Profil Lipid
        fitur_lipid = ["Chol", "TG", "HDL", "LDL", "VLDL"]
        nilai_lipid = [data_input[feat][0] for feat in fitur_lipid]
        df_lipid = pd.DataFrame({"Fitur": fitur_lipid, "Nilai": nilai_lipid})
        st.subheader("**🧪 Profil Lipid**")
        st.bar_chart(df_lipid.set_index("Fitur"), y="Nilai", height=250)

        # 3. Indikator Fungsi Ginjal & Risiko Diabetes
        fitur_glikemik = ["Urea/Cr", "BMI x HbA1c"]
        nilai_glikemik = [data_input[feat][0] for feat in fitur_glikemik]
        df_glikemik = pd.DataFrame({"Fitur": fitur_glikemik, "Nilai": nilai_glikemik})
        st.subheader("**🧬 Indikator Fungsi Ginjal & Risiko Diabetes**")
        st.bar_chart(df_glikemik.set_index("Fitur"), y="Nilai", height=250)

       # Prediksi
        prediction, prob = predict_diabetes(model, scaler, st.session_state.input_data)

        if prediction[0] == 0:
            st.success("✅ Pasien TIDAK BERISIKO terkena Diabetes")
        elif prediction[0] == 1:
            st.warning("⚠️ Pasien BERISIKO TINGGI terkena Diabetes Type 1")
        else:
            st.error("⚠️ Pasien BERISIKO TINGGI terkena Diabetes Type 2")
