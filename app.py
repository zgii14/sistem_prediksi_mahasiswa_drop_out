import streamlit as st
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# ===============================
# Load Model dan Preprocessing
# ===============================
model = joblib.load('random_forest_model.pkl')
scaler_gpa = joblib.load('scaler_gpa.pkl')
pca = joblib.load('pca.pkl')
scaler_features = joblib.load('scaler_features.pkl')

# ===============================
# Judul Aplikasi
# ===============================
st.title("Prediksi Mahasiswa Drop Out")

# ===============================
# Input Data Pengguna
# ===============================
st.header("Masukkan Data Mahasiswa")

# Input Numerik
attendance = st.slider("Tingkat Kehadiran (%)", 0, 100, 75)
retaken = st.number_input("Jumlah Mata Kuliah yang Diulang", min_value=0, max_value=20, value=0)
lms_score = st.slider("Skor Aktivitas LMS", 0, 100, 50)
work_hours = st.number_input("Jam Kerja per Minggu", min_value=0, max_value=60, value=0)

# Input GPA Semester 1 - 8
st.subheader("Nilai GPA per Semester")
gpa_values = []
for i in range(1, 9):
    gpa = st.number_input(f"GPA Semester {i}", min_value=0.0, max_value=4.0, value=3.0, step=0.01)
    gpa_values.append(gpa)

# Input Kategorikal
employment_status = st.selectbox("Status Pekerjaan", ['Employed', 'Unemployed'])
socioeconomic_status = st.selectbox("Status Sosial Ekonomi", ['High', 'Middle', 'Low'])

# ===============================
# Proses Prediksi
# ===============================
if st.button("Prediksi"):
    # --- PCA pada GPA ---
    gpa_array = np.array(gpa_values).reshape(1, -1)
    gpa_scaled = scaler_gpa.transform(gpa_array)
    gpa_pca = pca.transform(gpa_scaled)

    # --- One-hot encoding manual ---
    emp_unemployed = 1 if employment_status == 'Unemployed' else 0
    socio_low = 1 if socioeconomic_status == 'Low' else 0
    socio_middle = 1 if socioeconomic_status == 'Middle' else 0

    # --- Gabungkan fitur numerik ---
    numerical_features = np.array([
        attendance, retaken, lms_score, work_hours,
        gpa_pca[0][0], gpa_pca[0][1]
    ]).reshape(1, -1)

    scaled_numerical = scaler_features.transform(numerical_features)

    # --- Gabungkan numerik + one-hot ---
    features_final = np.hstack([
        scaled_numerical,
        np.array([[emp_unemployed, socio_low, socio_middle]])
    ])

    # --- Prediksi ---
    prediction = model.predict(features_final)[0]
    probability = model.predict_proba(features_final)[0][1]

    # --- Tampilkan Hasil ---
    st.subheader("Hasil Prediksi")
    if prediction == 1:
        st.error(f"Mahasiswa diprediksi akan Drop Out dengan probabilitas {probability:.2f}")
    else:
        st.success(f"Mahasiswa diprediksi akan Bertahan dengan probabilitas {1 - probability:.2f}")
