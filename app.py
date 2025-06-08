import streamlit as st
import numpy as np
import pandas as pd
import joblib

# === Load Model dan Scaler ===
model = joblib.load("model_rf.pkl")
scaler_gpa = joblib.load("scaler_gpa.pkl")
pca = joblib.load("pca.pkl")
scaler_features = joblib.load("scaler_features.pkl")

# === Halaman Streamlit ===
st.set_page_config(page_title="Prediksi Mahasiswa Dropout", layout="centered")
st.title("🎓 Prediksi Mahasiswa Dropout")
st.write("Masukkan data mahasiswa berikut:")

# === Form Input ===
with st.form("prediction_form"):
    gpa_inputs = [st.number_input(f"GPA Semester {i+1}", min_value=0.0, max_value=4.0, step=0.01, key=f"gpa{i}") for i in range(8)]
    
    attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
    retaken = st.number_input("Jumlah Mata Kuliah yang Diulang", min_value=0, max_value=20, value=1)
    lms_score = st.slider("Skor Aktivitas LMS", 0, 100, 75)
    work_hours = st.slider("Jam Kerja per Minggu", 0, 40, 0)
    
    employment_status = st.selectbox("Status Pekerjaan", ["Unemployed", "Employed"])
    socioeconomic = st.selectbox("Status Sosial Ekonomi", ["Low", "Middle", "High"])

    submitted = st.form_submit_button("🔍 Prediksi Dropout")

# === Mapping dan Prediksi ===
if submitted:
    # GPA PCA transform
    gpa_array = np.array(gpa_inputs).reshape(1, -1)
    gpa_scaled = scaler_gpa.transform(gpa_array)
    gpa_pca = pca.transform(gpa_scaled)  # menghasilkan PC1 dan PC2

    # One-hot encoding manual untuk 2 kolom kategori
    emp_unemployed = 1 if employment_status == "Unemployed" else 0
    socio_low = 1 if socioeconomic == "Low" else 0
    socio_middle = 1 if socioeconomic == "Middle" else 0
    # socio_high tidak dimasukkan karena drop_first=True saat one-hot

    # Gabungkan semua fitur
    features = np.array([attendance, retaken, lms_score, work_hours,
                         gpa_pca[0][0], gpa_pca[0][1],
                         emp_unemployed, socio_low, socio_middle]).reshape(1, -1)

    # Scaling fitur numerik (tanpa kolom one-hot)
    features_scaled = scaler_features.transform(features)

    # Prediksi
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    # === Output Hasil ===
    st.markdown("---")
    st.subheader("Hasil Prediksi")

    if prediction == 1:
        st.error(f"Mahasiswa diprediksi berisiko **Dropout** dengan probabilitas {probability:.2%}")
    else:
        st.success(f"✅ Mahasiswa diprediksi **tidak Dropout** dengan probabilitas {probability:.2%}")

    st.caption("Model menggunakan Random Forest Classifier berdasarkan data akademik, aktivitas LMS, dan kondisi sosial ekonomi.")
