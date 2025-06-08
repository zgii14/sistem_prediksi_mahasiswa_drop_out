import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Sistem Prediksi Mahasiswa Drop Out")

# Input user
st.header("Masukkan Data Mahasiswa:")
attendance_rate = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0)
retaken_courses = st.number_input("Jumlah Mata Kuliah Diulang", min_value=0)
lms_score = st.number_input("LMS Activity Score", min_value=0.0)
work_hours = st.number_input("Jam Kerja Mingguan", min_value=0)

# Komponen PCA
pc1 = st.number_input("Principal Component 1")
pc2 = st.number_input("Principal Component 2")

# Encoding one-hot
employment_status = st.selectbox("Status Pekerjaan", ["Tidak Bekerja", "Bekerja"])
socioeconomic_status = st.selectbox("Status Sosial Ekonomi", ["Rendah", "Menengah", "Tinggi"])

# Encoding
employment_encoded = 1 if employment_status == "Bekerja" else 0
socio_encoded = [0, 0]
if socioeconomic_status == "Menengah":
    socio_encoded = [1, 0]
elif socioeconomic_status == "Tinggi":
    socio_encoded = [0, 1]

# Gabung semua fitur jadi satu DataFrame
input_data = pd.DataFrame([[
    attendance_rate, retaken_courses, lms_score, work_hours, pc1, pc2,
    employment_encoded, *socio_encoded
]], columns=[
    "Attendance_Rate", "Retaken_Courses", "LMS_Activity_Score", "Work_Hours",
    "principal component 1", "principal component 2",
    "Employment_Status_Bekerja", "Socioeconomic_Status_Menengah", "Socioeconomic_Status_Tinggi"
])

# Standarisasi
scaled_data = scaler.transform(input_data)

# Prediksi
if st.button("Prediksi Drop Out"):
    prediction = model.predict(scaled_data)[0]
    proba = model.predict_proba(scaled_data)[0][1]
    st.write("**Hasil Prediksi:**", "Drop Out" if prediction == 1 else "Tidak Drop Out")
    st.write(f"**Probabilitas Drop Out:** {proba:.2f}")
