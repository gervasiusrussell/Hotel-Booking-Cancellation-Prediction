import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
@st.cache_resource
def load_model():
    with open("xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

model, label_encoders = load_model()

st.title("Hotel Booking Cancellation Prediction")
st.markdown("Masukkan detail booking untuk memprediksi kemungkinan pembatalan.")

# UI untuk input user
no_of_adults = st.number_input("Jumlah Dewasa", min_value=0, value=2)
no_of_children = st.number_input("Jumlah Anak", min_value=0, value=0)
no_of_weekend_nights = st.number_input("Jumlah Malam Akhir Pekan", min_value=0, value=1)
no_of_week_nights = st.number_input("Jumlah Malam Hari Kerja", min_value=0, value=2)
type_of_meal_plan = st.selectbox("Meal Plan", options=label_encoders['type_of_meal_plan'].classes_)
required_car_parking_space = st.selectbox("Butuh Parkir?", options=[0, 1])
room_type_reserved = st.selectbox("Tipe Kamar", options=label_encoders['room_type_reserved'].classes_)
lead_time = st.number_input("Lead Time (hari sebelum booking)", min_value=0, value=10)
arrival_year = st.selectbox("Tahun Kedatangan", options=[2022, 2023, 2024])
arrival_month = st.slider("Bulan Kedatangan", min_value=1, max_value=12, value=7)
arrival_date = st.slider("Tanggal Kedatangan", min_value=1, max_value=31, value=15)
market_segment_type = st.selectbox("Segment Pasar", options=label_encoders['market_segment_type'].classes_)
repeated_guest = st.selectbox("Tamu Berulang?", options=[0, 1])
no_of_previous_cancellations = st.number_input("Pembatalan Sebelumnya", min_value=0, value=0)
no_of_previous_bookings_not_canceled = st.number_input("Booking Sebelumnya yang Tidak Dibatalkan", min_value=0, value=0)
avg_price_per_room = st.number_input("Rata-rata Harga Kamar", min_value=0.0, value=100.0)
no_of_special_requests = st.number_input("Jumlah Permintaan Khusus", min_value=0, value=0)

# Transform categorical input with encoders
input_data = {
    "no_of_adults": no_of_adults,
    "no_of_children": no_of_children,
    "no_of_weekend_nights": no_of_weekend_nights,
    "no_of_week_nights": no_of_week_nights,
    "type_of_meal_plan": label_encoders['type_of_meal_plan'].transform([type_of_meal_plan])[0],
    "required_car_parking_space": required_car_parking_space,
    "room_type_reserved": label_encoders['room_type_reserved'].transform([room_type_reserved])[0],
    "lead_time": lead_time,
    "arrival_year": arrival_year,
    "arrival_month": arrival_month,
    "arrival_date": arrival_date,
    "market_segment_type": label_encoders['market_segment_type'].transform([market_segment_type])[0],
    "repeated_guest": repeated_guest,
    "no_of_previous_cancellations": no_of_previous_cancellations,
    "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
    "avg_price_per_room": avg_price_per_room,
    "no_of_special_requests": no_of_special_requests
}

input_df = pd.DataFrame([input_data])

# Prediksi
if st.button("Prediksi Pembatalan"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"Booking kemungkinan DIBATALKAN (Probabilitas: {probability:.2f})")
    else:
        st.success(f"Booking kemungkinan TIDAK DIBATALKAN (Probabilitas: {1 - probability:.2f})")

    st.subheader("Detail Input")
    st.dataframe(input_df)
else:
    st.info("Masukkan data dan klik 'Prediksi Pembatalan' untuk hasil.")
