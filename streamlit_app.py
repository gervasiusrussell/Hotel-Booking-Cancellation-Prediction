import streamlit as st
import pickle
import numpy as np
import pandas as pd

class HotelBookingApp:
    def __init__(self):
        self.model = self.load_model('xgboost_model.pkl')
        self.encoders = self.load_model('label_encoders.pkl')

    def load_model(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def encode_input(self, input_df):
        encoded_df = input_df.copy()
        for col in encoded_df.select_dtypes(include='object').columns:
            if col in self.encoders:
                le = self.encoders[col]
                encoded_df[col] = le.transform(encoded_df[col])
            else:
                encoded_df[col] = 0  # Fallback if encoder not found
        return encoded_df

    def predict(self, input_df):
        encoded_df = self.encode_input(input_df)
        prediction = self.model.predict(encoded_df)[0]
        probability = self.model.predict_proba(encoded_df)[0][1]
        return prediction, probability

    def run(self):
        st.title("Hotel Booking Cancellation Prediction")
        st.write("Masukkan data berikut untuk memprediksi status booking")

        test_cases = {
            "Test Case 1": {
                'no_of_adults': 2,
                'no_of_children': 0,
                'no_of_weekend_nights': 1,
                'no_of_week_nights': 2,
                'type_of_meal_plan': 'Meal Plan 1',
                'required_car_parking_space': 0.0,
                'room_type_reserved': 'Room_Type 1',
                'lead_time': 45,
                'arrival_year': 2017,
                'arrival_month': 7,
                'arrival_date': 15,
                'market_segment_type': 'Online',
                'repeated_guest': 0,
                'no_of_previous_cancellations': 0,
                'no_of_previous_bookings_not_canceled': 0,
                'avg_price_per_room': 100.0,
                'no_of_special_requests': 1
            },
            "Test Case 2": {
                'no_of_adults': 1,
                'no_of_children': 2,
                'no_of_weekend_nights': 2,
                'no_of_week_nights': 5,
                'type_of_meal_plan': 'Meal Plan 2',
                'required_car_parking_space': 1.0,
                'room_type_reserved': 'Room_Type 3',
                'lead_time': 100,
                'arrival_year': 2017,
                'arrival_month': 12,
                'arrival_date': 25,
                'market_segment_type': 'Offline',
                'repeated_guest': 1,
                'no_of_previous_cancellations': 1,
                'no_of_previous_bookings_not_canceled': 3,
                'avg_price_per_room': 150.0,
                'no_of_special_requests': 2
            }
        }

        selected_case = st.selectbox("Pilih Test Case", ["Manual Input"] + list(test_cases.keys()))

        if selected_case != "Manual Input":
            user_input = pd.DataFrame([test_cases[selected_case]])
        else:
            user_input = pd.DataFrame([{ 
                'no_of_adults': st.number_input('Jumlah Dewasa', min_value=1, max_value=10, value=2),
                'no_of_children': st.number_input('Jumlah Anak', min_value=0, max_value=10, value=0),
                'no_of_weekend_nights': st.number_input('Malam Akhir Pekan', min_value=0, max_value=10, value=1),
                'no_of_week_nights': st.number_input('Malam Hari Kerja', min_value=0, max_value=10, value=2),
                'type_of_meal_plan': st.selectbox('Paket Makan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']),
                'required_car_parking_space': float(st.selectbox('Butuh Parkir?', [0, 1])),
                'room_type_reserved': st.selectbox('Tipe Kamar', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7']),
                'lead_time': st.slider('Lead Time', 0, 500, 45),
                'arrival_year': st.selectbox('Tahun Kedatangan', [2017,2018]),
                'arrival_month': st.slider('Bulan Kedatangan', 1, 12, 7),
                'arrival_date': st.slider('Tanggal Kedatangan', 1, 31, 15),
                'market_segment_type': st.selectbox('Segment Pasar', ['Online', 'Offline', 'Corporate', 'Aviation', 'Complementary']),
                'repeated_guest': st.selectbox('Tamu Berulang?', [0, 1]),
                'no_of_previous_cancellations': st.slider('Jumlah Pembatalan Sebelumnya', 0, 10, 0),
                'no_of_previous_bookings_not_canceled': st.slider('Jumlah Booking Tidak Dibatalkan', 0, 10, 0),
                'avg_price_per_room': st.number_input('Harga Rata-rata per Kamar', min_value=0.0, max_value=1000.0, value=100.0),
                'no_of_special_requests': st.slider('Permintaan Khusus', 0, 5, 1)
            }])

        if st.button("Prediksi Booking Status"):
            pred, prob = self.predict(user_input)
            status = "Tidak Dibatalkan" if pred == 0 else "Dibatalkan"
            st.write(f"### Prediksi: {status}")
            st.write(f"### Probabilitas Dibatalkan: {prob:.2%}")
            st.write("\nData yang digunakan:")
            st.dataframe(user_input)

if __name__ == "__main__":
    app = HotelBookingApp()
    app.run()
