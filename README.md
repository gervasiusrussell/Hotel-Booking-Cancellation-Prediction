# 🏨 Hotel Booking Cancellation Prediction

This project is a **Streamlit web application** that predicts whether a hotel booking will be **cancelled** or **not cancelled**.
The model is trained using **XGBoost** and categorical features are encoded with **LabelEncoders**.

---

## 👤 Author

* **Name**: Gervasius Russell
* **University**: BINUS University
* **Major**: Data Science

---

## 📂 Project Structure

```
hotel-booking-app/
│── app.py                     # Main Streamlit app (OOP implementation)
│── xgboost_model.pkl          # Trained XGBoost model
│── label_encoders.pkl         # LabelEncoders for categorical features
│── Dataset_B_hotel.csv        # Dataset used (sample previewed in app)
│── requirements.txt           # Dependencies
│── README.md                  # Project documentation
```

---

## 🚀 Features

* 🖥️ User-friendly **Streamlit interface**.
* 📊 **Dataset preview** (first 50 rows).
* ✏️ Dynamic **user input form** for booking details:

  * Adults, children, nights, lead time, price, etc.
  * Meal plan, room type, market segment, etc.
* 🔮 **Prediction output**:

  * Booking status: ✅ Not Cancelled / ❌ Cancelled
  * Probability of cancellation.
* 📑 Display of the **exact input data used** for prediction.

---

## 📦 Installation & Setup

1. **Clone this repository**

```bash
git clone https://github.com/<your-username>/hotel-booking-app.git
cd hotel-booking-app
```

2. **Create and activate a virtual environment** (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate    # On Windows
source venv/bin/activate # On Mac/Linux
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run app.py
```

---

## 📊 Dataset

* **File**: `Dataset_B_hotel.csv`
* **Preview in App**: First 50 rows are displayed automatically.
* **Features** include:

  * `no_of_adults`, `no_of_children`, `no_of_weekend_nights`, `type_of_meal_plan`,
  * `room_type_reserved`, `lead_time`, `arrival_date`, `market_segment_type`, etc.

---

## 🧠 Model

* **Algorithm**: XGBoost (classification).
* **Encoders**: LabelEncoders for categorical features.
* **Saved as**:

  * `xgboost_model.pkl`
  * `label_encoders.pkl`

---

## 📸 App Preview

👉 Example prediction flow in the app:

1. Enter booking details in the form.
2. Click **Predict Booking Status**.
3. See output:

   * **Prediction**: ✅ Not Cancelled / ❌ Cancelled
   * **Probability**: e.g. 72% chance of cancellation.
   * **Data used for prediction** is displayed.

---

## ⚡ Future Improvements

* Add **data visualization dashboards** (e.g., cancellation trends).
* Support for **multiple models** (Random Forest, Neural Nets).
* Deploy to **Streamlit Cloud** or **Heroku**.

---

## 📌 Citation

If you use this project for research or coursework, please cite:
**Gervasius Russell, BINUS University – Hotel Booking Cancellation Prediction (2025).**

---
