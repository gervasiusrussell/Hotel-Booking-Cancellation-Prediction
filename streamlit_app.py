import streamlit as st
import pandas as pd
import numpy as np
import pickle
pip install xgboost
from xgboost import XGBClassifier

# Set page config
st.set_page_config(page_title="Hotel Booking Prediction", layout="wide")

# Load model and label encoders
@st.cache_resource
def load_artifacts():
    with open('xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    return model, encoders

model, encoders = load_artifacts()

# Load sample data
@st.cache_data
def load_data():
    return pd.read_csv('Dataset_B_hotel.csv')

data = load_data()

# Streamlit UI
st.title('🏨 Hotel Booking Cancellation Prediction')
st.markdown("""
Predict whether a hotel booking will be canceled based on booking characteristics.
""")

# Show raw data
with st.expander("📊 View Dataset"):
    st.dataframe(data, use_container_width=True)

# User input section
st.header("🔍 Make a Prediction")

col1, col2 = st.columns(2)

with col1:
    lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=500, value=30)
    arrival_date_year = st.selectbox("Arrival Year", [2015, 2016, 2017, 2018])
    arrival_date_month = st.selectbox("Arrival Month", [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    arrival_date_week_number = st.number_input("Arrival Week Number", min_value=1, max_value=53, value=25)
    stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=10, value=1)
    stays_in_week_nights = st.number_input("Week Nights", min_value=0, max_value=20, value=2)

with col2:
    adults = st.number_input("Adults", min_value=0, max_value=10, value=2)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    babies = st.number_input("Babies", min_value=0, max_value=5, value=0)
    meal = st.selectbox("Meal Type", ['BB', 'FB', 'HB', 'SC', 'Undefined'])
    country = st.selectbox("Country", ['PRT', 'GBR', 'USA', 'ESP', 'FRA', 'DEU', 'ITA'])
    market_segment = st.selectbox("Market Segment", [
        'Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 
        'Complementary', 'Groups', 'Undefined'
    ])

# Preprocessing function
def preprocess_input(input_df):
    # Encode categorical features
    cat_cols = ['arrival_date_month', 'meal', 'country', 'market_segment']
    for col in cat_cols:
        if col in encoders:
            input_df[col] = encoders[col].transform(input_df[col])
    
    return input_df

# Prediction function
def make_prediction(input_data):
    try:
        # Prepare input
        input_df = pd.DataFrame([input_data])
        processed_input = preprocess_input(input_df)
        
        # Make prediction
        prediction = model.predict(processed_input)
        proba = model.predict_proba(processed_input)[0]
        
        return prediction[0], proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

# Test cases
test_case_1 = {
    'lead_time': 120,
    'arrival_date_year': 2017,
    'arrival_date_month': 'July',
    'arrival_date_week_number': 28,
    'stays_in_weekend_nights': 2,
    'stays_in_week_nights': 5,
    'adults': 2,
    'children': 0,
    'babies': 0,
    'meal': 'BB',
    'country': 'PRT',
    'market_segment': 'Online TA'
}

test_case_2 = {
    'lead_time': 7,
    'arrival_date_year': 2018,
    'arrival_date_month': 'December',
    'arrival_date_week_number': 50,
    'stays_in_weekend_nights': 1,
    'stays_in_week_nights': 3,
    'adults': 1,
    'children': 1,
    'babies': 0,
    'meal': 'HB',
    'country': 'USA',
    'market_segment': 'Direct'
}

# User input dictionary
user_input = {
    'lead_time': lead_time,
    'arrival_date_year': arrival_date_year,
    'arrival_date_month': arrival_date_month,
    'arrival_date_week_number': arrival_date_week_number,
    'stays_in_weekend_nights': stays_in_weekend_nights,
    'stays_in_week_nights': stays_in_week_nights,
    'adults': adults,
    'children': children,
    'babies': babies,
    'meal': meal,
    'country': country,
    'market_segment': market_segment
}

# Prediction button
if st.button("🚀 Predict Cancellation"):
    prediction, proba = make_prediction(user_input)
    
    if prediction is not None:
        st.subheader("📝 Prediction Result")
        result = "❌ Canceled" if prediction == 1 else "✅ Not Canceled"
        st.success(f"Prediction: {result}")
        
        st.metric("Cancellation Probability", f"{proba[1]*100:.2f}%")
        
        # Show probabilities
        proba_df = pd.DataFrame({
            'Status': ['Not Canceled', 'Canceled'],
            'Probability': [proba[0], proba[1]]
        })
        st.bar_chart(proba_df.set_index('Status'))

# Test cases section
st.header("🧪 Test Cases")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Test Case 1 (High Cancellation Risk)")
    st.json(test_case_1)
    if st.button("Run Test Case 1"):
        prediction, proba = make_prediction(test_case_1)
        if prediction == 1:
            st.error(f"✅ Correct! Prediction: Canceled ({proba[1]*100:.2f}% probability)")
        else:
            st.warning(f"❌ Incorrect! Prediction: Not Canceled")

with col2:
    st.subheader("Test Case 2 (Low Cancellation Risk)")
    st.json(test_case_2)
    if st.button("Run Test Case 2"):
        prediction, proba = make_prediction(test_case_2)
        if prediction == 0:
            st.success(f"✅ Correct! Prediction: Not Canceled ({proba[0]*100:.2f}% probability)")
        else:
            st.warning(f"❌ Incorrect! Prediction: Canceled")

# Add some styling
st.markdown("""
<style>
    .st-bq {
        border-left: 5px solid #4e79a7;
        padding: 0.5em;
    }
    .st-c0 {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)
