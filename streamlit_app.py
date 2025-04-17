import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import sys
import subprocess

# Check and install xgboost if not available
try:
    from xgboost import XGBClassifier
except ImportError:
    st.warning("XGBoost not found! Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"])
    from xgboost import XGBClassifier

class HotelBookingPredictor:
    def __init__(self, model_path, encoder_path):
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.model = None
        self.label_encoders = None

    def load_model_and_encoders(self):
        """Load the trained model and label encoders"""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.encoder_path, 'rb') as f:
            self.label_encoders = pickle.load(f)
        return self

    def preprocess_input(self, input_df):
        """Preprocess user input for prediction"""
        # Copy to avoid modifying original
        processed_df = input_df.copy()
        
        # Encode categorical features
        cat_cols = processed_df.select_dtypes(include='object').columns.tolist()
        
        for col in cat_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                processed_df[col] = le.transform(processed_df[col])
        
        return processed_df

    def predict(self, input_data):
        """Make prediction on processed input"""
        processed_input = self.preprocess_input(input_data)
        prediction = self.model.predict(processed_input)
        proba = self.model.predict_proba(processed_input)
        return prediction[0], proba[0]

# Initialize app
st.set_page_config(page_title="🏨 Hotel Booking Cancellation Predictor", layout="wide")

# Title and description
st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("""
Predict whether a hotel booking will be canceled based on booking characteristics.
""")

# Sidebar for file upload
with st.sidebar:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {len(batch_df)} records")

# Load model (cached)
@st.cache_resource
def load_predictor():
    try:
        return HotelBookingPredictor(
            model_path='xgboost_model.pkl',
            encoder_path='label_encoders.pkl'
        ).load_model_and_encoders()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

predictor = load_predictor()

# Single prediction form
st.header("🔍 Single Booking Prediction")
col1, col2 = st.columns(2)

with col1:
    lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=500, value=30)
    arrival_year = st.selectbox("Arrival Year", [2017, 2018, 2019])
    arrival_month = st.selectbox("Arrival Month", [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=10, value=1)
    week_nights = st.number_input("Week Nights", min_value=0, max_value=20, value=2)

with col2:
    adults = st.number_input("Adults", min_value=0, max_value=10, value=2)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    meal = st.selectbox("Meal Type", ['BB', 'FB', 'HB', 'SC', 'Undefined'])
    country = st.selectbox("Country", ['PRT', 'GBR', 'USA', 'ESP', 'FRA'])
    market_segment = st.selectbox("Market Segment", [
        'Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary'
    ])

# Prepare input data
input_data = pd.DataFrame([{
    'lead_time': lead_time,
    'arrival_year': arrival_year,
    'arrival_month': arrival_month,
    'stays_in_weekend_nights': weekend_nights,
    'stays_in_week_nights': week_nights,
    'adults': adults,
    'children': children,
    'meal': meal,
    'country': country,
    'market_segment': market_segment
}])

# Test cases
test_case_1 = {
    'lead_time': 210,
    'arrival_year': 2018,
    'arrival_month': 'July',
    'stays_in_weekend_nights': 2,
    'stays_in_week_nights': 5,
    'adults': 2,
    'children': 0,
    'meal': 'BB',
    'country': 'PRT',
    'market_segment': 'Online TA'
}

test_case_2 = {
    'lead_time': 14,
    'arrival_year': 2019,
    'arrival_month': 'December',
    'stays_in_weekend_nights': 1,
    'stays_in_week_nights': 3,
    'adults': 1,
    'children': 1,
    'meal': 'HB',
    'country': 'USA',
    'market_segment': 'Direct'
}

# Prediction button
if st.button("🚀 Predict Cancellation"):
    with st.spinner("Making prediction..."):
        try:
            prediction, proba = predictor.predict(input_data)
            result = "❌ Canceled" if prediction == 1 else "✅ Not Canceled"
            
            st.subheader("📝 Prediction Result")
            st.success(f"Prediction: {result}")
            
            # Display probabilities
            proba_df = pd.DataFrame({
                'Status': ['Not Canceled', 'Canceled'],
                'Probability': [proba[0], proba[1]]
            })
            
            st.metric("Cancellation Probability", f"{proba[1]*100:.2f}%")
            st.bar_chart(proba_df.set_index('Status'))
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# Test cases section
st.header("🧪 Test Cases")

col1, col2 = st.columns(2)

with col1:
    st.subheader("High Cancellation Risk")
    st.json(test_case_1)
    if st.button("Run Test Case 1"):
        with st.spinner("Testing..."):
            test_df = pd.DataFrame([test_case_1])
            prediction, proba = predictor.predict(test_df)
            if prediction == 1:
                st.success(f"✅ Correct! Prediction: Canceled ({proba[1]*100:.2f}% probability)")
            else:
                st.error(f"❌ Incorrect! Should be Canceled")

with col2:
    st.subheader("Low Cancellation Risk")
    st.json(test_case_2)
    if st.button("Run Test Case 2"):
        with st.spinner("Testing..."):
            test_df = pd.DataFrame([test_case_2])
            prediction, proba = predictor.predict(test_df)
            if prediction == 0:
                st.success(f"✅ Correct! Prediction: Not Canceled ({proba[0]*100:.2f}% probability)")
            else:
                st.error(f"❌ Incorrect! Should be Not Canceled")

# Batch prediction section
if uploaded_file:
    st.header("📊 Batch Prediction Results")
    with st.spinner("Processing batch predictions..."):
        try:
            # Process batch file
            batch_processed = predictor.preprocess_input(batch_df)
            batch_predictions = predictor.model.predict(batch_processed)
            batch_proba = predictor.model.predict_proba(batch_processed)
            
            # Add to dataframe
            results_df = batch_df.copy()
            results_df['Prediction'] = ['Canceled' if x == 1 else 'Not Canceled' for x in batch_predictions]
            results_df['Cancel_Probability'] = batch_proba[:,1]
            
            # Show results
            st.dataframe(results_df)
            
            # Download button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name='batch_predictions.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Batch processing failed: {str(e)}")

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
    .stAlert {
        padding: 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)
