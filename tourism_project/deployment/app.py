import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="SuriyaSR/TourismPackagePurchases", filename="tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Purchases Prediction App")
st.write("""
This application predicts the likelihood of a customer purchases based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Free Lancer', 'Large Business'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    income = st.number_input("Monthly Income", min_value=0, value=25000)

with col2:
    contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
    product = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
    marital = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])
    designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
    passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Additional fields used in the dataset
duration = st.slider("Duration of Pitch", 5, 40, 15)
followups = st.slider("Number of Follow-ups", 1, 6, 3)
trips = st.number_input("Number of Trips", min_value=1, max_value=20, value=3)

# 4. Assemble input into DataFrame 
# IMPORTANT: The column names here MUST match the X_train columns used during training
input_dict = {
    'Age': age,
    'TypeofContact': contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfFollowups': followups,
    'ProductPitched': product,
    'MaritalStatus': marital,
    'NumberOfTrips': trips,
    'Passport': passport,
    'MonthlyIncome': income,
    'Designation': designation
}

# Note: If your model was trained on more columns (like PitchSatisfactionScore, OwnCar, etc.), 
# you must include them here with default values or UI inputs.
input_df = pd.DataFrame([input_dict])

# 5. Prediction Logic
if st.button("Predict Purchase Likelihood"):
    prediction = model.predict(input_df)[0]
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success("🎯 **The customer is likely to purchase the package!**")
    else:
        st.warning("📉 **The customer is unlikely to purchase the package.**")

# 6. Optional: Show input data for debugging
if st.checkbox("Show Raw Input Data"):
    st.write(input_df)
