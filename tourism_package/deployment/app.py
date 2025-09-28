
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
import os

# The Dockerfile handles HF_HOME, but this ensures Streamlit cache is also safe.
if 'HF_HOME' in os.environ:
    os.environ['STREAMLIT_SERVER_DIR'] = os.environ['HF_HOME']
    os.environ['STREAMLIT_SERVER_DATA_FRAME_CACHE_DIR'] = os.path.join(os.environ['HF_HOME'], 'st_cache/data_frame')
    os.environ['STREAMLIT_SERVER_FILE_CACHE_PATH'] = os.path.join(os.environ['HF_HOME'], 'st_cache/file_cache')
# -------------------------------------------------------------

# --- Configuration ---
# Replace with your actual Hugging Face username

HF_USERNAME = "RajendrakumarPachaiappan"
HF_REPO_MODEL = f"{HF_USERNAME}/tourism-package-prediction-model"

# --- Model Loading ---
@st.cache_resource
def load_model_and_preprocessor():
    """Downloads and loads the model and preprocessor from Hugging Face Hub."""
    
    # 1. Download Preprocessor
    preprocessor_path = hf_hub_download(
        repo_id=HF_REPO_MODEL,
        filename="preprocessor.joblib",
        repo_type="model"
    )
    preprocessor = joblib.load(preprocessor_path)

    # 2. Download Model
    model_path = hf_hub_download(
        repo_id=HF_REPO_MODEL,
        filename="model.joblib",
        repo_type="model"
    )
    model = joblib.load(model_path)
    
    return model, preprocessor

# --- Inference Function ---
def make_prediction(model, preprocessor, input_df):
    """Preprocesses input data and makes a prediction."""
    try:
        # Preprocess the input DataFrame
        processed_data = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(processed_data)
        prediction_proba = model.predict_proba(processed_data)
        
        return prediction[0], prediction_proba[0]
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# --- Streamlit UI ---
st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.stButton>button {
    font-size: 20px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.title("✈️ Visit with Us: Wellness Tourism Package Predictor")
st.markdown("---")

# Load artifacts
model, preprocessor = load_model_and_preprocessor()

if model and preprocessor:
    st.sidebar.header("Customer Profile Input")

    # --- Input Fields ---
    
    # Column 1
    with st.sidebar:
        age = st.slider("Age", 18, 70, 35)
        monthly_income = st.number_input("Monthly Income (INR)", 10000.0, 50000.0, 25000.0, step=100.0)
        occupation = st.selectbox("Occupation", ('Salaried', 'Small Business', 'Large Business', 'Free Lancer'))
        marital_status = st.selectbox("Marital Status", ('Single', 'Married', 'Divorced', 'Unmarried'))
        city_tier = st.selectbox("City Tier (1=Highest)", (1, 2, 3))
        
    # Column 2
    col1, col2 = st.columns(2)
    with col1:
        type_of_contact = st.selectbox("Type of Contact", ('Self Enquiry', 'Company Invited'))
        duration_of_pitch = st.number_input("Pitch Duration (min)", 1.0, 60.0, 10.0, step=1.0)
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 3)
        product_pitched = st.selectbox("Product Pitched", ('Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'))
        
    with col2:
        gender = st.selectbox("Gender", ('Male', 'Female'))
        designation = st.selectbox("Designation", ('Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'))
        num_persons = st.number_input("Number of Persons Visiting", 1, 6, 3)
        num_children = st.number_input("Number of Children Visiting", 0, 4, 0)
        num_trips = st.number_input("Average Number of Trips Annually", 0, 20, 3)
        
    # Binary/Ordinal Features
    preferred_star = st.sidebar.selectbox("Preferred Hotel Star", (3.0, 4.0, 5.0, np.nan))
    has_passport = st.sidebar.checkbox("Passport Holder", True)
    owns_car = st.sidebar.checkbox("Owns Car", True)
    num_followups = st.sidebar.number_input("Number of Follow-ups", 0, 10, 3)
    
    # --- Create Input DataFrame ---
    data = {
        'Age': age,
        'MonthlyIncome': monthly_income,
        'Occupation': occupation,
        'MaritalStatus': marital_status,
        'CityTier': city_tier,
        'TypeofContact': type_of_contact,
        'DurationOfPitch': duration_of_pitch,
        'PitchSatisfactionScore': pitch_satisfaction_score,
        'ProductPitched': product_pitched,
        'Gender': gender,
        'Designation': designation,
        'NumberOfPersonVisiting': num_persons,
        'NumberOfChildrenVisiting': num_children,
        'NumberOfTrips': num_trips,
        'PreferredPropertyStar': preferred_star,
        'Passport': int(has_passport),
        'OwnCar': int(owns_car),
        'NumberOfFollowups': num_followups
    }
    
    input_df = pd.DataFrame([data])
    
    st.markdown("### 📊 Prediction Result")
    
    if st.button("Predict Package Purchase"):
        
        prediction, prediction_proba = make_prediction(model, preprocessor, input_df)
        
        if prediction is not None:
            if prediction == 1:
                # FIX: Use st.markdown() with custom styling for success look and feel
                st.markdown(f"""
                <div style='
                    padding: 1rem; 
                    border: 1px solid #00c000; 
                    border-radius: 0.5rem; 
                    background-color: #e6ffe6;
                    color: #00c000;
                '>
                    <p class='big-font'>Prediction: Customer **WILL** likely purchase the Wellness Tourism Package! 🎉</p>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                # Already fixed in the previous step, using st.markdown() for info look and feel
                st.markdown(f"""
                <div style='
                    padding: 1rem; 
                    border: 1px solid #007bff; 
                    border-radius: 0.5rem; 
                    background-color: #e6f3ff;
                    color: #007bff;
                '>
                <p class='big-font'>Prediction: Customer is NOT likely to purchase the package. 😔</p>
                </div>
                """, unsafe_allow_html=True)


            # Display Probability
            st.markdown("#### Confidence Score")
            st.code(f"Probability of NOT purchasing (0): {prediction_proba[0]:.4f}")
            st.code(f"Probability of purchasing (1): {prediction_proba[1]:.4f}")
            
            # Display Input Data
            st.markdown("#### Customer Data Summary")
            st.dataframe(input_df.T, use_container_width=True)

else:
    st.warning("Model and Preprocessor failed to load from Hugging Face Hub. Please check your HF_REPO_MODEL configuration.")
