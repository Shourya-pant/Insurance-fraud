import streamlit as st
import pickle
import numpy as np

# Load the trained model
def load_model():
    with open("svm_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# Streamlit app title and subtitle with a description
st.set_page_config(page_title="Medical Insurance Fraud Detection", page_icon=":hospital:", layout="wide")
st.title("Medical Insurance Fraud Detection System :hospital:")

st.markdown("""
    This system helps predict whether an insurance claim is fraudulent or not. 
    Fill in the details below and click on **'Predict'** to get the result.
""")
st.markdown("---")

# Sidebar with instructions
st.sidebar.title("Instructions")
st.sidebar.info("""
    - Fill in the required fields.
    - Make sure all the data is accurate.
    - Once you’ve entered all the details, click on **'Predict'** to check if the claim is fraudulent.
""")

# Create a better input structure with expanders and columns for better UX

# Section 1: Basic Claim Information
with st.expander("Claim Information", expanded=True):
    provider = st.number_input("Provider ID", min_value=0, step=1, help="Enter the unique provider ID.")
    ins_claim_amt_reimbursed = st.number_input("Insurance Claim Amount Reimbursed", min_value=0.0, step=0.01, help="Amount reimbursed by insurance.")
    admit_for_days = st.slider("Admit For Days", min_value=0, max_value=365, step=1, help="Number of days admitted to the hospital.")
    deductible_amt_paid = st.slider("Deductible Amount Paid", min_value=0.0, max_value=100000.0, step=0.01, help="Amount paid for deductible.")
    n_procedure = st.slider("Number of Procedures", min_value=0, max_value=10, step=1, help="Number of medical procedures performed.")
    operating_physician = st.text_input("Operating Physician", help="Enter the name of the operating physician.")

# Section 2: Date Information
with st.expander("Date Information", expanded=True):
    admission_dt = st.date_input("Admission Date")
    discharge_dt = st.date_input("Discharge Date")
    dob = st.date_input("Date of Birth")
    dod = st.date_input("Date of Death (if applicable)", value=None)

# Section 3: Demographics
with st.expander("Demographic Information", expanded=True):
    gender = st.selectbox("Gender", options=["M", "F", "U"], help="Select the gender of the patient.")
    race = st.number_input("Race", min_value=0, step=1, help="Enter the race value of the patient.")

# Section 4: Chronic Conditions
with st.expander("Chronic Conditions", expanded=True):
    chronic_cond_alzheimer = st.selectbox("Chronic Condition: Alzheimer", options=[0, 1], help="Does the patient have Alzheimer’s disease?")
    chronic_cond_heart_failure = st.selectbox("Chronic Condition: Heart Failure", options=[0, 1], help="Does the patient have heart failure?")
    
    # Dynamic Dropdowns: If Alzheimer's is selected, show options related to Alzheimer's
    if chronic_cond_alzheimer == 1:
        st.selectbox("Alzheimer's Care Options", options=["Memory Care", "Medication", "Therapy"])
    
    chronic_cond_kidney_disease = st.selectbox("Chronic Condition: Kidney Disease", options=[0, 1], help="Does the patient have kidney disease?")
    chronic_cond_cancer = st.selectbox("Chronic Condition: Cancer", options=[0, 1], help="Does the patient have cancer?")
    chronic_cond_obstr_pulmonary = st.selectbox("Chronic Condition: Obstructive Pulmonary", options=[0, 1], help="Does the patient have obstructive pulmonary disease?")
    chronic_cond_depression = st.selectbox("Chronic Condition: Depression", options=[0, 1], help="Does the patient have depression?")
    chronic_cond_diabetes = st.selectbox("Chronic Condition: Diabetes", options=[0, 1], help="Does the patient have diabetes?")
    chronic_cond_ischemic_heart = st.selectbox("Chronic Condition: Ischemic Heart", options=[0, 1], help="Does the patient have ischemic heart disease?")
    chronic_cond_osteoporasis = st.selectbox("Chronic Condition: Osteoporosis", options=[0, 1], help="Does the patient have osteoporosis?")
    chronic_cond_rheumatoid_arthritis = st.selectbox("Chronic Condition: Rheumatoid Arthritis", options=[0, 1], help="Does the patient have rheumatoid arthritis?")
    chronic_cond_stroke = st.selectbox("Chronic Condition: Stroke", options=[0, 1], help="Does the patient have stroke?")

# Section 5: Additional Info
with st.expander("Additional Information", expanded=True):
    op_annual_reimbursement_amt = st.slider("OP Annual Reimbursement Amount", min_value=0.0, max_value=100000.0, step=0.01, help="Annual reimbursement for outpatient procedures.")
    op_annual_deductible_amt = st.slider("OP Annual Deductible Amount", min_value=0.0, max_value=100000.0, step=0.01, help="Annual deductible amount for outpatient procedures.")
    age = st.slider("Age", min_value=0, max_value=100, step=1, help="Enter the patient's age.")
    whether_dead = st.selectbox("Whether Dead (0 or 1)", options=[0, 1], help="Indicate whether the patient is deceased (1) or not (0).")

# Preprocess inputs
gender_map = {"M": 1, "F": 0, "U": -1}
gender_value = gender_map[gender]

# Convert date inputs to numeric values (e.g., days since a reference date)
reference_date = np.datetime64('1970-01-01')
admission_dt_value = (np.datetime64(admission_dt) - reference_date).astype(int)
discharge_dt_value = (np.datetime64(discharge_dt) - reference_date).astype(int)
dob_value = (np.datetime64(dob) - reference_date).astype(int)
dod_value = (np.datetime64(dod) - reference_date).astype(int) if dod else 0

# Validate and convert operating_physician to a numeric value (e.g., hash)
operating_physician_value = hash(operating_physician) % (10 ** 8)

# Create feature array
input_features = np.array([
    provider, ins_claim_amt_reimbursed, admit_for_days, deductible_amt_paid,
    n_procedure, operating_physician_value,
    admission_dt_value, discharge_dt_value, dob_value, dod_value, gender_value, race,
    chronic_cond_alzheimer,
    chronic_cond_heart_failure, chronic_cond_kidney_disease, chronic_cond_cancer,
    chronic_cond_obstr_pulmonary, chronic_cond_depression, chronic_cond_diabetes,
    chronic_cond_ischemic_heart, chronic_cond_osteoporasis, chronic_cond_rheumatoid_arthritis,
    chronic_cond_stroke,
    op_annual_reimbursement_amt, op_annual_deductible_amt, age, whether_dead
]).reshape(1, -1)

# Check the number of features
expected_num_features = model.n_features_in_
if input_features.shape[1] != expected_num_features:
    st.error(f"Expected {expected_num_features} features, but got {input_features.shape[1]} features.")
else:
    # Interactive prediction button
    if st.button("Predict", key="predict_button"):
        prediction = model.predict(input_features)
        prediction_result = "Fraudulent" if prediction[0] == 1 else "Not Fraudulent"
        
        # Display prediction result with icons and colors
        st.markdown(f"### Prediction: **{prediction_result}**")
        
        if prediction[0] == 1:
            st.warning("⚠️ **Fraudulent** claim detected! ⚠️")
        else:
            st.success("✅ This claim is **legitimate**. ✅")

# Footer
st.markdown("---")
st.markdown("© 2025 Medical Insurance Fraud Detection System")
