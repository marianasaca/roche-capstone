import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Roche Lab Operations - Delay Predictor",
    page_icon="🔬",
    layout="wide"
)

# --- Title & Header ---
st.title("🔬 Roche Lab Operations - Delay Prediction System")
st.markdown("### Advanced AI-Driven Operations Management")
st.markdown("---")

# --- Sidebar: Operational Inputs ---
st.sidebar.header("Operational Parameters")

# 1. Operational inputs
lab_occupancy = st.sidebar.slider("Lab Occupancy (%)", 0, 100, 70, help="Current utilization of lab resources.")
scientist_workload = st.sidebar.slider("Scientist Workload (Active Projects)", 1, 20, 5, help="Number of concurrent experiments per scientist.")

st.sidebar.markdown("---")
st.sidebar.header("Simulation & Supply Chain (V2)")

# 2. V2 Features: Drift & Supply
days_since_start = st.sidebar.slider("Days Since Installation (Simulate Aging)", 0, 730, 100, help="Higher values simulate older machines (Drift).")

# Reagent Batch Selection (Top 5 list + Bad Ones)
# We simulate a list of batches. BATCH_392 is the famous 'Bad Batch'.
batch_options = [f"BATCH_{i:03d}" for i in range(1, 11)] + ["BATCH_392", "BATCH_042", "BATCH_007"]
reagent_batch_id = st.sidebar.selectbox("Reagent Batch ID", sorted(batch_options), index=0, help="Select the batch used for this experiment.")

st.sidebar.markdown("---")
st.sidebar.header("Experiment Details")

# 3. Categorical Inputs
experiment_type = st.sidebar.selectbox("Experiment Type", ["Validation", "QC", "Pilot", "Screening", "R&D"])
instrument_type = st.sidebar.selectbox("Instrument Type", ["Microscope", "Centrifuge", "Spectrometer", "HPLC", "Incubator", "PCR"])
scientist_experience = st.sidebar.selectbox("Scientist Experience", ["Junior", "Mid", "Senior"])
priority_level = st.sidebar.selectbox("Priority Level", ["Low", "Medium", "High", "Critical"])

# --- Advanced / Hidden Inputs ---
# We need 'mean_ambient_temp' and 'expected_duration' for the model
# We set reasonable defaults or infer them
with st.expander("Advanced Configuration (Hidden Variables)"):
    mean_ambient_temp = st.slider("Mean Ambient Temp (°C)", 18.0, 30.0, 22.0)
    
    # Duration Map (Inferred from Type)
    duration_map = {'Validation': 60, 'QC': 45, 'Pilot': 90, 'Screening': 30, 'R&D': 120}
    default_duration = duration_map.get(experiment_type, 60)
    expected_duration = st.number_input("Expected Duration (mins)", value=default_duration)

# --- Main Dashboard Logic ---

# Create a DataFrame for the model
input_data = pd.DataFrame({
    'scientist_workload': [scientist_workload],
    'lab_occupancy_level': [lab_occupancy],
    'expected_duration': [expected_duration],
    'mean_ambient_temp': [mean_ambient_temp],
    'days_since_start': [days_since_start],
    'experiment_type': [experiment_type],
    'instrument_type': [instrument_type],
    'scientist_experience_level': [scientist_experience],
    'reagent_batch_id': [reagent_batch_id]
    # Note: priority_level is excluded as it's not in the V2 model
})

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Experimental Overview")
    st.table(input_data)

    if st.button("Predict Delay Risk", type="primary"):
        try:
            # Load Model
            model = joblib.load('lab_delay_model_v2.pkl')
            
            # Predict
            prediction = model.predict(input_data)[0]
            
            # Display Result
            st.markdown("### Prediction Result")
            
            # Risk Logic
            if prediction < 15:
                risk_color = "green"
                risk_label = "LOW RISK"
                advice = "Process is standard. No intervention needed."
            elif 15 <= prediction < 45:
                risk_color = "orange"
                risk_label = "MODERATE RISK"
                advice = "Monitor closely. Potential minor delays expected."
            else:
                risk_color = "red"
                risk_label = "HIGH RISK (BOTTLENECK)"
                advice = "CRITICAL: Reschedule or allocate backup resources immediately."

            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 10px solid {risk_color};">
                <h2 style="color: {risk_color}; margin:0;">{risk_label}</h2>
                <h3 style="margin:0;">Expected Delay: {prediction:.1f} minutes</h3>
                <p style="margin-top: 10px;"><strong>Action:</strong> {advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except FileNotFoundError:
            st.error("Error: Model file 'lab_delay_model_v2.pkl' not found. Please train the model first.")
        except Exception as e:
            st.error(f"Prediction Error: {e}")

with col2:
    st.markdown("### Manager's Insight")
    with st.expander("Model Logic & heuristics", expanded=True):
        st.info("""
        **Delay Drivers Identified:**
        
        1.  **Lab Occupancy**:
            *   Levels > 85% cause exponential blocking.
        
        2.  **Machine Aging (Drift)**:
            *   Instruments older than **600 days** (approx 2 years) show significant performance degradation.
            
        3.  **Supply Chain**:
            *   **BATCH_392** has been flagged as **DEFECTIVE**. Using this batch increases failure probability by 500%.
        """)
    
    st.markdown("---")
    st.caption("v2.0.1 | Roche Capstone AI")
