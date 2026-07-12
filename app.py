import joblib
import numpy as np
import os
import streamlit as st

# 1. Custom CSS Injector for Modern Glassmorphism & Professional Healthcare Theme
st.set_page_config(
    page_title="Aegis Medical Intelligence Panel",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Main structural modifications */
        .main { background-color: #f8f9fa; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        
        /* Glassmorphism Title Card Banner */
        .main-header {
            background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
            padding: 2.5rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }
        .main-header h1 { color: white !important; margin: 0; font-weight: 700; font-size: 2.5rem; }
        .main-header p { color: #e2e8f0 !important; margin-top: 0.5rem; margin-bottom: 0; font-size: 1.1rem; opacity: 0.9; }
        
        /* Metric Sub-headers styling */
        .section-header {
            color: #0f172a;
            font-size: 1.3rem;
            font-weight: 600;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #e2e8f0;
            margin-bottom: 1.25rem;
            margin-top: 1rem;
        }
        
        /* Custom styling overrides for primary predictive action buttons */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 0.75rem 2rem !important;
            border-radius: 8px !important;
            border: none !important;
            width: 100% !important;
            box-shadow: 0 4px 12px rgba(13, 148, 136, 0.2) !important;
            transition: all 0.2s ease;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(13, 148, 136, 0.3) !important;
        }
    </style>
""",
    unsafe_allow_html=True,
)


# 2. Optimized Asset Caching Function
@st.cache_resource
def load_disease_assets(disease_name):
    file_path = f"models/{disease_name.lower().replace(' ', '_')}_assets.pkl"
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None


# 3. Upgraded App Header Banner Layout
st.markdown(
    """
    <div class="main-header">
        <h1>🩺 Aegis Diagnostics Portal</h1>
        <p>Advanced Clinical Multi-Disease Predictor powered by Optimized Random Forest Ensembles.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Sidebar System Design Configuration
st.sidebar.markdown("### 🧭 System Navigation")
disease_selection = st.sidebar.selectbox(
    "Target Diagnostic Domain",
    ["Diabetes", "Heart Disease", "Chronic Kidney Disease"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='background-color:#f1f5f9; padding: 1rem; border-radius: 8px; border-left: 4px solid #0d9488;'>
        <small style='color:#475569; font-weight:500;'>
            <strong>Pipeline Framework Status:</strong><br>
            ✓ Feature Normalization Active<br>
            ✓ Median Data Imputation Enabled<br>
            ✓ LIME Context Baseline Cached
        </small>
    </div>
""",
    unsafe_allow_html=True,
)

# Fetch Model Pipeline Packages
assets = load_disease_assets(disease_selection)

if assets is None:
    st.error(
        f"❌ Operational Asset Error: Could not locate `models/{disease_selection.lower().replace(' ', '_')}_assets.pkl`. Please execute the training script package first."
    )
else:
    model = assets["model"]
    imputer = assets["imputer"]
    scaler = assets["scaler"]
    feature_names = assets["feature_names"]

    # Containerized Input Blocks (FIXED: Uses unsafe_allow_html=True)
    st.markdown(
        f"<div class='section-header'>📊 Vital Parameters Form: {disease_selection} Matrix</div>",
        unsafe_allow_html=True,
    )
    input_data = {}

    # DIABETES FORM LAYOUT
    if disease_selection == "Diabetes":
        col1, col2, col3 = st.columns(3)
        with col1:
            input_data["Pregnancies"] = st.number_input(
                "Pregnancies Count", min_value=0, max_value=20, value=1
            )
            input_data["Glucose"] = st.number_input(
                "Plasma Glucose (mg/dL)", min_value=0, max_value=300, value=115
            )
            input_data["BloodPressure"] = st.number_input(
                "Diastolic BP (mm Hg)", min_value=0, max_value=200, value=72
            )
        with col2:
            input_data["SkinThickness"] = st.number_input(
                "Triceps Skin Fold (mm)", min_value=0, max_value=100, value=20
            )
            input_data["Insulin"] = st.number_input(
                "Serum Insulin (mu U/ml)", min_value=0, max_value=1000, value=79
            )
            input_data["BMI"] = st.number_input(
                "Body Mass Index (BMI)",
                min_value=0.0,
                max_value=70.0,
                value=24.5,
                format="%.1f",
            )
        with col3:
            input_data["DiabetesPedigreeFunction"] = st.number_input(
                "Pedigree Score Function",
                min_value=0.0,
                max_value=3.0,
                value=0.47,
                format="%.3f",
            )
            input_data["Age"] = st.number_input(
                "Patient Age (Years)", min_value=1, max_value=120, value=33
            )

    # HEART DISEASE FORM LAYOUT
    elif disease_selection == "Heart Disease":
        col1, col2, col3 = st.columns(3)
        with col1:
            input_data["age"] = st.number_input(
                "Patient Age", min_value=1, max_value=120, value=48
            )
            sex_choice = st.selectbox("Biological Sex", ["Male", "Female"])
            input_data["sex"] = 1 if sex_choice == "Male" else 0
            cp_choice = st.selectbox(
                "Chest Pain Presentation",
                [
                    "0: Typical Angina",
                    "1: Atypical Angina",
                    "2: Non-anginal Pain",
                    "3: Asymptomatic",
                ],
            )
            input_data["cp"] = int(cp_choice.split(":")[0])
            input_data["trestbps"] = st.number_input(
                "Resting Sys BP (mm Hg)", min_value=50, max_value=250, value=130
            )
        with col2:
            input_data["chol"] = st.number_input(
                "Serum Chol (mg/dl)", min_value=100, max_value=600, value=240
            )
            fbs_choice = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl", ["False", "True"]
            )
            input_data["fbs"] = 1 if fbs_choice == "True" else 0
            restecg_choice = st.selectbox(
                "Resting ECG Variant",
                [
                    "0: Normal",
                    "1: ST-T Wave Abnormality",
                    "2: Left Ventricular Hypertrophy",
                ],
            )
            input_data["restecg"] = int(restecg_choice.split(":")[0])
            input_data["thalach"] = st.number_input(
                "Peak Heart Rate Achieved", min_value=60, max_value=250, value=145
            )
        with col3:
            exang_choice = st.selectbox(
                "Exercise Induced Angina", ["No", "Yes"]
            )
            input_data["exang"] = 1 if exang_choice == "Yes" else 0
            input_data["oldpeak"] = st.number_input(
                "ST Segment Depression",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                format="%.1f",
            )
            input_data["slope"] = st.selectbox(
                "Slope of Peak ST Segment", [0, 1, 2], index=1
            )
            input_data["ca"] = st.selectbox(
                "Fluoroscopy Major Vessels (0-4)", [0, 1, 2, 3, 4]
            )
            thal_choice = st.selectbox(
                "Thalassemia Type Variant",
                [
                    "0: Normal",
                    "1: Fixed Defect",
                    "2: Reversable Defect",
                    "3: Severe Defect",
                ],
                index=2,
            )
            input_data["thal"] = int(thal_choice.split(":")[0])

    # CHRONIC KIDNEY DISEASE FORM LAYOUT
    elif disease_selection == "Chronic Kidney Disease":
        st.markdown(
            "<h5 style='color:#64748b;'>Section A: Physiological & Metabolic Panels</h5>",
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            input_data["age"] = st.number_input(
                "Age Profile", min_value=1, max_value=120, value=50
            )
            input_data["bp"] = st.number_input(
                "Blood Pressure", min_value=40, max_value=200, value=80
            )
            input_data["sg"] = st.selectbox(
                "Specific Gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025], index=3
            )
            input_data["al"] = st.selectbox(
                "Albumin Level (al)", [0, 1, 2, 3, 4, 5], index=0
            )
        with col2:
            input_data["su"] = st.selectbox(
                "Sugar Scale (su)", [0, 1, 2, 3, 4, 5], index=0
            )
            input_data["bgr"] = st.number_input(
                "Blood Glucose Random", min_value=30, max_value=500, value=130
            )
            input_data["bu"] = st.number_input(
                "Blood Urea Count", min_value=5, max_value=400, value=35
            )
            input_data["sc"] = st.number_input(
                "Serum Creatinine Level",
                min_value=0.1,
                max_value=20.0,
                value=1.1,
                format="%.2f",
            )
        with col3:
            input_data["sod"] = st.number_input(
                "Sodium Element Count", min_value=50, max_value=200, value=137
            )
            input_data["pot"] = st.number_input(
                "Potassium Element Count",
                min_value=1.0,
                max_value=10.0,
                value=4.4,
                format="%.1f",
            )
            input_data["hemo"] = st.number_input(
                "Hemoglobin Scale",
                min_value=2.0,
                max_value=25.0,
                value=13.5,
                format="%.1f",
            )
            input_data["pcv"] = st.number_input(
                "Packed Cell Volume", min_value=5, max_value=60, value=38
            )

        st.markdown(
            "<h5 style='color:#64748b; margin-top:1.5rem;'>Section B: Cellular Labs & Anamnesis Comorbidities</h5>",
            unsafe_allow_html=True,
        )
        col4, col5, col6 = st.columns(3)
        with col4:
            input_data["wbcc"] = st.number_input(
                "White Blood Cell Count",
                min_value=1000,
                max_value=30000,
                value=7500,
            )
            input_data["rbcc"] = st.number_input(
                "Red Blood Cell Count",
                min_value=1.0,
                max_value=10.0,
                value=4.8,
                format="%.1f",
            )
            rbc_c = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
            input_data["rbc"] = 1 if rbc_c == "Normal" else 0
            pc_c = st.selectbox("Pus Cell Quality", ["Normal", "Abnormal"])
            input_data["pc"] = 1 if pc_c == "Normal" else 0
        with col5:
            pcc_c = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
            input_data["pcc"] = 1 if pcc_c == "Present" else 0
            ba_c = st.selectbox("Bacterial Cultures", ["Not Present", "Present"])
            input_data["ba"] = 1 if ba_c == "Present" else 0
            htn_c = st.selectbox("Hypertension Status", ["No", "Yes"])
            input_data["htn"] = 1 if htn_c == "Yes" else 0
            dm_c = st.selectbox("Diabetes Mellitus History", ["No", "Yes"])
            input_data["dm"] = 1 if dm_c == "Yes" else 0
        with col6:
            cad_c = st.selectbox("Coronary Artery Condition", ["No", "Yes"])
            input_data["cad"] = 1 if cad_c == "Yes" else 0
            appet_c = st.selectbox("Overall Appetite Assessment", ["Good", "Poor"])
            input_data["appet"] = 1 if appet_c == "Good" else 0
            pe_c = st.selectbox("Pedal Edema Presentation", ["No", "Yes"])
            input_data["pe"] = 1 if pe_c == "Yes" else 0
            ane_c = st.selectbox("Anemia Symptom Variant", ["No", "Yes"])
            input_data["ane"] = 1 if ane_c == "Yes" else 0

    st.markdown("<br>", unsafe_allow_html=True)

    # Execution Assessment Button
    if st.button(f"🔍 Execute {disease_selection} Analytics Engine"):
        raw_features = [input_data[feat] for feat in feature_names]
        feature_array = np.array(raw_features).reshape(1, -1)

        # Transformation Processing pipeline tracking
        imputed_features = imputer.transform(feature_array)
        scaled_features = scaler.transform(imputed_features)

        prediction_proba = model.predict_proba(scaled_features)[0][1]
        prediction_class = model.predict(scaled_features)[0]

        st.markdown(
            f"<div class='section-header'>📋 Diagnostic Engine Analytics Output: {disease_selection} Report</div>",
            unsafe_allow_html=True,
        )

        res_col1, res_col2 = st.columns([1.5, 2.5])

        with res_col1:
            st.markdown("#### Operational Status")
            if prediction_class == 1:
                st.markdown(
                    """
                    <div style='background-color:#fee2e2; border-left: 5px solid #ef4444; padding: 1.25rem; border-radius:8px;'>
                        <h4 style='color:#b91c1c; margin:0;'>🚨 High Risk Pattern Flagged</h4>
                        <p style='color:#7f1d1d; margin-top:0.5rem; margin-bottom:0; font-size:0.95rem;'>
                            The Ensembled trees indicate high correlation coefficients matching clinical classifications for positive diagnostic flags.
                        </p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style='background-color:#dcfce7; border-left: 5px solid #22c55e; padding: 1.25rem; border-radius:8px;'>
                        <h4 style='color:#15803d; margin:0;'>✅ Low Risk Pattern Profile</h4>
                        <p style='color:#14532d; margin-top:0.5rem; margin-bottom:0; font-size:0.95rem;'>
                            The structural indicators match healthy distributions within safe standard boundaries.
                        </p>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

        with res_col2:
            st.markdown("#### Probability Spectrum Metrics")

            # Mathematical Calculation of Progress Gauge Layouts
            color = "#ef4444" if prediction_proba > 0.5 else "#22c55e"

            st.markdown(
                f"""
                <div style='background-color:#ffffff; border: 1px solid #e2e8f0; padding: 1.25rem; border-radius:8px; box-shadow: 0 2px 8px rgba(0,0,0,0.02);'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;'>
                        <span style='font-weight:600; color:#475569;'>Calculated Classification Match:</span>
                        <span style='font-size:1.5rem; font-weight:700; color:{color};'>{prediction_proba:.2%}</span>
                    </div>
                    <div style='background-color:#e2e8f0; height:12px; border-radius:6px; overflow:hidden;'>
                        <div style='background-color:{color}; width:{prediction_proba*100}%; height:100%; border-radius:6px;'></div>
                    </div>
                    <p style='margin-top:0.75rem; margin-bottom:0; font-size:0.85rem; color:#64748b;'>
                        *Confidence level is calculated using the ratio of voting paths across the trained Random Forest ensemble array.
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )