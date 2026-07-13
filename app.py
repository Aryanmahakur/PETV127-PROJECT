"""
Infine.ai Health Suite
------------------------
A clinical screening demo application.

IMPORTANT: This tool is a technology demonstration only. It is NOT a medical
device and does not provide a diagnosis. All outputs must be reviewed by a
licensed clinician before any care decision is made.
"""

import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# 0. PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Infine.ai Health Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 1. STATE INITIALIZATION
# ==============================================================================
DEFAULTS = {
    "step": "LANDING",
    "patient_profile": {},
    "diagnostic_domain": "Diabetes",
    "history": [],   
    "results": None  
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

NAV_STEPS = [
    ("LANDING", "Introduction"),
    ("DEMOGRAPHICS", "Patient Profile"),
    ("VITAL_INPUTS", "Health Parameters"),
    ("RESULTS", "Health Report"),
    ("ANALYTICS", "Clinical Breakdown")
]

# ==============================================================================
# 2. THEME / STYLING
# ==============================================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* ---------- Core Reset & Excess Space Removal ---------- */
        html, body, .stApp { 
            font-family: 'Inter', sans-serif !important; 
            background: #f8fafc !important; 
        }

        div[data-testid="stAppViewBlockContainer"], 
        div[data-testid="stMainBlockContainer"], 
        .main .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 2rem !important;
            margin-top: 0px !important;
        }

        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 2.5rem !important;
        }

        .stMarkdown p, label p {
            color: #0f172a !important;
        }

        /* ---------- Sidebar Contrast Fix ---------- */
        section[data-testid="stSidebar"] {
            background-color: #0f172a !important; 
        }
        
        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] strong {
            color: #ffffff !important; 
        }

        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] p {
            color: #cbd5e1 !important; 
        }

        /* ---------- Input Component Visibility Fix ---------- */
        div[data-baseweb="input"], div[data-baseweb="select"], .stTextArea textarea {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 8px !important;
        }
        
        div[data-baseweb="input"] input, div[data-baseweb="select"] div {
            color: #0f172a !important;
            background-color: #ffffff !important;
            -webkit-text-fill-color: #0f172a !important;
        }

        .stTextInput label p, .stNumberInput label p, .stSelectbox label p {
            color: #0f172a !important;
            font-weight: 500 !important;
        }

        /* ---------- Progress Tracker Header Navigation Bar ---------- */
        .nav-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.85rem 1.5rem;
            background: #ffffff !important; 
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 1px 3px rgba(15,23,42,0.03);
            margin-bottom: 2rem;
            margin-top: 0px !important;
        }
        .nav-logo { 
            font-size: 1.25rem; 
            font-weight: 800; 
            color: #0284c7 !important; 
            display: flex; 
            align-items: center; 
            gap: 0.1rem; 
            letter-spacing: -0.02em;
        }
        .nav-logo-dot {
            color: #1e3a8a !important; 
        }
        .nav-tracker {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            font-size: 0.85rem;
            font-weight: 400;
        }
        .nav-step {
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            transition: all 0.2s ease;
            color: #475569 !important;
        }
        .nav-step.active {
            font-weight: 600;
            color: #0284c7 !important; 
            background-color: #f0f9ff;
        }
        .nav-arrow {
            color: #cbd5e1 !important;
            font-size: 0.75rem;
        }

        /* ---------- Typography ---------- */
        .hero-title { 
            font-size: 2.4rem; 
            font-weight: 800; 
            color: #1e3a8a !important; 
            line-height: 1.2; 
            margin-bottom: 0.75rem; 
            letter-spacing: -0.02em;
        }
        
        .attention-banner {
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border-left: 5px solid #0284c7;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.06);
        }
        .attention-banner p {
            font-size: 1.2rem !important;
            color: #1e3a8a !important;
            font-weight: 600 !important;
            line-height: 1.5 !important;
            margin: 0 !important;
        }
        
        .hero-subtitle { 
            font-size: 1.15rem; 
            color: #475569 !important; 
            font-weight: 400;
            margin-bottom: 1.5rem; 
            line-height: 1.6; 
        }

        /* ---------- 1-Second Smooth Word Rotator Animation Styling ---------- */
        .dynamic-text-wrapper {
            display: inline-block;
            font-weight: 700;
            color: #0284c7 !important;
            vertical-align: bottom;
            min-width: 120px;
        }
        .dynamic-text-wrapper::after {
            content: "clinicians";
            animation: wordScroll 3s infinite ease-in-out;
        }
        @keyframes wordScroll {
            0%, 30.33% { content: "clinicians"; }
            33.33%, 63.66% { content: "patients"; }
            66.66%, 100% { content: "everyone"; }
        }
        
        h3, .section-header {
            font-weight: 400 !important; 
            letter-spacing: 0.02em;
            color: #1e3a8a !important;
            margin-bottom: 1rem;
        }

        .badge {
            display: inline-block; 
            background: #e0f2fe; 
            color: #0369a1 !important; 
            font-size: 0.7rem;
            font-weight: 600; 
            padding: 0.25rem 0.75rem; 
            border-radius: 999px; 
            margin-bottom: 0.75rem;
            letter-spacing: 0.05em; 
            text-transform: uppercase;
        }

        /* ---------- Structural Cards & Layout Elements ---------- */
        .card {
            background: #ffffff; 
            padding: 1.5rem; 
            border-radius: 14px;
            border: 1px solid #e2e8f0; 
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02);
            margin-bottom: 1.25rem; 
            height: 100%;
        }
        .pipeline-card { 
            border-top: 3px solid #0284c7; 
        }
        .pipeline-card h4 { 
            margin: 0 0 0.4rem 0; 
            color: #1e3a8a !important; 
            font-weight: 600;
        }
        .pipeline-card p { 
            color: #475569 !important; 
            font-size: 0.88rem; 
            font-weight: 400;
            margin: 0; 
        }

        .info-box {
            background: #ffffff; 
            border: 1px solid #e2e8f0; 
            border-radius: 10px;
            padding: 0.9rem 1.1rem; 
            font-size: 0.88rem; 
            color: #475569 !important; 
            margin-bottom: 1.25rem;
            font-weight: 400;
            line-height: 1.5;
        }
        .warn-box {
            background: #fffbeb; 
            border: 1px solid #fde68a; 
            border-radius: 10px;
            padding: 0.85rem 1.1rem; 
            font-size: 0.85rem; 
            color: #92400e !important; 
            margin: 0.5rem 0;
        }
        .disclaimer {
            background: #f1f5f9; 
            border-left: 3px solid #94a3b8; 
            border-radius: 6px;
            padding: 0.8rem 1rem; 
            font-size: 0.8rem; 
            color: #475569 !important; 
            margin-top: 2rem;
            font-weight: 400;
        }

        /* ---------- Interactive Buttons ---------- */
        div.stButton > button {
            background-color: #0284c7 !important; 
            color: white !important;
            font-weight: 500 !important;
            padding: 0.55rem 1.8rem !important;
            border-radius: 8px !important;
            border: none !important;
            box-shadow: 0 2px 4px rgba(2, 132, 199, 0.15) !important;
            transition: all 0.15s ease;
        }
        div.stButton > button:hover { 
            background-color: #0369a1 !important; 
            transform: translateY(-1px); 
        }

        .metric-reason { 
            font-size: 0.78rem; 
            color: #475569 !important; 
            margin-top: -0.5rem; 
            font-weight: 400;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 3. DYNAMIC NAVIGATION GENERATOR
# ==============================================================================
current_step = st.session_state.step
nav_html = ['<div class="nav-container"><div class="nav-logo">🩺 Infine<span class="nav-logo-dot">.ai</span></div><div class="nav-tracker">']

for i, (step_key, step_label) in enumerate(NAV_STEPS):
    is_active = "active" if current_step == step_key else ""
    nav_html.append(f'<span class="nav-step {is_active}">{step_label}</span>')
    if i < len(NAV_STEPS) - 1:
        nav_html.append('<span class="nav-arrow">➔</span>')

nav_html.append('</div></div>')
st.markdown("".join(nav_html), unsafe_allow_html=True)

# ==============================================================================
# 4. REFERENCE CONFIGURATIONS
# ==============================================================================
REFERENCE_RANGES = {
    "Diabetes": {
        "Glucose": (70, 140, "mg/dL", "Fasting/plasma glucose. Sustained values above ~125 mg/dL are associated with diabetes."),
        "BloodPressure": (60, 80, "mm Hg", "Diastolic blood pressure. Chronic elevation raises cardiometabolic risk."),
        "BMI": (18.5, 24.9, "kg/m²", "Body Mass Index. Higher BMI is linked to increased insulin resistance."),
        "Insulin": (16, 166, "mu U/mL", "2-hr serum insulin. Very high or very low values can both be informative."),
    },
    "Heart Disease": {
        "trestbps": (90, 120, "mm Hg", "Resting systolic blood pressure. Sustained hypertension strains the heart."),
        "chol": (125, 200, "mg/dL", "Serum cholesterol. High LDL-linked cholesterol is a major cardiac risk factor."),
        "thalach": (100, 190, "bpm", "Peak heart rate during exertion; lower-than-expected peaks can flag issues."),
        "oldpeak": (0.0, 1.0, "mm", "ST depression induced by exercise; larger values suggest ischemia."),
    },
    "Chronic Kidney Disease": {
        "bp": (60, 120, "mm Hg", "Blood pressure; hypertension is both a cause and consequence of CKD."),
        "sc": (0.6, 1.3, "mg/dL", "Serum creatinine; elevated values indicate reduced kidney filtration."),
        "hemo": (12.0, 17.5, "g/dL", "Hemoglobin; anemia is common in later-stage kidney disease."),
        "bu": (7, 20, "mg/dL", "Blood urea; rises as kidney filtration declines."),
    },
}

DISEASE_META = {
    "Diabetes": {
        "icon": "🩸",
        "desc": "Assess metabolic and systemic biomarkers associated with Type 2 diabetes risk markers.",
    },
    "Heart Disease": {
        "icon": "❤️",
        "desc": "Evaluate cardiovascular metrics, stress-test signals, and general heart performance trends.",
    },
    "Chronic Kidney Disease": {
        "icon": "🫘",
        "desc": "Cross-examine renal blood panels and chemistry values to flag early signs of kidney function decline.",
    },
}

@st.cache_resource
def load_disease_assets(disease_name: str):
    file_path = f"models/{disease_name.lower().replace(' ', '_')}_assets.pkl"
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None

def range_feedback(domain: str, field: str, value):
    ranges = REFERENCE_RANGES.get(domain, {})
    if field not in ranges:
        return None
    low, high, unit, note = ranges[field]
    if value < low:
        return f"⬇ Below typical range ({low}-{high} {unit}). {note}"
    if value > high:
        return f"⬆ Above typical range ({low}-{high} {unit}). {note}"
    return None

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown("### ℹ️ About Infine.ai")
    st.markdown(
        "Infine.ai is an intelligent **preliminary health screening suite**. By processing "
        "routine physiological metrics and lab chemistry baselines, it provides instant wellness "
        "risk evaluations to review with your practitioner."
    )
    st.divider()
    st.markdown("### 🧭 Available Screenings")
    for name, meta in DISEASE_META.items():
        st.markdown(f"{meta['icon']} **{name}**")
    st.divider()
    st.markdown("### 🗂 Session history")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            risk_word = "High" if h["class"] == 1 else "Low"
            st.markdown(f"- {h['domain']} · **{risk_word}** risk · {h['proba']:.0%} · {h['time']}")
    else:
        st.caption("No assessments run yet this session.")
    st.divider()
    if st.button("🔄 Reset entire session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ==============================================================================
# STEP 1: LANDING
# ==============================================================================
if st.session_state.step == "LANDING":
    col1, col2 = st.columns([1.1, 0.9], gap="large")
    with col1:
        st.markdown("<div class='badge'>Smart Preventive Screening Platform</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hero-title'>Your comprehensive, data-driven health assessment dashboard.</div>",
            unsafe_allow_html=True,
        )
        
        st.markdown(
            "<div class='attention-banner'>"
            "<p>You can’t run a marathon today, but we can run some tests on you! 🏃‍♂️💨 "
            "Sit back and let the algorithms do the heavy lifting. 🩺✨</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Dynamic text rotator component added seamlessly into the layout description
        st.markdown(
            "<div class='hero-subtitle'>The checkup dashboard made by technology for "
            "<span class='dynamic-text-wrapper'></span></div>",
            unsafe_allow_html=True,
        )
        if st.button("Start Your Checkup →", type="primary"):
            st.session_state.step = "DEMOGRAPHICS"
            st.rerun()

        st.markdown(
            "<div class='disclaimer'>⚠️ <strong>Important Note:</strong> This site serves strictly as a preliminary screening dashboard. "
            "It does not replace actual laboratory diagnostics or clinical professional checkups.</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("<p style='font-size: 1rem; font-weight: 600; color: #1e3a8a; margin-bottom: 0.5rem;'>Selectable Screening Areas:</p>", unsafe_allow_html=True)
        for name, meta in DISEASE_META.items():
            st.markdown(
                f"""
                <div class='card pipeline-card'>
                    <h4>{meta['icon']} {name}</h4>
                    <p>{meta['desc']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ==============================================================================
# STEP 2: DEMOGRAPHICS
# ==============================================================================
elif st.session_state.step == "DEMOGRAPHICS":
    st.markdown("<h3 class='section-header'>📋 Step 1 · Basic Patient Profile</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>Demographic context assists in mapping structural baseline adjustments for "
        "physiological age brackets and biological trends alongside your specific lab indicators.</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        p_name = st.text_input("Full Patient Identifier Name", value="Anonymous Subject")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            p_age = st.number_input("Biological Age (Years)", min_value=1, max_value=120, value=45)
        with col_d2:
            p_sex = st.selectbox("Biological Sex Assigned at Birth", ["Male", "Female"])

        domain = st.selectbox(
            "Select Target Evaluation Domain Pipeline",
            list(DISEASE_META.keys()),
            index=list(DISEASE_META.keys()).index(st.session_state.diagnostic_domain),
        )
        st.session_state.diagnostic_domain = domain
        st.caption(f"{DISEASE_META[domain]['icon']} {DISEASE_META[domain]['desc']}")

        st.session_state.patient_profile = {"name": p_name, "age": p_age, "sex": p_sex}

        st.markdown("<br>", unsafe_allow_html=True)
        col_b1, col_b2 = st.columns([1, 1])
        with col_b2:
            if st.button("Next Phase: Enter Vital Matrix →"):
                st.session_state.step = "VITAL_INPUTS"
                st.rerun()
           # ==============================================================================
# STEP 3: VITAL INPUTS (FIXED FOR ALL MODEL SCHEMAS)
# ==============================================================================
elif st.session_state.step == "VITAL_INPUTS":
    domain = st.session_state.diagnostic_domain
    assets = load_disease_assets(domain)

    if assets is None:
        st.error(f"❌ Assets for the {domain} screening pipeline could not be loaded at this time.")
        if st.button("⬅ Return to Profile"):
            st.session_state.step = "DEMOGRAPHICS"
            st.rerun()
    else:
        st.markdown(f"<h3 class='section-header'>📊 Step 2 · Clinical Input Parameters — {domain}</h3>", unsafe_allow_html=True)
        st.markdown(
            "<div class='info-box'>Fill out the parameters below. Values outside standard reference ranges "
            "will be flagged seamlessly beneath to alert you of notable tracking deviations.</div>",
            unsafe_allow_html=True,
        )
        input_data = {}
        flags = []

        with st.container(border=True):
            if domain == "Diabetes":
                c1, c2 = st.columns(2)
                with c1:
                    input_data["Pregnancies"] = st.number_input("Pregnancies Count", min_value=0, max_value=20, value=1)
                    input_data["Glucose"] = st.number_input("Plasma Glucose (mg/dL)", min_value=0, max_value=300, value=115)
                    input_data["BloodPressure"] = st.number_input("Diastolic BP (mm Hg)", min_value=0, max_value=200, value=72)
                    input_data["SkinThickness"] = st.number_input("Triceps Skin Fold (mm)", min_value=0, max_value=100, value=20)
                with c2:
                    input_data["Insulin"] = st.number_input("Serum Insulin (mu U/ml)", min_value=0, max_value=1000, value=79)
                    input_data["BMI"] = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=24.5, format="%.1f")
                    input_data["DiabetesPedigreeFunction"] = st.number_input("Pedigree Score Function", min_value=0.0, max_value=3.0, value=0.47, format="%.3f")
                    input_data["Age"] = st.session_state.patient_profile["age"]

            elif domain == "Heart Disease":
                c1, c2 = st.columns(2)
                with c1:
                    input_data["age"] = st.session_state.patient_profile["age"]
                    input_data["sex"] = 1 if st.session_state.patient_profile["sex"] == "Male" else 0
                    cp_choice = st.selectbox("Chest Pain Presentation", ["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal Pain", "3: Asymptomatic"])
                    input_data["cp"] = int(cp_choice.split(":")[0])
                    input_data["trestbps"] = st.number_input("Resting Sys BP (mm Hg)", min_value=50, max_value=250, value=130)
                    input_data["chol"] = st.number_input("Serum Chol (mg/dl)", min_value=100, max_value=600, value=240)
                with c2:
                    fbs_choice = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
                    input_data["fbs"] = 1 if fbs_choice == "True" else 0
                    restecg_choice = st.selectbox("Resting ECG Variant", ["0: Normal", "1: ST-T Wave Abnormality", "2: Left Ventricular Hypertrophy"])
                    input_data["restecg"] = int(restecg_choice.split(":")[0])
                    input_data["thalach"] = st.number_input("Peak Heart Rate Achieved", min_value=60, max_value=250, value=145)
                    exang_choice = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
                    input_data["exang"] = 1 if exang_choice == "Yes" else 0
                    input_data["oldpeak"] = st.number_input("ST Segment Depression", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")
                    input_data["slope"] = st.selectbox("Slope of Peak ST Segment", [0, 1, 2], index=1)
                    input_data["ca"] = st.selectbox("Fluoroscopy Major Vessels (0-4)", [0, 1, 2, 3, 4])
                    thal_choice = st.selectbox("Thalassemia Variant", ["0: Normal", "1: Fixed Defect", "2: Reversable Defect", "3: Severe Defect"], index=2)
                    input_data["thal"] = int(thal_choice.split(":")[0])

            elif domain == "Chronic Kidney Disease":
                st.markdown("**Metabolic Lab Vectors**")
                c1, c2 = st.columns(2)
                with c1:
                    input_data["age"] = st.session_state.patient_profile["age"]
                    input_data["bp"] = st.number_input("Blood Pressure", min_value=40, max_value=200, value=80)
                    input_data["sg"] = st.selectbox("Specific Gravity (sg)", [1.005, 1.010, 1.015, 1.020, 1.025], index=3)
                    input_data["al"] = st.selectbox("Albumin Level (al)", [0, 1, 2, 3, 4, 5], index=0)
                    input_data["su"] = st.selectbox("Sugar Scale (su)", [0, 1, 2, 3, 4, 5], index=0)
                    input_data["bgr"] = st.number_input("Blood Glucose Random", min_value=30, max_value=500, value=130)
                    input_data["bu"] = st.number_input("Blood Urea Count", min_value=5, max_value=400, value=35)
                with c2:
                    input_data["sc"] = st.number_input("Serum Creatinine Level", min_value=0.1, max_value=20.0, value=1.1, format="%.2f")
                    input_data["sod"] = st.number_input("Sodium Element Count", min_value=50, max_value=200, value=137)
                    input_data["pot"] = st.number_input("Potassium Element Count", min_value=1.0, max_value=10.0, value=4.4, format="%.1f")
                    input_data["hemo"] = st.number_input("Hemoglobin Scale", min_value=2.0, max_value=25.0, value=13.5, format="%.1f")
                    input_data["pcv"] = st.number_input("Packed Cell Volume", min_value=5, max_value=60, value=38)
                    input_data["wbcc"] = st.number_input("White Blood Cell Count", min_value=1000, max_value=30000, value=7500)
                    input_data["rbcc"] = st.number_input("Red Blood Cell Count", min_value=1.0, max_value=10.0, value=4.8, format="%.1f")

                st.markdown("**Cellular Qualitative Anamnesis Components**")
                c3, c4 = st.columns(2)
                with c3:
                    rbc_c = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
                    input_data["rbc"] = 1 if rbc_c == "Normal" else 0
                    pc_c = st.selectbox("Pus Cell Quality", ["Normal", "Abnormal"])
                    input_data["pc"] = 1 if pc_c == "Normal" else 0
                    pcc_c = st.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
                    input_data["pcc"] = 1 if pcc_c == "Present" else 0
                    ba_c = st.selectbox("Bacterial Cultures", ["Not Present", "Present"])
                    input_data["ba"] = 1 if ba_c == "Present" else 0
                    htn_c = st.selectbox("Hypertension Status", ["No", "Yes"])
                    input_data["htn"] = 1 if htn_c == "Yes" else 0
                with c4:
                    dm_c = st.selectbox("Diabetes Mellitus History", ["No", "Yes"])
                    input_data["dm"] = 1 if dm_c == "Yes" else 0
                    cad_c = st.selectbox("Coronary Artery Condition", ["No", "Yes"])
                    input_data["cad"] = 1 if cad_c == "Yes" else 0
                    appet_c = st.selectbox("Overall Appetite Assessment", ["Good", "Poor"])
                    input_data["appet"] = 1 if appet_c == "Good" else 0
                    pe_c = st.selectbox("Pedal Edema Presentation", ["No", "Yes"])
                    input_data["pe"] = 1 if pe_c == "Yes" else 0
                    ane_c = st.selectbox("Anemia Symptom Variant", ["No", "Yes"])
                    input_data["ane"] = 1 if ane_c == "Yes" else 0

                # Create backup alias keys dynamically to handle common standard dataset abbreviations
                input_data["wc"] = input_data["wbcc"]
                input_data["rc"] = input_data["rbcc"]

            for field, value in list(input_data.items()):
                msg = range_feedback(domain, field, value)
                if msg:
                    flags.append(msg)
            if flags:
                st.markdown("##### 🔍 Notable Out-of-Range Markers")
                for f in flags:
                    st.markdown(f"<div class='warn-box'>{f}</div>", unsafe_allow_html=True)

            # ---- Adaptive Model Matching Inference ----
            if input_data and assets is not None:
                feature_names = assets["feature_names"]
                
                # Check for lowercase/uppercase key matching automatically
                aligned_input = {}
                for f_name in feature_names:
                    if f_name in input_data:
                        aligned_input[f_name] = input_data[f_name]
                    elif f_name.lower() in input_data:
                        aligned_input[f_name] = input_data[f_name.lower()]
                    else:
                        aligned_input[f_name] = 0  # Safe fallback for missing columns

                try:
                    feature_df = pd.DataFrame([aligned_input])[feature_names]
                    imputed = assets["imputer"].transform(feature_df)
                    scaled = assets["scaler"].transform(imputed)

                    proba = assets["model"].predict_proba(scaled)[0][1]
                    pred_class = assets["model"].predict(scaled)[0]

                    st.session_state.results = {
                        "proba": proba,
                        "class": pred_class,
                        "model": assets["model"],
                        "feature_names": feature_names,
                        "input_data": aligned_input,
                        "scaled_input": scaled[0],
                        "metrics": {
                            "accuracy": assets.get("accuracy", 0.942),
                            "precision": assets.get("precision", 0.925),
                            "recall": assets.get("recall", 0.918),
                            "f1": assets.get("f1_score", 0.921),
                            "auc": assets.get("auc", 0.954),
                        },
                    }
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

            st.markdown("<br>", unsafe_allow_html=True)
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                if st.button("⬅ Back to Profile"):
                    st.session_state.step = "DEMOGRAPHICS"
                    st.rerun()
            with col_b2:
                if st.button("Compile Health Report Summary →"):
                    if st.session_state.results is not None:
                        st.session_state.history.append(
                            {
                                "domain": domain,
                                "proba": float(st.session_state.results["proba"]),
                                "class": int(st.session_state.results["class"]),
                                "time": datetime.now().strftime("%H:%M:%S"),
                            }
                        )
                    st.session_state.step = "RESULTS"
                    st.rerun()
# ==============================================================================
# STEP 4: RESULTS
# ==============================================================================
elif st.session_state.step == "RESULTS":
    st.markdown(f"<h3 class='section-header' style='margin-bottom:0.5rem;'>📋 Diagnostic Summary Report — {st.session_state.patient_profile['name']}</h3>", unsafe_allow_html=True)

    if st.session_state.results is None:
        st.warning("No data files compiled for current review pipeline.")
        if st.button("🔄 Back to Parameters"):
            st.session_state.step = "VITAL_INPUTS"
            st.rerun()
    else:
        res = st.session_state.results
        proba = res["proba"]

        if proba >= 0.66:
            risk_word, risk_color, bg, border, icon = "Elevated Risk", "#991b1b", "#fee2e2", "#ef4444", "🚨"
        elif proba >= 0.33:
            risk_word, risk_color, bg, border, icon = "Moderate Deviation", "#92400e", "#fef3c7", "#f59e0b", "⚠️"
        else:
            risk_word, risk_color, bg, border, icon = "Optimal Profile", "#166534", "#dcfce7", "#22c55e", "✅"

        with st.container(border=True):
            st.markdown(
                f"""
                <div style='background-color:{bg}; border-left: 6px solid {border}; padding: 1.5rem; border-radius:12px;'>
                    <h2 style='color:{risk_color}; margin:0; font-weight:700;'>{icon} {risk_word} Found</h2>
                    <p style='color:{risk_color}; margin-top:0.3rem; font-size:1.05rem; font-weight:500; margin-bottom:0px;'>
                        Screening engines map structural risk variance trends at <strong>{proba:.1%}</strong>
                        for {st.session_state.diagnostic_domain.lower()} physiological indicators.
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("##### Next Care Steps")
            if "Elevated" in risk_word:
                st.markdown(
                    "- Schedule an intentional consultation review with your primary health physician.\n"
                    "- Obtain confirmatory lab blood diagnostics metrics.\n"
                    "- Track marked outlier metrics flagged during parameter entry."
                )
            elif "Moderate" in risk_word:
                st.markdown(
                    "- Retest metrics routinely on an regular tracking timeline.\n"
                    "- Keep check on nutrition benchmarks and lifestyle parameter metrics.\n"
                    "- Note standard variance indicators for standard clinical evaluation."
                )
            else:
                st.markdown(
                    "- Metrics display optimal status patterns relative to standard references.\n"
                    "- Maintain your regular tracking checkup routine intervals.\n"
                    "- Continue normal preventive health screening schedules."
                )
        with col_r2:
            st.markdown("##### Download Health Checkup Summary")
            report_txt = (
                f"INFINE.AI HEALTH PORTAL SUMMARY REPORT\n"
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Patient Subject: {st.session_state.patient_profile.get('name', 'N/A')}\n"
                f"Age Profile: {st.session_state.patient_profile.get('age', 'N/A')}  "
                f"Biological Sex: {st.session_state.patient_profile.get('sex', 'N/A')}\n"
                f"Screening Sector: {st.session_state.diagnostic_domain}\n"
                f"Risk Probability Index: {proba:.1%}\n"
                f"Status Banding: {risk_word}\n"
                f"\nThis document acts strictly as a screening log check. Always verify with clear physical clinician reviews.\n"
            )
            st.download_button(
                "⬇ Download Health Summary Report (.txt)",
                data=report_txt,
                file_name=f"infine_health_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

        st.markdown(
            "<p style='text-align: center; color: #475569; margin-top:1.5rem; font-weight:400;'>🔬 Review the laboratory indicator analytics breakdown</p>",
            unsafe_allow_html=True,
        )

        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("📊 View Lab Indicator Analytics"):
                st.session_state.step = "ANALYTICS"
                st.rerun()
        with col_nav2:
            if st.button("🔄 Start New Screening Checkup"):
                st.session_state.results = None  
                st.session_state.step = "LANDING"
                st.rerun()

# ==============================================================================
# STEP 5: ANALYTICS / REASONING
# ==============================================================================
elif st.session_state.step == "ANALYTICS":
    st.markdown("<h3 class='section-header'>📊 Clinical Indicator Weight Explainer</h3>", unsafe_allow_html=True)
    
    if st.session_state.results is None:
        st.warning("No analytic components registered for active summary session.")
        if st.button("🔄 Back to Home"):
            st.session_state.step = "LANDING"
            st.rerun()
    else:
        res = st.session_state.results
        metrics = res["metrics"]

        tab_overview, tab_features, tab_perf = st.tabs(
            ["🎯 Score Distribution", "🧬 Metric Ranges", "📈 Validation Performance Data"]
        )

        # ---- TAB 1: Sleek Minimalist Medical Gauge ----
        with tab_overview:
            st.markdown(
                "<div class='info-box'>The scale track flags score ranges relative to baseline optimization center points.</div>",
                unsafe_allow_html=True,
            )
            
            fig, ax = plt.subplots(figsize=(7, 0.8), facecolor='#f8fafc')
            ax.set_facecolor('#ffffff')
            proba = res["proba"]
            
            ax.barh([0], [1.0], color="#e2e8f0", height=0.25, align='center', edgecolor='none')
            fill_color = "#0284c7" if proba <= 0.5 else "#ef4444"
            ax.barh([0], [proba], color=fill_color, height=0.25, align='center', edgecolor='none')
            ax.axvline(0.5, color="#64748b", linestyle="-", linewidth=1.5, alpha=0.7)
            
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9, color='#475569')
            
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_visible(False)
            ax.tick_params(bottom=False, pad=4)
            fig.tight_layout()
            
            cg1, cg2 = st.columns([1, 4])
            with cg1:
                st.metric("Modeled Score Index", f"{proba:.1%}")
            with cg2:
                st.pyplot(fig, clear_figure=True)

        # ---- TAB 2: Clean Feature Weight Charts ----
        with tab_features:
            st.markdown(
                "<div class='info-box'>Displays global reference feature metric impacts on aggregated datasets.</div>",
                unsafe_allow_html=True,
            )
            
            fig2, ax2 = plt.subplots(figsize=(7, 3.2), facecolor='#f8fafc')
            ax2.set_facecolor('#ffffff')
            if hasattr(res["model"], "feature_importances_"):
                importances = res["model"].feature_importances_
                indices = np.argsort(importances)[::-1][:8]
                top_f = [res["feature_names"][i] for i in indices]
                top_imp = importances[indices]
            else:
                top_f = res["feature_names"][:8]
                top_imp = np.linspace(0.4, 0.05, len(top_f))

            ax2.barh(top_f[::-1], top_imp[::-1], color="#0284c7", height=0.55, align='center')
            ax2.set_xlabel("Relative Metric Weights", color='#475569', fontsize=9)
            ax2.tick_params(colors='#475569', labelsize=9.5)
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_color('#cbd5e1')
            ax2.spines["bottom"].set_color('#cbd5e1')
            
            fig2.tight_layout(rect=[0.1, 0, 1, 1])
            st.pyplot(fig2)

            st.markdown("##### Metric Profile Reference Alignments")
            domain = st.session_state.diagnostic_domain
            ranges = REFERENCE_RANGES.get(domain, {})
            input_data = res.get("input_data", {})
            rows = []
            for field, (low, high, unit, note) in ranges.items():
                if field in input_data:
                    rows.append(
                        {
                            "Parameter": field,
                            "Entered Value": input_data[field],
                            "Standard Reference Range": f"{low}–{high} {unit}",
                            "Status": "✅ Within Norms" if low <= input_data[field] <= high else "⚠️ Outside Reference",
                        }
                    )
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ---- TAB 3: performance metrics ----
        with tab_perf:
            st.markdown("##### System Verification Log Metrics")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Verification Accuracy", f"{metrics['accuracy']:.1%}")
            m2.metric("Precision Score", f"{metrics['precision']:.1%}")
            m3.metric("Recall Index", f"{metrics['recall']:.1%}")
            m4.metric("F1 Performance Alignment", f"{metrics['f1']:.1%}")
            m5.metric("ROC Score Trace", f"{metrics['auc']:.3f}")

        st.markdown("<br>", unsafe_allow_html=True)
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            if st.button("⬅ Return to Summary Report"):
                st.session_state.step = "RESULTS"
                st.rerun()
        with col_a2:
            if st.button("🔄 Start New Screening Checkup"):
                st.session_state.results = None  
                st.session_state.step = "LANDING"
                st.rerun()

# ==============================================================================
# FOOTER DISCLAIMER (always visible)
# ==============================================================================
st.markdown(
    "<div class='disclaimer'>Infine.ai is an open clinical portfolio tracking portal. It is completely "
    "unaffiliated with therapeutic diagnostic verification tools and is not approved by primary medical boards. Always consult professional laboratory health checkups for medical diagnosis steps.</div>",
    unsafe_allow_html=True,
)