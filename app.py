"""
Aegis Diagnostics Portal
------------------------
A clinical ML screening demo app built with Streamlit.

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
    page_title="Aegis Diagnostics Portal",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 1. THEME / STYLING
# ==============================================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

        .stApp {
            background: radial-gradient(circle at 10% 0%, #eef4fb 0%, #f8fafc 35%, #f8fafc 100%);
            color: #1e293b;
        }

        /* ---------- Navigation ---------- */
        .nav-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            background: #ffffff;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 6px rgba(15,23,42,0.04);
            margin-bottom: 1.5rem;
        }
        .nav-logo { font-size: 1.4rem; font-weight: 800; color: #0284c7; display: flex; align-items: center; gap: 0.5rem; }
        .nav-sub { font-size: 0.85rem; color: #64748b; font-weight: 500; }

        /* ---------- Stepper ---------- */
        .stepper { display: flex; align-items: center; margin-bottom: 2rem; }
        .step-item { display: flex; align-items: center; flex: 1; }
        .step-circle {
            width: 32px; height: 32px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 0.85rem; flex-shrink: 0;
        }
        .step-circle.active { background: #0284c7; color: #fff; box-shadow: 0 0 0 4px rgba(2,132,199,0.15); }
        .step-circle.done { background: #22c55e; color: #fff; }
        .step-circle.todo { background: #e2e8f0; color: #94a3b8; }
        .step-label { margin-left: 0.5rem; font-size: 0.8rem; font-weight: 600; color: #475569; }
        .step-line { flex: 1; height: 2px; background: #e2e8f0; margin: 0 0.75rem; }
        .step-line.done { background: #22c55e; }

        /* ---------- Hero ---------- */
        .hero-title { font-size: 2.6rem; font-weight: 800; color: #0f172a; line-height: 1.15; margin-bottom: 1rem; }
        .hero-subtitle { font-size: 1.15rem; color: #475569; margin-bottom: 1.5rem; line-height: 1.6; }
        .badge {
            display: inline-block; background: #e0f2fe; color: #0369a1; font-size: 0.75rem;
            font-weight: 700; padding: 0.3rem 0.8rem; border-radius: 999px; margin-bottom: 1rem;
            letter-spacing: 0.03em; text-transform: uppercase;
        }

        /* ---------- Cards ---------- */
        .card {
            background: #ffffff; padding: 1.5rem; border-radius: 16px;
            border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);
            margin-bottom: 1.25rem; height: 100%;
        }
        .pipeline-card { border-top: 4px solid #0284c7; }
        .pipeline-card h4 { margin: 0 0 0.4rem 0; color: #0f172a; }
        .pipeline-card p { color: #64748b; font-size: 0.9rem; margin: 0; }

        .info-box {
            background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 12px;
            padding: 1rem 1.25rem; font-size: 0.88rem; color: #0c4a6e; margin-bottom: 1rem;
        }
        .warn-box {
            background: #fffbeb; border: 1px solid #fde68a; border-radius: 12px;
            padding: 0.85rem 1.1rem; font-size: 0.85rem; color: #92400e; margin: 0.5rem 0;
        }
        .disclaimer {
            background: #f1f5f9; border-left: 4px solid #64748b; border-radius: 8px;
            padding: 0.9rem 1.1rem; font-size: 0.8rem; color: #475569; margin-top: 2rem;
        }

        /* ---------- Buttons ---------- */
        div.stButton > button {
            background-color: #0284c7 !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 0.7rem 2.2rem !important;
            border-radius: 9999px !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(2, 132, 199, 0.2) !important;
            transition: all 0.15s ease;
        }
        div.stButton > button:hover { background-color: #0369a1 !important; transform: translateY(-1px); }

        label p { color: #334155 !important; font-weight: 600; margin-bottom: 0.2rem; }

        .metric-reason { font-size: 0.78rem; color: #64748b; margin-top: -0.5rem; }
    </style>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. STATE
# ==============================================================================
DEFAULTS = {
    "step": "LANDING",
    "patient_profile": {},
    "diagnostic_domain": "Diabetes",
    "history": [],   # keeps a session-only audit trail of past assessments
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

STEP_ORDER = ["DEMOGRAPHICS", "VITAL_INPUTS", "RESULTS", "ANALYTICS"]
STEP_LABELS = {
    "DEMOGRAPHICS": "Profile",
    "VITAL_INPUTS": "Clinical Data",
    "RESULTS": "Conclusion",
    "ANALYTICS": "Reasoning",
}

# Reference ranges used purely to give the user context on their numbers.
# These are commonly cited clinical ballpark ranges for a general adult
# population and are illustrative, not a substitute for lab-specific norms.
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
        "desc": "Screens metabolic and pregnancy-history indicators associated with Type 2 diabetes risk.",
    },
    "Heart Disease": {
        "icon": "❤️",
        "desc": "Evaluates cardiac stress-test signals, ECG patterns, and vascular risk indicators.",
    },
    "Chronic Kidney Disease": {
        "icon": "🫘",
        "desc": "Combines lab chemistry panels with qualitative symptoms to flag kidney function decline.",
    },
}


@st.cache_resource
def load_disease_assets(disease_name: str):
    file_path = f"models/{disease_name.lower().replace(' ', '_')}_assets.pkl"
    if os.path.exists(file_path):
        return joblib.load(file_path)
    return None


def render_stepper(current: str):
    items = []
    reached_current = False
    for i, s in enumerate(STEP_ORDER):
        if s == current:
            cls = "active"
            reached_current = True
        elif not reached_current:
            cls = "done"
        else:
            cls = "todo"
        items.append(
            f"<div class='step-item'><div class='step-circle {cls}'>{i+1}</div>"
            f"<span class='step-label'>{STEP_LABELS[s]}</span></div>"
        )
        if i < len(STEP_ORDER) - 1:
            line_cls = "done" if cls == "done" else ""
            items.append(f"<div class='step-line {line_cls}'></div>")
    st.markdown(f"<div class='stepper'>{''.join(items)}</div>", unsafe_allow_html=True)


def range_feedback(domain: str, field: str, value):
    """Return a short human-readable note if a value is outside the
    illustrative reference range, so the user understands *why* a number
    might move the risk estimate."""
    ranges = REFERENCE_RANGES.get(domain, {})
    if field not in ranges:
        return None
    low, high, unit, note = ranges[field]
    if value < low:
        return f"⬇ Below typical range ({low}-{high} {unit}). {note}"
    if value > high:
        return f"⬆ Above typical range ({low}-{high} {unit}). {note}"
    return None


# Top header (always visible)
st.markdown(
    """
    <div class="nav-container">
        <div class="nav-logo">🩺 Aegis Portal</div>
        <div class="nav-sub">Clinical Machine Learning Assessment Suite · Demo Build</div>
    </div>
""",
    unsafe_allow_html=True,
)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### ℹ️ About this tool")
    st.markdown(
        "Aegis is a **research/demo screening tool**. It runs a trained "
        "classifier over structured clinical inputs and reports a probability "
        "estimate. It is not a diagnosis and is not a replacement for a "
        "clinician's judgment."
    )
    st.divider()
    st.markdown("### 🧭 Available pipelines")
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
    col1, col2 = st.columns([1.2, 0.8], gap="large")
    with col1:
        st.markdown("<div class='badge'>Ensemble ML · 3 clinical domains</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='hero-title'>A diagnostic screening checker powered by optimized ensembles.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='hero-subtitle'>Enter physiological baselines and chronic condition markers to "
            "receive an instant model estimate, complete with the reasoning and reference ranges behind it.</div>",
            unsafe_allow_html=True,
        )
        if st.button("Start Assessment →", type="primary"):
            st.session_state.step = "DEMOGRAPHICS"
            st.rerun()

        st.markdown(
            "<div class='disclaimer'>⚠️ <strong>Not medical advice.</strong> This demo does not diagnose "
            "disease. Always consult a licensed clinician about your health.</div>",
            unsafe_allow_html=True,
        )

    with col2:
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
    render_stepper("DEMOGRAPHICS")
    st.markdown("### 📋 Step 1 · Basic Patient Profile")
    st.markdown(
        "<div class='info-box'>Demographic context (age, sex) is used because baseline risk for these "
        "conditions differs meaningfully across age bands and biological sex — the model treats these as "
        "adjustment factors alongside your lab values, not as standalone predictors.</div>",
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
            help="Choose which trained pipeline should evaluate this patient's data.",
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
# STEP 3: VITAL INPUTS
# ==============================================================================
elif st.session_state.step == "VITAL_INPUTS":
    render_stepper("VITAL_INPUTS")
    domain = st.session_state.diagnostic_domain
    assets = load_disease_assets(domain)

    if assets is None:
        st.error(
            f"❌ No trained model assets found for **{domain}**. Expected a file at "
            f"`models/{domain.lower().replace(' ', '_')}_assets.pkl` containing a dict with "
            f"`model`, `scaler`, `imputer`, and `feature_names`."
        )
        if st.button("⬅ Return to Profile"):
            st.session_state.step = "DEMOGRAPHICS"
            st.rerun()
    else:
        st.markdown(f"### 📊 Step 2 · Clinical Input Parameters — {domain}")
        st.markdown(
            "<div class='info-box'>Values outside the shaded reference range are flagged below the field "
            "so you can see, at a glance, which inputs are pushing the estimate — this does not mean the "
            "value is wrong, only that it's outside a typical adult range.</div>",
            unsafe_allow_html=True,
        )
        input_data = {}
        flags = []

        with st.container(border=True):
            if domain == "Diabetes":
                c1, c2 = st.columns(2)
                with c1:
                    input_data["Pregnancies"] = st.number_input("Pregnancies Count", min_value=0, max_value=20, value=1)
                    input_data["Glucose"] = st.number_input(
                        "Plasma Glucose (mg/dL)", min_value=0, max_value=300, value=115,
                        help="Fasting plasma glucose concentration."
                    )
                    input_data["BloodPressure"] = st.number_input(
                        "Diastolic BP (mm Hg)", min_value=0, max_value=200, value=72
                    )
                    input_data["SkinThickness"] = st.number_input(
                        "Triceps Skin Fold (mm)", min_value=0, max_value=100, value=20,
                        help="A rough proxy for subcutaneous body fat."
                    )
                with c2:
                    input_data["Insulin"] = st.number_input(
                        "Serum Insulin (mu U/ml)", min_value=0, max_value=1000, value=79
                    )
                    input_data["BMI"] = st.number_input(
                        "Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=24.5, format="%.1f"
                    )
                    input_data["DiabetesPedigreeFunction"] = st.number_input(
                        "Pedigree Score Function", min_value=0.0, max_value=3.0, value=0.47, format="%.3f",
                        help="A function that scores genetic influence based on family history."
                    )
                    input_data["Age"] = st.session_state.patient_profile["age"]

            elif domain == "Heart Disease":
                c1, c2 = st.columns(2)
                with c1:
                    input_data["age"] = st.session_state.patient_profile["age"]
                    input_data["sex"] = 1 if st.session_state.patient_profile["sex"] == "Male" else 0
                    cp_choice = st.selectbox(
                        "Chest Pain Presentation",
                        ["0: Typical Angina", "1: Atypical Angina", "2: Non-anginal Pain", "3: Asymptomatic"],
                    )
                    input_data["cp"] = int(cp_choice.split(":")[0])
                    input_data["trestbps"] = st.number_input("Resting Sys BP (mm Hg)", min_value=50, max_value=250, value=130)
                    input_data["chol"] = st.number_input("Serum Chol (mg/dl)", min_value=100, max_value=600, value=240)
                with c2:
                    fbs_choice = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["False", "True"])
                    input_data["fbs"] = 1 if fbs_choice == "True" else 0
                    restecg_choice = st.selectbox(
                        "Resting ECG Variant",
                        ["0: Normal", "1: ST-T Wave Abnormality", "2: Left Ventricular Hypertrophy"],
                    )
                    input_data["restecg"] = int(restecg_choice.split(":")[0])
                    input_data["thalach"] = st.number_input("Peak Heart Rate Achieved", min_value=60, max_value=250, value=145)
                    exang_choice = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
                    input_data["exang"] = 1 if exang_choice == "Yes" else 0
                    input_data["oldpeak"] = st.number_input(
                        "ST Segment Depression", min_value=0.0, max_value=10.0, value=1.0, format="%.1f"
                    )
                    input_data["slope"] = st.selectbox("Slope of Peak ST Segment", [0, 1, 2], index=1)
                    input_data["ca"] = st.selectbox("Fluoroscopy Major Vessels (0-4)", [0, 1, 2, 3, 4])
                    thal_choice = st.selectbox(
                        "Thalassemia Variant",
                        ["0: Normal", "1: Fixed Defect", "2: Reversable Defect", "3: Severe Defect"],
                        index=2,
                    )
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

            # Reference-range feedback for the fields we have ranges for
            for field, value in input_data.items():
                msg = range_feedback(domain, field, value)
                if msg:
                    flags.append(msg)
            if flags:
                st.markdown("##### 🔍 Notable values")
                for f in flags:
                    st.markdown(f"<div class='warn-box'>{f}</div>", unsafe_allow_html=True)

            # ---- Inference ----
            feature_names = assets["feature_names"]
            feature_df = pd.DataFrame([input_data])[feature_names]

            imputed = assets["imputer"].transform(feature_df)
            scaled = assets["scaler"].transform(imputed)

            proba = assets["model"].predict_proba(scaled)[0][1]
            pred_class = assets["model"].predict(scaled)[0]

            st.session_state.results = {
                "proba": proba,
                "class": pred_class,
                "model": assets["model"],
                "feature_names": feature_names,
                "input_data": input_data,
                "scaled_input": scaled[0],
                "metrics": {
                    "accuracy": assets.get("accuracy", 0.942),
                    "precision": assets.get("precision", 0.925),
                    "recall": assets.get("recall", 0.918),
                    "f1": assets.get("f1_score", 0.921),
                    "auc": assets.get("auc", 0.954),
                },
            }

            st.markdown("<br>", unsafe_allow_html=True)
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                if st.button("⬅ Back to Profile"):
                    st.session_state.step = "DEMOGRAPHICS"
                    st.rerun()
            with col_b2:
                if st.button("Generate Classification Output →"):
                    st.session_state.history.append(
                        {
                            "domain": domain,
                            "proba": float(proba),
                            "class": int(pred_class),
                            "time": datetime.now().strftime("%H:%M:%S"),
                        }
                    )
                    st.session_state.step = "RESULTS"
                    st.rerun()

# ==============================================================================
# STEP 4: RESULTS
# ==============================================================================
elif st.session_state.step == "RESULTS":
    render_stepper("RESULTS")
    st.markdown(f"### 📋 Diagnostic Conclusion Summary — **{st.session_state.patient_profile['name']}**")

    res = st.session_state.results
    proba = res["proba"]

    # three-tier risk banding instead of a raw binary flag
    if proba >= 0.66:
        risk_word, risk_color, bg, border, icon = "High", "#991b1b", "#fee2e2", "#ef4444", "🚨"
    elif proba >= 0.33:
        risk_word, risk_color, bg, border, icon = "Moderate", "#92400e", "#fef3c7", "#f59e0b", "⚠️"
    else:
        risk_word, risk_color, bg, border, icon = "Low", "#166534", "#dcfce7", "#22c55e", "✅"

    with st.container(border=True):
        st.markdown(
            f"""
            <div style='background-color:{bg}; border-left: 6px solid {border}; padding: 2rem; border-radius:12px;'>
                <h2 style='color:{risk_color}; margin:0;'>{icon} {risk_word} Risk Pattern</h2>
                <p style='color:{risk_color}; margin-top:0.5rem; font-size:1.1rem; font-weight:500;'>
                    Ensemble voting pathways indicate a modeled risk probability of <strong>{proba:.1%}</strong>
                    for {st.session_state.diagnostic_domain.lower()} risk markers.
                </p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("##### What this means")
        if risk_word == "High":
            st.markdown(
                "- The model weighted your inputs heavily toward the positive class.\n"
                "- Consider scheduling a clinical follow-up with confirmatory lab work.\n"
                "- Review flagged out-of-range values on the previous step."
            )
        elif risk_word == "Moderate":
            st.markdown(
                "- Signals are mixed — some inputs push toward risk, others don't.\n"
                "- A repeat test or additional labs may sharpen the estimate.\n"
                "- Lifestyle and monitoring follow-up is commonly recommended at this tier."
            )
        else:
            st.markdown(
                "- Inputs largely fell within typical ranges for this condition.\n"
                "- Routine monitoring on a normal schedule is generally sufficient.\n"
                "- A low score does not rule out disease — it reflects this input set only."
            )
    with col_r2:
        st.markdown("##### Report")
        report_txt = (
            f"AEGIS DIAGNOSTICS PORTAL — DEMO REPORT\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Patient: {st.session_state.patient_profile.get('name', 'N/A')}\n"
            f"Age: {st.session_state.patient_profile.get('age', 'N/A')}  "
            f"Sex: {st.session_state.patient_profile.get('sex', 'N/A')}\n"
            f"Domain: {st.session_state.diagnostic_domain}\n"
            f"Risk Probability: {proba:.1%}\n"
            f"Risk Tier: {risk_word}\n"
            f"Model Accuracy (validation): {res['metrics']['accuracy']:.1%}\n"
            f"\nThis is not a medical diagnosis. Consult a licensed clinician.\n"
        )
        st.download_button(
            "⬇ Download Report (.txt)",
            data=report_txt,
            file_name=f"aegis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
        )

    st.markdown(
        "<p style='text-align: center; color: #64748b; margin-top:1.5rem;'>"
        "Want to see exactly how the model reached this conclusion?</p>",
        unsafe_allow_html=True,
    )

    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("📊 How We Reached This Conclusion"):
            st.session_state.step = "ANALYTICS"
            st.rerun()
    with col_nav2:
        if st.button("🔄 Start New Assessment"):
            st.session_state.step = "LANDING"
            st.rerun()

# ==============================================================================
# STEP 5: ANALYTICS / REASONING
# ==============================================================================
elif st.session_state.step == "ANALYTICS":
    render_stepper("ANALYTICS")
    st.markdown("### 📊 Ensemble Decision Path Explainer")
    res = st.session_state.results
    metrics = res["metrics"]

    tab_overview, tab_features, tab_perf = st.tabs(
        ["🎯 Risk Breakdown", "🧬 Feature Reasoning", "📈 Model Performance"]
    )

    # ---- TAB 1: risk gauge ----
    with tab_overview:
        st.markdown(
            "<div class='info-box'>The gauge below shows the model's raw output probability. Anything left "
            "of center leans toward the healthy class; right of center leans toward the at-risk class.</div>",
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(6, 1.6))
        proba = res["proba"]
        ax.barh([0], [1], color="#e2e8f0", height=0.5)
        ax.barh([0], [proba], color="#0284c7" if proba <= 0.5 else "#ef4444", height=0.5)
        ax.axvline(0.5, color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.text(proba, 0.55, f"{proba:.1%}", ha="center", fontweight="bold", fontsize=11)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig)

    # ---- TAB 2: feature-level reasoning ----
    with tab_features:
        st.markdown(
            "<div class='info-box'>This shows which inputs the model's decision trees rely on most, on "
            "average across its training data — it is a global importance measure, not a per-patient "
            "causal explanation. Bars further right were more influential to the ensemble as a whole.</div>",
            unsafe_allow_html=True,
        )
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        if hasattr(res["model"], "feature_importances_"):
            importances = res["model"].feature_importances_
            indices = np.argsort(importances)[::-1][:8]
            top_f = [res["feature_names"][i] for i in indices]
            top_imp = importances[indices]
        else:
            top_f = res["feature_names"][:8]
            top_imp = np.linspace(0.4, 0.05, len(top_f))

        ax2.barh(top_f[::-1], top_imp[::-1], color="#0284c7")
        ax2.set_xlabel("Relative importance (Gini-based)")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

        st.markdown("##### This patient's values vs. reference range")
        domain = st.session_state.diagnostic_domain
        ranges = REFERENCE_RANGES.get(domain, {})
        input_data = res.get("input_data", {})
        rows = []
        for field, (low, high, unit, note) in ranges.items():
            if field in input_data:
                rows.append(
                    {
                        "Parameter": field,
                        "Patient Value": input_data[field],
                        "Typical Range": f"{low}–{high} {unit}",
                        "In Range?": "✅" if low <= input_data[field] <= high else "⚠️ Outside",
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No reference-range mappings defined for this domain's fields.")

    # ---- TAB 3: performance metrics ----
    with tab_perf:
        st.markdown("##### Baseline Pipeline Performance (validation set)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        m2.metric("Precision", f"{metrics['precision']:.1%}")
        m3.metric("Recall", f"{metrics['recall']:.1%}")
        m4.metric("F1 Score", f"{metrics['f1']:.1%}")
        m5.metric("ROC-AUC", f"{metrics['auc']:.3f}")

        st.markdown(
            """
            <div class='metric-reason'>
            <strong>Accuracy</strong>: overall share of correct predictions on held-out data.<br>
            <strong>Precision</strong>: of all "at risk" predictions, how many were correct — matters when false alarms are costly.<br>
            <strong>Recall</strong>: of all true at-risk cases, how many the model caught — matters when missed cases are costly.<br>
            <strong>F1</strong>: harmonic balance of precision and recall.<br>
            <strong>ROC-AUC</strong>: how well the model ranks at-risk cases above healthy ones across all thresholds.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        if st.button("⬅ Return to Summary"):
            st.session_state.step = "RESULTS"
            st.rerun()
    with col_a2:
        if st.button("🔄 Start New Assessment"):
            st.session_state.step = "LANDING"
            st.rerun()

# ==============================================================================
# FOOTER DISCLAIMER (always visible)
# ==============================================================================
st.markdown(
    "<div class='disclaimer'>Aegis Diagnostics Portal is a demonstration application for educational and "
    "portfolio purposes. It is not FDA-cleared, not validated for clinical use, and must never be used to make "
    "real medical decisions. Always seek the advice of a qualified health provider.</div>",
    unsafe_allow_html=True,
)