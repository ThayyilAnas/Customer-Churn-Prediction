"""
NEXUS // CHURN INTELLIGENCE - PRODUCTION STREAMLIT WEB APPLICATION
Next-Generation Telecommunications AI & Customer Retention Platform
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.pipeline import (
    clean_raw_data, engineer_features, prepare_inference_df,
    get_risk_tier, generate_retention_recommendations,
    RAW_FEATURE_COLUMNS, ALL_MODEL_FEATURES
)

# ---------------------------------------------------------
# Page Configuration & Metadata
# ---------------------------------------------------------
st.set_page_config(
    page_title="NEXUS // Churn Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Next-Generation Telecom Intelligence CSS ("Midnight Network")
# ---------------------------------------------------------
TELECOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg-midnight: #070b14;
        --bg-surface: #0b1120;
        --bg-panel: #0f172a;
        --bg-panel-hover: #131d33;
        --border-subtle: rgba(56, 189, 248, 0.12);
        --border-cyan: rgba(0, 242, 254, 0.35);
        --accent-cyan: #00f2fe;
        --accent-blue: #38bdf8;
        --accent-violet: #8b5cf6;
        --status-green: #10b981;
        --status-amber: #f59e0b;
        --status-coral: #f43f5e;
        --text-white: #f8fafc;
        --text-slate: #94a3b8;
        --text-dim: #64748b;
    }

    /* Global application background & base font */
    .stApp {
        background-color: var(--bg-midnight);
        color: var(--text-white);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Header & Network SVG Banner */
    .telecom-hero {
        position: relative;
        background: linear-gradient(135deg, rgba(11, 17, 32, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .telecom-hero::before {
        content: "";
        position: absolute;
        top: 0; right: 0; bottom: 0; left: 0;
        background-image: radial-gradient(rgba(0, 242, 254, 0.08) 1px, transparent 1px),
                          radial-gradient(rgba(139, 92, 246, 0.05) 1px, transparent 1px);
        background-size: 24px 24px;
        background-position: 0 0, 12px 12px;
        opacity: 0.6;
        pointer-events: none;
    }
    .hero-pretitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--accent-cyan);
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-white);
        margin: 0 0 6px 0;
        line-height: 1.15;
    }
    .hero-title span {
        background: linear-gradient(90deg, #00f2fe 0%, #38bdf8 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: var(--text-slate);
        max-width: 650px;
        margin: 0;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #34d399;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
    }
    .status-pulse {
        width: 7px;
        height: 7px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }

    /* Telecom Metric Cards */
    .telecom-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        position: relative;
        transition: transform 0.15s ease, border-color 0.15s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }
    .telecom-card:hover {
        border-color: var(--border-cyan);
        transform: translateY(-1px);
    }
    .telecom-card-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-dim);
        margin-bottom: 6px;
    }
    .telecom-card-value {
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
        color: var(--text-white);
    }
    .telecom-card-sub {
        font-size: 0.75rem;
        color: var(--text-dim);
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Risk Badges */
    .badge-high {
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-low {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Section Division Headers */
    .section-header-box {
        margin: 20px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(56, 189, 248, 0.15);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-index {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--accent-cyan);
        background: rgba(0, 242, 254, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(0, 242, 254, 0.2);
    }
    .section-title {
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-white);
    }

    /* Prediction Result Panel */
    .prediction-focal-panel {
        background: linear-gradient(135deg, #0b1120 0%, #0f172a 100%);
        border: 1px solid var(--border-cyan);
        border-radius: 12px;
        padding: 24px;
        margin-top: 16px;
        margin-bottom: 24px;
        box-shadow: 0 4px 24px rgba(0, 242, 254, 0.08);
    }
    .panel-tagline {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--accent-cyan);
        margin-bottom: 6px;
    }
    .prediction-verdict {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 6px;
    }
    .prediction-subtext {
        font-size: 0.9rem;
        color: var(--text-slate);
        line-height: 1.5;
        margin-bottom: 16px;
    }

    /* Strategy Retention Card */
    .telecom-strategy-card {
        background: rgba(15, 23, 42, 0.75);
        border-left: 3px solid var(--accent-cyan);
        border-top: 1px solid var(--border-subtle);
        border-right: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 10px;
    }
    .strategy-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--accent-blue);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .strategy-desc {
        color: var(--text-slate);
        font-size: 0.88rem;
        margin: 6px 0;
    }
    .strategy-impact {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--accent-violet);
    }

    /* Streamlit widget overrides */
    div[data-testid="stSidebar"] {
        background-color: #060a12 !important;
        border-right: 1px solid var(--border-subtle);
    }
    .stButton>button {
        background: linear-gradient(90deg, #00f2fe 0%, #0ea5e9 100%) !important;
        color: #070b14 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.04em !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        box-shadow: 0 2px 12px rgba(0, 242, 254, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(0, 242, 254, 0.45) !important;
    }
    .stDownloadButton>button {
        background: #0f172a !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    .stDownloadButton>button:hover {
        background: rgba(56, 189, 248, 0.1) !important;
        border-color: #00f2fe !important;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--bg-panel) !important;
        color: var(--text-white) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
</style>
"""
st.markdown(TELECOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Production Artifacts (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_production_artifacts():
    model_path = "models/model.pkl"
    preprocessor_path = "models/preprocessor.pkl"
    metadata_path = "models/model_metadata.json"
    features_path = "config/feature_names.json"

    if not all(os.path.exists(p) for p in [model_path, preprocessor_path, metadata_path, features_path]):
        return None, None, None, None

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    with open(features_path, "r") as f:
        feature_names = json.load(f)

    return model, preprocessor, metadata, feature_names


model, preprocessor, metadata, feature_names = load_production_artifacts()

# Guard for missing model
if model is None:
    st.error("⚠️ **PRODUCTION TELEMETRY ARTIFACTS OFFLINE**")
    st.info("Execute `train.py` to regenerate production models, scalers, and configuration files.")
    st.stop()

# ---------------------------------------------------------
# Sidebar: NEXUS Platform Control
# ---------------------------------------------------------
with st.sidebar:
    # Branding header
    st.markdown(
        """
        <div style="padding: 10px 0 18px 0; border-bottom: 1px solid rgba(56, 189, 248, 0.15); margin-bottom: 16px;">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #00f2fe; letter-spacing: 0.15em;">
                ENTERPRISE TELECOM AI
            </div>
            <div style="font-size: 1.5rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em; margin-top: 2px;">
                NEXUS <span style="color: #00f2fe;">//</span> CHURN
            </div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">
                Predict. Understand. Retain.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    mode = st.radio(
        "OPERATIONAL WORKFLOW",
        [
            "👤 Single Account Risk Scoring",
            "📁 Batch Portfolio Analysis",
            "📊 AI Model Diagnostics & SHAP",
            "💡 Strategic Retention Playbook"
        ]
    )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #38bdf8; text-transform: uppercase; margin-bottom: 8px;">
            DECISION THRESHOLD
        </div>
        """,
        unsafe_allow_html=True
    )
    threshold = st.slider(
        "Classification Sensitivity",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability cut-off for flagging churn risk. Lower values prioritize Recall (capturing more churners)."
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div class="telecom-card" style="margin-top: 10px; padding: 14px;">
            <div class="telecom-card-header">ENGINE TELEMETRY</div>
            <div style="font-size: 0.85rem; color: #f8fafc; font-weight: 600; margin-bottom: 2px;">
                {metadata.get('model_name', 'Tuned XGBoost')}
            </div>
            <div style="font-size: 0.75rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                ROC-AUC: <span style="color: #00f2fe;">{metadata['test_metrics']['ROC-AUC']:.4f}</span><br>
                RECALL: <span style="color: #34d399;">{metadata['test_metrics']['Recall']:.4f}</span><br>
                PRECISION: <span style="color: #a855f7;">{metadata['test_metrics']['Precision']:.4f}</span>
            </div>
            <div style="margin-top: 8px;">
                <span class="status-badge"><span class="status-pulse"></span> SYSTEM ACTIVE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Hero Section
# ---------------------------------------------------------
st.markdown(
    """
    <div class="telecom-hero">
        <div class="hero-pretitle">
            <span class="status-pulse"></span>
            NETWORK OPERATIONS INTELLIGENCE &bull; PREDICTIVE CHURN ENGINE
        </div>
        <div class="hero-title">
            CUSTOMER CHURN <span>INTELLIGENCE</span>
        </div>
        <div class="hero-subtitle">
            Next-generation telecommunications AI platform delivering real-time account risk telemetry,
            multi-factor explainability, and automated retention interventions.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 1. SINGLE ACCOUNT RISK SCORING
# =========================================================
if mode == "👤 Single Account Risk Scoring":
    st.markdown(
        """
        <div class="section-header-box">
            <span class="section-index">SCENARIO EVALUATION</span>
            <span class="section-title">Manual Account Telemetry Assessment</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Input subscriber parameters across profile, account terms, active services, and billing telemetry.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="section-header-box" style="margin-top: 10px;">
                <span class="section-index">01</span>
                <span class="section-title">CUSTOMER PROFILE</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        gender = st.selectbox("Customer Gender", ["Female", "Male"], index=0)
        senior = st.selectbox("Senior Citizen Cohort (65+)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        partner = st.selectbox("Has Registered Partner", ["Yes", "No"], index=1)
        dependents = st.selectbox("Has Household Dependents", ["Yes", "No"], index=1)

        st.markdown(
            """
            <div class="section-header-box" style="margin-top: 24px;">
                <span class="section-index">02</span>
                <span class="section-title">ACCOUNT & CONTRACT</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        tenure = st.slider("Network Tenure (Months with Provider)", min_value=0, max_value=72, value=4, help="Cumulative duration of active service.")
        contract = st.selectbox("Contract Term Commitment", ["Month-to-month", "One year", "Two year"], index=0)
        paperless = st.selectbox("Paperless Billing Active", ["Yes", "No"], index=0)
        payment = st.selectbox(
            "Payment Method Channel",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            index=0
        )

    with col2:
        st.markdown(
            """
            <div class="section-header-box" style="margin-top: 10px;">
                <span class="section-index">03</span>
                <span class="section-title">SERVICES & NETWORK ADD-ONS</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        phone = st.selectbox("Voice / Phone Service", ["Yes", "No"], index=0)
        multiple_lines = st.selectbox("Multiple Voice Lines", ["No", "Yes", "No phone service"], index=0)
        internet = st.selectbox("Broadband Infrastructure Type", ["Fiber optic", "DSL", "No"], index=0)

        c_sec1, c_sec2 = st.columns(2)
        with c_sec1:
            security = st.selectbox("Online Cyber Security", ["No", "Yes", "No internet service"], index=0)
            backup = st.selectbox("Cloud Storage Backup", ["No", "Yes", "No internet service"], index=0)
            device = st.selectbox("Hardware Protection Plan", ["No", "Yes", "No internet service"], index=0)
        with c_sec2:
            tech_support = st.selectbox("Dedicated Tech Support", ["No", "Yes", "No internet service"], index=0)
            stream_tv = st.selectbox("IPTV / TV Streaming", ["No", "Yes", "No internet service"], index=0)
            stream_movies = st.selectbox("On-Demand Streaming Movies", ["No", "Yes", "No internet service"], index=0)

        st.markdown(
            """
            <div class="section-header-box" style="margin-top: 14px;">
                <span class="section-index">04</span>
                <span class="section-title">BILLING & FINANCIAL TELEMETRY</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        monthly_charges = st.number_input("Current Monthly Recurring Charge ($)", min_value=18.0, max_value=150.0, value=85.50, step=0.50)
        auto_est = round(float(monthly_charges * max(tenure, 1)), 2)
        total_charges = st.number_input("Cumulative Total Invoiced Charges ($)", min_value=0.0, max_value=10000.0, value=auto_est, step=10.0)

    # Format customer dictionary
    customer_dict = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': security,
        'OnlineBackup': backup,
        'DeviceProtection': device,
        'TechSupport': tech_support,
        'StreamingTV': stream_tv,
        'StreamingMovies': stream_movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    input_df = pd.DataFrame([customer_dict])

    st.markdown("###")
    if st.button("⚡ EXECUTE RISK TELEMETRY ANALYSIS", use_container_width=True):
        X_inf = prepare_inference_df(input_df, preprocessor, feature_names)
        churn_prob = float(model.predict_proba(X_inf)[0, 1])
        prediction = 1 if churn_prob >= threshold else 0
        risk_tier, _, _ = get_risk_tier(churn_prob)
        confidence = churn_prob if prediction == 1 else (1.0 - churn_prob)

        # Focal Prediction Result Panel
        is_high = (risk_tier == "High")
        is_med = (risk_tier == "Medium")
        verdict_color = "#f43f5e" if is_high else ("#f59e0b" if is_med else "#10b981")
        badge_style = f"badge-{risk_tier.lower()}"

        st.markdown(
            f"""
            <div class="prediction-focal-panel">
                <div class="panel-tagline">PREDICTION VERDICT &bull; RETENTION TELEMETRY</div>
                <div class="prediction-verdict" style="color: {verdict_color};">
                    {'HIGH CHURN RISK DETECTED' if prediction == 1 else 'RETAINED & STABLE ACCOUNT'}
                </div>
                <div class="prediction-subtext">
                    The machine learning engine identifies this subscriber as having a <strong>{risk_tier.upper()}</strong> 
                    propensity for service de-activation (<strong>{churn_prob:.1%}</strong> probability) based on submitted 
                    contract duration, service utilization patterns, and historical billing friction.
                </div>
                <div style="display: flex; gap: 12px; align-items: center;">
                    <span class="{badge_style}">RISK TIER: {risk_tier.upper()}</span>
                    <span class="status-badge" style="border-color: rgba(56, 189, 248, 0.4); color: #38bdf8; background: rgba(56, 189, 248, 0.1);">
                        MODEL CONFIDENCE: {confidence:.1%}
                    </span>
                    <span class="status-badge" style="border-color: rgba(139, 92, 246, 0.4); color: #a855f7; background: rgba(139, 92, 246, 0.1);">
                        DECISION CUT-OFF: {threshold:.2f}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Telemetry KPI Row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""
                <div class="telecom-card">
                    <div class="telecom-card-header">CHURN PROBABILITY</div>
                    <div class="telecom-card-value" style="color: {verdict_color};">{churn_prob:.1%}</div>
                    <div class="telecom-card-sub">ESTIMATED DEFECT PROBABILITY</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                f"""
                <div class="telecom-card">
                    <div class="telecom-card-header">RISK CLASSIFICATION</div>
                    <div class="telecom-card-value">{risk_tier.upper()}</div>
                    <div class="telecom-card-sub">TIER THRESHOLD: {'&ge; 0.60' if is_high else ('0.30 - 0.59' if is_med else '< 0.30')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with k3:
            st.markdown(
                f"""
                <div class="telecom-card">
                    <div class="telecom-card-header">CONTRACT BINDING</div>
                    <div class="telecom-card-value" style="font-size: 1.45rem;">{contract}</div>
                    <div class="telecom-card-sub">TENURE: {tenure} MONTHS</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with k4:
            st.markdown(
                f"""
                <div class="telecom-card">
                    <div class="telecom-card-header">MONTHLY RECURRING</div>
                    <div class="telecom-card-value" style="color: #00f2fe;">${monthly_charges:.2f}</div>
                    <div class="telecom-card-sub">TOTAL: ${total_charges:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Risk Meter & Local SHAP Signals
        g_col, s_col = st.columns([1, 1])

        with g_col:
            st.markdown(
                """
                <div class="section-header-box">
                    <span class="section-index">TELEMETRY GAUGE</span>
                    <span class="section-title">Risk Probability Gauge</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Custom dark-themed Plotly gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                number={'suffix': "%", 'font': {'size': 36, 'color': '#ffffff', 'family': 'JetBrains Mono'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#38bdf8", 'tickfont': {'color': '#94a3b8'}},
                    'bar': {'color': "#00f2fe", 'thickness': 0.22},
                    'bgcolor': "#070b14",
                    'borderwidth': 1,
                    'bordercolor': "rgba(56, 189, 248, 0.2)",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.25)'},
                        {'range': [30, 60], 'color': 'rgba(245, 158, 11, 0.25)'},
                        {'range': [60, 100], 'color': 'rgba(244, 63, 94, 0.25)'}
                    ],
                    'threshold': {
                        'line': {'color': "#f43f5e", 'width': 3},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(11, 17, 32, 0.0)',
                plot_bgcolor='rgba(11, 17, 32, 0.0)',
                height=260,
                margin=dict(l=20, r=20, t=30, b=20)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with s_col:
            st.markdown(
                """
                <div class="section-header-box">
                    <span class="section-index">AI EXPLANATION</span>
                    <span class="section-title">Top Churn Signals</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            importances = model.feature_importances_
            feat_contribs = []
            for feat_name, imp, val in zip(feature_names, importances, X_inf.iloc[0]):
                if imp > 0.015:
                    impact = imp * val
                    feat_contribs.append({'Feature': feat_name, 'Contribution': impact})

            contrib_df = pd.DataFrame(feat_contribs).sort_values(by='Contribution', key=abs, ascending=False).head(6)

            fig_signals = px.bar(
                contrib_df,
                x='Contribution',
                y='Feature',
                orientation='h',
                color='Contribution',
                color_continuous_scale=['#10b981', '#f43f5e'],
                title=None
            )
            fig_signals.update_layout(
                paper_bgcolor='rgba(11, 17, 32, 0.0)',
                plot_bgcolor='rgba(11, 17, 32, 0.0)',
                height=260,
                margin=dict(l=20, r=20, t=20, b=20),
                showlegend=False,
                xaxis=dict(gridcolor='rgba(56, 189, 248, 0.1)', tickfont=dict(color='#94a3b8')),
                yaxis=dict(tickfont=dict(color='#cbd5e1', size=11))
            )
            st.plotly_chart(fig_signals, use_container_width=True)
            st.caption("Values reflect relative statistical association with churn prediction, not proven direct causation.")

        # Actionable Business Protocols
        st.markdown(
            """
            <div class="section-header-box" style="margin-top: 24px;">
                <span class="section-index">INTERVENTION PROTOCOLS</span>
                <span class="section-title">Automated Retention Strategies</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        recommendations = generate_retention_recommendations(customer_dict, churn_prob)
        for rec in recommendations:
            st.markdown(
                f"""
                <div class="telecom-strategy-card">
                    <div class="strategy-title">
                        <span>{rec['strategy']}</span>
                        <span class="strategy-impact">{rec['impact']}</span>
                    </div>
                    <div class="strategy-desc">{rec['action']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# 2. BATCH PORTFOLIO ANALYSIS
# =========================================================
elif mode == "📁 Batch Portfolio Analysis":
    st.markdown(
        """
        <div class="section-header-box">
            <span class="section-index">PORTFOLIO TELEMETRY</span>
            <span class="section-title">Batch Customer Risk Analysis</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Upload enterprise customer cohorts to evaluate segment risk distribution and identify churn clusters.")

    # Sample CSV Download Button
    sample_csv_path = "data/sample_customers.csv"
    if os.path.exists(sample_csv_path):
        with open(sample_csv_path, "rb") as f:
            sample_bytes = f.read()
        st.download_button(
            label="📥 DOWNLOAD SAMPLE TELECOM BATCH CSV (20 CUSTOMERS)",
            data=sample_bytes,
            file_name="sample_telecom_customers.csv",
            mime="text/csv",
            help="Download pre-validated CSV template to test batch inference instantly."
        )

    uploaded_file = st.file_uploader("Upload Customer Records (CSV format)", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✓ Stream connected: Loaded {len(batch_df):,} customer records successfully.")

            # Column validation
            missing_cols = [c for c in RAW_FEATURE_COLUMNS if c not in batch_df.columns]
            if missing_cols:
                st.error(f"❌ Schema validation failed: Missing required fields: `{', '.join(missing_cols)}`")
                st.info(f"Required schema: `{', '.join(RAW_FEATURE_COLUMNS)}`")
                st.stop()

            # Execute batch prediction
            with st.spinner("Executing pipeline transformations and neural inference..."):
                X_batch = prepare_inference_df(batch_df, preprocessor, feature_names)
                batch_probs = model.predict_proba(X_batch)[:, 1]
                batch_preds = (batch_probs >= threshold).astype(int)

                results_df = batch_df.copy()
                results_df['churn_probability'] = np.round(batch_probs, 4)
                results_df['churn_prediction'] = np.where(batch_preds == 1, 'Churn Risk', 'Retained')
                results_df['risk_level'] = [get_risk_tier(p)[0] for p in batch_probs]

            # Aggregate KPIs
            total_customers = len(results_df)
            predicted_churners = int(batch_preds.sum())
            high_risk_customers = int((results_df['risk_level'] == 'High').sum())
            avg_churn_prob = float(batch_probs.mean())
            revenue_at_risk = float(results_df[results_df['churn_prediction'] == 'Churn Risk']['MonthlyCharges'].sum())

            st.markdown("###")
            st.markdown(
                """
                <div class="section-header-box">
                    <span class="section-index">PORTFOLIO KPIS</span>
                    <span class="section-title">Cohort Executive Summary</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)
            with kpi_c1:
                st.markdown(
                    f"""
                    <div class="telecom-card">
                        <div class="telecom-card-header">CUSTOMERS ANALYZED</div>
                        <div class="telecom-card-value">{total_customers:,}</div>
                        <div class="telecom-card-sub">EVALUATED COHORT</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with kpi_c2:
                st.markdown(
                    f"""
                    <div class="telecom-card">
                        <div class="telecom-card-header">HIGH-RISK ACCOUNTS</div>
                        <div class="telecom-card-value" style="color: #f43f5e;">{high_risk_customers:,}</div>
                        <div class="telecom-card-sub">PROBABILITY &ge; 60%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with kpi_c3:
                st.markdown(
                    f"""
                    <div class="telecom-card">
                        <div class="telecom-card-header">PREDICTED CHURNERS</div>
                        <div class="telecom-card-value" style="color: #fb7185;">{predicted_churners:,}</div>
                        <div class="telecom-card-sub">AT THRESHOLD {threshold:.2f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with kpi_c4:
                st.markdown(
                    f"""
                    <div class="telecom-card">
                        <div class="telecom-card-header">AVG CHURN PROB</div>
                        <div class="telecom-card-value" style="color: #38bdf8;">{avg_churn_prob:.1%}</div>
                        <div class="telecom-card-sub">COHORT MEAN</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with kpi_c5:
                st.markdown(
                    f"""
                    <div class="telecom-card">
                        <div class="telecom-card-header">REVENUE AT RISK</div>
                        <div class="telecom-card-value" style="color: #f43f5e;">${revenue_at_risk:,.2f}</div>
                        <div class="telecom-card-sub">MONTHLY RECURRING (MRR)</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Portfolio Visualizations
            bc1, bc2 = st.columns(2)
            with bc1:
                tier_counts = results_df['risk_level'].value_counts().reset_index()
                tier_counts.columns = ['Risk Tier', 'Count']
                fig_pie = px.pie(
                    tier_counts,
                    names='Risk Tier',
                    values='Count',
                    hole=0.55,
                    color='Risk Tier',
                    color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#f43f5e'},
                    title="Risk Tier Distribution"
                )
                fig_pie.update_layout(
                    paper_bgcolor='rgba(11, 17, 32, 0.0)',
                    plot_bgcolor='rgba(11, 17, 32, 0.0)',
                    font=dict(color='#cbd5e1', family='Inter'),
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with bc2:
                fig_hist = px.histogram(
                    results_df,
                    x='churn_probability',
                    nbins=25,
                    color='risk_level',
                    color_discrete_map={'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#f43f5e'},
                    title="Churn Probability Density Distribution"
                )
                fig_hist.add_vline(x=threshold, line_dash="dash", line_color="#00f2fe", annotation_text="Cut-off", annotation_font_color="#00f2fe")
                fig_hist.update_layout(
                    paper_bgcolor='rgba(11, 17, 32, 0.0)',
                    plot_bgcolor='rgba(11, 17, 32, 0.0)',
                    font=dict(color='#cbd5e1', family='Inter'),
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(gridcolor='rgba(56, 189, 248, 0.1)'),
                    yaxis=dict(gridcolor='rgba(56, 189, 248, 0.1)')
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # High-Risk Prioritization Table
            st.markdown(
                """
                <div class="section-header-box" style="margin-top: 20px;">
                    <span class="section-index">PRIORITY ACCOUNTS</span>
                    <span class="section-title">Top High-Risk Accounts Requiring Immediate Outreach</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            r_filter = st.selectbox("Filter Display by Risk Classification:", ["All Accounts", "High Risk Only", "Medium Risk Only", "Low Risk Only"], index=1)
            filtered = results_df.copy()
            if r_filter == "High Risk Only":
                filtered = filtered[filtered['risk_level'] == 'High']
            elif r_filter == "Medium Risk Only":
                filtered = filtered[filtered['risk_level'] == 'Medium']
            elif r_filter == "Low Risk Only":
                filtered = filtered[filtered['risk_level'] == 'Low']

            disp_cols = [c for c in ['customerID', 'churn_prediction', 'churn_probability', 'risk_level', 'tenure', 'Contract', 'MonthlyCharges', 'PaymentMethod', 'InternetService'] if c in filtered.columns]
            st.dataframe(
                filtered[disp_cols].sort_values(by='churn_probability', ascending=False),
                use_container_width=True,
                height=340
            )

            # Download Predictions CSV
            csv_export = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 EXPORT ENRICHED TELEMETRY CSV (WITH CHURN PROBABILITIES)",
                data=csv_export,
                file_name="telecom_churn_predictions_export.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Error during batch ingestion: {str(e)}")


# =========================================================
# 3. AI MODEL DIAGNOSTICS & SHAP
# =========================================================
elif mode == "📊 AI Model Diagnostics & SHAP":
    st.markdown(
        """
        <div class="section-header-box">
            <span class="section-index">MODEL DIAGNOSTICS</span>
            <span class="section-title">Cross-Model Benchmark & Game-Theoretic SHAP Telemetry</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("Deep technical inspection into classification curves, error matrices, and SHAP explainability.")

    # Benchmark table
    st.markdown("#### 🏆 Test Set Model Benchmark (Holdout: 1,409 Accounts)")
    bench_data = metadata.get('benchmark_comparison', {})
    b_df = pd.DataFrame(bench_data).T.reset_index().rename(columns={'index': 'Architecture'})
    st.dataframe(b_df.sort_values(by='ROC-AUC', ascending=False), use_container_width=True)

    t1, t2, t3 = st.tabs(["📈 ROC & PR Curves", "🧩 Confusion Matrix", "🐝 SHAP Global Explainability"])

    with t1:
        rc1, rc2 = st.columns(2)
        with rc1:
            roc_path = "outputs/evaluation/roc_curves.png"
            if os.path.exists(roc_path):
                st.image(roc_path, caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)
        with rc2:
            pr_path = "outputs/evaluation/pr_curves.png"
            if os.path.exists(pr_path):
                st.image(pr_path, caption="Precision-Recall Curve Comparison (Imbalanced Focus)", use_container_width=True)

    with t2:
        cm_c1, cm_c2 = st.columns(2)
        with cm_c1:
            cm_path = "outputs/evaluation/confusion_matrix.png"
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix (Tuned XGBoost at 0.50 Threshold)", use_container_width=True)
        with cm_c2:
            st.markdown("#### Operational Trade-Off Analysis")
            st.markdown(
                """
                - **True Negatives (753)**: Correctly classified loyal accounts; no unnecessary retention discount spend.
                - **False Positives (282)**: Retained customers flagged for retention. Minor cost of proactive outreach.
                - **False Negatives (75)**: Churners missed by the engine. Represents permanent customer loss.
                - **True Positives (299)**: Churners successfully intercepted (**80% Recall**).
                """
            )

    with t3:
        st.markdown("#### Global Feature Impact via SHapley Additive exPlanations")
        sh1, sh2 = st.columns(2)
        with sh1:
            sh_sum = "outputs/shap/shap_summary.png"
            if os.path.exists(sh_sum):
                st.image(sh_sum, caption="SHAP Summary Beeswarm Plot", use_container_width=True)
        with sh2:
            sh_bar = "outputs/shap/shap_bar.png"
            if os.path.exists(sh_bar):
                st.image(sh_bar, caption="SHAP Global Mean Absolute Importance", use_container_width=True)


# =========================================================
# 4. STRATEGIC RETENTION PLAYBOOK
# =========================================================
elif mode == "💡 Strategic Retention Playbook":
    st.markdown(
        """
        <div class="section-header-box">
            <span class="section-index">STRATEGIC PLAYBOOK</span>
            <span class="section-title">Data-Driven Retention Architecture</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    ed1, ed2 = st.columns(2)
    with ed1:
        f1 = "outputs/figures/churn_by_contract.png"
        if os.path.exists(f1):
            st.image(f1, caption="Churn Rates by Contract Type (%)", use_container_width=True)
        f2 = "outputs/figures/churn_vs_tenure.png"
        if os.path.exists(f2):
            st.image(f2, caption="Customer Tenure Distribution vs Churn Status", use_container_width=True)

    with ed2:
        f3 = "outputs/figures/churn_vs_monthly_charges.png"
        if os.path.exists(f3):
            st.image(f3, caption="Monthly Recurring Charges Density by Churn Status", use_container_width=True)
        f4 = "outputs/figures/feature_importance.png"
        if os.path.exists(f4):
            st.image(f4, caption="Top 15 Predictive Signals by Model Gain", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🎯 Enterprise Retention Architecture & Intervention Pillars")
    st.markdown(
        """
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 12px;">
            <div class="telecom-card">
                <div class="telecom-card-header" style="color: #00f2fe;">PILLAR 01 &bull; CONTRACT TRANSITION</div>
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;">Month-to-Month Contract Conversion</div>
                <div style="font-size: 0.88rem; color: #94a3b8;">
                    Subscribers on month-to-month terms exhibit an alarming <strong>42.7% churn rate</strong>, compared to 
                    11.3% for 1-year and 2.8% for 2-year contracts. Deploy targeted 15% discount incentives to lock in 12-month commitments.
                </div>
            </div>
            <div class="telecom-card">
                <div class="telecom-card-header" style="color: #8b5cf6;">PILLAR 02 &bull; EARLY ONBOARDING SHIELD</div>
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;">Tenure &lt; 12 Months Intervention</div>
                <div style="font-size: 0.88rem; color: #94a3b8;">
                    Over 50% of customer defections occur within the first 12 months of service. Establish automated customer 
                    success checkpoints at Days 14, 45, and Month 6 with dedicated technician dispatch support.
                </div>
            </div>
            <div class="telecom-card">
                <div class="telecom-card-header" style="color: #10b981;">PILLAR 03 &bull; AUTOPAY ENROLLMENT</div>
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;">Frictionless Automated Billing</div>
                <div style="font-size: 0.88rem; color: #94a3b8;">
                    Subscribers paying via electronic checks churn at double the frequency of automated credit card or bank transfer 
                    accounts. Incentivize autopay enrollment with a one-time $10 account credit.
                </div>
            </div>
            <div class="telecom-card">
                <div class="telecom-card-header" style="color: #f59e0b;">PILLAR 04 &bull; PROTECTIVE BUNDLING</div>
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 6px;">Tech Support & Cyber Security Add-ons</div>
                <div style="font-size: 0.88rem; color: #94a3b8;">
                    Broadband accounts with active Tech Support and Online Security exhibit superior retention metrics due to 
                    higher perceived switching barriers. Provide 3-month complimentary trials to new subscribers.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
