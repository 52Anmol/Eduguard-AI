import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time


st.set_page_config(
    page_title="EduGuard AI",
    page_icon="🚀",
    layout="wide"
)


if "page" not in st.session_state:
    st.session_state.page = "dashboard"

if "risk_score" not in st.session_state:
    st.session_state.risk_score = None


# Styling

st.markdown("""
<style>

/* ===== GLOBAL ===== */
.glow-card {
    padding: 20px;
    border-radius: 15px;
    transition: 0.3s ease;
}

.result-card {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 10px;
    border-left: 4px solid #0ea5e9;
}

/* ===== DARK MODE ===== */
[data-theme="dark"] .stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

[data-theme="dark"] .glow-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    color: white;
}

[data-theme="dark"] .result-card {
    background: rgba(255,255,255,0.05);
    color: white;
}

[data-theme="dark"] .stButton>button {
    background: #0ea5e9;
    color: white;
}

/* ===== LIGHT MODE ===== */
[data-theme="light"] .stApp {
    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
}

[data-theme="light"] .glow-card {
    background: rgba(255,255,255,0.9);
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    color: #1e293b;
}

[data-theme="light"] .result-card {
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    color: #1e293b;
}

[data-theme="light"] .stButton>button {
    background: #2563eb;
    color: white;
}

.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.2s ease;
}

</style>
""", unsafe_allow_html=True)


df = pd.read_csv("StudentPerformanceFactors.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

score_column = "exam_score"
median_score = df[score_column].median()
df["risk"] = df[score_column].apply(lambda x: 1 if x < median_score else 0)


# Load Model

model = joblib.load("trained_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ==========================================================
# DASHBOARD PAGE
# ==========================================================
if st.session_state.page == "dashboard":

    col1, col2 = st.columns([8,2])

    with col1:
        st.title("🚀 EduGuard AI - Institutional Dashboard")

    with col2:
        if st.button("🔍 AI Prediction"):
            st.session_state.page = "prediction"
            st.rerun()

    st.divider()

    total = len(df)
    at_risk = int(df["risk"].sum())
    safe = total - at_risk
    risk_percent = round((at_risk / total) * 100, 2)
    avg_score = round(df[score_column].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    def kpi(title, value):
        st.markdown(f"""
        <div class="glow-card">
            <div>{title}</div>
            <div style="font-size:24px;color:#38bdf8;"><b>{value}</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        kpi("Total Students", total)
    with col2:
        kpi("At Risk %", f"{risk_percent}%")
    with col3:
        kpi("Average Score", avg_score)
    with col4:
        kpi("Model Status", "Active")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            values=[safe, at_risk],
            names=["Safe", "At Risk"],
            hole=0.6
        )
        fig_pie.update_layout(template="plotly_dark", title="Risk Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_hist = px.histogram(df, x=score_column, nbins=30)
        fig_hist.update_layout(template="plotly_dark", title="Score Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

# ==========================================================
# PREDICTION PAGE
# ==========================================================
elif st.session_state.page == "prediction":

    col1, col2 = st.columns([8,2])

    with col1:
        st.title("🔍 AI Risk Assessment Engine")

    with col2:
        if st.button("⬅ Back"):
            st.session_state.page = "dashboard"
            st.session_state.risk_score = None
            st.rerun()

    st.divider()

    # -----------------------------
    # Validation Function
    # -----------------------------
    def validate(value, min_val, max_val):
        try:
            val = int(value)
        except:
            return min_val, False
        if val < min_val or val > max_val:
            return max(min(val, max_val), min_val), False
        return val, True

    col1, col2 = st.columns(2)

    with col1:
        hours_raw = st.text_input("Hours Studied (0–60)", "0")
        attendance_raw = st.text_input("Attendance % (0–100)", "0")

    with col2:
        sleep_raw = st.text_input("Sleep Hours (0–12)", "0")
        score_raw = st.text_input("Previous Score (0–100)", "0")

    hours_studied, h_valid = validate(hours_raw, 0, 60)
    attendance, a_valid = validate(attendance_raw, 0, 100)
    sleep_hours, s_valid = validate(sleep_raw, 0, 12)
    previous_score, p_valid = validate(score_raw, 0, 100)

    # -----------------------------
    # Prediction Button
    # -----------------------------
    if st.button("Run AI Analysis"):

        if not (h_valid and a_valid and s_valid and p_valid):
            st.error("Please enter valid integer values within allowed ranges.")
        else:

            with st.spinner("🧠 AI analyzing behavioral patterns..."):
                time.sleep(1)

            input_dict = {
                "hours_studied": hours_studied,
                "attendance": attendance,
                "sleep_hours": sleep_hours,
                "previous_scores": previous_score
            }

            input_df = pd.DataFrame([input_dict])

            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[feature_columns]

            prob = model.predict_proba(input_df)[0][1]
            st.session_state.risk_score = int(prob * 100)

    # -----------------------------
    # Display Results
    # -----------------------------
    if st.session_state.risk_score is not None:

        risk_score = st.session_state.risk_score

        # Detect theme
        theme = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"

        # Risk Color + Label
        if risk_score <= 30:
            bar_color = "#22c55e"
            risk_label = "LOW RISK"
        elif risk_score <= 60:
            bar_color = "#facc15"
            risk_label = "MODERATE RISK"
        else:
            bar_color = "#ef4444"
            risk_label = "HIGH RISK"

        # -----------------------------
        # PREMIUM GAUGE
        # -----------------------------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={'font': {'size': 48}, 'suffix': "%"},
            title={'text': "<b>AI Risk Intelligence Score</b>", 'font': {'size': 20}},
            gauge={
                'shape': "angular",
                'axis': {'range': [0, 100]},
                'bar': {'color': bar_color, 'thickness': 0.35},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(34,197,94,0.2)"},
                    {'range': [30, 60], 'color': "rgba(250,204,21,0.2)"},
                    {'range': [60, 100], 'color': "rgba(239,68,68,0.2)"}
                ],
                'threshold': {
                    'line': {'color': bar_color, 'width': 6},
                    'thickness': 0.75,
                    'value': risk_score
                }
            }
        ))

        fig.update_layout(
            template=theme,
            height=420,
            margin=dict(t=60, b=0, l=20, r=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Risk Badge
        st.markdown(f"""
        <div style="
        padding:12px 20px;
        border-radius:30px;
        display:inline-block;
        font-weight:bold;
        background:{bar_color};
        color:white;
        margin-top:10px;
        font-size:16px;">
        {risk_label}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 🚀 Strategic Academic Roadmap")

        # -----------------------------
        # Roadmap
        # -----------------------------
        if risk_score <= 10:
            st.success("Excellent performance stability detected.")
            roadmap = [
                "Maintain structured study habits.",
                "Continue high attendance consistency.",
                "Engage in advanced skill development.",
                "Track monthly progress."
            ]
        elif risk_score <= 30:
            roadmap = [
                "Maintain attendance above 85%.",
                "Increase study hours slightly.",
                "Perform weekly academic review."
            ]
        elif risk_score <= 60:
            roadmap = [
                "Create structured timetable.",
                "Seek faculty mentorship.",
                "Reduce distractions."
            ]
        else:
            roadmap = [
                "Immediate academic counseling recommended.",
                "Develop supervised study plan.",
                "Track weekly improvement."
            ]

        for step in roadmap:
            st.markdown(f'<div class="result-card">• {step}</div>', unsafe_allow_html=True)





import serial
import time

try:
    arduino = serial.Serial("COM3", 9600)  # Change COM port
    time.sleep(2)
    arduino.write(str(risk_score).encode())
    arduino.close()
except:
    pass
