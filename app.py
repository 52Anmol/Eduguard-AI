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
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.glow-card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    transition: 0.3s ease;
}
.glow-card:hover {
    box-shadow: 0 0 20px #38bdf8;
}
.result-card {
    padding: 15px;
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
    margin-bottom: 10px;
    border-left: 4px solid #0ea5e9;
}
.stButton>button {
    background: #0ea5e9;
    color: white;
    border-radius: 8px;
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

    # Validation Function
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
            risk_score = int(prob * 100)

            # Save to session
            st.session_state.risk_score = risk_score

    # -------------------------------------------------
    # Display Results (After Prediction)
    # -------------------------------------------------
    if st.session_state.risk_score is not None:

        risk_score = st.session_state.risk_score

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "AI Risk Score"},
            gauge={'axis': {'range': [0, 100]}}
        ))
        fig.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("## 🚀 Strategic Academic Roadmap")

        # Tier-Based Strategy
        if risk_score <= 10:

            st.success("Excellent performance stability detected. Keep maintaining this discipline.")

            roadmap = [
                "Maintain structured study habits.",
                "Continue high attendance consistency.",
                "Engage in advanced skill development.",
                "Mentor peers to reinforce learning.",
                "Track monthly progress to sustain excellence."
            ]

        elif 10 < risk_score <= 30:

            roadmap = [
                "Maintain attendance above 85%.",
                "Slightly increase weekly study hours.",
                "Review weak topics proactively.",
                "Improve sleep consistency.",
                "Perform weekly academic review."
            ]

        elif 30 < risk_score <= 60:

            roadmap = [
                "Create structured academic timetable.",
                "Seek faculty mentorship.",
                "Reduce distractions.",
                "Strengthen fundamental concepts."
            ]

        else:

            roadmap = [
                "Immediate academic counseling recommended.",
                "Develop daily supervised study plan.",
                "Prioritize weakest subjects first.",
                "Minimize non-academic distractions.",
                "Track weekly performance improvement."
            ]

        for step in roadmap:
            st.markdown(f'<div class="result-card">• {step}</div>', unsafe_allow_html=True)
