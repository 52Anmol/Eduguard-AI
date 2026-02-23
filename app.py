import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import time

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="EduGuard AI",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

if "risk_score" not in st.session_state:
    st.session_state.risk_score = None

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv("StudentPerformanceFactors.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

score_column = "exam_score"
median_score = df[score_column].median()
df["risk"] = df[score_column].apply(lambda x: 1 if x < median_score else 0)

# =====================================================
# LOAD MODEL
# =====================================================
model = joblib.load("trained_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# =====================================================
# DASHBOARD PAGE
# =====================================================
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

    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            values=[safe, at_risk],
            names=["Safe", "At Risk"],
            hole=0.6
        )
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_hist = px.histogram(df, x=score_column, nbins=30)
        fig_hist.update_layout(template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

# =====================================================
# PREDICTION PAGE
# =====================================================
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

    # INPUTS
    col1, col2 = st.columns(2)

    with col1:
        hours = st.number_input("Hours Studied (0–60)", 0, 60, 10)
        attendance = st.number_input("Attendance % (0–100)", 0, 100, 70)

    with col2:
        sleep = st.number_input("Sleep Hours (0–12)", 0, 12, 7)
        previous = st.number_input("Previous Score (0–100)", 0, 100, 60)

    # =====================================================
    # RUN PREDICTION
    # =====================================================
    if st.button("Run AI Analysis"):

        with st.spinner("🧠 AI analyzing behavioral patterns..."):
            time.sleep(1)

        input_dict = {
            "hours_studied": hours,
            "attendance": attendance,
            "sleep_hours": sleep,
            "previous_scores": previous
        }

        input_df = pd.DataFrame([input_dict])

        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[feature_columns]

        prob = model.predict_proba(input_df)[0][1]
        st.session_state.risk_score = int(prob * 100)

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================
    if st.session_state.risk_score is not None:

        risk_score = st.session_state.risk_score

        # Risk color
        if risk_score <= 30:
            bar_color = "#22c55e"
            risk_label = "LOW RISK"
        elif risk_score <= 60:
            bar_color = "#facc15"
            risk_label = "MODERATE RISK"
        else:
            bar_color = "#ef4444"
            risk_label = "HIGH RISK"

        # PREMIUM GAUGE
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={'suffix': "%"},
            title={'text': "AI Risk Intelligence Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': bar_color},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(34,197,94,0.2)"},
                    {'range': [30, 60], 'color': "rgba(250,204,21,0.2)"},
                    {'range': [60, 100], 'color': "rgba(239,68,68,0.2)"}
                ]
            }
        ))

        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Risk Badge
        st.markdown(f"""
        <div style="
        padding:10px 20px;
        border-radius:25px;
        background:{bar_color};
        color:white;
        display:inline-block;
        font-weight:bold;">
        {risk_label}
        </div>
        """, unsafe_allow_html=True)

        # =====================================================
        # ROADMAP
        # =====================================================
        st.markdown("## 🚀 Strategic Academic Roadmap")

        if risk_score <= 10:
            roadmap = [
                "Maintain current disciplined routine.",
                "Continue consistent attendance.",
                "Engage in advanced skill development."
            ]
        elif risk_score <= 30:
            roadmap = [
                "Maintain attendance above 85%.",
                "Increase structured study hours.",
                "Weekly progress review."
            ]
        elif risk_score <= 60:
            roadmap = [
                "Create structured timetable.",
                "Seek faculty mentorship.",
                "Reduce distractions."
            ]
        else:
            roadmap = [
                "Immediate academic counseling.",
                "Daily supervised study plan.",
                "Weekly performance monitoring."
            ]

        for step in roadmap:
            st.write("•", step)

        # =====================================================
        # SEND TO ARDUINO
        # =====================================================
        try:
            import serial

            arduino = serial.Serial("COM3", 9600, timeout=1)  # CHANGE PORT
            time.sleep(2)
            arduino.write(f"{risk_score}\n".encode())
            arduino.close()

            st.success("Hardware Synced Successfully ✅")

        except:
            st.info("Arduino not connected or COM port incorrect.")
