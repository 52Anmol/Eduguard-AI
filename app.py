import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time

st.set_page_config(page_title="EduGuard AI", layout="wide")

# ----------------------------
# Session State
# ----------------------------
if "page" not in st.session_state:
    st.session_state.page = "dashboard"

if "risk_score" not in st.session_state:
    st.session_state.risk_score = None

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv("StudentPerformanceFactors.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

score_column = "exam_score"
median_score = df[score_column].median()
df["risk"] = df[score_column].apply(lambda x: 1 if x < median_score else 0)

model = joblib.load("trained_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

theme = "plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white"

# ==========================================================
# DASHBOARD
# ==========================================================
if st.session_state.page == "dashboard":

    st.title("🚀 EduGuard AI - Institutional Dashboard")

    if st.button("🔍 AI Prediction"):
        st.session_state.page = "prediction"
        st.rerun()

    total = len(df)
    at_risk = int(df["risk"].sum())
    safe = total - at_risk

    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(values=[safe, at_risk], names=["Safe", "At Risk"], hole=0.6)
        fig.update_layout(template=theme)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.histogram(df, x=score_column, nbins=30)
        fig2.update_layout(template=theme)
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# PREDICTION
# ==========================================================
elif st.session_state.page == "prediction":

    st.title("🔍 AI Risk Assessment Engine")

    if st.button("⬅ Back"):
        st.session_state.page = "dashboard"
        st.session_state.risk_score = None
        st.rerun()

    col1, col2 = st.columns(2)

    with col1:
        hours = st.number_input("Hours Studied", 0, 60, 10)
        attendance = st.number_input("Attendance %", 0, 100, 70)

    with col2:
        sleep = st.number_input("Sleep Hours", 0, 12, 7)
        previous = st.number_input("Previous Score", 0, 100, 60)

    if st.button("Run AI Analysis"):

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
        risk_score = int(prob * 100)
        st.session_state.risk_score = risk_score

    # ----------------------------
    # RESULTS
    # ----------------------------
    if st.session_state.risk_score is not None:

        risk_score = st.session_state.risk_score

        # Determine color
        if risk_score <= 30:
            color = "#22c55e"
        elif risk_score <= 60:
            color = "#facc15"
        else:
            color = "#ef4444"

        # Premium Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={'suffix': "%"},
            title={'text': "AI Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(34,197,94,0.2)"},
                    {'range': [30, 60], 'color': "rgba(250,204,21,0.2)"},
                    {'range': [60, 100], 'color': "rgba(239,68,68,0.2)"}
                ]
            }
        ))

        fig.update_layout(template=theme, height=350)
        st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # 🔥 AUTOMATIC AI OPTIMIZATION ENGINE
        # =====================================================
        st.markdown("## 🤖 AI Recommended Optimization Plan")

        # Auto improvements logic
        improved_attendance = max(attendance, 85)
        improved_hours = max(hours, 20)
        improved_sleep = max(sleep, 7)
        improved_score = max(previous, 75)

        optimized_input = {
            "hours_studied": improved_hours,
            "attendance": improved_attendance,
            "sleep_hours": improved_sleep,
            "previous_scores": improved_score
        }

        sim_df = pd.DataFrame([optimized_input])

        for col in feature_columns:
            if col not in sim_df.columns:
                sim_df[col] = 0

        sim_df = sim_df[feature_columns]

        new_prob = model.predict_proba(sim_df)[0][1]
        new_risk = int(new_prob * 100)

        st.markdown("### 📉 Animated Risk Reduction")

        placeholder = st.empty()

        for i in range(risk_score, new_risk - 1, -1):
            fig_anim = go.Figure(go.Indicator(
                mode="gauge+number",
                value=i,
                number={'suffix': "%"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            fig_anim.update_layout(template=theme, height=300)
            placeholder.plotly_chart(fig_anim, use_container_width=True)
            time.sleep(0.02)

        reduction = risk_score - new_risk

        st.success(f"AI Reduced Risk by {reduction}% through behavioral optimization.")

        colA, colB = st.columns(2)
        with colA:
            st.metric("Original Risk", f"{risk_score}%")
        with colB:
            st.metric("Optimized Risk", f"{new_risk}%", delta=f"-{reduction}%")

        st.markdown("### 📌 Recommended Improvements")

        st.write(f"- Increase Attendance to **{improved_attendance}%**")
        st.write(f"- Increase Study Hours to **{improved_hours} hours**")
        st.write(f"- Maintain Sleep at **{improved_sleep} hours**")
        st.write(f"- Improve Performance to **{improved_score}%**")
