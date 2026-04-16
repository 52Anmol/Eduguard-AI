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

df = pd.read_csv("StudentPerformanceFactors.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

score_column = "exam_score"
median_score = df[score_column].median()
df["risk"] = df[score_column].apply(lambda x: 1 if x < median_score else 0)


model = joblib.load("trained_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


if st.session_state.page == "dashboard":

    
    st.markdown("""
    <style>

    /* Glass Card */
    .glass-card {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0,255,255,0.15);
        transition: all 0.4s ease-in-out;
    }

    .glass-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 0 40px rgba(0,255,255,0.55);
    }

    /* Metric Number */
    .metric-value {
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(90deg, #00f5ff, #00ff95);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: glow 2s infinite alternate;
    }

    @keyframes glow {
        from { text-shadow: 0 0 10px #00f5ff; }
        to { text-shadow: 0 0 25px #00ff95; }
    }

    </style>
    """, unsafe_allow_html=True)

    
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

    
    colA, colB, colC = st.columns(3)

    with colA:
        st.markdown(f"""
        <div class="glass-card">
            <div>Total Students</div>
            <div class="metric-value">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown(f"""
        <div class="glass-card">
            <div>Safe Students</div>
            <div class="metric-value">{safe}</div>
        </div>
        """, unsafe_allow_html=True)

    with colC:
        st.markdown(f"""
        <div class="glass-card">
            <div>At Risk Students</div>
            <div class="metric-value">{at_risk}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    
    col1, col2 = st.columns(2)

    with col1:
        fig_pie = px.pie(
            values=[safe, at_risk],
            names=["Safe", "At Risk"],
            hole=0.6
        )
        fig_pie.update_layout(
            template="plotly_dark",
            transition_duration=800
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_hist = px.histogram(
            df,
            x=score_column,
            nbins=30
        )
        fig_hist.update_layout(
            template="plotly_dark",
            transition_duration=800
        )
        st.plotly_chart(fig_hist, use_container_width=True)


elif st.session_state.page == "prediction":

    col1, col2 = st.columns([8, 2])

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
