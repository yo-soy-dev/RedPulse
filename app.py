# import streamlit as st
# import pandas as pd
# import joblib
# import plotly.graph_objects as go

# # Load ML files
# model = joblib.load("KNN_heart.pkl")
# scaler = joblib.load("scaler.pkl")
# expected_columns = joblib.load("columns.pkl")

# st.set_page_config(
#     page_title="AI Heart Disease Dashboard",
#     page_icon="❤️",
#     layout="wide"
# )

# st.title("🏥 AI Heart Disease Prediction Dashboard")
# st.caption("Developed by Soy-Yo-Dev | Educational Medical AI System")

# st.write("---")

# # Sidebar
# st.sidebar.header("Patient Information")

# age = st.sidebar.slider("Age",18,100,40)
# sex = st.sidebar.selectbox("Sex",["M","F"])
# chest_pain = st.sidebar.selectbox("Chest Pain Type",["ATA","NAP","TA","ASY"])
# resting_bp = st.sidebar.number_input("Resting Blood Pressure",80,200,120)
# cholesterol = st.sidebar.number_input("Cholesterol",100,600,200)
# fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar >120",[0,1])
# resting_ecg = st.sidebar.selectbox("Resting ECG",["Normal","ST","LVH"])
# max_hr = st.sidebar.slider("Max Heart Rate",60,220,150)
# exercise_angina = st.sidebar.selectbox("Exercise Angina",["Y","N"])
# oldpeak = st.sidebar.slider("Oldpeak",0.0,6.0,1.0)
# st_slope = st.sidebar.selectbox("ST Slope",["Up","Flat","Down"])

# # Dashboard cards
# col1,col2,col3,col4 = st.columns(4)

# col1.metric("Age",age)
# col2.metric("Resting BP",resting_bp)
# col3.metric("Cholesterol",cholesterol)
# col4.metric("Max HR",max_hr)

# st.write("---")

# if st.button("🔍 Analyze Patient Risk"):

#     raw_input = {
#         "Age": age,
#         "RestingBP": resting_bp,
#         "Cholesterol": cholesterol,
#         "FastingBS": fasting_bs,
#         "MaxHR": max_hr,
#         "Oldpeak": oldpeak,
#         "Sex_" + sex: 1,
#         "ChestPainType_" + chest_pain: 1,
#         "RestingECG_" + resting_ecg: 1,
#         "ExerciseAngina_" + exercise_angina: 1,
#         "ST_Slope_" + st_slope: 1
#     }

#     input_df = pd.DataFrame([raw_input])

#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     input_df = input_df[expected_columns]

#     scaled_input = scaler.transform(input_df)

#     prediction = model.predict(scaled_input)[0]
#     probability = model.predict_proba(scaled_input)[0][1]*100

#     st.write("## AI Diagnosis Result")

#     if prediction == 1:
#         st.error(f"⚠️ High Risk of Heart Disease ({probability:.2f}%)")
#     else:
#         st.success(f"✅ Low Risk of Heart Disease ({probability:.2f}%)")

#     st.write("---")

#     # Gauge meter
#     st.subheader("Heart Disease Risk Meter")

#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=probability,
#         title={'text': "Risk Percentage"},
#         gauge={
#             'axis': {'range': [0,100]},
#             'bar': {'color': "red"},
#             'steps':[
#                 {'range':[0,30],'color':"green"},
#                 {'range':[30,70],'color':"yellow"},
#                 {'range':[70,100],'color':"red"}
#             ]
#         }
#     ))

#     st.plotly_chart(fig,use_container_width=True)

#     st.write("---")

#     # Patient report
#     st.subheader("Patient Report")

#     report = pd.DataFrame({
#         "Feature":[
#             "Age","RestingBP","Cholesterol","MaxHR","Oldpeak"
#         ],
#         "Value":[
#             age,resting_bp,cholesterol,max_hr,oldpeak
#         ]
#     })

#     st.dataframe(report)

#     csv = report.to_csv(index=False).encode("utf-8")

#     st.download_button(
#         label="Download Patient Report",
#         data=csv,
#         file_name="patient_report.csv",
#         mime="text/csv"
#     )

# st.write("---")

# st.caption("⚠️ This system is for educational purposes only and should not replace professional medical diagnosis.")





import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Load ML files
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.set_page_config(
    page_title="RedPulse — Heart Risk Analyzer",
    page_icon="🫀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0a0a0f;
    }
    [data-testid="stSidebar"] {
        background-color: #0f0f1a;
        border-right: 1px solid #1e1e3f;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff4b6e, #ff8c69);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #2d2d5e;
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card .metric-label {
        color: #9ca3af;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .metric-value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .result-high {
        background: linear-gradient(135deg, #3b0a0a, #5c1a1a);
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-low {
        background: linear-gradient(135deg, #0a3b1a, #0f5c2a);
        border: 1px solid #22c55e;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
    }
    .result-prob {
        font-size: 1rem;
        color: #d1d5db;
        margin-top: 0.4rem;
    }
    .section-title {
        color: #e5e7eb;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
        border-left: 4px solid #ff4b6e;
        padding-left: 0.8rem;
    }
    .tip-box {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #9ca3af;
        font-size: 0.9rem;
        margin-top: 1rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff4b6e, #ff8c69);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        opacity: 0.88;
    }
    div[data-testid="stMetric"] {
        background: #0f0f1a;
        border: 1px solid #1e1e3f;
        border-radius: 12px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🫀 RedPulse — Heart Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Heart Risk Prediction System | RedPulse</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.markdown("### 🧑‍⚕️ Patient Information")
st.sidebar.markdown("---")

age = st.sidebar.slider("🎂 Age", 18, 100, 40)
sex = st.sidebar.selectbox("⚧ Sex", ["M", "F"])
chest_pain = st.sidebar.selectbox("💢 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
    help="ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina, ASY: Asymptomatic")
resting_bp = st.sidebar.number_input("🩺 Resting Blood Pressure", 80, 200, 120)
cholesterol = st.sidebar.number_input("🧪 Cholesterol", 100, 600, 200)
fasting_bs = st.sidebar.selectbox("🍬 Fasting Blood Sugar >120", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No")
resting_ecg = st.sidebar.selectbox("📈 Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.slider("💓 Max Heart Rate", 60, 220, 150)
exercise_angina = st.sidebar.selectbox("🏃 Exercise Angina", ["Y", "N"],
    format_func=lambda x: "Yes" if x == "Y" else "No")
oldpeak = st.sidebar.slider("📉 Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.sidebar.selectbox("📊 ST Slope", ["Up", "Flat", "Down"])

# Metric Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎂 Age", age)
col2.metric("🩺 Resting BP", f"{resting_bp} mmHg")
col3.metric("🧪 Cholesterol", f"{cholesterol} mg/dL")
col4.metric("💓 Max HR", f"{max_hr} bpm")

st.markdown("---")

# Predict Button
if st.button("🔍 Analyze Patient Risk"):

    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1] * 100

    # Result Card
    if prediction == 1:
        st.markdown(f"""
        <div class="result-high">
            <div class="result-title">⚠️ High Risk of Heart Disease</div>
            <div class="result-prob">Risk Probability: <b>{probability:.2f}%</b> — Please consult a cardiologist immediately.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-low">
            <div class="result-title">✅ Low Risk of Heart Disease</div>
            <div class="result-prob">Risk Probability: <b>{probability:.2f}%</b> — Keep maintaining a healthy lifestyle!</div>
        </div>
        """, unsafe_allow_html=True)

    # Two column layout for charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="section-title">Risk Gauge</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={'text': "Risk %", 'font': {'color': 'white'}},
            number={'font': {'color': 'white'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': 'white'},
                'bar': {'color': "#ff4b6e"},
                'bgcolor': "#1a1a2e",
                'steps': [
                    {'range': [0, 30], 'color': "#14532d"},
                    {'range': [30, 70], 'color': "#713f12"},
                    {'range': [70, 100], 'color': "#7f1d1d"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': probability
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0a0a0f",
            font={'color': 'white'},
            height=300,
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with chart_col2:
        st.markdown('<div class="section-title">Risk Breakdown</div>', unsafe_allow_html=True)
        fig_bar = go.Figure(go.Bar(
            x=["Low Risk", "High Risk"],
            y=[100 - probability, probability],
            marker_color=["#22c55e", "#ef4444"],
            text=[f"{100-probability:.1f}%", f"{probability:.1f}%"],
            textposition="outside",
            textfont=dict(color="white")
        ))
        fig_bar.update_layout(
            paper_bgcolor="#0a0a0f",
            plot_bgcolor="#0f0f1a",
            font={'color': 'white'},
            height=300,
            margin=dict(t=40, b=20),
            yaxis=dict(range=[0, 110], gridcolor="#1e1e3f"),
            xaxis=dict(gridcolor="#1e1e3f")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Patient Report
    st.markdown('<div class="section-title">📋 Patient Report</div>', unsafe_allow_html=True)

    report = pd.DataFrame({
        "Feature": ["Age", "Sex", "Chest Pain", "Resting BP", "Cholesterol",
                    "Fasting BS", "Resting ECG", "Max HR", "Exercise Angina", "Oldpeak", "ST Slope"],
        "Value": [age, sex, chest_pain, resting_bp, cholesterol,
                  "Yes" if fasting_bs == 1 else "No", resting_ecg, max_hr,
                  exercise_angina, oldpeak, st_slope],
        "Risk Level": ["⚠️ High" if prediction == 1 else "✅ Low"] * 11
    })

    st.dataframe(report, use_container_width=True, hide_index=True)

    csv = report.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Full Patient Report",
        data=csv,
        file_name="patient_report.csv",
        mime="text/csv"
    )

    # Health Tips
    st.markdown('<div class="section-title">💡 Health Tips</div>', unsafe_allow_html=True)
    if prediction == 1:
        tips = [
            "🏥 Consult a cardiologist as soon as possible.",
            "🚭 Avoid smoking and alcohol completely.",
            "🥗 Switch to a heart-healthy diet — less salt, less fat.",
            "💊 Monitor your blood pressure and cholesterol regularly.",
            "🧘 Reduce stress through meditation and light exercise."
        ]
    else:
        tips = [
            "🥦 Maintain a balanced diet rich in fruits and vegetables.",
            "🏃 Exercise at least 30 minutes daily.",
            "😴 Get 7-8 hours of quality sleep.",
            "💧 Stay hydrated — drink 8 glasses of water daily.",
            "🩺 Get regular health checkups every 6 months."
        ]
    for tip in tips:
        st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

else:
    # Default state before prediction
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #6b7280;">
        <div style="font-size: 4rem;">🫀</div>
         <div style="font-size: 1.2rem; margin-top: 1rem;">Welcome to <b style="color:#ff4b6e">RedPulse</b></div>
        <div style="font-size: 1.2rem; margin-top: 1rem;">Fill in patient details on the left sidebar</div>
        <div style="font-size: 0.9rem; margin-top: 0.5rem;">and click <b style="color:#ff4b6e">Analyze Patient Risk</b> to get AI diagnosis</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("🫀 RedPulse | ⚠️ For educational purposes only — not a medical diagnosis tool.")