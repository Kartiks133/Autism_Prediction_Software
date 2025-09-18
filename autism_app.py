import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Autism Prediction", page_icon="🧠", layout="centered")

# Light background with soft header
st.markdown("""
<div style="background-color:#E3F2FD;padding:25px;border-radius:15px;text-align:center;">
<h1 style="color:#1565C0;">🧠 Autism Prediction Demo</h1>
<p style="color:#0D47A1;">Fill in your details and answer the AQ-10 questions</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("autism_model.joblib")

# Columns for basic info
col1, col2, col3, col4 = st.columns(4)
age = col1.number_input("Age", min_value=1, max_value=100, value=25)
gender = col2.selectbox("Gender", ["male","female"])
jundice = col3.selectbox("Jaundice", ["yes","no"])
used_app_before = col4.selectbox("Used app before", ["yes","no"])

# AQ-10 Questions (expanded for neat display)
questions = [
    "I often notice small sounds when others do not.",
    "I usually concentrate more on the whole picture, rather than the small details.",
    "I find it easy to do more than one thing at a time.",
    "I enjoy social occasions.",
    "I find it difficult to switch between activities.",
    "I frequently get so strongly absorbed in one thing that I lose sight of other things.",
    "I find it easy to 'read between the lines' when someone is talking to me.",
    "I usually notice car number plates or similar strings of information.",
    "I enjoy meeting new people.",
    "I find it difficult to work out people's intentions."
]

responses = {}
for i, q in enumerate(questions, 1):
    with st.expander(f"Q{i}: {q}"):
        responses[f"A{i}_Score"] = st.radio("Answer", ["No", "Yes"], index=0, key=f"A{i}")

# Predict button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Predict 🧠"):
    input_data = pd.DataFrame([{
        key: 1 if value == "Yes" else 0
        for key, value in responses.items()
    }])
    input_data["age"] = age
    input_data["gender"] = gender
    input_data["jundice"] = jundice
    input_data["used_app_before"] = used_app_before

    feature_order = [
        "age", "gender", "jundice", "used_app_before",
        "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
        "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score"
    ]
    input_data = input_data[feature_order]

    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Likely Autism")
    else:
        st.success("✅ No Autism detected")
