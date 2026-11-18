import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "my_model.pkl"
DATA_PATH = "merged_dataset.csv"

# 2. LOAD SAVED MODEL
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model file my_model.pkl not found.")
    st.stop()

model = load_model()

# 3. LOAD DATASET (FOR INPUT RANGES)
@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None

df = load_data()

# Extract features from the model
if hasattr(model, "feature_names_in_"):
    FEATURES = list(model.feature_names_in_)
else:
    # Fallback default features from merged_dataset.csv
    FEATURES = ['StudyHours', 'Attendance', 'Resources', 'Extracurricular', 'Motivation',
                'Internet', 'Gender', 'Age', 'LearningStyle', 'OnlineCourses',
                'Discussions', 'AssignmentCompletion', 'ExamScore', 'EduTech', 'StressLevel']

# 4. CUSTOM CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(rgba(5, 15, 35, 0.91), rgba(8, 25, 55, 0.94));
    color: #ffffff;
}
h1, h2, h3, p, label { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# 5. SIDEBAR
st.sidebar.title("🎓 STUDENT AI")
page = st.sidebar.radio("NAVIGATION", ["Home", "Student Prediction", "About Project"])

# 6. HOME PAGE
if page == "Home":
    st.title("STUDENT PERFORMANCE PREDICTION")
    if df is not None:
        st.subheader("📈 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

# 7. STUDENT PREDICTION PAGE
elif page == "Student Prediction":
    st.title("🎓 STUDENT PREDICTION")
    grade_meanings = {
        0: "A Grade - Excellent Performance",
        1: "B Grade - Good Performance",
        2: "C Grade - Average Performance",
        3: "D Grade - Below Average Performance",
        4: "F Grade - Low Performance"
    }

    input_values = {}
    with st.form("prediction_form"):
        columns = st.columns(2)
        for i, feature in enumerate(FEATURES):
            with columns[i % 2]:
                if df is not None and feature in df.columns:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    input_values[feature] = st.slider(feature, min_val, max_val, mean_val)
                else:
                    input_values[feature] = st.number_input(feature, value=0.0)
        submitted = st.form_submit_button("🔮 PREDICT PERFORMANCE")

    if submitted:
        input_df = pd.DataFrame([input_values], columns=FEATURES)
        prediction = model.predict(input_df)[0]

        # Standardize representation
        grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
        predicted_class = grade_map.get(prediction, prediction)

        st.subheader("✨ Prediction Result")
        st.write(f"Predicted Class: {predicted_class}")

# 8. ABOUT PROJECT
elif page == "About Project":
    st.title("ABOUT THE PROJECT")
    st.write("Multiclass student performance classification dashboard built with Streamlit.")
