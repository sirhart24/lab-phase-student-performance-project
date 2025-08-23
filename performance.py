import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
try:
    model = joblib.load("student_performance_model.pkl")
except FileNotFoundError:
    st.error("Error: The model file 'students_performance_model.joblib' was not found.")
    st.stop()

st.title("🎓 Student Exam Performance Prediction")
st.markdown("Enter student information to predict their average exam score.")

# Get user inputs from the Streamlit UI
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_education = st.selectbox(
    "Parental Education Level",
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
)
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

# Button to trigger prediction
if st.button("Predict Average Score"):
    # --- Data Preprocessing to Match the Model's Training Data ---
    # Create a DataFrame with the same columns as the training data, initialized to zero.
    # This is critical to ensure the feature order and names are correct.
    # The columns must match the one-hot encoded columns created during training.

    # These are the columns the model expects after one-hot encoding and drop_first=True
    model_columns = ['gender_male', 'race/ethnicity_group B', 'race/ethnicity_group C',
                     'race/ethnicity_group D', 'race/ethnicity_group E',
                     'parental level of education_bachelor\'s degree',
                     'parental level of education_high school',
                     'parental level of education_master\'s degree',
                     'parental level of education_some college',
                     'parental level of education_some high school',
                     'lunch_standard', 'test preparation course_none']

    # Create an empty DataFrame with the correct columns and a single row
    input_data = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

    # Set the appropriate one-hot encoded columns to 1 based on user selections
    if gender == "male":
        input_data['gender_male'] = 1

    if race != "group A":
        input_data[f'race/ethnicity_{race}'] = 1

    if parent_education != "associate's degree":
        # The 'associate\'s degree' is the reference category due to drop_first=True
        # We handle this with an if statement to avoid errors.
        if parent_education in ['bachelor\'s degree', 'high school', 'master\'s degree', 'some college',
                                'some high school']:
            input_data[f'parental level of education_{parent_education}'] = 1

    if lunch == "standard":
        input_data['lunch_standard'] = 1

    if test_prep == "none":
        input_data['test preparation course_none'] = 1

    # Predict the average score using the preprocessed data
    prediction = model.predict(input_data)

    # Display the prediction to the user
    st.success(f"📊 Predicted Average Score: {prediction[0]:.2f}")
