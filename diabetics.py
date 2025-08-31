import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset (for training once)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details to check diabetes risk:")

# Input fields
Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
Glucose = st.number_input("Glucose", 0, 200, 120)
BloodPressure = st.number_input("Blood Pressure", 0, 122, 70)
SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
Insulin = st.number_input("Insulin", 0, 900, 79)
BMI = st.number_input("BMI", 0.0, 67.1, 25.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
Age = st.number_input("Age", 1, 120, 25)

# Prediction button
if st.button("Predict"):
    features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age]])
    features = scaler.transform(features)

    prediction = model.predict(features)
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"

    st.success(f"Prediction: {result}")
