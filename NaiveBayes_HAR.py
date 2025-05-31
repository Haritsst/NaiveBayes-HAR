# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Load dataset
df = pd.read_csv("heart_attack_prediction_dataset.csv")

# Drop irrelevant columns
cols_to_drop = ['Exercise Hours Per Week', 'Previous Heart Problems', 'Medication Use',
                'Stress Level', 'Sedentary Hours Per Day', 'Income', 'BMI',
                'Physical Activity Days Per Week', 'Country', 'Continent', 'Hemisphere']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# One-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Diet'], drop_first=True)

# Split Blood Pressure
df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
df.drop('Blood Pressure', axis=1, inplace=True)

# Split features and target
X = df.drop('Heart Attack Risk', axis=1)
y = df['Heart Attack Risk']

# Handle imbalance
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train model
model = GaussianNB()
model.fit(X_scaled, y_resampled)

# Streamlit UI
st.title("Heart Attack Risk Prediction")
st.write("Input the patient's data below:")

# Input widgets
age = st.slider("Age", 20, 100, 50)
chol = st.slider("Cholesterol Level", 100, 400, 200)
systolic = st.slider("Systolic BP", 90, 200, 120)
diastolic = st.slider("Diastolic BP", 60, 140, 80)
smoker = st.selectbox("Do you smoke?", ['Yes', 'No'])
alcohol = st.selectbox("Do you consume alcohol?", ['Yes', 'No'])
sex = st.selectbox("Sex", ['Male', 'Female'])
diet = st.selectbox("Diet Type", ['Vegetarian', 'Non-Vegetarian'])

# Prepare input for prediction
input_data = pd.DataFrame([{
    'Age': age,
    'Cholesterol Level': chol,
    'Smoking': 1 if smoker == 'Yes' else 0,
    'Alcohol Consumption': 1 if alcohol == 'Yes' else 0,
    'Sex_Male': 1 if sex == 'Male' else 0,
    'Diet_Vegetarian': 1 if diet == 'Vegetarian' else 0,
    'Systolic_BP': systolic,
    'Diastolic_BP': diastolic
}])

# Ensure all model features are present
for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Reorder and scale
input_data = input_data[X.columns]
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    result = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]
    st.success(f"Predicted Risk: {'High' if result[0] == 1 else 'Low'} ({prob*100:.2f}% confidence)")
