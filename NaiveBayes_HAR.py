# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

st.set_page_config(
    page_title="Prediksi Risiko Serangan Jantung",
    page_icon="❤️",
    layout="wide"
)

@st.cache_data

def load_data():
    df = pd.read_csv("heart_attack_prediction_dataset.csv")
    cols_to_drop = ['Patient ID', 'Country', 'Continent', 'Hemisphere']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Diet'], drop_first=True)
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df.drop('Blood Pressure', axis=1, inplace=True)
    df.dropna(inplace=True)
    return df


def load_model_and_scaler():
    df = load_data()
    X = df.drop('Heart Attack Risk', axis=1)
    y = df['Heart Attack Risk']
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    model = GaussianNB()
    model.fit(X_scaled, y_resampled)
    return model, scaler, X, y

def plot_roc_curve(model, X, y, scaler):
    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {roc_auc:.2f}", line=dict(color='crimson')))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(title='Kurva ROC', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    return fig

def get_recommendation(prediction, prob):
    if prediction == 1:
        if prob > 0.8:
            return """
            ### Risiko Tinggi Serangan Jantung
            - Segera konsultasi dengan dokter
            - Perbaiki gaya hidup dan pola makan
            - Hindari stres dan olahraga berat
            """
        else:
            return """
            ### Risiko Sedang Serangan Jantung
            - Lakukan pemeriksaan lebih lanjut
            - Jaga pola makan dan aktivitas fisik
            """
    else:
        return """
        ### Risiko Rendah Serangan Jantung
        - Tetap jaga gaya hidup sehat
        - Lakukan pemeriksaan berkala
        """

# Main app logic
st.title("Dashboard Prediksi Risiko Serangan Jantung")

st.markdown("""
Aplikasi ini memprediksi kemungkinan risiko serangan jantung berdasarkan data medis menggunakan model Naive Bayes.
""")

df = load_data()
model, scaler, X, y = load_model_and_scaler()

# Statistik
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Jumlah Data", df.shape[0])
with col2:
    st.metric("Risiko Tinggi", df[df['Heart Attack Risk'] == 1].shape[0])
with col3:
    st.metric("Risiko Rendah", df[df['Heart Attack Risk'] == 0].shape[0])

st.subheader("Distribusi Risiko Serangan Jantung")
fig = px.pie(df, names='Heart Attack Risk', title='Distribusi Kelas', color_discrete_sequence=['#2ecc71', '#e74c3c'])
st.plotly_chart(fig)

st.subheader("Input Data Baru untuk Prediksi")
age = st.slider("Age", 20, 100, 50)
chol = st.slider("Cholesterol Level", 100, 400, 200)
systolic = st.slider("Systolic BP", 90, 200, 120)
diastolic = st.slider("Diastolic BP", 60, 140, 80)
smoker = st.selectbox("Do you smoke?", ['Yes', 'No'])
alcohol = st.selectbox("Do you consume alcohol?", ['Yes', 'No'])
sex = st.selectbox("Sex", ['Male', 'Female'])
diet = st.selectbox("Diet Type", ['Vegetarian', 'Non-Vegetarian'])

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

for col in X.columns:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[X.columns]
input_scaled = scaler.transform(input_data)

if st.button("Prediksi"):
    result = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1]
    st.markdown("---")
    st.subheader("Hasil Prediksi")

    if result[0] == 1:
        st.error(f"Prediksi: Risiko **Tinggi** ({prob*100:.2f}% confidence)")
    else:
        st.success(f"Prediksi: Risiko **Rendah** ({(1 - prob)*100:.2f}% confidence)")

    col1, col2 = st.columns(2)
    with col1:
        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(x=['Rendah', 'Tinggi'], y=[1 - prob, prob], marker_color=['#2ecc71', '#e74c3c']))
        bar_fig.update_layout(title='Probabilitas Prediksi', yaxis_range=[0, 1])
        st.plotly_chart(bar_fig)

    with col2:
        st.plotly_chart(plot_roc_curve(model, X, y, scaler))

    st.markdown(get_recommendation(result[0], prob))

st.markdown("---")
