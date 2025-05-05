import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder
from src.load_data import load_data
from src.preprocess import preprocess_data
from src.train_model import train_and_evaluate

# Set page config
st.set_page_config(page_title="Iris Classification", layout="centered")

# Title
st.title("🌸 Iris Classification")

# Sidebar input fields
st.header("Enter Flower Measurements")
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Load and preprocess data
@st.cache_data
def prepare_model():
    df = load_data("data/IRIS.csv")
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(df)
    model, _ = train_and_evaluate(X_train, X_test, y_train, y_test, label_encoder)
    return model, label_encoder

model, label_encoder = prepare_model()

# Predict on user input
if st.button("🔍 Predict"):
    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(user_input)
    predicted_species = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🌼 Predicted Species: **{predicted_species}**")
