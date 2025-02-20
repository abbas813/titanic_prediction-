import streamlit as st
import pickle
import numpy as np

# Load trained model & scaler
with open("logistic_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

st.title("🚢 Titanic Survival Prediction")

# User Inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, value=30.0)
embarked = st.radio("Embarked Port", ["C", "Q", "S"])

# Convert Inputs
sex = 1 if sex == "Female" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create Feature Array
features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked_Q, embarked_S]])
features = scaler.transform(features)  # Apply Standard Scaling

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(features)[0]
    prediction_prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"The passenger is likely to survive. Probability: {prediction_prob:.2f}")
    else:
        st.error(f"The passenger is unlikely to survive. Probability: {prediction_prob:.2f}")
