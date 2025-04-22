
import streamlit as st
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample training data
X_train = pd.DataFrame({
    'crown_type': ['PEKK', 'LD', 'PEKK', 'LD'],
    'gender': ['Female', 'Male', 'Female', 'Female'],
    'implant_site': ['Central', 'Lateral', 'Central', 'Central'],
    'baseline_HU': [1179, 695, 826, 954],
    'delta_HU': [-335, -126, 142, -232]
})
y_train = [0.6, 0.0, 1.9, 0.75]

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), ['crown_type', 'gender', 'implant_site']),
    ('num', StandardScaler(), ['baseline_HU', 'delta_HU'])
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

st.title("🦷 AI-Powered MBL Predictor")

st.sidebar.header("Enter Patient Features:")
crown_type = st.sidebar.selectbox("Crown Type", ['PEKK', 'LD'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
implant_site = st.sidebar.selectbox("Implant Site", ['Central', 'Lateral'])
baseline_HU = st.sidebar.slider("Baseline HU", 400, 1600, 800)
delta_HU = st.sidebar.slider("Change in HU", -500, 500, 0)

# Predict
user_input = pd.DataFrame([{
    'crown_type': crown_type,
    'gender': gender,
    'implant_site': implant_site,
    'baseline_HU': baseline_HU,
    'delta_HU': delta_HU
}])
prediction = model.predict(user_input)[0]
st.subheader(f"📈 Predicted Marginal Bone Loss: {prediction:.2f} mm")

# Explain prediction using SHAP
explainer = shap.Explainer(model.named_steps["regressor"])
transformed_input = model.named_steps["preprocessor"].transform(user_input)
shap_values = explainer(transformed_input)

st.subheader("🔍 Key Contributing Features (SHAP)")
fig = shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)
