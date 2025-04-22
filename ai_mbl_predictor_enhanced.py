
import streamlit as st
import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Custom page settings
st.set_page_config(page_title="AI-Powered MBL Predictor", page_icon="🦷")

st.image("https://upload.wikimedia.org/wikipedia/commons/8/88/Tooth_icon.png", width=60)
st.title("🦷 AI-Powered MBL Predictor")

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

# Sidebar for patient input
st.sidebar.header("Enter Patient Features:")
crown_type = st.sidebar.selectbox("Crown Type", ['PEKK', 'LD'])
gender = st.sidebar.selectbox("Gender", ['Male', 'Female'])
implant_site = st.sidebar.selectbox("Implant Site", ['Central', 'Lateral'])
baseline_HU = st.sidebar.slider("Baseline HU", 400, 1600, 800)
delta_HU = st.sidebar.slider("Change in HU", -500, 500, 0)

# Make prediction
user_input = pd.DataFrame([{
    'crown_type': crown_type,
    'gender': gender,
    'implant_site': implant_site,
    'baseline_HU': baseline_HU,
    'delta_HU': delta_HU
}])
prediction = model.predict(user_input)[0]
st.subheader(f"📈 Predicted Marginal Bone Loss: {prediction:.2f} mm")

# SHAP waterfall explanation
explainer = shap.Explainer(model.named_steps["regressor"])
transformed_input = model.named_steps["preprocessor"].transform(user_input)
shap_values = explainer(transformed_input)
st.subheader("🔍 Key Contributing Features (SHAP)")
shap.plots.waterfall(shap_values[0], show=True)

# Batch Prediction
st.header("📂 Batch Prediction (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file with patient data", type=["csv"])
if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    preds = model.predict(batch_df)
    batch_df["Predicted MBL"] = np.round(preds, 2)
    st.write(batch_df)
    csv = batch_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

# SHAP summary plot for batch mode
if uploaded_file is not None:
    st.subheader("📊 SHAP Summary Plot (Uploaded Data)")
    transformed_batch = model.named_steps["preprocessor"].transform(batch_df.drop(columns=["Predicted MBL"]))
    batch_shap_values = explainer(transformed_batch)
    shap.summary_plot(batch_shap_values, show=False)
    st.pyplot(bbox_inches="tight")
