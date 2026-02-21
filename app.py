import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Cancer Prediction App")

st.title("🧬 Breast Cancer Prediction using Gradient Boosting")

# Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train Model
model = GradientBoostingClassifier()
model.fit(X, y)

st.subheader("Enter Tumor Features")

mean_radius = st.number_input("Mean Radius", float(X.mean()[0]))
mean_texture = st.number_input("Mean Texture", float(X.mean()[1]))
mean_perimeter = st.number_input("Mean Perimeter", float(X.mean()[2]))
mean_area = st.number_input("Mean Area", float(X.mean()[3]))

# Fill remaining features with mean values
input_data = [mean_radius, mean_texture, mean_perimeter, mean_area] + list(X.mean()[4:])
input_data = [input_data]

if st.button("Predict Tumor Type"):
    prediction = model.predict(input_data)[0]
    result = "Malignant (Cancerous)" if prediction == 0 else "Benign (Non-Cancerous)"
    if prediction == 0:
        st.error(result)
    else:
        st.success(result)