import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Load the saved artifacts
# -------------------------------
artifact_dir = "diabetes_app_artifacts"
model_path = os.path.join(artifact_dir, 'xgb_model.joblib')
preprocessor_path = os.path.join(artifact_dir, 'preprocessor.joblib')

# Load model + preprocessor
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

# -------------------------------
# App Title
# -------------------------------
st.title('AI-Driven Diabetes Risk Profiling and Stratification')
st.write("Input your health indicators to get your predicted diabetes risk, top influencing factors, complication risk tier, and personalized recommendations.")

# -------------------------------
# Feature groups
# -------------------------------
binary_cols = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare',
    'NoDocbcCost', 'DiffWalk', 'Sex'
]
ordinal_cols = ['GenHlth', 'Education', 'Income', 'Age']
continuous_cols = ['BMI', 'MentHlth', 'PhysHlth']

# -------------------------------
# Collect Inputs
# -------------------------------
st.header("User Health Indicators")

# Binary inputs
binary_inputs = {}
st.subheader("Binary Indicators")
for col in binary_cols:
    if col == 'Sex':
        binary_inputs[col] = st.selectbox(
            f"{col} (0: Female, 1: Male)",
            [0.0, 1.0],
            format_func=lambda x: 'Female' if x == 0.0 else 'Male',
            key=f"{col}_selectbox"
        )
    else:
        binary_inputs[col] = st.selectbox(
            f"{col} (0: No, 1: Yes)",
            [0.0, 1.0],
            format_func=lambda x: 'No' if x == 0.0 else 'Yes',
            key=f"{col}_selectbox"
        )

# Ordinal inputs
ordinal_inputs = {}
st.subheader("Ordinal Indicators")
ordinal_inputs['GenHlth'] = st.selectbox(
    "General Health (1: Excellent, 5: Poor)", 
    list(range(1, 6)), 
    format_func=lambda x: ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'][x-1]
)
ordinal_inputs['Education'] = st.selectbox(
    "Education Level", 
    list(range(1, 7)), 
    format_func=lambda x: [
        'Never attended school/kindergarten',
        'Grades 1-8',
        'Grades 9-11',
        'High school graduate',
        'Some college/technical school',
        'College graduate'
    ][x-1]
)
ordinal_inputs['Income'] = st.selectbox(
    "Income Level", 
    list(range(1, 9)), 
    format_func=lambda x: [
        '< $10k', '$10-15k', '$15-20k', '$20-25k',
        '$25-35k', '$35-50k', '$50-75k', '$75k+'
    ][x-1]
)
ordinal_inputs['Age'] = st.selectbox(
    "Age Group", 
    list(range(1, 14)), 
    format_func=lambda x: f"{17 + x*5}-{23 + x*5}" if x < 13 else "80+"
)

# Continuous inputs
continuous_inputs = {}
st.subheader("Continuous Indicators")
continuous_inputs['BMI'] = st.number_input("BMI", min_value=12.0, max_value=98.0, value=25.0, step=0.1)
continuous_inputs['MentHlth'] = st.number_input("Days of poor mental health (last 30 days)", min_value=0.0, max_value=30.0, value=0.0, step=1.0)
continuous_inputs['PhysHlth'] = st.number_input("Days of poor physical health (last 30 days)", min_value=0.0, max_value=30.0, value=0.0, step=1.0)

# -------------------------------
# Prediction Section
# -------------------------------
st.header("Predict and Analyze")
predict_button = st.button('Predict Risk and Get Recommendations')

# Placeholders for output
st.header("Results")
probability_placeholder = st.empty()
shap_placeholder = st.empty()
risk_tier_placeholder = st.empty()
recommendations_placeholder = st.empty()
cluster_placeholder = st.empty()

# Risk thresholds (adjusted to probability scale)
low_risk_threshold = 0.3
moderate_risk_threshold = 0.6

# Personalized recommendations
personalized_recommendations = {
    'Sex': 'Consider targeted health screening based on demographic risk.',
    'BMI': 'Focus on weight management through diet and exercise.',
    'HighBP': 'Work with a healthcare provider to manage blood pressure.',
    'Income': 'Explore resources for affordable healthy food and healthcare access.',
    'Smoker': 'Seek support for smoking cessation.',
    'Education': 'Increase awareness of diabetes risk factors and healthy lifestyle choices.',
    'DiffWalk': 'Incorporate regular, gentle physical activity tailored to mobility levels.',
    'Stroke': 'Focus on managing cardiovascular health to prevent further complications.',
    'Fruits': 'Increase daily fruit intake as part of a balanced diet.',
    'HighChol': 'Work with a healthcare provider to manage cholesterol levels.'
}

if predict_button:
    # Merge inputs
    input_data = {**binary_inputs, **ordinal_inputs, **continuous_inputs}
    input_df = pd.DataFrame([input_data])

    # Preprocess
    X_processed = preprocessor.transform(input_df)

    # Predict
    prob = model.predict_proba(X_processed)[0][1]
    probability_placeholder.write(f"**Predicted Probability of Diabetes:** {prob:.2f}")

    # Risk Tier
    if prob < low_risk_threshold:
        risk_tier = "Low"
    elif prob < moderate_risk_threshold:
        risk_tier = "Moderate"
    else:
        risk_tier = "High"
    risk_tier_placeholder.write(f"**Risk Tier:** {risk_tier}")

    # Recommendations
    recs = []
    for col in input_data:
        if col in personalized_recommendations:
            recs.append(f"- {personalized_recommendations[col]}")
    recommendations_placeholder.markdown("\n".join(recs))

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)

    st.subheader("Feature Importance (SHAP)")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    shap.summary_plot(shap_values, input_df, show=False, plot_type="bar")
    st.pyplot(fig)
    plt.close(fig)